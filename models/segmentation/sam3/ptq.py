# Copyright (c) 2026 HOUMO AI
#
# File: ptq.py
# Description:
#   SAM3 export tool for loading a local Transformers checkpoint, exporting
#   and simplifying ONNX, and converting the model to HMONNX.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import copy
import gc
import importlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import helper, numpy_helper

from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import first_not_none, get_model_configs
from xhquant.api import (
	DeviceType,
	HMONNXGoldenInference,
	QuantScheme,
	convert_onnx_to_hmonnx,
	create_quant_config,
	get_root_logger,
	xhquant_init,
)
from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx

THIS_DIR = Path(__file__).resolve().parent
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = THIS_DIR / "config.yml"
DEFAULT_WORK_DIR = THIS_DIR / "work_dirs"
DEFAULT_OUT_DIR = THIS_DIR / "output" / HOUMO_TARGET / "hmquant"
MIN_EXPORT_GPU_MEMORY_GB = 20
ONNX_IR_VERSION = 8
ONNX_OPSET_VERSION = 18
TOKEN_CONTEXT_LENGTH = 32
INPUT_NAMES = [
	"image",
	"token_ids",
	"token_valid_mask",
	"input_boxes",
	"input_boxes_mask",
	"input_boxes_label",
]
OUTPUT_NAMES = [
	"pred_logits",
	"pred_boxes",
	"pred_boxes_xyxy",
	"pred_masks",
	"presence_logit_dec",
]


def get_default_model_dir(model_config: dict) -> str:
	repo_ids = model_config.get("modelscope_repo", [])
	if repo_ids:
		return repo_ids[0].rsplit("/", maxsplit=1)[-1]
	return "sam3"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yml")
	parser.add_argument("--model", type=Path, default=None, help="local ModelScope model directory")
	parser.add_argument("--model_name", type=str, default=None, help="output model name")
	parser.add_argument("--model_size", type=str, default=None, help="model size")
	parser.add_argument("--work_dir", type=Path, default=DEFAULT_WORK_DIR, help="working directory")
	parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
	parser.add_argument("--quant_type", type=str, default=None, help="quantization type")
	parser.add_argument("--num_boxes", type=int, default=None, help="maximum number of box prompts")
	parser.add_argument("--max_size_w", type=int, default=None, help="maximum image width")
	parser.add_argument("--max_size_h", type=int, default=None, help="maximum image height")
	parser.add_argument("--dump_golden", action="store_true", help="dump HMONNX golden data on GPU")
	parser.add_argument("--debug", action="store_true", help="debug mode")
	args = parser.parse_args()

	default_model_size, default_model_name, model_configs = get_model_configs(
		str(args.config)
	)
	args.model_name = first_not_none(args.model_name, default_model_name)
	args.model_size = first_not_none(args.model_size, default_model_size)
	model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
	args.model = first_not_none(
		args.model, THIS_DIR / get_default_model_dir(model_config)
	)
	args.quant_type = first_not_none(
		args.quant_type, model_config.get("quant_type", "w8a8")
	)
	args.num_boxes = first_not_none(args.num_boxes, model_config.get("num_boxes", 8))
	args.max_size_w = first_not_none(args.max_size_w, model_config.get("max_size_w", 1008))
	args.max_size_h = first_not_none(args.max_size_h, model_config.get("max_size_h", 1008))
	if args.model_name != "sam3":
		raise ValueError(f"model_name must be sam3, got: {args.model_name}")
	if args.num_boxes <= 0:
		raise ValueError(f"num_boxes must be positive, got: {args.num_boxes}")
	return args


def _pair2(value) -> tuple[int, int]:
	if isinstance(value, int):
		return value, value
	return int(value[0]), int(value[1])


def _format_rois(input_data: torch.Tensor, boxes) -> torch.Tensor:
	if not isinstance(boxes, (list, tuple)):
		return boxes
	chunks = []
	for batch_idx, per_image in enumerate(boxes):
		if per_image.numel() == 0:
			continue
		batch_col = torch.full(
			(per_image.shape[0], 1),
			batch_idx,
			dtype=per_image.dtype,
			device=per_image.device,
		)
		chunks.append(torch.cat((batch_col, per_image), dim=1))
	return torch.cat(chunks, dim=0) if chunks else input_data.new_zeros((0, 5))


def roi_align_grid_sample_maxmask(
	input_data: torch.Tensor,
	boxes,
	output_size,
	spatial_scale: float = 1.0,
	sampling_ratio: int = -1,
	aligned: bool = False,
):
	"""Replace adaptive RoiAlign with a static GridSample implementation."""
	out_h, out_w = _pair2(output_size)
	rois = _format_rois(input_data, boxes)
	if rois.numel() == 0:
		return input_data.new_zeros((0, input_data.shape[1], out_h, out_w))
	if rois.shape[-1] != 5:
		raise ValueError(f"Expected RoIs [K,5], got {tuple(rois.shape)}")

	batch_idx = rois[:, 0].long()
	xyxy = rois[:, 1:].to(input_data.dtype)
	height, width = input_data.shape[-2:]
	offset = 0.5 if aligned else 0.0
	start_x = xyxy[:, 0] * spatial_scale - offset
	start_y = xyxy[:, 1] * spatial_scale - offset
	end_x = xyxy[:, 2] * spatial_scale - offset
	end_y = xyxy[:, 3] * spatial_scale - offset
	roi_w = end_x - start_x
	roi_h = end_y - start_y
	if aligned:
		roi_w = roi_w.clamp_min(1e-6)
		roi_h = roi_h.clamp_min(1e-6)
	else:
		roi_w = roi_w.clamp_min(1.0)
		roi_h = roi_h.clamp_min(1.0)

	bin_w = roi_w / out_w
	bin_h = roi_h / out_h
	if sampling_ratio > 0:
		grid_w = torch.full_like(roi_w, float(sampling_ratio))
		grid_h = torch.full_like(roi_h, float(sampling_ratio))
		max_grid_w = max_grid_h = int(sampling_ratio)
	else:
		grid_w = torch.ceil(roi_w / out_w).clamp_min(1.0)
		grid_h = torch.ceil(roi_h / out_h).clamp_min(1.0)
		max_grid_w = int(np.ceil(width * spatial_scale / out_w))
		max_grid_h = int(np.ceil(height * spatial_scale / out_h))

	ph = torch.arange(out_h, device=input_data.device, dtype=input_data.dtype).view(1, out_h, 1)
	pw = torch.arange(out_w, device=input_data.device, dtype=input_data.dtype).view(1, out_w, 1)
	iy = torch.arange(max_grid_h, device=input_data.device, dtype=input_data.dtype).view(1, 1, max_grid_h)
	ix = torch.arange(max_grid_w, device=input_data.device, dtype=input_data.dtype).view(1, 1, max_grid_w)
	sample_y = start_y[:, None, None] + ph * bin_h[:, None, None] + (
		iy + 0.5
	) * bin_h[:, None, None] / grid_h[:, None, None]
	sample_x = start_x[:, None, None] + pw * bin_w[:, None, None] + (
		ix + 0.5
	) * bin_w[:, None, None] / grid_w[:, None, None]
	sample_y = sample_y.reshape(-1, out_h * max_grid_h, 1).expand(
		-1, -1, out_w * max_grid_w
	)
	sample_x = sample_x.reshape(-1, 1, out_w * max_grid_w).expand(
		-1, out_h * max_grid_h, -1
	)
	grid = torch.stack(
		(
			((sample_x + 0.5) * 2.0 / width) - 1.0,
			((sample_y + 0.5) * 2.0 / height) - 1.0,
		),
		dim=-1,
	)
	sampled = torch.nn.functional.grid_sample(
		input_data.index_select(0, batch_idx),
		grid,
		mode="bilinear",
		padding_mode="border",
		align_corners=False,
	)
	sampled = sampled.reshape(
		rois.shape[0], input_data.shape[1], out_h, max_grid_h, out_w, max_grid_w
	)
	mask = (
		(iy < grid_h[:, None, None])[:, None, :, :, None, None]
		& (ix < grid_w[:, None, None])[:, None, None, None, :, :]
	).to(input_data.dtype)
	summed = (sampled * mask).sum(dim=(3, 5))
	return summed / (grid_h * grid_w)[:, None, None, None]


def patch_transformers_for_onnx() -> None:
	"""Patch Transformers operators that cannot be exported safely."""
	try:
		modeling_sam3 = importlib.import_module("transformers.models.sam3.modeling_sam3")
	except ImportError as exc:
		raise RuntimeError(
			"Transformers with native SAM3 support is required."
		) from exc
	modeling_sam3.torchvision.ops.roi_align = roi_align_grid_sample_maxmask

	original_bidirectional_mask = modeling_sam3.create_bidirectional_mask

	def create_bidirectional_mask_onnx_safe(
		config,
		inputs_embeds,
		attention_mask,
		past_key_values=None,
		encoder_hidden_states=None,
		**kwargs,
	):
		if not torch.onnx.is_in_onnx_export():
			return original_bidirectional_mask(
				config=config,
				inputs_embeds=inputs_embeds,
				attention_mask=attention_mask,
				past_key_values=past_key_values,
				encoder_hidden_states=encoder_hidden_states,
				**kwargs,
			)
		if attention_mask is None:
			return None
		query_len = inputs_embeds.shape[1]
		key_len = (
			encoder_hidden_states.shape[1]
			if encoder_hidden_states is not None
			else inputs_embeds.shape[1]
		)
		min_value = torch.finfo(inputs_embeds.dtype).min
		additive = (1.0 - attention_mask.to(inputs_embeds.dtype)) * min_value
		return additive[:, None, None, :].expand(-1, 1, query_len, key_len)

	modeling_sam3.create_bidirectional_mask = create_bidirectional_mask_onnx_safe

	modeling_clip = importlib.import_module("transformers.models.clip.modeling_clip")
	if hasattr(modeling_clip.CLIPTextModel, "_sam3_onnx_original_forward"):
		return
	modeling_clip.CLIPTextModel._sam3_onnx_original_forward = (
		modeling_clip.CLIPTextModel.forward
	)

	def clip_text_forward_onnx_safe(
		self,
		input_ids=None,
		attention_mask=None,
		position_ids=None,
		**kwargs,
	):
		if not torch.onnx.is_in_onnx_export():
			return modeling_clip.CLIPTextModel._sam3_onnx_original_forward(
				self,
				input_ids=input_ids,
				attention_mask=attention_mask,
				position_ids=position_ids,
				**kwargs,
			)
		if input_ids is None:
			raise ValueError("input_ids is required")
		input_shape = input_ids.size()
		input_ids = input_ids.view(-1, input_shape[-1])
		hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
		seq_len = hidden_states.shape[1]
		min_value = torch.finfo(hidden_states.dtype).min
		causal = torch.triu(
			torch.full(
				(seq_len, seq_len),
				min_value,
				dtype=hidden_states.dtype,
				device=hidden_states.device,
			),
			diagonal=1,
		).view(1, 1, seq_len, seq_len)
		if attention_mask is not None:
			padding = 1.0 - attention_mask.to(hidden_states.dtype)
			causal = causal + padding[:, None, None, :] * min_value
		kwargs.pop("is_causal", None)
		encoder_outputs = self.encoder(
			inputs_embeds=hidden_states,
			attention_mask=causal,
			is_causal=False,
			**kwargs,
		)
		last_hidden_state = self.final_layer_norm(encoder_outputs.last_hidden_state)
		pooled_index = (
			input_ids.to(dtype=torch.int, device=last_hidden_state.device)
			== self.eos_token_id
		).int().argmax(dim=-1)
		pooled_output = last_hidden_state[
			torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
			pooled_index,
		]
		return modeling_clip.BaseModelOutputWithPooling(
			last_hidden_state=last_hidden_state,
			pooler_output=pooled_output,
		)

	modeling_clip.CLIPTextModel.forward = clip_text_forward_onnx_safe


class TransformersSam3Wrapper(nn.Module):
	"""Expose a fixed tensor-only SAM3 image-grounding interface."""

	def __init__(self, model: nn.Module):
		super().__init__()
		self.model = model

	def forward(
		self,
		image: torch.Tensor,
		token_ids: torch.Tensor,
		token_valid_mask: torch.Tensor,
		input_boxes: torch.Tensor,
		input_boxes_mask: torch.Tensor,
		input_boxes_label: torch.Tensor,
	) -> Tuple[torch.Tensor, ...]:
		boxes = input_boxes.transpose(0, 1).contiguous()
		labels = input_boxes_label.transpose(0, 1).contiguous().long()
		valid = input_boxes_mask <= 0.5
		labels = torch.where(valid, labels, torch.full_like(labels, -10))
		outputs = self.model(
			pixel_values=image,
			input_ids=token_ids.long(),
			attention_mask=token_valid_mask > 0.5,
			input_boxes=boxes,
			input_boxes_labels=labels,
			return_dict=True,
		)
		pred_boxes_xyxy = outputs.pred_boxes
		x1, y1, x2, y2 = pred_boxes_xyxy.unbind(-1)
		pred_boxes = torch.stack(
			((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1
		)
		return (
			outputs.pred_logits.unsqueeze(-1),
			pred_boxes,
			pred_boxes_xyxy,
			outputs.pred_masks,
			outputs.presence_logits,
		)


def select_export_device(logger) -> torch.device:
	if not torch.cuda.is_available():
		logger.info("CUDA is unavailable; exporting ONNX on CPU.")
		return torch.device("cpu")

	best_device = None
	best_free_bytes = -1
	for device_index in range(torch.cuda.device_count()):
		try:
			free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
		except RuntimeError as exc:
			logger.warning(f"Cannot query cuda:{device_index}: {exc}")
			continue
		logger.info(
			f"cuda:{device_index} {torch.cuda.get_device_name(device_index)}: "
			f"free={free_bytes / 2**30:.2f} GiB, total={total_bytes / 2**30:.2f} GiB"
		)
		if free_bytes > best_free_bytes:
			best_device = torch.device(f"cuda:{device_index}")
			best_free_bytes = free_bytes

	minimum_bytes = MIN_EXPORT_GPU_MEMORY_GB * 2**30
	if best_device is None or best_free_bytes < minimum_bytes:
		logger.info(
			f"No GPU has at least {MIN_EXPORT_GPU_MEMORY_GB} GiB free memory; "
			"exporting ONNX on CPU."
		)
		return torch.device("cpu")
	logger.info(f"Exporting ONNX on {best_device}.")
	return best_device


def load_model(model_dir: Path, device: torch.device):
	try:
		from transformers import Sam3Model, Sam3Processor
	except ImportError as exc:
		raise RuntimeError(
			"Transformers with Sam3Model and Sam3Processor support is required."
		) from exc
	if not model_dir.is_dir():
		raise FileNotFoundError(f"Missing local SAM3 model directory: {model_dir}")

	patch_transformers_for_onnx()
	processor = Sam3Processor.from_pretrained(
		str(model_dir), local_files_only=True, trust_remote_code=False
	)
	model = Sam3Model.from_pretrained(
		str(model_dir),
		local_files_only=True,
		trust_remote_code=False,
		use_safetensors=True,
		dtype=torch.float32,
		attn_implementation="sdpa",
	)
	model.text_encoder.set_attn_implementation("eager")
	model.to(device).eval()
	return model, processor


def make_inputs(
	processor,
	input_height: int,
	input_width: int,
	num_boxes: int,
	device: torch.device,
) -> tuple[torch.Tensor, ...]:
	image = torch.randn(
		1, 3, input_height, input_width, dtype=torch.float32, device=device
	).clamp_(-1.0, 1.0)
	token_ids = processor.tokenizer(
		["shoe"],
		padding="max_length",
		max_length=TOKEN_CONTEXT_LENGTH,
		return_tensors="pt",
	)["input_ids"].to(device)
	token_valid_mask = token_ids.ne(0).to(torch.float32)
	input_boxes = torch.zeros((num_boxes, 1, 4), dtype=torch.float32, device=device)
	input_boxes_mask = torch.ones((1, num_boxes), dtype=torch.float32, device=device)
	input_boxes_label = torch.zeros((num_boxes, 1), dtype=torch.int64, device=device)
	input_boxes[0, 0] = torch.tensor(
		[0.535, 0.47, 0.11, 0.36], dtype=torch.float32, device=device
	)
	input_boxes_mask[0, 0] = 0.0
	input_boxes_label[0, 0] = 1
	return (
		image,
		token_ids,
		token_valid_mask,
		input_boxes,
		input_boxes_mask,
		input_boxes_label,
	)


def _remove_model_files(model_path: Path) -> None:
	if model_path.exists():
		model_path.unlink()
	data_paths = (
		model_path.with_suffix(model_path.suffix + ".data"),
		model_path.with_name(f"{model_path.stem}_external_data"),
	)
	for data_path in data_paths:
		if data_path.exists():
			data_path.unlink()


def save_external_onnx(model: onnx.ModelProto, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	_remove_model_files(output_path)
	data_path = output_path.with_suffix(output_path.suffix + ".data")
	onnx.save_model(
		model,
		str(output_path),
		save_as_external_data=True,
		all_tensors_to_one_file=True,
		location=data_path.name,
		size_threshold=1024,
		convert_attribute=False,
	)


@contextmanager
def limit_graphsurgeon_ir_version(max_ir_version: int):
	"""Limit temporary ONNX models exported during constant folding."""
	module_names = (
		"onnx_graphsurgeon.exporters.onnx_exporter",
		"onnx_graphsurgeon.ir.graph",
		"onnxslim.third_party.onnx_graphsurgeon.exporters.onnx_exporter",
		"onnxslim.third_party.onnx_graphsurgeon.ir.graph",
	)
	patched_modules = []
	for module_name in module_names:
		try:
			graph_module = importlib.import_module(module_name)
		except ModuleNotFoundError:
			continue
		if not hasattr(graph_module, "export_onnx"):
			continue
		original_export_onnx = graph_module.export_onnx

		def export_onnx_with_compatible_ir(*args, _export=original_export_onnx, **kwargs):
			model = _export(*args, **kwargs)
			model.ir_version = min(model.ir_version, max_ir_version)
			return model

		graph_module.export_onnx = export_onnx_with_compatible_ir
		patched_modules.append((graph_module, original_export_onnx))
	try:
		yield
	finally:
		for graph_module, original_export_onnx in patched_modules:
			graph_module.export_onnx = original_export_onnx


def _resolve_constant_bool(
	name: str,
	initializer_by_name: dict,
	producer_by_output: dict,
	visited=None,
):
	visited = set() if visited is None else visited
	if name in visited:
		return None
	visited.add(name)
	if name in initializer_by_name:
		value = numpy_helper.to_array(initializer_by_name[name])
		return bool(value.reshape(-1)[0]) if value.size == 1 else None
	producer = producer_by_output.get(name)
	if producer is None:
		return None
	if producer.op_type == "Identity" and producer.input:
		return _resolve_constant_bool(
			producer.input[0], initializer_by_name, producer_by_output, visited
		)
	if producer.op_type == "Constant" and producer.attribute:
		attribute = producer.attribute[0]
		if attribute.type == onnx.AttributeProto.TENSOR:
			value = numpy_helper.to_array(attribute.t)
			return bool(value.reshape(-1)[0]) if value.size == 1 else None
	return None


def _inline_if_branch(node, condition: bool, folded: int):
	branch_name = "then_branch" if condition else "else_branch"
	branch = next(
		(attribute.g for attribute in node.attribute if attribute.name == branch_name),
		None,
	)
	if branch is None or len(branch.output) != len(node.output):
		return None
	output_map = {
		branch_output.name: graph_output
		for branch_output, graph_output in zip(branch.output, node.output)
	}
	new_nodes = []
	produced = set()
	for branch_node in branch.node:
		cloned = copy.deepcopy(branch_node)
		cloned.input[:] = [output_map.get(value, value) for value in cloned.input]
		cloned.output[:] = [output_map.get(value, value) for value in cloned.output]
		produced.update(value for value in cloned.output if value)
		new_nodes.append(cloned)
	for branch_output, graph_output in output_map.items():
		if graph_output not in produced:
			new_nodes.append(
				helper.make_node(
					"Identity",
					[branch_output],
					[graph_output],
					name=f"{node.name}_folded_identity_{folded}",
				)
			)
	return new_nodes


def fold_constant_ifs(model: onnx.ModelProto) -> int:
	"""Inline constant ONNX If branches before large-model simplification."""
	graph = model.graph
	initializer_by_name = {value.name: value for value in graph.initializer}
	producer_by_output = {
		output: node for node in graph.node for output in node.output if output
	}
	new_nodes = []
	folded = 0
	for node in graph.node:
		if node.op_type != "If":
			new_nodes.append(node)
			continue
		condition = _resolve_constant_bool(
			node.input[0], initializer_by_name, producer_by_output
		)
		if condition is None:
			new_nodes.append(node)
			continue
		inlined_nodes = _inline_if_branch(node, condition, folded)
		if inlined_nodes is None:
			new_nodes.append(node)
			continue
		new_nodes.extend(inlined_nodes)
		folded += 1
	if folded:
		del graph.node[:]
		graph.node.extend(new_nodes)
	return folded


class Sam3OnnxExporter:
	"""Load, wrap, export and simplify a local Transformers SAM3 model."""

	def __init__(
		self,
		model_dir: Path,
		input_height: int,
		input_width: int,
		num_boxes: int,
		device: torch.device,
	):
		self.model_dir = model_dir
		self.input_height = input_height
		self.input_width = input_width
		self.num_boxes = num_boxes
		self.device = device

	def export(self, simplified_path: Path) -> tuple[torch.Tensor, ...]:
		_remove_model_files(simplified_path)
		model, processor = load_model(self.model_dir, self.device)
		wrapper = TransformersSam3Wrapper(model).to(self.device).eval()
		inputs = make_inputs(
			processor,
			self.input_height,
			self.input_width,
			self.num_boxes,
			self.device,
		)
		with tempfile.TemporaryDirectory(prefix="sam3_export_", dir=simplified_path.parent) as tmp:
			raw_path = Path(tmp) / "sam3_raw.onnx"
			with torch.inference_mode():
				torch.onnx.export(
					wrapper,
					inputs,
					str(raw_path),
					input_names=INPUT_NAMES,
					output_names=OUTPUT_NAMES,
					opset_version=ONNX_OPSET_VERSION,
					do_constant_folding=True,
					external_data=True,
					dynamic_axes=None,
				)
			raw_model = onnx.load(str(raw_path), load_external_data=True)
			raw_model.ir_version = ONNX_IR_VERSION
			# The simplifier performs topological sorting itself, but its generic
			# constant folding cannot safely consume these exported If subgraphs.
			fold_constant_ifs(raw_model)
			with limit_graphsurgeon_ir_version(ONNX_IR_VERSION):
				simplified_model, checked = simplify_large_onnx(raw_model)
			if not checked:
				raise RuntimeError("xhquant large-model ONNX simplification failed")
			simplified_model.ir_version = ONNX_IR_VERSION
			save_external_onnx(simplified_model, simplified_path)

		cpu_inputs = tuple(value.detach().cpu() for value in inputs)
		del wrapper, model, processor, inputs
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		return cpu_inputs


def export_hmonnx(
	onnx_path: Path,
	hmonnx_path: Path,
	input_tensors: tuple[torch.Tensor, ...],
	quant_type: str,
) -> Path:
	quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
	quant_config = create_quant_config(quant_scheme)
	quant_config.ops_cfg["Sign"] = dict(force_fp32=True)
	_remove_model_files(hmonnx_path)
	convert_onnx_to_hmonnx(
		str(onnx_path),
		input_tensors,
		device_type=DeviceType.XH2a,
		out_hmonnx_file=str(hmonnx_path),
		quant_config=quant_config,
		input_names=INPUT_NAMES,
		output_names=OUTPUT_NAMES,
	)
	return hmonnx_path


def dump_golden(
	hmonnx_path: Path,
	input_tensors: tuple[torch.Tensor, ...],
	output_dir: Path,
) -> None:
	if not torch.cuda.is_available():
		raise RuntimeError("dump_golden requires a CUDA GPU")
	device = max(
		(torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())),
		key=lambda value: torch.cuda.mem_get_info(value)[0],
	)
	output_dir.mkdir(parents=True, exist_ok=True)
	session = HMONNXGoldenInference(str(hmonnx_path))
	session.to(device)
	session.save_golden = True
	session.golden_dir = str(output_dir)
	with torch.inference_mode():
		session(*(value.to(device) for value in input_tensors))


def main() -> None:
	torch.manual_seed(42)
	args = parse_args()
	args.work_dir.mkdir(parents=True, exist_ok=True)
	args.out_dir.mkdir(parents=True, exist_ok=True)
	xhquant_init(None, debug=args.debug)
	logger = get_root_logger()
	logger.info(args)

	model_shape_name = (
		f"{args.model_name}_{args.model_size}_"
		f"{args.max_size_w}x{args.max_size_h}"
	)
	simplified_path = args.work_dir / f"{model_shape_name}_sim.onnx"
	hmonnx_path = args.out_dir / f"hmquant_{model_shape_name}_{args.quant_type}.onnx"
	device = select_export_device(logger)
	exporter = Sam3OnnxExporter(
		model_dir=args.model,
		input_height=args.max_size_h,
		input_width=args.max_size_w,
		num_boxes=args.num_boxes,
		device=device,
	)

	with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
		input_tensors = exporter.export(simplified_path)
		export_hmonnx(
			simplified_path,
			hmonnx_path,
			input_tensors,
			args.quant_type,
		)
		if args.dump_golden:
			dump_golden(hmonnx_path, input_tensors, args.work_dir / "golden")

	logger.info(f"Simplified ONNX model export to {simplified_path}")
	logger.info(f"HMONNX model export to {hmonnx_path}")
	print(f"\n=== Export completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ===")


if __name__ == "__main__":
	main()
