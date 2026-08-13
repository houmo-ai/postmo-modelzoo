# Copyright (c) 2026 HOUMO AI
#
# File: sam3_engine.py
# Description:
#   SAM3 engine implementation with tcim_lite HMM backend.
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

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from hmatc.python.get_hm_devices import get_hm_devices
from sam3_processor import (
	DEFAULT_MODEL_DIR,
	TOKEN_CONTEXT_LENGTH,
	load_sam3_tokenizer,
	tokenize_prompt,
)

try:
	import tcim_lite as tcim  # type: ignore[import-not-found]
except ImportError:
	tcim = None


HMM_MODEL_NOT_LOADED = "HMM model has not been loaded"


class BaseRuntimeModel:
	"""Abstract single-model runtime that runs set_input, run, and get_output."""

	def __init__(self, output_names):
		self.output_names = output_names

	def load(self, model_path):
		raise NotImplementedError

	def set_input(self, name, data):
		raise NotImplementedError

	def run(self):
		raise NotImplementedError

	def get_output(self, name):
		raise NotImplementedError


class HMMRuntimeModel(BaseRuntimeModel):
	"""HMM runtime wrapper with interface discovery and staged profiling."""

	def __init__(self, output_names: list[str], ndevice: int = 1):
		super().__init__(output_names)
		self.ndevice = ndevice
		self.model = None
		self.input_infos = {}

	def load(self, model_path: str):
		if tcim is None:
			raise ImportError("Please install tcim_lite before using the HMM backend")
		dev_manager = tcim.runtime.DevManager(
			get_hm_devices(self.ndevice), "Xh2HalBackend"
		)
		option = tcim.runtime.Option(tcim.runtime.WeightManager(dev_manager))
		self.model = tcim.runtime.load(model_path, option=option)
		self.input_infos = {
			self.model.get_input_name(index): self.model.get_input_info(
				self.model.get_input_name(index)
			)
			for index in range(self.model.get_num_inputs())
		}
		return self

	@property
	def input_names(self) -> set[str]:
		return set(self.input_infos)

	def input_shape(self, name: str) -> tuple[int, ...]:
		return tuple(self.input_infos[name].shape)

	def set_input(self, name, data):
		if self.model is None:
			raise RuntimeError(HMM_MODEL_NOT_LOADED)
		self.model.set_input(name, data.astype(self.input_infos[name].dtype))

	def run(self):
		if self.model is None:
			raise RuntimeError(HMM_MODEL_NOT_LOADED)
		self.model.run()
		self.model.sync()

	def get_output(self, name):
		if self.model is None:
			raise RuntimeError(HMM_MODEL_NOT_LOADED)
		return self.model.get_output(name).numpy()

	def infer(
		self, inputs: dict[str, np.ndarray]
	) -> tuple[dict[str, np.ndarray], dict[str, float]]:
		if self.model is None:
			raise RuntimeError(HMM_MODEL_NOT_LOADED)
		missing = self.input_names.difference(inputs)
		if missing:
			raise ValueError(f"Missing HMM inputs: {sorted(missing)}")

		start = time.perf_counter()
		for name in self.input_infos:
			self.set_input(name, inputs[name])
		set_input_ms = (time.perf_counter() - start) * 1000.0

		start = time.perf_counter()
		self.run()
		infer_ms = (time.perf_counter() - start) * 1000.0

		start = time.perf_counter()
		outputs = {name: self.get_output(name) for name in self.output_names}
		get_output_ms = (time.perf_counter() - start) * 1000.0
		return outputs, {
			"set_input_ms": set_input_ms,
			"infer_ms": infer_ms,
			"get_output_ms": get_output_ms,
		}


class SAM3Engine:
	"""SAM3 image-grounding engine supporting old and new HMM interfaces."""

	OUTPUTS = [
		"pred_logits",
		"pred_boxes",
		"pred_boxes_xyxy",
		"pred_masks",
		"presence_logit_dec",
	]

	def __init__(
		self,
		backend: str = "hmm",
		model_dir: Path = DEFAULT_MODEL_DIR,
		ndevice: int = 1,
		threshold: float = 0.5,
		max_size_w: int = 1008,
		max_size_h: int = 1008,
	):
		self.backend = backend
		self.model_dir = Path(model_dir)
		self.ndevice = ndevice
		self.threshold = threshold
		self.max_size_w = max_size_w
		self.max_size_h = max_size_h
		self.model: HMMRuntimeModel | None = None
		self.tokenizer = None
		self.last_profile: dict[str, float] = {}

	def load(self, model_path: str):
		if self.backend not in ("hmm", "xh2"):
			raise ValueError(f"Unsupported backend: {self.backend}")
		if not Path(model_path).is_file():
			raise FileNotFoundError(f"model not found: {model_path}")
		print(f"[info] Backend: {self.backend}")
		print(f"[info] Loading model: {model_path}")
		start = time.perf_counter()
		self.model = HMMRuntimeModel(self.OUTPUTS, self.ndevice).load(model_path)
		self.last_profile = {"load_ms": (time.perf_counter() - start) * 1000.0}
		print(f"[info] HMM inputs: {sorted(self.model.input_names)}")
		self._load_tokenizer()
		return self

	def _load_tokenizer(self) -> None:
		self.tokenizer = load_sam3_tokenizer(self.model_dir)

	@staticmethod
	def _box_to_normalized_cxcywh(box_xywh, width: int, height: int) -> np.ndarray:
		x, y, box_width, box_height = box_xywh
		return np.asarray(
			[
				(x + box_width / 2.0) / width,
				(y + box_height / 2.0) / height,
				box_width / width,
				box_height / height,
			],
			dtype=np.float32,
		)

	def _add_box_inputs(
		self,
		inputs: dict[str, np.ndarray],
		image: np.ndarray,
		boxes_xywh,
		box_labels,
	) -> None:
		if self.model is None or "input_boxes" not in self.model.input_names:
			return
		num_boxes = self.model.input_shape("input_boxes")[0]
		input_boxes = np.zeros((num_boxes, 1, 4), dtype=np.float32)
		input_boxes_mask = np.ones((1, num_boxes), dtype=np.float32)
		input_boxes_label = np.zeros((num_boxes, 1), dtype=np.int64)
		boxes_xywh = boxes_xywh or []
		box_labels = box_labels or []
		valid_num = min(len(boxes_xywh), num_boxes)
		height, width = image.shape[:2]
		for index in range(valid_num):
			input_boxes[index, 0] = self._box_to_normalized_cxcywh(
				boxes_xywh[index], width, height
			)
			input_boxes_mask[0, index] = 0.0
			input_boxes_label[index, 0] = int(box_labels[index])
		inputs.update(
			{
				"input_boxes": input_boxes,
				"input_boxes_mask": input_boxes_mask,
				"input_boxes_label": input_boxes_label,
			}
		)

	def preprocess(
		self,
		image: np.ndarray,
		prompt: str,
		boxes_xywh=None,
		box_labels=None,
	) -> dict[str, np.ndarray]:
		if self.model is None:
			raise RuntimeError(HMM_MODEL_NOT_LOADED)
		if self.tokenizer is None:
			raise RuntimeError("SAM3 tokenizer has not been loaded")
		pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		transform = v2.Compose(
			[
				v2.ToDtype(torch.uint8, scale=True),
				v2.Resize(size=(self.max_size_h, self.max_size_w)),
				v2.ToDtype(torch.float32, scale=True),
				v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
			]
		)
		image_tensor = (
			transform(v2.functional.to_image(pil_image))
			.unsqueeze(0)
			.numpy()
			.astype(np.float32)
		)
		token_ids = tokenize_prompt(self.tokenizer, prompt)
		token_valid_mask = (token_ids != 0).astype(np.float32)
		inputs = {
			"image": image_tensor,
			"token_ids": token_ids,
			"token_valid_mask": token_valid_mask,
		}
		if "prompt_mask" in self.model.input_names:
			prompt_shape = self.model.input_shape("prompt_mask")
			prompt_mask = np.ones(prompt_shape, dtype=np.float32)
			prompt_mask[:, :TOKEN_CONTEXT_LENGTH] = 1.0 - token_valid_mask
			inputs["prompt_mask"] = prompt_mask
		self._add_box_inputs(inputs, image, boxes_xywh, box_labels)
		if (
			"prompt_mask" in inputs
			and "input_boxes" in self.model.input_names
			and boxes_xywh
		):
			valid_num = min(
				len(boxes_xywh), self.model.input_shape("input_boxes")[0]
			)
			geometry_end = min(
				TOKEN_CONTEXT_LENGTH + valid_num + 2,
				inputs["prompt_mask"].shape[1],
			)
			inputs["prompt_mask"][:, TOKEN_CONTEXT_LENGTH:geometry_end] = 0.0
		return inputs

	def infer_raw(
		self,
		image: np.ndarray,
		prompt: str,
		boxes_xywh=None,
		box_labels=None,
	) -> dict[str, np.ndarray]:
		if self.model is None:
			raise RuntimeError(HMM_MODEL_NOT_LOADED)
		total_start = time.perf_counter()
		start = time.perf_counter()
		inputs = self.preprocess(image, prompt, boxes_xywh, box_labels)
		preprocess_ms = (time.perf_counter() - start) * 1000.0
		outputs, runtime_profile = self.model.infer(inputs)
		self.last_profile = {
			**self.last_profile,
			"preprocess_ms": preprocess_ms,
			**runtime_profile,
			"total_ms": (time.perf_counter() - total_start) * 1000.0,
		}
		return outputs

	def postprocess(self, outputs, orig_height: int, orig_width: int):
		logits = outputs["pred_logits"][0, :, 0].astype(np.float32)
		presence = outputs["presence_logit_dec"].reshape(-1).astype(np.float32)[0]
		scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
		presence_score = 1.0 / (1.0 + np.exp(-np.clip(presence, -20.0, 20.0)))
		scores *= presence_score
		boxes = outputs["pred_boxes_xyxy"][0].astype(np.float32)
		masks = outputs["pred_masks"][0].astype(np.float32)
		indices = np.nonzero(scores >= self.threshold)[0]
		if len(indices) == 0:
			indices = np.argsort(scores)[-1:]
		results = []
		for index in indices:
			mask = cv2.resize(
				masks[index], (orig_width, orig_height), interpolation=cv2.INTER_LINEAR
			)
			mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -20.0, 20.0)))
			box = boxes[index].copy()
			if np.nanmax(np.abs(box)) <= 2.0:
				box[[0, 2]] *= orig_width
				box[[1, 3]] *= orig_height
			results.append(
				{
					"query_id": int(index),
					"score": float(scores[index]),
					"box": box,
					"mask": mask,
				}
			)
		return results

	def infer(self, image, prompt, boxes_xywh=None, box_labels=None):
		orig_height, orig_width = image.shape[:2]
		outputs = self.infer_raw(image, prompt, boxes_xywh, box_labels)
		return self.postprocess(outputs, orig_height, orig_width)

	def benchmark(
		self,
		image: np.ndarray,
		prompt: str,
		warmup: int = 1,
		repeat: int = 10,
	) -> dict[str, float]:
		"""Benchmark HMM input setup, inference, and output retrieval."""
		if self.model is None:
			raise RuntimeError(HMM_MODEL_NOT_LOADED)
		if warmup < 0 or repeat < 1:
			raise ValueError("warmup must be >= 0 and repeat must be >= 1")
		inputs = self.preprocess(image, prompt)
		for _ in range(warmup):
			self.model.infer(inputs)
		profiles = [self.model.infer(inputs)[1] for _ in range(repeat)]
		result = {
			name: sum(profile[name] for profile in profiles) / repeat
			for name in ("set_input_ms", "infer_ms", "get_output_ms")
		}
		result["total_ms"] = sum(result.values())
		result["fps"] = 1000.0 / result["total_ms"] if result["total_ms"] else 0.0
		return result
