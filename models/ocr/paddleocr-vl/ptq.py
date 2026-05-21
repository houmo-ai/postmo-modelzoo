# Copyright (c) 2026 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# PaddleOCR-VL models using post-training quantization techniques.
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

import argparse
import os
import shutil
import os.path as osp
from pathlib import Path
import time
import psutil
import random
import numpy as np
from typing import Any, Dict

import torch
import json
import types
import huggingface_hub.dataclasses as hf_dataclasses
from PIL import Image
import gc
from loguru import logger
from tqdm import tqdm

from xhquant.api import (
    Config,
    DeviceType,
    PrecisionMode,
    ptq_quantize,
    QuantScheme,
    xhquant_init,
    convert_onnx_to_hmonnx,
    create_quant_config,
)

from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import (
    check_gpu,
    first_not_none,
    get_model_configs,
    parse_context_length,
)


def _patch_huggingface_union_validator() -> None:
    union_validator = hf_dataclasses._BASIC_TYPE_VALIDATORS.get(__import__("typing").Union)
    if union_validator is not None and types.UnionType not in hf_dataclasses._BASIC_TYPE_VALIDATORS:
        hf_dataclasses._BASIC_TYPE_VALIDATORS[types.UnionType] = union_validator


_patch_huggingface_union_validator()

from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.paddleocr_vl import (
   XHPaddleOCRVLLLMModel, XHPaddleOCRVLVisionModel
)
from image_processing import SiglipImageProcessor
from processing_paddleocr_vl import PaddleOCRVLProcessor

HOUMO_DATASETS_PATH = os.getenv(
    "HOUMO_DATASETS_PATH",
    str(Path(__file__).resolve().parents[3] / "data" / "datasets"),
)
HOUMO_PIC_PATH = os.getenv(
    "HOUMO_PIC_PATH", str(Path(__file__).resolve().parents[3] / "data" / "pic")
)
HOUMO_TARGET = os.getenv("HOUMO_TARGET", "")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "copaw-flash").upper()
    model_size = model_config.get("model_size", "9b").upper()
    return f"{model_name}-{model_size}"

class ChildProcessMemoryMonitor(ProcessMemoryMonitor):
    """Process memory monitor that optionally includes child processes."""

    def __init__(self, *args, include_children: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_children = include_children

    @property
    def process(self):
        return self._process

    def get_memory_info(self) -> Dict[str, float]:
        """Gets current memory usage information."""
        try:
            rss = self.process.memory_info().rss
            if self.include_children:
                for child in self.process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (
                        psutil.NoSuchProcess,
                        psutil.AccessDenied,
                        psutil.ZombieProcess,
                    ):
                        continue

            rss_mb = rss / (1024 * 1024)
            percent = self.process.memory_percent()
            return {"rss_mb": rss_mb, "percent": percent}
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return {"rss_mb": 0.0, "percent": 0.0}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1", ""):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cleanup_cpu():
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def save_json(file_path: Path, data: Dict[str, Any]):
    with open(file_path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=4, ensure_ascii=False)


def _unwrap_model_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(output, dict):
        for key in ("image_embeds", "last_hidden_state", "hidden_states", "logits"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported model output type: {type(output)}")


def _resolve_visual_projector(native_model):
    candidates = [
        getattr(native_model, "projector", None),
        getattr(native_model, "mlp_AR", None),
        getattr(getattr(native_model, "model", None), "projector", None),
        getattr(getattr(native_model, "model", None), "mlp_AR", None),
    ]
    for module in candidates:
        if isinstance(module, torch.nn.Module):
            return module
    raise AttributeError("Cannot find PaddleOCR-VL projector (projector/mlp_AR)")


class VisionProjectorWrapModel(torch.nn.Module):
    def __init__(
        self,
        vision_model: torch.nn.Module,
        projector: torch.nn.Module,
        patch_size: int,
        temporal_patch_size: int = 1,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.projector = projector
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size

    @property
    def dtype(self):
        return getattr(self.vision_model, "dtype", torch.float16)

    def _build_image_grid_thw(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, temporal, _, height, width = pixel_values.shape
        grid_t = max(1, temporal // max(self.temporal_patch_size, 1))
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        return torch.tensor(
            [[grid_t, grid_h, grid_w]] * batch_size,
            dtype=torch.long,
            device=pixel_values.device,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)
        image_grid_thw = self._build_image_grid_thw(pixel_values)
        vision_outputs = self.vision_model(pixel_values.to(dtype=self.dtype))
        vision_hidden_states = _unwrap_model_output(vision_outputs)
        if vision_hidden_states.ndim == 3:
            batch_size, seq_len, hidden_size = vision_hidden_states.shape
            vision_hidden_states = vision_hidden_states.reshape(
                batch_size * seq_len, hidden_size
            )
        image_embeds = self.projector(vision_hidden_states, image_grid_thw)
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        if image_embeds.ndim == 2:
            image_embeds = image_embeds.reshape(
                pixel_values.shape[0], image_embeds.shape[-1], -1
            )
        return image_embeds


def get_jsonl_texts(path, nsamples, text_key="text"):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get(text_key)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid `{text_key}` at {path}:{line_no}")
            samples.append(text)
            if len(samples) >= nsamples:
                break
    if len(samples) == 0:
        raise ValueError(f"No calibration samples found in {path}")
    return samples


def get_wikitext2(nsamples, seqlen, local_dir=None, tokenizer=None):
    if local_dir is None:
        from datasets import load_dataset

        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
            lambda x: len(x["text"]) >= seqlen
        )
        return [example["text"] for example in traindata.select(range(nsamples))]
    else:
        from datasets import load_from_disk
        import random

        train_data = load_from_disk(local_dir)["train"]
        if tokenizer is None:
            ValueError("tokenizer must be provided when local_dir is specified")
        trainenc = tokenizer("n\n".join(train_data["text"]))
        random.seed(0)
        train_samples = []
        input_ids = trainenc.input_ids
        data_seq_len = len(input_ids)
        nsamples = min(nsamples, data_seq_len // seqlen)
        for i in tqdm(range(nsamples)):
            start = i * seqlen
            end = start + seqlen
            inp = trainenc.input_ids[start:end]
            inp_mask = trainenc["attention_mask"][start:end]
            train_samples.append(
                {
                    "input_ids": inp,
                    "attention_mask": inp_mask,
                }
            )
        return train_samples

def export_prefill_decode(
    xh_model, data_batch, model_name, work_dir, quant_type, mode="prefill"
):
    xh_model.set_exec_device(torch.device("cpu"))
    if hasattr(xh_model, "kv_cache_to"):
        xh_model.kv_cache_to("cpu")
    if hasattr(xh_model, "rope_deltas") and xh_model.rope_deltas is not None:
        xh_model.rope_deltas = xh_model.rope_deltas.to(torch.device("cpu"))
    xh_model.convert_to_export_graph(data_batch)
    xh_model.change_eval_type(EvalModelType.EXPORTED)
    xh_model.to_export_onnx(data_batch, work_dir, f"{model_name}_{mode}_{quant_type}")
    xh_model.release_exported_model()
    return work_dir / f"{model_name}_{mode}_{quant_type}.onnx"

def houmo_export_llm(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = Config.fromfile(args.llm_config)

    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a, quant_type=args.quant_type
    )
    QConfig = create_quant_config(quant_scheme)
    cfg.model.quant_config.w_schema.bits = QConfig.w_schema.man_bit
    cfg.model.quant_config.w_schema.fp_mode = QConfig.w_schema.fp_mode
    cfg.model.quant_config.w_schema.hidden_bit = QConfig.w_schema.hidden_bit
    cfg.model.quant_config.act_schema.bits = QConfig.act_schema.man_bit
    cfg.model.quant_config.act_schema.fp_mode = QConfig.act_schema.fp_mode
    cfg.model.quant_config.act_schema.hidden_bit = QConfig.act_schema.hidden_bit

    prefix = "{}-XH2a-{}".format(args.model_name, format_number(args.context_length))
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)

    cfg.hf_model_dir = hf_model_path
    cfg.model.hf_model = hf_model_path

    cfg.model.wrap_cfg.max_sequence_length = args.context_length
    cfg.model.wrap_cfg.input_sequence_length = args.input_sequence_length

    paddleocr_vl_llm_model: XHPaddleOCRVLLLMModel = MODELS.build(cfg.model)
    native_model = paddleocr_vl_llm_model.get_hf_model()

    token_embedding = native_model.model.get_input_embeddings()
    token_embedding_file = Path(work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))

    processor = PaddleOCRVLProcessor.from_pretrained(hf_model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 147384
    processor.image_processor.max_pixels = 2822400

    image = Image.open(f"{HOUMO_PIC_PATH}/ocr.jpeg").convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    # tokenizer = processor.tokenizer
    native_model.to(device)
    inputs.to(device)

    visual = native_model.visual
    visual.to(device)
    visual.to(torch.float16)

    with torch.no_grad():
        pixel_values = inputs["pixel_values"].to(device).to(torch.float16)
        image_grid_thw = inputs["image_grid_thw"].to(device)
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)  # Add temporal dimension for vision model
        siglip_position_ids = []
        image_grid_hws = []
        sample_indices = []
        cu_seqlens = [0]
        for idx, thw in enumerate(image_grid_thw):
            thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
            numel = np.prod(thw_tuple)
            image_grid_hws.append(thw_tuple)
            image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
            siglip_position_ids.append(image_position_ids)
            sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
            cu_seqlens.append(cu_seqlens[-1] + numel)

        siglip_position_ids = torch.cat(siglip_position_ids, dim=0).to(device)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(device)
        sample_indices = torch.cat(sample_indices, dim=0).to(device)

        visual_outputs = visual(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_hws,
            position_ids=siglip_position_ids,
            vision_return_embed_list=True,
            interpolate_pos_encoding=True,
            sample_indices=sample_indices,
            cu_seqlens=cu_seqlens,
            return_pooler_output=False,
            use_rope=True,
            window_size=-1,
        )
        image_embeds = visual_outputs.last_hidden_state
        image_embeds = native_model.mlp_AR(image_embeds, image_grid_thw)
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
    
    del visual
    del visual_outputs, pixel_values, siglip_position_ids, sample_indices, cu_seqlens

    native_model.cpu()
    image_embeds = image_embeds.cpu()
    image_grid_thw = image_grid_thw.cpu()
    cleanup_cuda()

    paddleocr_vl_llm_model.init_wrap_model(native_model)

    kv_cache_shape = paddleocr_vl_llm_model.past_key_caches[0].shape
    num_hidden_layers = len(paddleocr_vl_llm_model.past_key_caches)
    hidden_size = token_embedding.weight.shape[1]
    
    del native_model
    
    data_prefill = {
        "input_ids": inputs["input_ids"].to(torch.device("cpu")),
        "image_embeds": image_embeds,
        "past_seq_length": 0,
        "image_grid_thw": image_grid_thw,
    }
    paddleocr_vl_llm_model.change_eval_type(EvalModelType.WRAPED)
    paddleocr_vl_llm_model.to(torch.float16)
    paddleocr_vl_llm_model.to(torch.device("cpu"))

    paddleocr_vl_llm_model.convert_to_fronted_graph(data_prefill, release_wraped_model=False)
    paddleocr_vl_llm_model.convert_to_quant_graph(DeviceType.XH2a)

    calib_data = []
    for arg in paddleocr_vl_llm_model.prepare_inputs_for_graph(data_prefill):
        if isinstance(arg, (list, tuple)):
            calib_data.extend(arg)
        else:
            calib_data.append(arg)
    ptq_quantize(
        paddleocr_vl_llm_model.quanted_model,
        [calib_data],
        PrecisionMode.ALIGNED,
        [torch.device("cpu")],
    )

    paddleocr_vl_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    paddleocr_vl_llm_model.to(torch.float16)

    prefill_hmonnx_dir = Path(work_dir) / "prefill"
    prefill_hmonnx_dir.mkdir(exist_ok=True, parents=True)

    logger.info("*************** Start exporting prefill model ***************")
    prefill_hmonnx_file = export_prefill_decode(
        paddleocr_vl_llm_model, data_prefill, args.model_name, prefill_hmonnx_dir, args.quant_type, mode="prefill"
    )
    
    logger.info("*************** Start exporting decode model ***************")
    paddleocr_vl_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    paddleocr_vl_llm_model.set_input_sequence_length(1)
    data_decode = {
        "input_ids": torch.randint(0, 1000, (1, 1)),
        "past_seq_length": 256,
    }
    
    decode_hmonnx_dir = Path(work_dir) / "decode"
    decode_hmonnx_dir.mkdir(exist_ok=True, parents=True)
    decode_hmonnx_file = export_prefill_decode(
        paddleocr_vl_llm_model, data_decode, args.model_name, decode_hmonnx_dir, args.quant_type, mode="decode"
    )

    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": "text_llm",
        "model_name": args.model_name,
        "prefill_onnx": str(prefill_hmonnx_file),
        "decode_onnx": str(decode_hmonnx_file),
        "embedding_file": str(token_embedding_file),
        "kv_cache": {"shape": kv_cache_shape, "num_decoder_layers": num_hidden_layers},
        "hidden_size": int(hidden_size),
        "input_sequence_length": int(inputs["input_ids"].shape[-1]),
    }
    meta_file = work_dir / "meta.json"
    save_json(meta_file, meta_info)
    logger.info(f"Text LLM export complete. Meta saved to {meta_file}")


def houmo_export_vision(args):
    import onnx
    from xhquant.utils.onnxsim_large_model.simplify_large_onnx import (
        simplify_large_onnx,
    )

    hf_model_path = osp.normpath(osp.abspath(args.model))

    cfg = Config.fromfile(args.vision_config)

    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a, quant_type=args.quant_type
    )
    QConfig = create_quant_config(quant_scheme)
    cfg.model.quant_config.w_schema.bits = QConfig.w_schema.man_bit
    cfg.model.quant_config.w_schema.fp_mode = QConfig.w_schema.fp_mode
    cfg.model.quant_config.w_schema.hidden_bit = QConfig.w_schema.hidden_bit
    cfg.model.quant_config.act_schema.bits = QConfig.act_schema.man_bit
    cfg.model.quant_config.act_schema.fp_mode = QConfig.act_schema.fp_mode
    cfg.model.quant_config.act_schema.hidden_bit = QConfig.act_schema.hidden_bit
    cfg.model.wrap_cfg.max_size_w = args.max_size_w
    cfg.model.wrap_cfg.max_size_h = args.max_size_h

    prefix = "{}-XH2a-{}".format(args.model_name, format_number(args.context_length))
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)

    cfg.model.hf_model = hf_model_path
    cfg.hf_model_dir = hf_model_path

    paddleocr_vl_vision_model: XHPaddleOCRVLVisionModel = MODELS.build(cfg.model)
    native_model = paddleocr_vl_vision_model.get_hf_model()

    paddleocr_vl_vision_model.init_wrap_model(native_model)

    wraped_model = paddleocr_vl_vision_model._wrap_model
    projector_model = _resolve_visual_projector(native_model)

    processor = PaddleOCRVLProcessor.from_pretrained(hf_model_path, trust_remote_code=True)
    processor.image_processor = SiglipImageProcessor.from_pretrained(hf_model_path)
    fixed_output_width = int(getattr(cfg.model.wrap_cfg, "max_size_w", 0) or 0)
    fixed_output_height = int(getattr(cfg.model.wrap_cfg, "max_size_h", 0) or 0)
    if fixed_output_width > 0 and fixed_output_height > 0:
        processor.image_processor.fixed_output_width = fixed_output_width
        processor.image_processor.fixed_output_height = fixed_output_height
    processor.image_processor.min_pixels = 147384
    processor.image_processor.max_pixels = 2822400
    image = Image.open(f"{HOUMO_PIC_PATH}/ocr.jpeg").convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if "pixel_values" in inputs:
        logger.info(
            f"pixel_values shape from processor: {tuple(inputs['pixel_values'].shape)}"
        )

    native_model.to(torch.device("cpu"))
    native_model.to(torch.float16)
    inputs = {
        k: (
            v.to(torch.device("cpu")).to(torch.float16)
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
            else v.to(torch.device("cpu")) if isinstance(v, torch.Tensor) else v
        )
        for k, v in inputs.items()
    }
    if hasattr(native_model, "visual"):
        native_model.visual.to(torch.device("cpu")).to(torch.float16)
    if hasattr(native_model, "mlp_AR"):
        native_model.mlp_AR.to(torch.device("cpu")).to(torch.float16)
    if hasattr(native_model, "projector"):
        native_model.projector.to(torch.device("cpu")).to(torch.float16)
    if hasattr(native_model, "model") and hasattr(native_model.model, "visual"):
        native_model.model.visual.to(torch.device("cpu")).to(torch.float16)
    if hasattr(native_model, "model") and hasattr(native_model.model, "mlp_AR"):
        native_model.model.mlp_AR.to(torch.device("cpu")).to(torch.float16)
    if hasattr(native_model, "model") and hasattr(native_model.model, "projector"):
        native_model.model.projector.to(torch.device("cpu")).to(torch.float16)
    
    hm_pixel_values = (
        inputs["pixel_values"].unsqueeze(0) if inputs["pixel_values"].ndim == 4 else inputs["pixel_values"]
    )
    hm_pixel_values = hm_pixel_values.type(wraped_model.dtype).to(wraped_model.device)

    fused_model = VisionProjectorWrapModel(
        wraped_model,
        projector_model,
        patch_size=cfg.model.wrap_cfg.patch_size,
        temporal_patch_size=cfg.model.wrap_cfg.temporal_patch_size,
    )

    logger.info("Start export vision onnx ...")

    vision_tmp_onnx_dir = work_dir / "vision_tmp"
    vision_tmp_onnx_dir.mkdir(exist_ok=True, parents=True)
    vision_tmp_onnx_file = str(vision_tmp_onnx_dir / f"{args.model_name}_visual.onnx")

    fused_model.float().eval()
    fused_model.cpu()
    torch.onnx.export(
        fused_model,
        (hm_pixel_values.float().cpu()),
        vision_tmp_onnx_file,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        verbose=True,
    )
    onnx_model = onnx.load(vision_tmp_onnx_file, load_external_data=True)
    onnx_model, _ = simplify_large_onnx(onnx_model)
    vision_onnx_file = str(work_dir / f"{args.model_name}_visual.onnx")
    onnx.save(
        onnx_model,
        vision_onnx_file,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{args.model_name}_visual_external_data",
        convert_attribute=True,
    )

    os.system("rm -rf {}".format(vision_tmp_onnx_dir))

    vision_hmonnx_dir = work_dir / "visual"
    vision_hmonnx_dir.mkdir(exist_ok=True, parents=True)
    vision_hmonnx_file = (
        vision_hmonnx_dir / f"{args.model_name}_visual_{args.quant_type}.onnx"
    )

    convert_onnx_to_hmonnx(
        vision_onnx_file,
        [hm_pixel_values.float().cpu()],
        device_type=DeviceType.XH2a,
        out_hmonnx_file=vision_hmonnx_file,
        quant_config=cfg.model.quant_config,
    )
    logger.info(
        "Export vision model successful, saved to {}".format(vision_hmonnx_file)
    )

    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": "vision",
        "model_name": args.model_name,
        "vision_encoder_onnx": str(vision_hmonnx_file.relative_to(work_dir)),
        "vision_patch_size": cfg.model.wrap_cfg.patch_size,
        "vision_input_size": [
            cfg.model.wrap_cfg.max_size_h,
            cfg.model.wrap_cfg.max_size_w,
        ],
        "vision_channels": 3,
        "vision_temporal_patch_size": cfg.model.wrap_cfg.temporal_patch_size,
        "vision_output_name": "image_embeds",
        "projector_fused": True,
        "export_method": "two_stage_onnx_simplify_with_projector",  # Mark as using improved method
    }
    meta_file = work_dir / "meta_vision.json"
    save_json(meta_file, meta_info)
    logger.info(f"Vision export complete. Meta saved to {meta_file}")


def move_models(
    work_dir: Path,
    source: str = "prefill",
    model: str = "prefill",
    target_name: str = "hmquant_paddleocr-vl_with_act.onnx",
):
    source_dir = work_dir / "hmquant/{}".format(source)
    matched_files = list(source_dir.glob("*{}*.onnx".format(model)))

    if not matched_files:
        raise FileNotFoundError(f"No matching ONNX files found in {source_dir}")

    target_path = source_dir / target_name
    if target_path.exists():
        target_path.unlink()

    shutil.move(matched_files[0], target_path)
    return target_path


def format_number(n):
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)}m"
    elif n >= 1024:
        return f"{n // 1024}k"
    else:
        return f"0k"


def move_llm(args):
    work_dir = Path(args.work_dir)
    dest_dir = Path(args.out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_name = os.path.basename(args.model_name)
    hm_model_name = "hmquant_{}_with_act.onnx".format(model_name)
    hmm_model_dir = "{}-XH2a-{}".format(model_name, format_number(args.context_length))
    logger.info(
        msg_output_format("Start move from {} to {}").format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    shutil.move(work_dir / hmm_model_dir / "prefill", dest_dir / "hmquant/prefill")
    move_models(dest_dir, "prefill", target_name=hm_model_name)
    shutil.move(work_dir / hmm_model_dir / "decode", dest_dir / "hmquant/decode")
    move_models(dest_dir, "decode", "decode", target_name=hm_model_name)
    shutil.move(
        work_dir / hmm_model_dir / "token_embedding.pt",
        dest_dir / "hmquant/quant_embedding.pt",
    )
    shutil.move(work_dir / hmm_model_dir / "visual", dest_dir / "hmquant/visual")
    move_models(dest_dir, "visual", "visual", target_name=hm_model_name)
    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="output hmonnx model name",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="model size",
    )
    parser.add_argument("--work-dir", type=str, default="work_dirs/")
    parser.add_argument(
        "--out-dir", type=str, default="output/{}".format(HOUMO_TARGET)
    )
    parser.add_argument("--llm_config", type=str, default="llm_config.py")
    parser.add_argument("--vision_config", type=str, default="vision_config.py")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--context-length", type=int, default=4096, help="max sequence length"
    )
    parser.add_argument(
        "--input-sequence-length",
        type=int,
        default=None,
        help="input sequence length",
    )
    parser.add_argument(
        "--quant-type",
        default=None,
        help="quant type, default is w8a8h0_ssfp",
    )
    args = parser.parse_args()
    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model = first_not_none(args.model, get_default_model_dir(model_config))
    args.context_length = first_not_none(
        args.context_length,
        parse_context_length(model_config.get("context_length", "4k")),
    )
    args.input_sequence_length = first_not_none(
        args.input_sequence_length, model_config.get("prefill_length", 256)
    )
    args.quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "w8a8h0_ssfp")
    )
    args.max_size_w = model_config.get("max_size_w", 448)
    args.max_size_h = model_config.get("max_size_h", 448)
    return args

if __name__ == "__main__":
    assert check_gpu() is True, "Error: Not found GPU device."

    args = parse_args()
    set_seed(42)
    with ChildProcessMemoryMonitor(interval=2, log_file="./cpu_monitor.log", include_children=True) as monitor:
        houmo_export_llm(args)
        houmo_export_vision(args)
        move_llm(args)
    print(f"Peak memory usage: {monitor.peak_memory_mb:.2f} MB")
