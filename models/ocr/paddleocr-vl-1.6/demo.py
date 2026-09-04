#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   PaddleOCR-VL-1.6 HMM inference demo on HOUMO AI devices.
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
import glob
import importlib
import json
import os
import math
import re
import sys
import time
import types
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
import tcim_lite as tcim
import huggingface_hub.dataclasses as hf_dataclasses
from transformers import AutoTokenizer
from hmatc.utils.perf_infomations import (
    InferencePerformanceTracker,
    PERFTYPE,
)
from hmatc.utils.utils import first_not_none, get_model_configs

from hmatc.python.get_hm_devices import get_hm_devices

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"
HOUMO_PIC_PATH = os.getenv(
    "HOUMO_PIC_PATH", str(Path(__file__).resolve().parents[3] / "data" / "pic")
)
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 1
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


def get_default_tokenizer_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "paddleocr-vl").upper()
    model_size = model_config.get("model_size", "0.9b").upper()
    return f"{model_name}-{model_size}"


def get_default_image_inputs() -> list[list[str]]:
    return [[os.path.join(HOUMO_PIC_PATH, "ocr.jpeg")]]


def normalize_image_inputs(
    image_args: Optional[List[List[str]]],
) -> Optional[List[str]]:
    if not image_args:
        return None
    image_paths = []
    values = (value for group in image_args for value in group)
    for value in values:
        _append_image_paths(image_paths, value)
    return image_paths or None


def _append_image_paths(image_paths: list[str], value: str) -> None:
    for item in value.split(","):
        image_path = item.strip()
        if not image_path:
            continue
        matched_paths = sorted(glob.glob(image_path))
        if matched_paths:
            image_paths.extend(matched_paths)
        else:
            image_paths.append(image_path)


sys.path.insert(0, str(SCRIPT_DIR))
from processing_paddleocr_vl import PaddleOCRVLProcessor


def _patch_huggingface_union_validator() -> None:
    validator = hf_dataclasses._BASIC_TYPE_VALIDATORS.get(__import__("typing").Union)
    if validator is not None and types.UnionType not in hf_dataclasses._BASIC_TYPE_VALIDATORS:
        hf_dataclasses._BASIC_TYPE_VALIDATORS[types.UnionType] = validator


_patch_huggingface_union_validator()


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        help="model name",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size",
    )
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default=None,
        help="tokenizer / hf model dir",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"),
        help="houmo embedding weight path",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=None,
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default=None,
        help="houmo decode model path",
    )
    parser.add_argument(
        "--vision_path",
        dest="vision_path",
        type=str,
        default=None,
        help="houmo vision model path (.hmm)",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number, only xh2 support",
    )
    parser.add_argument(
        "--repetition_penalty",
        dest="repetition_penalty",
        type=float,
        default=1.0,
        help="sampling repetition_penalty",
    )
    parser.add_argument(
        "--topk",
        dest="topk",
        type=int,
        default=None,
        help="sampling top-k",
    )
    parser.add_argument(
        "--topp",
        dest="topp",
        type=float,
        default=1.0,
        help="sampling top-p",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=1.0,
        help="sampling temperature",
    )
    parser.add_argument(
        "--task",
        dest="task",
        type=str,
        default="ocr",
        choices=["ocr", "table", "formula", "chart", "spotting", "seal"],
        help="default PaddleOCR-VL task prompt",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt",
        type=str,
        default=None,
        help="custom user prompt",
    )
    parser.add_argument(
        "--max_new_tokens",
        dest="max_new_tokens",
        type=int,
        default=1024,
        help="max new tokens for generation",
    )
    parser.add_argument(
        "--image_path",
        dest="image_path",
        nargs="+",
        action="append",
        default=None,
        help="one or more image paths, supports repeated usage and comma-separated values",
    )
    parser.add_argument(
        "--max_size_w",
        dest="max_size_w",
        type=int,
        default=None,
        help="max image width for vision",
    )
    parser.add_argument(
        "--max_size_h",
        dest="max_size_h",
        type=int,
        default=None,
        help="max image height for vision",
    )
    parser.add_argument(
        "--static_resize_mode",
        dest="static_resize_mode",
        choices=["stretch", "letterbox"],
        default="stretch",
        help="static image resize mode; default stretch, optional letterbox",
    )
    parser.add_argument(
        "--patch_size",
        dest="patch_size",
        type=int,
        default=14,
        help="vision patch size",
    )
    parser.add_argument(
        "--history",
        dest="history",
        action="store_true",
        help="keep chat history",
    )
    args = parser.parse_args()

    default_size, default_name, configs = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_name)
    args.model_size = first_not_none(args.model_size, default_size)
    config = configs.get(args.model_name, {}).get(args.model_size, {})
    args.ndevice = first_not_none(args.ndevice, config.get("ndevice", 1))
    args.max_size_w = first_not_none(args.max_size_w, config.get("max_size_w", 896))
    args.max_size_h = first_not_none(args.max_size_h, config.get("max_size_h", 896))
    args.max_size_t = config.get("max_size_t", TEMPORAL_PATCH_SIZE)
    if args.tokenizer_dir is None:
        args.tokenizer_dir = get_default_tokenizer_dir(config)
    if args.prefill_path is None:
        args.prefill_path = os.path.join(
            "output", HOUMO_TARGET, f"{args.model_name}-{args.model_size}_prefill.hmm"
        )
    if args.decode_path is None:
        args.decode_path = os.path.join(
            "output", HOUMO_TARGET, f"{args.model_name}-{args.model_size}_decode.hmm"
        )
    if args.vision_path is None:
        args.vision_path = os.path.join(
            "output",
            HOUMO_TARGET,
            f"{args.model_name}-{args.model_size}_visual_{args.max_size_w}x{args.max_size_h}.hmm",
        )
    args.image_path = first_not_none(args.image_path, get_default_image_inputs())
    if args.ndevice > 1:
        args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        args.decode_path = args.decode_path.replace(".hmm", ".hmms")
    args.image_path = normalize_image_inputs(args.image_path)
    return args


def _dtype(info) -> np.dtype:
    return np.dtype(info.dtype)


def _set_scalar(module, name: str, value: int) -> None:
    info = module.get_input_info(name)
    module.set_input(name, np.asarray([value], dtype=_dtype(info)))


def _find_input(names: Sequence[str], *keys: str) -> Optional[str]:
    for name in names:
        if name in keys or any(key in name for key in keys):
            return name
    return None


def _set_length_inputs(
    module, names: Sequence[str], past_length: int, current_length: int
) -> None:
    for name in names:
        if name in ("valid_length", "past_seq_length"):
            _set_scalar(module, name, past_length)
        elif name in ("current_length", "current_input_length"):
            _set_scalar(module, name, current_length)


def _count_layers(names: Sequence[str]) -> int:
    ids = []
    for name in names:
        match = re.search(r"model_layers_(\d+)_self_attn_[kv]cache_input", name)
        if match:
            ids.append(int(match.group(1)))
    if not ids:
        raise ValueError("No KV cache inputs found in HMM graph")
    return max(ids) + 1


def show_ttft_breakdown(ttft_time: float, perf_tracker: InferencePerformanceTracker) -> None:
    metrics = perf_tracker.current_metrics
    ttft_ms = ttft_time * 1000
    components = [
        ("Prompt/Tokenize", metrics.prefill_perf_infos.tokenizer_time),
        ("Vision SetInput", metrics.vision_perf_infos.setinput_time),
        ("Vision Infer", metrics.vision_perf_infos.infer_time),
        ("Vision GetOutput", metrics.vision_perf_infos.getoutput_time),
        ("Prefill Embedding", metrics.prefill_perf_infos.embedding_time),
        ("Prefill SetInput", metrics.prefill_perf_infos.setinput_time),
        ("Prefill Infer", metrics.prefill_perf_infos.infer_time),
        ("Prefill GetOutput", metrics.prefill_perf_infos.getoutput_time),
    ]
    tracked_ms = sum(value for _, value in components)
    residual_ms = max(ttft_ms - tracked_ms, 0.0)
    logger.success("TTFT Breakdown:")
    for label, value in components:
        if value > 0:
            logger.success("  {}: {:.3f} ms", label, value)
    if residual_ms > 0:
        logger.success("  Other/Untracked: {:.3f} ms", residual_ms)
    logger.success("  Total TTFT: {:.3f} ms", ttft_ms)


class HmPaddleOCRVL:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.perf_tracker = InferencePerformanceTracker()
        if args.ndevice != 1:
            raise ValueError("PaddleOCR-VL HMM demo currently supports --ndevice 1")
        devices = get_hm_devices(args.ndevice)
        manager = tcim.runtime.WeightManager(tcim.runtime.DevManager(devices, "Xh2HalBackend"))
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self.prefill = tcim.runtime.load(args.prefill_path, option=tcim.runtime.Option(manager))
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)
        option = tcim.runtime.Option(manager)
        cache_names = [self.prefill.get_input_name(i) for i in range(self.prefill.get_num_inputs())]
        option.set_dummy_tensors([name for name in cache_names if "model_layers" in name])
        self.perf_tracker.perf_start(PERFTYPE.DECODE_LOAD_TIME)
        self.decode = tcim.runtime.load(args.decode_path, option=option)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_LOAD_TIME)
        self.perf_tracker.perf_start(PERFTYPE.VISION_LOAD_TIME)
        self.vision = tcim.runtime.load(args.vision_path, option=tcim.runtime.Option(manager))
        self.perf_tracker.perf_end(PERFTYPE.VISION_LOAD_TIME)

        self.prefill_names = [self.prefill.get_input_name(i) for i in range(self.prefill.get_num_inputs())]
        self.decode_names = [self.decode.get_input_name(i) for i in range(self.decode.get_num_inputs())]
        self.prefill_input = _find_input(self.prefill_names, "input_1") or self.prefill_names[0]
        self.decode_input = _find_input(self.decode_names, "input_1") or self.decode_names[0]
        self.prefill_length = int(self.prefill.get_input_info(self.prefill_input).shape[1])
        self.hidden_size = int(self.prefill.get_input_info(self.prefill_input).shape[-1])
        self.context_max_length = self._cache_shape()[2]
        self.num_layers = _count_layers(self.prefill_names)
        for name in self.prefill_names:
            if "model_layers" in name and name in self.decode_names:
                self.decode.set_dev_input(name, self.prefill.get_dev_input(name))

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=False)
        self.embedding = self._load_embedding(args.embedding_path)
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids("<|IMAGE_PLACEHOLDER|>"))
        self.vision_start_token_id = int(self.tokenizer.convert_tokens_to_ids("<|IMAGE_START|>"))
        config_path = Path(args.tokenizer_dir) / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        eos = config.get("eos_token_id", self.tokenizer.eos_token_id)
        self.eos_token_ids = {int(eos)} if isinstance(eos, int) else {int(x) for x in eos}
        self.processor = PaddleOCRVLProcessor.from_static_pretrained(
            args.tokenizer_dir,
            max_size_w=args.max_size_w,
            max_size_h=args.max_size_h,
            patch_size=args.patch_size,
            static_resize_mode=args.static_resize_mode,
        )
        self._reset()
        self.ttft_time = 0.0
        logger.info("HMM ready: prefill_length={}, hidden={}, layers={}, context={}", self.prefill_length, self.hidden_size, self.num_layers, self.context_max_length)

    def _cache_shape(self):
        for name in self.prefill_names:
            if name.endswith("_kcache_input"):
                return tuple(self.prefill.get_input_info(name).shape)
        raise ValueError("Cannot find cache shape")

    @staticmethod
    def _load_embedding(path: Optional[str]):
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"Embedding checkpoint not found: {path}")
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, dict):
            value = value.get("weight", value)
        if isinstance(value, torch.nn.Embedding):
            value = value.weight
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Unsupported embedding checkpoint: {type(value)}")
        return value.float()

    def _reset(self):
        self.context_length = 0
        self.rope_delta = 0
        for name in self.prefill_names:
            if "model_layers" in name:
                info = self.prefill.get_input_info(name)
                zeros = np.zeros(info.shape, dtype=_dtype(info))
                self.prefill.set_input(name, zeros)
                if name in self.decode_names:
                    self.decode.set_input(name, zeros)

    def _get_rope_index(self, input_ids, mm_token_type_ids, image_grid_thw):
        parts = []
        current = 0
        image_index = 0
        types_ = mm_token_type_ids[0].tolist()
        start = 0
        while start < len(types_):
            modality = types_[start]
            end = start + 1
            while end < len(types_) and types_[end] == modality:
                end += 1
            length = end - start
            if modality == 0:
                parts.append(torch.arange(length).view(1, -1).expand(3, -1) + current)
                current += length
            elif modality == 1:
                t, h, w = [int(x) for x in image_grid_thw[image_index].tolist()]
                image_index += 1
                lh, lw = h // 2, w // 2
                if length != t * lh * lw:
                    raise ValueError(f"Image token/grid mismatch: {length}/{t * lh * lw}")
                ti = torch.arange(t).view(-1, 1).expand(-1, lh * lw).flatten()
                hi = torch.arange(lh).view(1, -1, 1).expand(t, -1, lw).flatten()
                wi = torch.arange(lw).view(1, 1, -1).expand(t, lh, -1).flatten()
                parts.append(torch.stack((ti, hi, wi)) + current)
                current += max(h, w) // 2
            else:
                raise ValueError(f"Unsupported multimodal token type: {modality}")
            start = end
        positions = torch.cat(parts, dim=1)
        self.rope_delta = int(positions.max().item() + 1 - input_ids.shape[1])
        return positions

    def _run_visual(self, pixel_values, image_grid_thw):
        name = self.vision.get_input_name(0)
        info = self.vision.get_input_info(name)
        data = pixel_values.unsqueeze(1).numpy().astype(_dtype(info)) if pixel_values.ndim == 4 else pixel_values.numpy().astype(_dtype(info))
        if data.shape != tuple(info.shape):
            raise ValueError(
                "Processor visual input shape does not match the loaded graph: "
                f"processor={data.shape}, graph={tuple(info.shape)}"
            )
        self.perf_tracker.perf_start(PERFTYPE.VISION_INPUT_TIME)
        self.vision.set_input(name, data)
        self.perf_tracker.perf_end(PERFTYPE.VISION_INPUT_TIME)
        self.perf_tracker.perf_start(PERFTYPE.VISION_INFER_TIME)
        self.vision.run()
        self.vision.sync()
        self.perf_tracker.perf_end(PERFTYPE.VISION_INFER_TIME)
        self.perf_tracker.perf_start(PERFTYPE.VISION_OUTPUT_TIME)
        output = self.vision.get_output(self.vision.get_output_name(0)).numpy()
        output = torch.from_numpy(output).float().reshape(-1, self.hidden_size)
        self.perf_tracker.perf_end(PERFTYPE.VISION_OUTPUT_TIME)
        return output

    def _set_positions(self, module, names, positions):
        mapping = {"time_position_ids": 0, "hight_position_ids": 1, "height_position_ids": 1, "width_position_ids": 2}
        for name in names:
            if name not in mapping:
                continue
            info = module.get_input_info(name)
            value = positions[mapping[name]].numpy().astype(_dtype(info))
            module.set_input(name, value.reshape(info.shape))

    def _run_llm(self, module, names, embeds, positions, valid_length, current_length):
        input_name = _find_input(names, "input_1") or names[0]
        info = module.get_input_info(input_name)
        module.set_input(input_name, embeds.numpy().astype(_dtype(info)))
        self._set_positions(module, names, positions)
        _set_length_inputs(module, names, valid_length, current_length)
        module.run()
        module.sync()
        return module.get_output(module.get_output_name(0)).numpy()

    def _prepare_prefill(self, image_path: str, prompt: str):
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOKEN_TIME)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOKEN_TIME)
        input_ids = inputs["input_ids"]
        grid = inputs["image_grid_thw"]
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
        token_embeds = F.embedding(input_ids, self.embedding)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)
        self.perf_tracker.perf_start(PERFTYPE.VISION_TOTAL_TIME)
        image_embeds = self._run_visual(inputs["pixel_values"], grid)
        self.perf_tracker.perf_end(PERFTYPE.VISION_TOTAL_TIME)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(token_embeds)
        image_token_count = int((input_ids == self.image_token_id).sum())
        if image_token_count != image_embeds.shape[0]:
            raise ValueError(
                f"Image token/features mismatch: {image_token_count}/{image_embeds.shape[0]}"
            )
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
        token_embeds = token_embeds.masked_scatter(
            image_mask, image_embeds.to(token_embeds.dtype)
        )
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)
        positions = self._get_rope_index(
            input_ids, inputs["mm_token_type_ids"], grid
        )
        return input_ids, token_embeds, positions

    def _run_prefill(self, token_embeds, positions, prompt_len: int) -> int:
        next_id = None
        for start in range(0, prompt_len, self.prefill_length):
            current = min(self.prefill_length, prompt_len - start)
            chunk = token_embeds[:, start : start + current]
            pos = positions[:, start : start + current]
            if current < self.prefill_length:
                pad = self.prefill_length - current
                chunk = torch.cat(
                    [chunk, torch.zeros(1, pad, self.hidden_size)], dim=1
                )
                pos = torch.cat([pos, pos[:, -1:].expand(3, pad)], dim=1)
            logits = self._run_timed_prefill(chunk, pos, current)
            if logits.ndim == 2:
                logits = logits.reshape(1, logits.shape[0], logits.shape[1])
            if logits.ndim != 3 or logits.shape[0] != 1:
                raise ValueError(f"Unexpected prefill logits shape: {tuple(logits.shape)}")
            logits_length = int(logits.shape[1])
            if logits_length == 1:
                next_logits = logits[0, 0]
            elif current <= logits_length:
                # The last prefill chunk is padded to the static graph length.
                # Its next-token logits are at the last real token, not at the
                # padded tail.  This matters for resolutions whose image-token
                # count does not make the prompt length a multiple of the chunk.
                next_logits = logits[0, current - 1]
            else:
                raise ValueError(
                    "Prefill logits are shorter than the requested chunk: "
                    f"{logits_length}/{current}"
                )
            next_id = int(next_logits.argmax())
            self.context_length += current
        return next_id

    def _run_timed_prefill(self, chunk, pos, current: int) -> np.ndarray:
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
        input_name = _find_input(self.prefill_names, "input_1") or self.prefill_names[0]
        info = self.prefill.get_input_info(input_name)
        self.prefill.set_input(input_name, chunk.numpy().astype(_dtype(info)))
        self._set_positions(self.prefill, self.prefill_names, pos)
        _set_length_inputs(
            self.prefill, self.prefill_names, self.context_length, current
        )
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
        self.prefill.run()
        self.prefill.sync()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
        logits = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)
        return np.asarray(logits)

    def _run_decode(self, first_id: int) -> list[int]:
        generated = [first_id]
        next_id = first_id
        while (
            len(generated) < self.args.max_new_tokens
            and next_id not in self.eos_token_ids
            and self.context_length < self.context_max_length
        ):
            next_id = self._run_decode_token(next_id)
            generated.append(next_id)
            self.context_length += 1
        return generated

    def _run_decode_token(self, token_id: int) -> int:
        self.perf_tracker.perf_start(PERFTYPE.DECODE_TOTAL_TIME)
        self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
        token = torch.tensor([[token_id]], dtype=torch.long)
        embed = F.embedding(token, self.embedding)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)
        position = torch.full(
            (3, 1), self.context_length + self.rope_delta, dtype=torch.float32
        )
        logits = self._run_timed_decode(embed, position)
        if logits.ndim == 2:
            logits = logits.reshape(1, logits.shape[0], logits.shape[1])
        self.perf_tracker.perf_start(PERFTYPE.DECODE_TOKEN_TIME)
        next_id = int(logits[0, -1].argmax())
        self.perf_tracker.perf_end(PERFTYPE.DECODE_TOKEN_TIME)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)
        return next_id

    def _run_timed_decode(self, embed, position) -> np.ndarray:
        self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
        input_name = _find_input(self.decode_names, "input_1") or self.decode_names[0]
        info = self.decode.get_input_info(input_name)
        self.decode.set_input(input_name, embed.numpy().astype(_dtype(info)))
        self._set_positions(self.decode, self.decode_names, position)
        _set_length_inputs(self.decode, self.decode_names, self.context_length, 1)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INPUT_TIME)
        self.perf_tracker.perf_start(PERFTYPE.DECODE_INFER_TIME)
        self.decode.run()
        self.decode.sync()
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INFER_TIME)
        self.perf_tracker.perf_start(PERFTYPE.DECODE_OUTPUT_TIME)
        logits = self.decode.get_output(self.decode.get_output_name(0)).numpy()
        self.perf_tracker.perf_end(PERFTYPE.DECODE_OUTPUT_TIME)
        return np.asarray(logits)

    @torch.no_grad()
    def chat(self, image_paths: List[str], prompt: str) -> str:
        if len(image_paths) != 1:
            raise ValueError("PaddleOCR-VL HMM demo currently supports one image")
        self._reset()
        self.perf_tracker.reset_perf_time()
        self.ttft_time = 0.0
        start_time = time.time()
        input_ids, token_embeds, positions = self._prepare_prefill(
            image_paths[0], prompt
        )
        prompt_len = input_ids.shape[1]
        next_id = self._run_prefill(token_embeds, positions, prompt_len)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)
        self.ttft_time = time.time() - start_time
        generated = self._run_decode(next_id)
        self.perf_tracker.set_basic_info(
            batch_size=1,
            input_seq_length=prompt_len,
            output_seq_length=max(len(generated) - 1, 0),
            num_images=len(image_paths),
        )
        return self.tokenizer.decode(generated, skip_special_tokens=self.args.task != "spotting", clean_up_tokenization_spaces=False)


def post_process(result: str, args: argparse.Namespace) -> str:
    if args.task == "table":
        try:
            utils = importlib.import_module("paddlex.inference.pipelines.paddleocr_vl.uilts")
            return utils.convert_otsl_to_html(result)
        except (ImportError, AttributeError):
            logger.warning("PaddleX is unavailable; printing raw table tokens")
    if args.task == "spotting":
        try:
            utils = importlib.import_module("paddlex.inference.pipelines.paddleocr_vl.uilts")
            with Image.open(args.image_path[0]) as image:
                result, spotting = utils.post_process_for_spotting(result, *image.size)
            logger.info("spotting polygons: {}", spotting)
        except (ImportError, AttributeError):
            logger.warning("PaddleX spotting post-process is unavailable; printing raw tokens")
    return result


def main() -> None:
    args = get_args()
    prompt = args.prompt or PROMPTS[args.task]
    model = HmPaddleOCRVL(args)
    start = time.time()
    result = post_process(model.chat(args.image_path, prompt), args)
    print(result)
    logger.info("total_time={:.3f}s", time.time() - start)
    show_ttft_breakdown(model.ttft_time, model.perf_tracker)
    model.perf_tracker.show_summary()


if __name__ == "__main__":
    main()
