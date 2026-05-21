#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   PaddleOCR-VL Inference Demo - Python script for running PaddleOCR-VL
#   hmm inference on HOUMO AI device.
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

import glob
import os
import sys
import math
import time
import argparse
import types
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Union
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import huggingface_hub.dataclasses as hf_dataclasses
from PIL import Image
from loguru import logger
import tcim_lite as tcim
from transformers import AutoTokenizer

from hmatc.python.get_hm_devices import get_hm_devices
from hmatc.utils.perf_infomations import (
    InferencePerformanceTracker,
    PERFTYPE,
)
from hmatc.utils.utils import first_not_none, get_model_configs

def _patch_huggingface_union_validator() -> None:
    union_validator = hf_dataclasses._BASIC_TYPE_VALIDATORS.get(__import__("typing").Union)
    if union_validator is not None and types.UnionType not in hf_dataclasses._BASIC_TYPE_VALIDATORS:
        hf_dataclasses._BASIC_TYPE_VALIDATORS[types.UnionType] = union_validator


_patch_huggingface_union_validator()

sys.path.append(str(Path(__file__).parent))
from processing_paddleocr_vl import PaddleOCRVLProcessor
from image_processing import SiglipImageProcessor

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_PIC_PATH = os.getenv(
    "HOUMO_PIC_PATH", str(Path(__file__).resolve().parents[3] / "data" / "pic")
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
IMAGE_TOKEN_ID = 100295
VIDEO_TOKEN_ID = 101307
VISION_START_TOKEN_ID = 101305
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 1
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_tokenizer_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "paddleocr-vl").upper()
    model_size = model_config.get("model_size", "0.9b").upper()
    return f"{model_name}-{model_size}"


def get_default_image_inputs() -> list[list[str]]:
    return [[os.path.join(HOUMO_PIC_PATH, "ocr.jpeg")]]

def is_valid_char(cp):
    if cp in (0x09, 0x0A, 0x0D):
        return True
    if 0x20 <= cp <= 0x7E:
        return True
    if (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    ):
        return True
    return False


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
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
        choices=["ocr", "table", "formula", "chart"],
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
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="enable debug logs including TTFT breakdown",
    )
    args = parser.parse_args()
    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    args.max_size_w = first_not_none(
        args.max_size_w, model_config.get("max_size_w", 448)
    )
    args.max_size_h = first_not_none(
        args.max_size_h, model_config.get("max_size_h", 448)
    )
    args.max_size_t = model_config.get("max_size_t", TEMPORAL_PATCH_SIZE)
    if args.tokenizer_dir is None:
        args.tokenizer_dir = get_default_tokenizer_dir(model_config)
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


def build_processor(
    tokenizer_dir: str, max_size_h: int, max_size_w: int, patch_size: int
) -> PaddleOCRVLProcessor:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        chat_template_file = Path(tokenizer_dir) / "chat_template.jinja"
        if chat_template_file.exists():
            chat_template = chat_template_file.read_text(encoding="utf-8")
    image_processor = SiglipImageProcessor.from_pretrained(tokenizer_dir)
    image_processor.patch_size = patch_size
    image_processor.temporal_patch_size = TEMPORAL_PATCH_SIZE
    image_processor.merge_size = SPATIAL_MERGE_SIZE
    image_processor.min_pixels = max_size_h * max_size_w
    image_processor.max_pixels = max_size_h * max_size_w
    image_processor.fixed_output_height = max_size_h
    image_processor.fixed_output_width = max_size_w
    return PaddleOCRVLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )


def build_messages(
    prompt: str,
    image_paths: Sequence[str],
    max_size_h: int,
    max_size_w: int,
    system_prompt: str = "",
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    content = [
        {
            "type": "image",
            "image": image_path,
            "resized_height": max_size_h,
            "resized_width": max_size_w,
        }
        for image_path in image_paths
    ]
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})
    return messages


def build_default_vision_prompt(task: str, prompt: Optional[str] = None) -> str:
    if prompt is not None and prompt.strip():
        return prompt
    return PROMPTS[task]


def normalize_image_inputs(
    image_args: Optional[List[List[str]]],
) -> Optional[List[str]]:
    if not image_args:
        return None
    image_paths = []
    for group in image_args:
        for value in group:
            for item in value.split(","):
                image_path = item.strip()
                if not image_path:
                    continue
                matched_paths = sorted(glob.glob(image_path))
                if matched_paths:
                    image_paths.extend(matched_paths)
                else:
                    image_paths.append(image_path)
    return image_paths or None


def load_images(image_paths: Sequence[str]) -> List[Image.Image]:
    images = []
    for image_path in image_paths:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def scatter_image_embeds(input_ids, token_embeds, image_embeds, image_token_id):
    """Replace image placeholder embeddings with actual vision features."""
    n_image_tokens = int((input_ids == image_token_id).sum().item())
    if n_image_tokens == 0:
        return token_embeds
    if image_embeds.shape[0] != n_image_tokens:
        raise ValueError(
            f"Image tokens/features mismatch: tokens={n_image_tokens}, "
            f"features={image_embeds.shape[0]}"
        )
    if image_embeds.shape[-1] != token_embeds.shape[-1]:
        raise ValueError(
            "Vision embedding hidden size mismatch: "
            f"vision_output_dim={image_embeds.shape[-1]}, "
            f"text_embedding_dim={token_embeds.shape[-1]}."
        )
    image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(token_embeds)
    image_embeds = image_embeds.to(token_embeds.device, token_embeds.dtype)
    return token_embeds.masked_scatter(image_mask, image_embeds)


class SamplingManager:
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        min_tokens_to_keep: int = 1,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.min_tokens_to_keep = min_tokens_to_keep

    def apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            raise ValueError("Temperature must larger than 0")
        return logits / self.temperature

    def apply_repetition_penalty(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        if self.repetition_penalty == 1.0 or not previous_tokens:
            return logits
        adjusted_logits = logits.copy()
        for token_id in set(previous_tokens):
            if 0 <= token_id < len(logits):
                if logits[token_id] < 0:
                    adjusted_logits[token_id] = (
                        logits[token_id] * self.repetition_penalty
                    )
                else:
                    adjusted_logits[token_id] = (
                        logits[token_id] / self.repetition_penalty
                    )
        return adjusted_logits

    def apply_top_k(self, probs: np.ndarray) -> np.ndarray:
        if self.top_k is None or self.top_k <= 0:
            return probs
        top_k = min(self.top_k, len(probs))
        top_k_indices = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.ones_like(probs, dtype=bool)
        mask[top_k_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0
        if np.sum(filtered_probs) > 0:
            return filtered_probs / np.sum(filtered_probs)
        return np.ones_like(probs) / len(probs)

    def apply_top_p(self, probs: np.ndarray) -> np.ndarray:
        if self.top_p >= 1.0:
            return probs
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_indices = np.where(cumulative_probs >= self.top_p)[0]
        if len(cutoff_indices) > 0:
            cutoff_index = max(cutoff_indices[0], self.min_tokens_to_keep - 1)
            selected_indices = sorted_indices[: cutoff_index + 1]
        else:
            selected_indices = sorted_indices
        mask = np.ones_like(probs, dtype=bool)
        mask[selected_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0
        if np.sum(filtered_probs) > 0:
            return filtered_probs / np.sum(filtered_probs)
        return np.ones_like(probs) / len(probs)

    def process_logits(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        probs = self.apply_repetition_penalty(logits.copy(), previous_tokens)
        probs = self.apply_top_k(probs)
        probs = self.apply_top_p(probs)
        probs = self.apply_temperature(probs)
        return probs

    def sample(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        logits = logits[0]
        if HOUMO_TARGET == "xh2":
            logits = logits[0]
        probs = self.process_logits(logits, previous_tokens)
        if np.all(probs == 0):
            probs = np.ones_like(probs) / len(probs)
        sampled_index = int(probs.argmax(-1))
        return np.array([[sampled_index]])


def show_ttft_breakdown(
    ttft_time: float,
    perf_tracker: InferencePerformanceTracker,
    nested_timings: Optional[dict[str, float]] = None,
    extra_timings: Optional[dict[str, float]] = None,
) -> None:
    metrics = perf_tracker.current_metrics
    ttft_ms = ttft_time * 1000
    components = [
        ("Prompt/Tokenize", metrics.prefill_perf_infos.tokenizer_time),
        ("Vision Preprocess", metrics.vision_perf_infos.vision_preprocess_time),
        ("Vision SetInput", metrics.vision_perf_infos.setinput_time),
        ("Vision Infer", metrics.vision_perf_infos.infer_time),
        ("Vision GetOutput", metrics.vision_perf_infos.getoutput_time),
        ("Prefill Embedding", metrics.prefill_perf_infos.embedding_time),
        ("Prefill SetInput", metrics.prefill_perf_infos.setinput_time),
        ("Prefill Infer", metrics.prefill_perf_infos.infer_time),
        ("Prefill GetOutput", metrics.prefill_perf_infos.getoutput_time),
    ]
    tracked_ms = sum(value for _, value in components)
    if extra_timings:
        tracked_ms += sum(extra_timings.values())
    residual_ms = max(ttft_ms - tracked_ms, 0.0)
    logger.success("TTFT Breakdown:")
    for label, value in components:
        if value > 0:
            logger.success(f"  {label}: {value:.3f} ms")
    if nested_timings:
        logger.success("  Debug Details (included above):")
        for label, value in nested_timings.items():
            if value > 0:
                logger.success(f"    {label}: {value:.3f} ms")
    if extra_timings:
        logger.success("  Debug Details (newly split from residual):")
        for label, value in extra_timings.items():
            if value > 0:
                logger.success(f"    {label}: {value:.3f} ms")
    if residual_ms > 0:
        logger.success(f"  Other/Untracked: {residual_ms:.3f} ms")
    logger.success(f"  Total TTFT: {ttft_ms:.3f} ms")


class HmPaddleOCRVL:
    def __init__(
        self,
        prefill_path,
        decode_path,
        embedding_path,
        tokenizer_dir,
        ndevice=1,
        vision_path=None,
    ):
        self.perf_tracker = InferencePerformanceTracker()
        self.ndevice = ndevice
        self.vision = None
        self.rope_deltas = None
        dev_manager = tcim.runtime.DevManager(get_hm_devices(self.ndevice), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self.prefill = tcim.runtime.load(prefill_path, option=option1)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)
        logger.info("prefill model loaded")
        dummy_tensor_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_name = self.prefill.get_input_name(i)
            if "model_layers" in input_name:
                dummy_tensor_names.append(input_name)
        option2.set_dummy_tensors(dummy_tensor_names)
        self.perf_tracker.perf_start(PERFTYPE.DECODE_LOAD_TIME)
        self.decode = tcim.runtime.load(decode_path, option=option2)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_LOAD_TIME)
        logger.info("decode model loaded")
        self.samplingmanager = SamplingManager(
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            repetition_penalty=args.repetition_penalty,
        )
        self.prefill_length = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        self.embedding_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]
        self.context_max_length = self.decode.get_input_info(
            self.decode.get_input_name(7)
        ).shape[2]

        self.prefill_input_name = self.prefill.get_input_name(0)
        self.prefill_time_position_ids_name = self.prefill.get_input_name(1)
        self.prefill_height_position_ids_name = self.prefill.get_input_name(2)
        self.prefill_width_position_ids_name = self.prefill.get_input_name(3)
        self.prefill_valid_length_name = self.prefill.get_input_name(4)
        self.prefill_current_length_name = self.prefill.get_input_name(5)
        self.prefill_time_position_ids_info = self.prefill.get_input_info(
            self.prefill_time_position_ids_name
        )
        self.prefill_height_position_ids_info = self.prefill.get_input_info(
            self.prefill_height_position_ids_name
        )
        self.prefill_width_position_ids_info = self.prefill.get_input_info(
            self.prefill_width_position_ids_name
        )

        self.decode_input_name = self.decode.get_input_name(0)
        self.decode_time_position_ids_name = self.decode.get_input_name(1)
        self.decode_height_position_ids_name = self.decode.get_input_name(2)
        self.decode_width_position_ids_name = self.decode.get_input_name(3)
        self.decode_valid_length_name = self.decode.get_input_name(4)
        self.decode_current_length_name = self.decode.get_input_name(5)
        self.decode_time_position_ids_info = self.decode.get_input_info(
            self.decode_time_position_ids_name
        )
        self.decode_height_position_ids_info = self.decode.get_input_info(
            self.decode_height_position_ids_name
        )
        self.decode_width_position_ids_info = self.decode.get_input_info(
            self.decode_width_position_ids_name
        )

        self.batch = self.decode.get_input_info(self.decode.get_input_name(0)).shape[0]
        for i in range(self.prefill.get_num_inputs()):
            input_name = self.prefill.get_input_name(i)
            if "model_layers" in input_name:
                cache = self.prefill.get_dev_input(input_name)
                self.decode.set_dev_input(input_name, cache)
        self.clear_cache()
        self.decode.set_input(
            self.decode_current_length_name, np.array([1], dtype=np.int32)
        )

        processor_height = args.max_size_h
        processor_width = args.max_size_w
        if vision_path and os.path.exists(vision_path):
            option_v = tcim.runtime.Option(weight_manager)
            self.perf_tracker.perf_start(PERFTYPE.VISION_LOAD_TIME)
            self.vision = tcim.runtime.load(vision_path, option=option_v)
            self.perf_tracker.perf_end(PERFTYPE.VISION_LOAD_TIME)
            logger.info("vision model loaded")
            vision_input_shape = self.vision.get_input_info(
                self.vision.get_input_name(0)
            ).shape
            if len(vision_input_shape) >= 5:
                processor_height = int(vision_input_shape[-2])
                processor_width = int(vision_input_shape[-1])

        self.processor = build_processor(
            tokenizer_dir=tokenizer_dir,
            max_size_h=processor_height,
            max_size_w=processor_width,
            patch_size=args.patch_size,
        )
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = int(
            self.tokenizer.convert_tokens_to_ids("<|IMAGE_PLACEHOLDER|>")
        )
        self.video_token_id = int(
            self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        )
        self.vision_start_token_id = int(
            self.tokenizer.convert_tokens_to_ids("<|IMAGE_START|>")
        )
        self.spatial_merge_size = SPATIAL_MERGE_SIZE
        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=False
        )
        if isinstance(embedding_weight, torch.nn.Embedding):
            self.embedding_weight = embedding_weight.weight.reshape(
                -1, self.embedding_len
            ).float()
        elif isinstance(embedding_weight, torch.Tensor):
            self.embedding_weight = embedding_weight.reshape(-1, self.embedding_len).float()
        elif (
            isinstance(embedding_weight, (dict, OrderedDict))
            and "weight" in embedding_weight
        ):
            self.embedding_weight = embedding_weight["weight"].reshape(
                -1, self.embedding_len
            ).float()
        else:
            raise ValueError("Unsupported embedding weight format in the checkpoint")
        self.context_length = 0
        self.vision_time = 0
        self.ttft_debug_nested_timings = {}
        self.ttft_debug_extra_timings = {}
        self.perf_tracker.reset_perf_time()
        if self.vision is not None:
            self.validate_vision_output_dim()
        self.perf_tracker.reset_perf_time()

    def create_linear_attn_mask(
        self, fill_length: int, new_cache_length: int
    ) -> np.ndarray:
        mask = np.zeros((1, fill_length), dtype=np.float16)
        mask[0, :new_cache_length] = 1.0
        return mask

    def clear_cache(self):
        for i in range(self.prefill.get_num_inputs()):
            input_name = self.prefill.get_input_name(i)
            if "model_layers" in input_name:
                info = self.prefill.get_dev_input(input_name).info
                zeros = np.zeros(info.shape, dtype=info.dtype)
                self.prefill.set_input(input_name, zeros)
                self.decode.set_input(input_name, zeros)

    def validate_vision_output_dim(self):
        if self.vision is None:
            return
        vision_output_shape = tuple(
            self.vision.get_output_info(self.vision.get_output_name(0)).shape
        )
        if not vision_output_shape:
            raise ValueError("Vision model output shape is empty")
        if len(vision_output_shape) < 2:
            raise ValueError(
                f"Unexpected vision model output shape: {vision_output_shape}"
            )
        last_dim = int(vision_output_shape[-1])
        if last_dim == self.embedding_len:
            self.vision_output_layout = "token_major"
            return
        self.vision_output_layout = None
        raise ValueError(
            "Vision model output hidden size mismatch: "
            f"vision_output={vision_output_shape}, expected_last_dim={self.embedding_len}."
        )

    def run_vision(
        self, hm_pixel_values: Union[Sequence[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if self.vision is None:
            raise RuntimeError("Vision model not loaded. Provide --vision_path.")
        if isinstance(hm_pixel_values, torch.Tensor):
            hm_pixel_values = [hm_pixel_values]
        image_embeds_list = []
        vision_input_name = self.vision.get_input_name(0)
        vision_input_shape = list(self.vision.get_input_info(vision_input_name).shape)
        for pixel_values in hm_pixel_values:
            pixel_values_np = pixel_values.detach().cpu().numpy().astype(np.float16)
            if list(pixel_values_np.shape) != vision_input_shape:
                if pixel_values_np.size != int(np.prod(vision_input_shape)):
                    raise ValueError(
                        f"Vision model input shape mismatch: got {tuple(pixel_values_np.shape)}, expected {tuple(vision_input_shape)}"
                    )
                pixel_values_np = pixel_values_np.reshape(vision_input_shape)
            self.perf_tracker.perf_start(PERFTYPE.VISION_INPUT_TIME)
            self.vision.set_input(vision_input_name, pixel_values_np)
            self.perf_tracker.perf_end(PERFTYPE.VISION_INPUT_TIME)
            self.perf_tracker.perf_start(PERFTYPE.VISION_INFER_TIME)
            self.vision.run()
            self.vision.sync()
            self.perf_tracker.perf_end(PERFTYPE.VISION_INFER_TIME)
            self.perf_tracker.perf_start(PERFTYPE.VISION_OUTPUT_TIME)
            image_embeds = self.vision.get_output(self.vision.get_output_name(0)).numpy()
            self.perf_tracker.perf_end(PERFTYPE.VISION_OUTPUT_TIME)
            image_embeds = torch.from_numpy(image_embeds).float()
            if image_embeds.dim() == 3 and image_embeds.shape[0] == 1:
                image_embeds = image_embeds.squeeze(0)
            image_embeds_list.append(image_embeds)
        return torch.cat(image_embeds_list, dim=0)

    def maybe_fix_image_token_id(
        self, input_ids: torch.Tensor, image_embeds: torch.Tensor
    ):
        token_count = int((input_ids == self.image_token_id).sum().item())
        if token_count == int(image_embeds.shape[0]):
            return
        unique_ids, counts = torch.unique(input_ids, return_counts=True)
        matched = unique_ids[counts == int(image_embeds.shape[0])]
        if matched.numel() == 1:
            self.image_token_id = int(matched[0].item())
            logger.warning(
                f"Adjusted image_token_id to {self.image_token_id} based on processor output"
            )
            return
        raise ValueError(
            f"Image token count mismatch: configured={token_count}, feature_tokens={image_embeds.shape[0]}"
        )

    def get_rope_index_text(
        self,
        valid_length: int,
        current_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_1d = torch.arange(
            valid_length, valid_length + current_length, dtype=torch.long
        )
        position_ids = pos_1d.unsqueeze(0).unsqueeze(0).expand(3, 1, current_length)
        return position_ids, torch.tensor([[0]], dtype=torch.long)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            image_grid_thw_list = image_grid_thw.tolist() if image_grid_thw is not None else None
            video_grid_thw_list = video_grid_thw.tolist() if video_grid_thw is not None else None
            mrope_position_deltas = []
            for batch_index, ids in enumerate(total_input_ids):
                ids = ids[attention_mask[batch_index] == 1]
                vision_start_indices = torch.argwhere(
                    ids == self.vision_start_token_id
                ).squeeze(1)
                vision_tokens = ids[vision_start_indices + 1]
                image_nums = int((vision_tokens == self.image_token_id).sum().item())
                video_nums = int((vision_tokens == self.video_token_id).sum().item())
                input_tokens = ids.tolist()
                llm_pos_ids_list = []
                start = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    end_image = (
                        input_tokens.index(self.image_token_id, start)
                        if self.image_token_id in input_tokens and remain_images > 0
                        else len(input_tokens) + 1
                    )
                    end_video = (
                        input_tokens.index(self.video_token_id, start)
                        if self.video_token_id in input_tokens and remain_videos > 0
                        else len(input_tokens) + 1
                    )
                    if end_image < end_video:
                        t, h, w = image_grid_thw_list[image_index]
                        image_index += 1
                        remain_images -= 1
                        end = end_image
                    else:
                        t, h, w = video_grid_thw_list[video_index]
                        video_index += 1
                        remain_videos -= 1
                        end = end_video
                    llm_grid_t = t
                    llm_grid_h = h // self.spatial_merge_size
                    llm_grid_w = w // self.spatial_merge_size
                    text_len = end - start
                    start_index = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device)
                        .view(1, -1)
                        .expand(3, -1)
                        + start_index
                    )
                    t_index = (
                        torch.arange(llm_grid_t, device=input_ids.device)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h, device=input_ids.device)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w, device=input_ids.device)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + start_index
                    )
                    start = end + llm_grid_t * llm_grid_h * llm_grid_w
                if start < len(input_tokens):
                    start_index = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - start
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device)
                        .view(1, -1)
                        .expand(3, -1)
                        + start_index
                    )
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., batch_index, attention_mask[batch_index] == 1] = llm_positions
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[batch_index])
                )
            return position_ids, torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        return position_ids, torch.zeros([input_ids.shape[0], 1], dtype=input_ids.dtype, device=input_ids.device)

    def _set_prefill_position_inputs(self, chunk_pos: torch.Tensor):
        time_pos = chunk_pos[0].numpy().astype(np.float16)
        if len(self.prefill_time_position_ids_info.shape) > len(time_pos.shape):
            time_pos = time_pos[np.newaxis, :]
        self.prefill.set_input(self.prefill_time_position_ids_name, time_pos)

        height_pos = chunk_pos[1].numpy().astype(np.float16)
        if len(self.prefill_height_position_ids_info.shape) > len(height_pos.shape):
            height_pos = height_pos[np.newaxis, :]
        self.prefill.set_input(self.prefill_height_position_ids_name, height_pos)

        width_pos = chunk_pos[2].numpy().astype(np.float16)
        if len(self.prefill_width_position_ids_info.shape) > len(width_pos.shape):
            width_pos = width_pos[np.newaxis, :]
        self.prefill.set_input(self.prefill_width_position_ids_name, width_pos)

    def chat(self, question, image_paths):
        if not image_paths:
            raise ValueError("PaddleOCR-VL demo requires image input")
        if self.vision is None:
            raise RuntimeError("Vision model not loaded. Provide --vision_path.")
        self.generated_ids = []
        if not args.history:
            self.context_length = 0
            self.clear_cache()
        self.prefill_time = 0
        self.decode_time = 0
        self.ttft_time = 0
        self.vision_time = 0
        self.ttft_debug_nested_timings = {}
        self.ttft_debug_extra_timings = {}
        self.rope_deltas = None
        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))
        logger.info(f"images: {image_paths}")

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOKEN_TIME)
        start_time = time.time()

        messages = build_messages(
            prompt=question,
            image_paths=image_paths,
            max_size_h=self.processor.image_processor.fixed_output_height,
            max_size_w=self.processor.image_processor.fixed_output_width,
        )
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOKEN_TIME)

        self.perf_tracker.perf_start(PERFTYPE.VISION_TOTAL_TIME)
        vision_start = time.time()
        self.perf_tracker.perf_start(PERFTYPE.VISION_PREPROCESS_TIME)
        image_inputs = load_images(image_paths)
        model_inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        all_input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        image_grid_thw = model_inputs.get("image_grid_thw", None)
        pixel_values = model_inputs.get("pixel_values", None)
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("Processor output missing pixel_values or image_grid_thw")
        hm_pixel_values = pixel_values.unsqueeze(0) if pixel_values.ndim == 4 else pixel_values
        self.perf_tracker.perf_end(PERFTYPE.VISION_PREPROCESS_TIME)

        input_echo_len = int(all_input_ids.shape[1])
        if input_echo_len >= self.context_max_length:
            raise ValueError(
                f"Input length {input_echo_len} exceeds max {self.context_max_length}!"
            )

        logger.info(f"processor image_grid_thw: {image_grid_thw}")
        logger.info(
            f"processor image token count: {int((all_input_ids == self.image_token_id).sum().item())}"
        )
        image_embeds = self.run_vision(hm_pixel_values)
        self.perf_tracker.perf_end(PERFTYPE.VISION_TOTAL_TIME)
        self.vision_time = time.time() - vision_start
        logger.info(f"image_embeds shape: {image_embeds.shape}")

        self.maybe_fix_image_token_id(all_input_ids, image_embeds)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
        token_embeds = F.embedding(all_input_ids, self.embedding_weight)
        scatter_start = time.time()
        token_embeds = scatter_image_embeds(
            all_input_ids, token_embeds, image_embeds, self.image_token_id
        )
        self.ttft_debug_nested_timings["Scatter Image Embeds"] = (
            time.time() - scatter_start
        ) * 1000
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)

        rope_index_start = time.time()
        position_ids, self.rope_deltas = self.get_rope_index(
            all_input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        self.ttft_debug_extra_timings["Get Rope Index"] = (
            time.time() - rope_index_start
        ) * 1000
        position_ids = position_ids + self.context_length

        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        for round_idx in range(prefill_loop_round):
            valid_length = round_idx * self.prefill_length + self.context_length
            if round_idx == prefill_loop_round - 1:
                current_length = input_echo_len - round_idx * self.prefill_length
                chunk_end = input_echo_len
            else:
                current_length = self.prefill_length
                chunk_end = (round_idx + 1) * self.prefill_length
            chunk_start = round_idx * self.prefill_length
            chunk_embeds = token_embeds[:, chunk_start:chunk_end]
            effective_length = chunk_embeds.size(1)
            pad_embeds = torch.zeros(
                1,
                self.prefill_length - effective_length,
                chunk_embeds.size(-1),
                dtype=chunk_embeds.dtype,
                device=chunk_embeds.device,
            )
            input_data = torch.cat([chunk_embeds, pad_embeds], dim=1).reshape(
                1, self.prefill_length, self.embedding_len
            )
            chunk_pos = position_ids[:, 0, chunk_start:chunk_end]
            if chunk_pos.size(-1) < self.prefill_length:
                pad_pos = chunk_pos[:, -1:].expand(-1, self.prefill_length - chunk_pos.size(-1))
                chunk_pos = torch.cat([chunk_pos, pad_pos], dim=-1)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
            self.prefill.set_input(
                self.prefill_input_name,
                input_data.detach().cpu().numpy().astype(np.float16),
            )
            self.prefill.set_input(
                self.prefill_valid_length_name, np.array([valid_length], dtype=np.int32)
            )
            self.prefill.set_input(
                self.prefill_current_length_name, np.array([current_length], dtype=np.int32)
            )
            self._set_prefill_position_inputs(chunk_pos)
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
            prefill_start = time.time()
            self.prefill.run()
            self.prefill.sync()
            self.prefill_time += time.time() - prefill_start
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
        logits = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)
        next_id = int(logits.argmax(-1)[0][0])
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)

        prefill_response = self.tokenizer.decode([next_id], clean_up_tokenization_spaces=False)
        logger.success("response:")
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)
        self.ttft_time = time.time() - start_time

        chat_history_ids = all_input_ids[0]
        next_id_tensor = torch.tensor([next_id], dtype=torch.long)
        self.generated_ids.append(next_id)
        chat_history_ids = torch.cat([chat_history_ids, next_id_tensor], dim=-1)
        self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
        input_data = F.embedding(next_id_tensor.unsqueeze(0), self.embedding_weight).reshape(
            1, 1, -1
        )
        self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)

        all_response = prefill_response
        self.context_length += input_echo_len
        decode_count = 0
        skip_tokens = 0
        slide_len = 10
        last_response = self.tokenizer.decode(
            chat_history_ids.tolist()[-slide_len:], clean_up_tokenization_spaces=False
        )
        decode_response = ""
        while True:
            if self.context_length >= self.context_max_length:
                logger.info(f"context length greater than {self.context_max_length}, break!")
                break
            if decode_count >= args.max_new_tokens - 1:
                logger.info(f"reach max_new_tokens={args.max_new_tokens}, break!")
                break
            self.perf_tracker.perf_start(PERFTYPE.DECODE_TOTAL_TIME)
            self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)

            decode_pos = self.context_length + int(self.rope_deltas.item())
            time_position_ids_data = np.array([decode_pos], dtype=np.float16)
            hight_position_ids_data = time_position_ids_data.copy()
            width_position_ids_data = time_position_ids_data.copy()
            if len(self.decode_time_position_ids_info.shape) > len(time_position_ids_data.shape):
                time_position_ids_data = time_position_ids_data[np.newaxis, :]
            if len(self.decode_height_position_ids_info.shape) > len(hight_position_ids_data.shape):
                hight_position_ids_data = hight_position_ids_data[np.newaxis, :]
            if len(self.decode_width_position_ids_info.shape) > len(width_position_ids_data.shape):
                width_position_ids_data = width_position_ids_data[np.newaxis, :]

            self.decode.set_input(
                self.decode_input_name,
                input_data.detach().cpu().numpy().astype(np.float16),
            )
            self.decode.set_input(self.decode_time_position_ids_name, time_position_ids_data)
            self.decode.set_input(self.decode_height_position_ids_name, hight_position_ids_data)
            self.decode.set_input(self.decode_width_position_ids_name, width_position_ids_data)
            self.decode.set_input(
                self.decode_valid_length_name, np.array([self.context_length], dtype=np.int32)
            )
            self.perf_tracker.perf_end(PERFTYPE.DECODE_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_INFER_TIME)
            decode_start = time.time()
            self.decode.run()
            self.decode.sync()
            self.decode_time += time.time() - decode_start
            self.perf_tracker.perf_end(PERFTYPE.DECODE_INFER_TIME)
            self.perf_tracker.perf_start(PERFTYPE.DECODE_OUTPUT_TIME)
            decode_logits = self.decode.get_output(self.decode.get_output_name(0)).numpy()
            self.perf_tracker.perf_end(PERFTYPE.DECODE_OUTPUT_TIME)

            decode_count += 1
            self.perf_tracker.perf_start(PERFTYPE.DECODE_TOKEN_TIME)
            next_id_np = self.samplingmanager.sample(decode_logits, self.generated_ids)
            next_id = int(next_id_np[0][0])
            if next_id == self.tokenizer.eos_token_id:
                if decode_response:
                    print(decode_response, end="", flush=True)
                    all_response += decode_response
                self.perf_tracker.perf_end(PERFTYPE.DECODE_TOKEN_TIME)
                self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)
                break

            next_id_tensor = torch.tensor([next_id], dtype=torch.long)
            chat_history_ids = torch.cat([chat_history_ids, next_id_tensor], dim=-1)
            self.generated_ids.append(next_id)
            decode_response = self.tokenizer.decode(
                chat_history_ids.tolist()[-(slide_len + 1) - skip_tokens :],
                clean_up_tokenization_spaces=False,
            )[len(last_response) :]
            self.perf_tracker.perf_end(PERFTYPE.DECODE_TOKEN_TIME)
            self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)

            if decode_response != "" and is_valid_char(ord(decode_response[-1])):
                print(decode_response, end="", flush=True)
                all_response += decode_response
                last_response = self.tokenizer.decode(
                    chat_history_ids.tolist()[-slide_len:],
                    clean_up_tokenization_spaces=False,
                )
                skip_tokens = 0
                decode_response = ""
            else:
                skip_tokens += 1

            self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
            input_data = F.embedding(next_id_tensor.unsqueeze(0), self.embedding_weight).reshape(
                1, 1, -1
            )
            self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)
            self.context_length += 1

        print("\033[0m")
        self.perf_tracker.set_basic_info(
            batch_size=1,
            input_seq_length=input_echo_len,
            output_seq_length=decode_count + 1,
            num_images=len(image_paths),
        )
        return all_response, input_echo_len, decode_count + 1


if __name__ == "__main__":
    args = get_args()
    hm_paddleocr_vl = HmPaddleOCRVL(
        args.prefill_path,
        args.decode_path,
        args.embedding_path,
        args.tokenizer_dir,
        args.ndevice,
        vision_path=args.vision_path,
    )
    question = build_default_vision_prompt(args.task, prompt=args.prompt)
    start_time = time.time()
    try:
        response, input_tokens, output_tokens = hm_paddleocr_vl.chat(
            question,
            image_paths=args.image_path,
        )
        del response, input_tokens, output_tokens
        total_time = time.time() - start_time
        logger.info(f"total_time: {total_time:.3f}s")
        if args.debug:
            show_ttft_breakdown(
                hm_paddleocr_vl.ttft_time,
                hm_paddleocr_vl.perf_tracker,
                nested_timings=hm_paddleocr_vl.ttft_debug_nested_timings,
                extra_timings=hm_paddleocr_vl.ttft_debug_extra_timings,
            )
        hm_paddleocr_vl.perf_tracker.show_summary()
    except Exception as e:
        print(f"Program execution failed: {e}")
        raise
