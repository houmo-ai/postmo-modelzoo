#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Qwen3.5 Inference Demo - Python script for running Qwen3.5
# Inference on HOUMO AI device.
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
import re
import sys
import math
import time
import argparse
from pathlib import Path
from typing import Any, List, Tuple, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.image_processing_utils import BatchFeature
from transformers.video_processing_utils import BaseVideoProcessor
from loguru import logger

import tcim_lite as tcim

from hmatc.utils.perf_infomations import (
    InferencePerformanceTracker,
    InferenceMetrics,
    PERFTYPE,
)
from hmatc.python.get_hm_devices import get_hm_devices
from hmatc.utils.utils import first_not_none, get_model_configs
from processing_qwen3_5 import Qwen3_5Processor
from image_processing_qwen3_5 import Qwen3_5ImageProcessor
from vision_process_qwen3_5 import process_vision_info

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
HOUMO_EXAMPLES_PATH = os.getenv("HOUMO_EXAMPLES_PATH")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

IMAGE_TOKEN_ID = 248056
VIDEO_TOKEN_ID = 248057
VISION_START_TOKEN_ID = 248053
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2


def get_default_tokenizer_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "qwen3.6").upper()
    model_size = model_config.get("model_size", "35b-a3b").upper()
    return f"{model_name}-{model_size}"


def is_valid_char(cp):
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x20000 and cp <= 0x2A6DF)
        or (cp >= 0x2A700 and cp <= 0x2B73F)
        or (cp >= 0x2B740 and cp <= 0x2B81F)
        or (cp >= 0x2B820 and cp <= 0x2CEAF)
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
        or (0x0041 <= cp and cp <= 0x005A)
        or (0x0061 <= cp and cp <= 0x007A)
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
        help="tokenizer dir",
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
        help="device number",
    )
    parser.add_argument(
        "--question",
        dest="question",
        type=str,
        default="请介绍一下存算一体技术的优势",
        help="question to ask",
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
        "--top_k",
        dest="topk",
        type=int,
        default=None,
        help="sampling top-k",
    )
    parser.add_argument(
        "--topp",
        "--top_p",
        dest="topp",
        type=float,
        default=1.0,
        help="sampling top-p",
    )
    parser.add_argument(
        "--min_p",
        dest="min_p",
        type=float,
        default=0.0,
        help="sampling min-p",
    )
    parser.add_argument(
        "--presence_penalty",
        dest="presence_penalty",
        type=float,
        default=1.5,
        help="sampling presence_penalty",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=1.0,
        help="sampling temperature",
    )
    parser.add_argument(
        "--image_path",
        dest="image_path",
        nargs="+",
        action="append",
        default=[[os.path.join(HOUMO_EXAMPLES_PATH, "data/pic/beach.jpeg")]],
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
        "--max_size_t",
        dest="max_size_t",
        type=int,
        default=None,
        help="max temporal size for vision filename suffix",
    )
    parser.add_argument(
        "--patch_size",
        dest="patch_size",
        type=int,
        default=16,
        help="vision patch size",
    )
    parser.add_argument(
        "--it",
        dest="it",
        action="store_true",
        help="interactive mode",
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
        args.max_size_w, model_config.get("max_size_w", 896)
    )
    args.max_size_h = first_not_none(
        args.max_size_h, model_config.get("max_size_h", 896)
    )
    args.max_size_t = first_not_none(args.max_size_t, model_config.get("max_size_t", 2))
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
            f"{args.model_name}-{args.model_size}_visual_{args.max_size_w}x{args.max_size_h}x{args.max_size_t}.hmm",
        )
        fallback_vision_path = os.path.join(
            "output", HOUMO_TARGET, f"{args.model_name}-{args.model_size}_visual.hmm"
        )
        if not os.path.exists(args.vision_path) and os.path.exists(
            fallback_vision_path
        ):
            args.vision_path = fallback_vision_path
    if args.ndevice > 1:
        if args.prefill_path.endswith(".hmm"):
            args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        if args.decode_path.endswith(".hmm"):
            args.decode_path = args.decode_path.replace(".hmm", ".hmms")
    args.image_path = normalize_image_inputs(args.image_path)
    return args


class _DummyVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __call__(self, videos=None, **kwargs):
        del videos, kwargs
        return BatchFeature(data={})


def build_processor(
    tokenizer_dir: str, max_size_h: int, max_size_w: int, patch_size: int
) -> Qwen3_5Processor:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        chat_template_file = Path(tokenizer_dir) / "chat_template.jinja"
        if chat_template_file.exists():
            chat_template = chat_template_file.read_text(encoding="utf-8")

    image_processor = Qwen3_5ImageProcessor(
        patch_size=patch_size,
        merge_size=SPATIAL_MERGE_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        min_pixels=max_size_h * max_size_w,
        max_pixels=max_size_h * max_size_w,
    )
    return Qwen3_5Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=_DummyVideoProcessor(),
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
    messages.append(
        {
            "role": "user",
            "content": content,
        }
    )
    return messages


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


def prompt_for_image_paths() -> Optional[List[str]]:
    image_paths = []
    image_index = 1

    while True:
        raw_value = input(
            f"Input image path #{image_index} for this turn (press Enter to run): "
        ).strip()
        if not raw_value:
            break

        normalized_paths = normalize_image_inputs([[raw_value]])
        if normalized_paths is None:
            continue

        image_paths.extend(normalized_paths)
        image_index += 1

    return image_paths or None


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

    image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(token_embeds)
    image_embeds = image_embeds.to(token_embeds.device, token_embeds.dtype)
    return token_embeds.masked_scatter(image_mask, image_embeds)


class SamplingManager:
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_tokens_to_keep: int = 1,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.min_tokens_to_keep = min_tokens_to_keep

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            raise ValueError("Temperature must larger than 0")

        return logits / self.temperature

    @staticmethod
    def iter_token_ids(tokens: Optional[Sequence[Any]] = None) -> List[int]:
        if tokens is None:
            return []

        token_ids = []
        for token in tokens:
            if isinstance(token, torch.Tensor):
                token_ids.extend(int(item) for item in token.detach().cpu().reshape(-1))
            elif isinstance(token, np.ndarray):
                token_ids.extend(int(item) for item in token.reshape(-1))
            elif isinstance(token, (list, tuple)):
                token_ids.extend(SamplingManager.iter_token_ids(token))
            else:
                token_ids.append(int(token))
        return token_ids

    def apply_repetition_penalty(
        self, logits: np.ndarray, previous_tokens: Optional[Sequence[Any]] = None
    ) -> np.ndarray:
        if self.repetition_penalty == 1.0 or not previous_tokens:
            return logits

        adjusted_logits = logits.copy()
        for token_id in set(self.iter_token_ids(previous_tokens)):
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

    def apply_presence_penalty(
        self, logits: np.ndarray, previous_tokens: Optional[Sequence[Any]] = None
    ) -> np.ndarray:
        if self.presence_penalty == 0.0 or not previous_tokens:
            return logits

        adjusted_logits = logits.copy()
        for token_id in set(self.iter_token_ids(previous_tokens)):
            if 0 <= token_id < len(logits):
                adjusted_logits[token_id] = logits[token_id] - self.presence_penalty

        return adjusted_logits

    def apply_top_k(self, probs: np.ndarray) -> np.ndarray:
        if self.top_k is None or self.top_k <= 0:
            return probs

        top_k = min(self.top_k, len(probs))

        if top_k <= 0:
            return probs

        top_k_indices = np.argpartition(probs, -top_k)[-top_k:]

        mask = np.ones_like(probs, dtype=bool)
        mask[top_k_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0

        if np.sum(filtered_probs) > 0:
            normalized_probs = filtered_probs / np.sum(filtered_probs)
        else:
            normalized_probs = np.ones_like(probs) / len(probs)

        return normalized_probs

    def apply_top_p(self, probs: np.ndarray) -> np.ndarray:
        if self.top_p >= 1.0:
            return probs

        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        cumulative_probs = np.cumsum(sorted_probs)

        cutoff_indices = np.where(cumulative_probs >= self.top_p)[0]

        if len(cutoff_indices) > 0:
            cutoff_index = cutoff_indices[0]
            if cutoff_index < self.min_tokens_to_keep - 1:
                cutoff_index = self.min_tokens_to_keep - 1

            selected_indices = sorted_indices[: cutoff_index + 1]
        else:
            selected_indices = sorted_indices

        mask = np.ones_like(probs, dtype=bool)
        mask[selected_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0

        if np.sum(filtered_probs) > 0:
            normalized_probs = filtered_probs / np.sum(filtered_probs)
        else:
            normalized_probs = np.ones_like(probs) / len(probs)

        return normalized_probs

    def apply_min_p(self, probs: np.ndarray) -> np.ndarray:
        if self.min_p <= 0.0:
            return probs

        max_prob = np.max(probs)
        threshold = max_prob * self.min_p
        selected_indices = np.where(probs >= threshold)[0]

        if selected_indices.size < self.min_tokens_to_keep:
            keep_count = min(self.min_tokens_to_keep, len(probs))
            selected_indices = np.argpartition(probs, -keep_count)[-keep_count:]

        mask = np.ones_like(probs, dtype=bool)
        mask[selected_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0

        if np.sum(filtered_probs) > 0:
            normalized_probs = filtered_probs / np.sum(filtered_probs)
        else:
            normalized_probs = np.ones_like(probs) / len(probs)

        return normalized_probs

    def process_logits(
        self, logits: np.ndarray, previous_tokens: Optional[Sequence[Any]] = None
    ) -> np.ndarray:
        processed_logits = logits.copy()
        # 1. apply repetition penalty
        processed_logits = self.apply_repetition_penalty(
            processed_logits, previous_tokens
        )

        # 2. apply presence penalty
        processed_logits = self.apply_presence_penalty(
            processed_logits, previous_tokens
        )

        # 3. apply softmax
        # not using softmax in case of long time cost
        probs = processed_logits
        # probs = self.softmax(processed_logits)

        # 4. apply top-k
        probs = self.apply_top_k(probs)

        # 5. apply top-p
        probs = self.apply_top_p(probs)

        # 6. apply min-p
        probs = self.apply_min_p(probs)

        # 7. apply temperature
        probs = self.apply_temperature(probs)
        return probs

    def sample(
        self, logits: np.ndarray, previous_tokens: Optional[Sequence[Any]] = None
    ) -> np.ndarray:
        logits = logits[0]
        if HOUMO_TARGET == "xh2":
            logits = logits[0]
        probs = self.process_logits(logits, previous_tokens)
        if np.all(probs == 0):
            probs = np.ones_like(probs) / len(probs)

        # sampled_index = np.random.choice(len(probs), p=probs)
        sampled_index = probs.argmax(-1)

        return np.array([[sampled_index]])

    def get_processed_probs(
        self, logits: np.ndarray, previous_tokens: Optional[Sequence[Any]] = None
    ) -> np.ndarray:
        return self.process_logits(logits, previous_tokens)


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


class HmQwen:

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
        dev_manager = tcim.runtime.DevManager(
            get_hm_devices(self.ndevice), "Xh2HalBackend"
        )
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
        self.batch = self.decode.get_input_info(self.decode.get_input_name(0)).shape[0]

        for i in range(self.prefill.get_num_inputs()):
            input_name = self.prefill.get_input_name(i)
            if "model_layers" in input_name:
                cache = self.prefill.get_dev_input(input_name)
                self.decode.set_dev_input(input_name, cache)
            if "conv_cache" in input_name:
                output_name = input_name.replace("past_conv_cache_", "conv_cache_out_")
                cache = self.prefill.get_dev_input(input_name)
                self.prefill.set_dev_output(output_name, cache)
                self.decode.set_dev_input(input_name, cache)
                self.decode.set_dev_output(output_name, cache)
            if "recurrent_state" in input_name:
                output_name = input_name.replace(
                    "past_recurrent_state_", "recurrent_state_out_"
                )
                cache = self.prefill.get_dev_input(input_name)
                self.prefill.set_dev_output(output_name, cache)
                self.decode.set_dev_input(input_name, cache)
                self.decode.set_dev_output(output_name, cache)

        self.clear_cache()

        # set decode input
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode.get_input_name(5)
        self.set_model_input(
            self.decode, decode_current_length_name, current_length_input_1
        )

        self.processor = build_processor(
            tokenizer_dir=tokenizer_dir,
            max_size_h=args.max_size_h,
            max_size_w=args.max_size_w,
            patch_size=args.patch_size,
        )
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = int(
            getattr(self.processor, "image_token_id", IMAGE_TOKEN_ID)
        )
        self.video_token_id = int(
            getattr(self.processor, "video_token_id", VIDEO_TOKEN_ID)
        )
        self.vision_start_token_id = int(
            getattr(self.processor, "vision_start_token_id", VISION_START_TOKEN_ID)
        )
        self.spatial_merge_size = SPATIAL_MERGE_SIZE
        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=False
        )
        if isinstance(embedding_weight, dict):
            if "weight" not in embedding_weight:
                raise KeyError(
                    f"Embedding state_dict at {embedding_path} does not contain 'weight'"
                )
            embedding_tensor = embedding_weight["weight"]
        else:
            embedding_tensor = embedding_weight.weight
        self.embedding_weight = embedding_tensor.reshape(-1, self.embedding_len).float()
        self.context_length = 0
        self.vision_time = 0
        self.ttft_debug_nested_timings = {}
        self.ttft_debug_extra_timings = {}

        self.perf_tracker.reset_perf_time()

        if vision_path and os.path.exists(vision_path):
            vision_suffix = Path(vision_path).suffix.lower()
            if vision_suffix == ".hmms":
                vision_devices = get_hm_devices(self.ndevice)
            elif vision_suffix == ".hmm":
                vision_devices = [0]
            else:
                raise ValueError(
                    f"Unsupported vision model suffix '{vision_suffix}': "
                    "expected .hmm for a single-device model or .hmms for a "
                    "multi-device model."
                )

            dev_manager_v = tcim.runtime.DevManager(vision_devices, "Xh2HalBackend")
            weight_manager_v = tcim.runtime.WeightManager(dev_manager_v)
            option_v = tcim.runtime.Option(weight_manager_v)
            self.perf_tracker.perf_start(PERFTYPE.VISION_LOAD_TIME)
            self.vision = tcim.runtime.load(vision_path, option=option_v)
            self.perf_tracker.perf_end(PERFTYPE.VISION_LOAD_TIME)
            logger.info("vision model loaded")

        # Start inference statistics from a clean slate for the first request.
        self.perf_tracker.reset_perf_time()

    def create_linear_attn_mask(
        self, fill_length: int, new_cache_length: int
    ) -> np.ndarray:
        """Create a linear attention mask: [1,1,...,1,0,...,0] where the number of 1s equals new_cache_length."""
        assert fill_length > 0, "fill_length must be > 0 for linear attn mask"
        assert (
            0 < new_cache_length <= fill_length
        ), f"new_cache_length({new_cache_length}) must be in (0, fill_length({fill_length})]"
        mask = np.zeros((1, fill_length), dtype=np.float16)
        mask[0, :new_cache_length] = 1.0
        return mask

    def get_model_input_shape(self, runtime_model, input_name: str) -> tuple[int, ...]:
        return tuple(int(dim) for dim in runtime_model.get_input_info(input_name).shape)

    def adapt_input_to_model_shape(self, runtime_model, input_name: str, value):
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)

        expected_shape = self.get_model_input_shape(runtime_model, input_name)
        if array.shape == expected_shape:
            return array

        expected_size = int(np.prod(expected_shape))
        if array.size != expected_size:
            raise RuntimeError(
                f"Input shape mismatch for '{input_name}': expected {expected_shape}, "
                f"got {array.shape} (size={array.size})"
            )

        if args.debug:
            logger.info(
                f"Auto-reshaping input '{input_name}' from {array.shape} to {expected_shape}"
            )
        return array.reshape(expected_shape)

    def set_model_input(self, runtime_model, input_name: str, value) -> None:
        runtime_model.set_input(
            input_name,
            self.adapt_input_to_model_shape(runtime_model, input_name, value),
        )

    def clear_cache(self):
        for i in range(self.prefill.get_num_inputs()):
            input_name = self.prefill.get_input_name(i)
            if "conv_cache" in input_name or "recurrent_state" in input_name:
                info = self.prefill.get_dev_input(input_name).info
                zeros = np.zeros(info.shape, dtype=np.float16)
                self.set_model_input(self.prefill, input_name, zeros)
                self.set_model_input(self.decode, input_name, zeros)

    def run_vision(
        self, hm_pixel_values: Union[Sequence[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if self.vision is None:
            raise RuntimeError("Vision model not loaded. Provide --vision_path.")

        if isinstance(hm_pixel_values, torch.Tensor):
            hm_pixel_values = [hm_pixel_values]

        if not hm_pixel_values:
            raise ValueError("hm_pixel_values is empty")

        image_embeds_list = []
        vision_input_name = self.vision.get_input_name(0)

        for pixel_values in hm_pixel_values:
            pixel_values_np = pixel_values.detach().cpu().numpy()
            pixel_values_np = self.adapt_input_to_model_shape(
                self.vision, vision_input_name, pixel_values_np
            )

            self.perf_tracker.perf_start(PERFTYPE.VISION_INPUT_TIME)
            self.set_model_input(self.vision, vision_input_name, pixel_values_np)
            self.perf_tracker.perf_end(PERFTYPE.VISION_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.VISION_INFER_TIME)
            self.vision.run()
            self.vision.sync()
            self.perf_tracker.perf_end(PERFTYPE.VISION_INFER_TIME)

            self.perf_tracker.perf_start(PERFTYPE.VISION_OUTPUT_TIME)
            vision_output_name = self.vision.get_output_name(0)
            image_embeds = self.vision.get_output(vision_output_name).numpy()
            self.perf_tracker.perf_end(PERFTYPE.VISION_OUTPUT_TIME)

            image_embeds = torch.from_numpy(image_embeds)
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
        assert valid_length >= 0, "valid_length must be >= 0"
        assert current_length > 0, "current_length must be > 0"

        pos_1d = torch.arange(
            valid_length, valid_length + current_length, dtype=torch.long
        )

        position_ids = pos_1d.unsqueeze(0).unsqueeze(0)
        position_ids = position_ids.expand(3, 1, current_length)

        mrope_position_deltas = torch.tensor([[0]], dtype=torch.long)

        return position_ids, mrope_position_deltas

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute M-RoPE position IDs for vision+text input.

        Ported from qwen3_5_vl_hmonnx_demo.py Qwen35VLHMONNXModel.get_rope_index.
        """
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
            image_grid_thw_list = (
                image_grid_thw.tolist() if image_grid_thw is not None else None
            )
            video_grid_thw_list = (
                video_grid_thw.tolist() if video_grid_thw is not None else None
            )

            mrope_position_deltas = []
            for batch_index, ids in enumerate(total_input_ids):
                ids = ids[attention_mask[batch_index] == 1]
                vision_start_indices = torch.argwhere(
                    ids == self.vision_start_token_id
                ).squeeze(1)
                vision_tokens = ids[vision_start_indices + 1]
                image_nums = (vision_tokens == self.image_token_id).sum()
                video_nums = (vision_tokens == self.video_token_id).sum()
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
                    start_index = (
                        llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    )

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
                        torch.stack([t_index, h_index, w_index])
                        + text_len
                        + start_index
                    )
                    start = end + llm_grid_t * llm_grid_h * llm_grid_w

                if start < len(input_tokens):
                    start_index = (
                        llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    )
                    text_len = len(input_tokens) - start
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device)
                        .view(1, -1)
                        .expand(3, -1)
                        + start_index
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., batch_index, attention_mask[batch_index] == 1] = (
                    llm_positions.to(position_ids.device)
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[batch_index])
                )

            deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, deltas

        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            deltas = max_position_ids + 1 - attention_mask.shape[-1]
            return position_ids, deltas

        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        deltas = torch.zeros(
            [input_ids.shape[0], 1], dtype=input_ids.dtype, device=input_ids.device
        )
        return position_ids, deltas

    def chat(self, question, image_paths=None):
        use_vision = image_paths is not None
        if use_vision and self.vision is None:
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
        if use_vision:
            logger.info(f"images: {image_paths}")

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOKEN_TIME)
        start_time = time.time()

        attention_mask = None
        position_ids = None
        token_embeds = None
        num_images = 0

        if use_vision:
            messages = build_messages(
                prompt=question,
                image_paths=image_paths,
                max_size_h=args.max_size_h,
                max_size_w=args.max_size_w,
                system_prompt="介绍一下这些图片",
            )
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOKEN_TIME)

            self.perf_tracker.perf_start(PERFTYPE.VISION_TOTAL_TIME)
            vision_start = time.time()
            self.perf_tracker.perf_start(PERFTYPE.VISION_PREPROCESS_TIME)
            image_inputs, video_inputs = process_vision_info(messages)
            model_inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                min_pixels=args.max_size_h * args.max_size_w,
                max_pixels=args.max_size_h * args.max_size_w,
                patch_size=args.patch_size,
                merge_size=SPATIAL_MERGE_SIZE,
                padding=True,
                return_tensors="pt",
            )
            all_input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs.get("attention_mask", None)
            image_grid_thw = model_inputs.get("image_grid_thw", None)
            hm_pixel_values = model_inputs.get("hm_pixel_values", None)
            if hm_pixel_values is None:
                raise ValueError("Processor output does not contain hm_pixel_values")
            if image_grid_thw is None:
                raise ValueError("Processor output does not contain image_grid_thw")

            logger.info(f"processor image_grid_thw: {image_grid_thw}")
            logger.info(
                f"processor image token count: {int((all_input_ids == self.image_token_id).sum().item())}"
            )
            self.perf_tracker.perf_end(PERFTYPE.VISION_PREPROCESS_TIME)

            input_echo_len = int(all_input_ids.shape[1])
            if input_echo_len >= self.context_max_length:
                logger.error(
                    f"Input length {input_echo_len} exceeds max {self.context_max_length}!"
                )
                sys.exit(1)

            logger.info("Running vision model...")
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
            num_images = (len(image_inputs) if image_inputs is not None else 0) + (
                len(video_inputs) if video_inputs is not None else 0
            )
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            all_input_ids = inputs["input_ids"]
            input_echo_len = all_input_ids.numel()
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOKEN_TIME)

            if input_echo_len >= self.context_max_length:
                logger.error(
                    f"Question long than {self.context_max_length}, please shorten it!"
                )
                sys.exit(1)

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

            if use_vision:
                chunk_embeds = token_embeds[:, chunk_start:chunk_end]
                effective_length = chunk_embeds.size(1)
                _pad_embeds = torch.zeros(
                    1,
                    self.prefill_length - effective_length,
                    chunk_embeds.size(-1),
                    dtype=chunk_embeds.dtype,
                    device=chunk_embeds.device,
                )
                input_data = torch.cat([chunk_embeds, _pad_embeds], dim=1).reshape(
                    1, self.prefill_length, self.embedding_len
                )
                chunk_pos = position_ids[:, 0, chunk_start:chunk_end]
                if chunk_pos.size(-1) < self.prefill_length:
                    pad_pos = chunk_pos[:, -1:].expand(
                        -1, self.prefill_length - chunk_pos.size(-1)
                    )
                    chunk_pos = torch.cat([chunk_pos, pad_pos], dim=-1)
            else:
                input_ids = all_input_ids[:, chunk_start:chunk_end]
                self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
                chunk_embeds = F.embedding(input_ids, self.embedding_weight)
                effective_length = input_ids.size(-1)
                _pad_embeds = torch.zeros(
                    1,
                    self.prefill_length - effective_length,
                    chunk_embeds.size(-1),
                    dtype=chunk_embeds.dtype,
                    device=chunk_embeds.device,
                )
                input_data = torch.cat([chunk_embeds, _pad_embeds], dim=1).reshape(
                    1, self.prefill_length, self.embedding_len
                )
                self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)
                text_position_ids, _ = self.get_rope_index_text(
                    valid_length, self.prefill_length
                )
                chunk_pos = text_position_ids[:, 0, :]

            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")
            linear_attn_mask_data = self.create_linear_attn_mask(
                self.prefill_length, current_length
            )
            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
            input_name = self.prefill.get_input_name(0)
            time_position_ids_name = self.prefill.get_input_name(1)
            hight_position_ids_name = self.prefill.get_input_name(2)
            width_position_ids_name = self.prefill.get_input_name(3)
            valid_length_name = self.prefill.get_input_name(4)
            current_length_name = self.prefill.get_input_name(5)
            linear_attn_mask_name = self.prefill.get_input_name(6)
            self.set_model_input(self.prefill, input_name, input_data)
            self.set_model_input(self.prefill, valid_length_name, valid_length_data)
            self.set_model_input(self.prefill, current_length_name, current_length_data)
            self.set_model_input(self.prefill, time_position_ids_name, chunk_pos[0:1])
            self.set_model_input(self.prefill, hight_position_ids_name, chunk_pos[1:2])
            self.set_model_input(self.prefill, width_position_ids_name, chunk_pos[2:3])
            self.set_model_input(
                self.prefill, linear_attn_mask_name, linear_attn_mask_data
            )
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
            prefill_start = time.time()
            self.prefill.run()
            self.prefill.sync()
            self.prefill_time += time.time() - prefill_start
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
        input_data = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)

        next_id = input_data.argmax(-1)[0]
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)

        first_token_decode_start = time.time()
        prefill_response = self.tokenizer.decode(next_id)
        self.ttft_debug_extra_timings["First Token Decode"] = (
            time.time() - first_token_decode_start
        ) * 1000
        logger.success("response:")
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)
        self.ttft_time = time.time() - start_time
        chat_history_ids = all_input_ids[0]
        next_id = torch.from_numpy(next_id)
        self.generated_ids.append(next_id)

        chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)
        self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
        input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(
            1, 1, -1
        )
        self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)
        all_response = prefill_response
        self.context_length += input_echo_len

        decode_count = 0
        skip_tokens = 0
        slide_len = 10
        last_response = self.tokenizer.decode(chat_history_ids.tolist()[-slide_len:])
        decode_response = ""

        while True:
            if self.context_length >= self.context_max_length:
                logger.info(
                    f"context length greater than {self.context_max_length}, break!"
                )
                break

            self.perf_tracker.perf_start(PERFTYPE.DECODE_TOTAL_TIME)
            self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
            input_name = self.decode.get_input_name(0)
            time_position_ids_name = self.decode.get_input_name(1)
            hight_position_ids_name = self.decode.get_input_name(2)
            width_position_ids_name = self.decode.get_input_name(3)
            valid_length_name = self.decode.get_input_name(4)
            linear_attn_mask_name = self.decode.get_input_name(6)

            valid_length_data = np.array([self.context_length]).astype("int32")
            linear_attn_mask_data = self.create_linear_attn_mask(1, 1)
            if use_vision:
                decode_pos = self.context_length + self.rope_deltas.item()
                time_position_ids_data = np.array([[decode_pos]]).astype("int32")
                hight_position_ids_data = time_position_ids_data
                width_position_ids_data = time_position_ids_data
            else:
                decode_position_ids, _ = self.get_rope_index_text(
                    self.context_length, 1
                )
                time_position_ids_data = decode_position_ids[0].numpy()
                hight_position_ids_data = decode_position_ids[1].numpy()
                width_position_ids_data = decode_position_ids[2].numpy()

            self.set_model_input(self.decode, input_name, input_data)
            self.set_model_input(
                self.decode, time_position_ids_name, time_position_ids_data
            )
            self.set_model_input(
                self.decode, hight_position_ids_name, hight_position_ids_data
            )
            self.set_model_input(
                self.decode, width_position_ids_name, width_position_ids_data
            )
            self.set_model_input(self.decode, valid_length_name, valid_length_data)
            self.set_model_input(
                self.decode, linear_attn_mask_name, linear_attn_mask_data
            )
            self.perf_tracker.perf_end(PERFTYPE.DECODE_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_INFER_TIME)
            decode_start = time.time()
            self.decode.run()
            self.decode.sync()
            self.decode_time += time.time() - decode_start
            self.perf_tracker.perf_end(PERFTYPE.DECODE_INFER_TIME)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_OUTPUT_TIME)
            input_data = self.decode.get_output(self.decode.get_output_name(0)).numpy()
            self.perf_tracker.perf_end(PERFTYPE.DECODE_OUTPUT_TIME)

            decode_count += 1

            self.perf_tracker.perf_start(PERFTYPE.DECODE_TOKEN_TIME)
            next_id = self.samplingmanager.sample(input_data, self.generated_ids)
            if HOUMO_TARGET == "xh1":
                next_id = np.array(next_id)
            next_id = torch.from_numpy(next_id[0])
            if next_id == self.tokenizer.eos_token_id:
                print(decode_response, end="", flush=True)
                all_response += decode_response
                self.perf_tracker.perf_end(PERFTYPE.DECODE_TOKEN_TIME)
                self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)
                break

            chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)
            self.generated_ids.append(next_id)
            decode_response = self.tokenizer.decode(
                chat_history_ids.tolist()[-(slide_len + 1) - skip_tokens :]
            )[len(last_response) :]
            self.perf_tracker.perf_end(PERFTYPE.DECODE_TOKEN_TIME)
            self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)

            if decode_response != "" and is_valid_char(ord(decode_response[-1])):
                print(decode_response, end="", flush=True)
                all_response += decode_response
                last_response = self.tokenizer.decode(
                    chat_history_ids.tolist()[-slide_len:]
                )
                skip_tokens = 0
            else:
                skip_tokens += 1

            self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
            input_data = F.embedding(
                next_id.unsqueeze(0), self.embedding_weight
            ).reshape(1, 1, -1)
            self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)

            self.context_length = self.context_length + 1

        print("\033[0m")

        self.perf_tracker.set_basic_info(
            batch_size=1,
            input_seq_length=input_echo_len,
            output_seq_length=decode_count,
            num_images=num_images,
        )

        return all_response, input_echo_len, decode_count + 1

    def chat_vision(self, question, image_paths):
        return self.chat(question, image_paths=image_paths)


if __name__ == "__main__":

    args = get_args()
    supports_vision = args.vision_path is not None
    is_vision = args.image_path is not None and supports_vision
    hmqwen = HmQwen(
        args.prefill_path,
        args.decode_path,
        args.embedding_path,
        args.tokenizer_dir,
        args.ndevice,
        vision_path=args.vision_path if (is_vision or args.it) else None,
    )
    if args.it:
        from prompt_toolkit import prompt
    try:
        while True:
            current_image_paths = args.image_path if is_vision else None
            if args.it:
                try:
                    question = prompt("Input your instruction here: ").strip()
                    if question.lower() in ("stop", "exit", "quit", ""):
                        break
                    if not question:
                        print("Input cannot be empty. Please try again.")
                        continue
                    if supports_vision:
                        current_image_paths = prompt_for_image_paths()
                except (EOFError, KeyboardInterrupt):
                    print("\nProgram terminated")
                    break
            else:
                if is_vision:
                    question = "描述这些图片"
                else:
                    question = args.question

            start_time = time.time()
            try:
                response, input_tokens, output_tokens = hmqwen.chat(
                    question,
                    image_paths=current_image_paths,
                )
                total_time = time.time() - start_time
                if args.debug:
                    show_ttft_breakdown(
                        hmqwen.ttft_time,
                        hmqwen.perf_tracker,
                        nested_timings=hmqwen.ttft_debug_nested_timings,
                        extra_timings=hmqwen.ttft_debug_extra_timings,
                    )
                hmqwen.perf_tracker.show_summary()
                hmqwen.perf_tracker.reset_perf_time()
            except Exception as e:
                print(f"Error during chat: {e}")
                import traceback

                traceback.print_exc()
                if not args.it:
                    break
                continue
            if not args.it:
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Program execution failed: {e}")
