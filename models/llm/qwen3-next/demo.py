#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Qwen3-Next Inference Demo - Python script for running Qwen3-Next
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
import os
import sys
import math
import time
import argparse
from typing import Any, List, Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from loguru import logger

import tcim_lite as tcim

from hmatc.utils.perf_infomations import (
    InferencePerformanceTracker,
    PERFTYPE,
)
from hmatc.python.get_hm_devices import get_hm_devices
from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_tokenizer_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "qwen3-next").upper()
    model_size = model_config.get("model_size", "80b-a3b").upper()
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
        or (0x0009 <= cp and cp <= 0x000D)
        or cp == 0x0020
        or (0x0021 <= cp and cp <= 0x002F)
        or (0x0030 <= cp and cp <= 0x0039)
        or (0x003A <= cp and cp <= 0x0040)
        or (0x0041 <= cp and cp <= 0x005A)
        or (0x005B <= cp and cp <= 0x0060)
        or (0x0061 <= cp and cp <= 0x007A)
        or (0x007B <= cp and cp <= 0x007E)
        or (0x2000 <= cp and cp <= 0x206F)
        or (0x3000 <= cp and cp <= 0x303F)
        or (0xFF00 <= cp and cp <= 0xFFEF)
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
        default="请介绍一下存算一体。",
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
    parser.add_argument(
        "--system_prompt",
        dest="system_prompt",
        type=str,
        default=None,
        help="system prompt to control assistant behavior",
    )
    args = parser.parse_args()
    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
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
    if args.ndevice > 1:
        args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        args.decode_path = args.decode_path.replace(".hmm", ".hmms")
    return args


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
    ):
        self.perf_tracker = InferencePerformanceTracker()
        self.ndevice = ndevice
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
            self.decode.get_input_name(4)
        ).shape[2]
        self.batch = self.decode.get_input_info(self.decode.get_input_name(0)).shape[0]

        prefill_output_names = {
            self.prefill.get_output_name(i) for i in range(self.prefill.get_num_outputs())
        }
        decode_input_names = {
            self.decode.get_input_name(i) for i in range(self.decode.get_num_inputs())
        }
        decode_output_names = {
            self.decode.get_output_name(i) for i in range(self.decode.get_num_outputs())
        }

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
                if output_name in prefill_output_names:
                    self.prefill.set_dev_output(output_name, cache)
                    self.decode.set_dev_input(input_name, cache)
                    self.decode.set_dev_output(output_name, cache)
                else:
                    cache = self.decode.get_dev_input(input_name)
                    self.decode.set_dev_output(output_name, cache)

        self.clear_cache()

        # set decode input
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode.get_input_name(2)
        self.set_model_input(
            self.decode, decode_current_length_name, current_length_input_1
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )
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
        self.ttft_debug_nested_timings = {}
        self.ttft_debug_extra_timings = {}

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

    def chat(self, question, system_prompt=None):
        self.generated_ids = []
        if not args.history:
            self.context_length = 0
            self.clear_cache()
        self.prefill_time = 0
        self.decode_time = 0
        self.ttft_time = 0
        self.ttft_debug_nested_timings = {}
        self.ttft_debug_extra_timings = {}
        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOKEN_TIME)
        start_time = time.time()

        effective_system_prompt = (
            "You are a helpful assistant."
            if system_prompt is None
            else system_prompt
        )
        messages = []
        if effective_system_prompt:
            messages.append({"role": "system", "content": effective_system_prompt})
        messages.append({"role": "user", "content": question})
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
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

            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")
            linear_attn_mask_data = self.create_linear_attn_mask(
                self.prefill_length, current_length
            )
            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
            input_name = self.prefill.get_input_name(0)
            valid_length_name = self.prefill.get_input_name(1)
            current_length_name = self.prefill.get_input_name(2)
            linear_attn_mask_name = self.prefill.get_input_name(3)
            self.set_model_input(self.prefill, input_name, input_data)
            self.set_model_input(self.prefill, valid_length_name, valid_length_data)
            self.set_model_input(self.prefill, current_length_name, current_length_data)
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
            valid_length_name = self.decode.get_input_name(1)
            linear_attn_mask_name = self.decode.get_input_name(3)

            valid_length_data = np.array([self.context_length]).astype("int32")
            linear_attn_mask_data = self.create_linear_attn_mask(1, 1)

            self.set_model_input(self.decode, input_name, input_data)
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
        )

        return all_response, input_echo_len, decode_count + 1


if __name__ == "__main__":

    args = get_args()
    print(args.tokenizer_dir)
    hmqwen = HmQwen(
        args.prefill_path,
        args.decode_path,
        args.embedding_path,
        args.tokenizer_dir,
        args.ndevice,
    )
    if args.it:
        from prompt_toolkit import prompt
    try:
        while True:
            if args.it:
                try:
                    question = prompt("Input your instruction here: ").strip()
                    if question.lower() in ("stop", "exit", "quit", ""):
                        break
                    if not question:
                        print("Input cannot be empty. Please try again.")
                        continue
                except (EOFError, KeyboardInterrupt):
                    print("\nProgram terminated")
                    break
            else:
                question = args.question

            start_time = time.time()
            try:
                response, input_tokens, output_tokens = hmqwen.chat(
                    question, system_prompt=args.system_prompt
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
