#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo_mtp.py
# Description:
#   Qwen3.5 HMM MTP speculative decoding demo on XH2.
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

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoTokenizer

import tcim_lite as tcim

from hmatc.utils.perf_infomations import (
    InferencePerformanceTracker,
    PERFTYPE,
)
from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
SUFFIX = ".hmcc.format"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


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
        "--prefill_mtp_path",
        dest="prefill_mtp_path",
        type=str,
        default=None,
        help="houmo MTP prefill model path",
    )
    parser.add_argument(
        "--decode_mtp_path",
        dest="decode_mtp_path",
        type=str,
        default=None,
        help="houmo MTP draft decode model path",
    )
    parser.add_argument(
        "--decode_verify_path",
        dest="decode_verify_path",
        type=str,
        default=None,
        help="houmo verify decode model path",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number",
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
    default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.tokenizer_dir = first_not_none(args.tokenizer_dir, get_default_tokenizer_dir(model_config))
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    if args.prefill_path is None:
        args.prefill_path = os.path.join("output", HOUMO_TARGET, f"{args.model_name}-{args.model_size}_prefill.hmm")
    if args.prefill_mtp_path is None:
        args.prefill_mtp_path = os.path.join(
            "output",
            HOUMO_TARGET,
            f"{args.model_name}-{args.model_size}_prefill_mtp.hmm",
        )
    if args.decode_mtp_path is None:
        args.decode_mtp_path = os.path.join(
            "output",
            HOUMO_TARGET,
            f"{args.model_name}-{args.model_size}_decode_mtp.hmm",
        )
    if args.decode_verify_path is None:
        args.decode_verify_path = os.path.join(
            "output", HOUMO_TARGET, f"{args.model_name}-{args.model_size}_decode.hmm"
        )
    if args.ndevice > 1:
        if args.prefill_path.endswith(".hmm"):
            args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        if args.prefill_mtp_path.endswith(".hmm"):
            args.prefill_mtp_path = args.prefill_mtp_path.replace(".hmm", ".hmms")
        if args.decode_mtp_path.endswith(".hmm"):
            args.decode_mtp_path = args.decode_mtp_path.replace(".hmm", ".hmms")
        if args.decode_verify_path.endswith(".hmm"):
            args.decode_verify_path = args.decode_verify_path.replace(".hmm", ".hmms")
    return args


def _numpy_dtype(info) -> np.dtype:
    dtype = info.dtype
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return np.dtype(dtype)
    return np.dtype(dtype)


def _zeros_like_input(model, name: str) -> np.ndarray:
    info = model.get_dev_input(name).info
    return np.zeros(info.shape, dtype=_numpy_dtype(info))


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "to_host"):
        value = value.to_host()
    return value.numpy()


def _as_list_eos(eos_token_id) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(token_id) for token_id in eos_token_id}
    return {int(eos_token_id)}


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

    def apply_repetition_penalty(self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None) -> np.ndarray:
        if self.repetition_penalty == 1.0 or not previous_tokens:
            return logits

        adjusted_logits = logits.copy()
        for token_id in set(previous_tokens):
            if 0 <= token_id < len(logits):
                if logits[token_id] < 0:
                    adjusted_logits[token_id] = logits[token_id] * self.repetition_penalty
                else:
                    adjusted_logits[token_id] = logits[token_id] / self.repetition_penalty
        return adjusted_logits

    def apply_top_k(self, logits: np.ndarray) -> np.ndarray:
        if self.top_k is None or self.top_k <= 0:
            return logits
        top_k = min(self.top_k, len(logits))
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        filtered_logits = np.full_like(logits, -np.inf)
        filtered_logits[top_k_indices] = logits[top_k_indices]
        return filtered_logits

    def apply_top_p(self, logits: np.ndarray) -> np.ndarray:
        if self.top_p >= 1.0:
            return logits
        finite_logits = logits.astype(np.float64)
        exp_logits = np.exp(finite_logits - np.nanmax(finite_logits))
        probs = exp_logits / np.nansum(exp_logits)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_indices = np.where(cumulative_probs >= self.top_p)[0]
        if len(cutoff_indices) > 0:
            cutoff_index = max(int(cutoff_indices[0]), self.min_tokens_to_keep - 1)
            selected_indices = sorted_indices[: cutoff_index + 1]
        else:
            selected_indices = sorted_indices
        filtered_logits = np.full_like(logits, -np.inf)
        filtered_logits[selected_indices] = logits[selected_indices]
        return filtered_logits

    def process_logits(self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None) -> np.ndarray:
        processed_logits = logits.copy()
        processed_logits = self.apply_repetition_penalty(processed_logits, previous_tokens)
        processed_logits = self.apply_top_k(processed_logits)
        processed_logits = self.apply_top_p(processed_logits)
        processed_logits = self.apply_temperature(processed_logits)
        return processed_logits

    def sample(self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None) -> int:
        last_logits = np.asarray(logits).reshape(-1, np.asarray(logits).shape[-1])[-1]
        processed_logits = self.process_logits(last_logits.astype(np.float32), previous_tokens)
        if np.all(~np.isfinite(processed_logits)):
            processed_logits = last_logits.astype(np.float32)
        return int(processed_logits.argmax(-1))


class PrefillNames:
    _CONV_KIND_ORDER = {"q": 0, "k": 1, "v": 2, "": 3}

    def __init__(self, model):
        in_names = [model.get_input_name(index) for index in range(model.get_num_inputs())]
        out_names = [model.get_output_name(index) for index in range(model.get_num_outputs())]

        self.activation = self._pick(in_names, "input_1")
        self.valid_length = self._pick(in_names, "valid_length")
        self.current_length = self._pick(in_names, "current_length")
        self.time_position_ids = self._pick(in_names, "time_position_ids")
        self.height_position_ids = self._pick(in_names, "hight_position_ids")
        self.width_position_ids = self._pick(in_names, "width_position_ids")
        self.linear_attn_mask = self._pick_optional(in_names, "linear_attn_mask")
        self.kv_in = sorted(name for name in in_names if "_attn_kcache_input" in name or "_attn_vcache_input" in name)
        conv_cache_in = self._conv_cache_names(in_names, "past_conv_cache")
        rec_state_in = self._indexed_names(in_names, r"past_recurrent_state_(\d+)$")
        conv_cache_out = self._conv_cache_names(out_names, "conv_cache_out")
        rec_state_out = self._indexed_names(out_names, r"recurrent_state_out_(\d+)$")

        if not conv_cache_in:
            raise RuntimeError("missing recurrent conv cache inputs")
        self.layer_indices = sorted({layer_idx for layer_idx, _, _ in conv_cache_in})
        self._require_same_indices("past_recurrent_state", self.layer_indices, rec_state_in)
        if rec_state_out:
            self._require_same_indices("recurrent_state_out", self.layer_indices, rec_state_out)

        self.conv_cache_keys_by_input = {name: (layer_idx, kind) for layer_idx, kind, name in conv_cache_in}
        self.conv_cache_in = [name for _, _, name in conv_cache_in]
        self.rec_state_in = [name for _, name in rec_state_in]
        self.conv_cache_out_by_key = {(layer_idx, kind): name for layer_idx, kind, name in conv_cache_out}
        self.conv_cache_out_by_input = {
            name: self.conv_cache_out_by_key[key]
            for name, key in self.conv_cache_keys_by_input.items()
            if key in self.conv_cache_out_by_key
        }
        self.conv_cache_out = [
            self.conv_cache_out_by_input[name] for name in self.conv_cache_in if name in self.conv_cache_out_by_input
        ]
        self.rec_state_keys_by_input = {name: layer_idx for layer_idx, name in rec_state_in}
        self.rec_state_out_by_layer = {layer_idx: name for layer_idx, name in rec_state_out}
        self.rec_state_out_by_input = {
            name: self.rec_state_out_by_layer[layer_idx]
            for name, layer_idx in self.rec_state_keys_by_input.items()
            if layer_idx in self.rec_state_out_by_layer
        }
        self.rec_state_out = [name for _, name in rec_state_out]
        self.split_conv_cache_out_by_key = self._split_conv_outputs(out_names, "conv_cache_out")
        self.split_conv_cache_out_by_input = {
            name: self.split_conv_cache_out_by_key[key]
            for name, key in self.conv_cache_keys_by_input.items()
            if key in self.split_conv_cache_out_by_key
        }
        self.split_conv_cache_out = self._split_outputs(out_names, r"conv_cache_out_(\d+)_(\d+)$")
        self.split_rec_state_out = self._split_outputs(out_names, r"recurrent_state_out_(\d+)_(\d+)$")
        self.split_rec_state_out_by_input = {
            name: self.split_rec_state_out[layer_idx]
            for name, layer_idx in self.rec_state_keys_by_input.items()
            if layer_idx in self.split_rec_state_out
        }
        self.logits_out = self._pick(out_names, "logits")
        self.hidden_out = self._pick_any(out_names, ("hidden_states", "post_norm_hidden", "pre_norm_hidden"))

    @staticmethod
    def _pick(names: Sequence[str], exact: str) -> str:
        if exact in names:
            return exact
        raise RuntimeError(f"input/output {exact!r} not found")

    @classmethod
    def _pick_any(cls, names: Sequence[str], keywords: Sequence[str]) -> str:
        bare_map = {cls._bare(name): name for name in names}
        for keyword in keywords:
            for bare_name, full_name in bare_map.items():
                if keyword in bare_name:
                    return full_name
        raise RuntimeError(f"failed to find output with keywords {list(keywords)}, candidates={list(names)}")

    @staticmethod
    def _pick_optional(names: Sequence[str], exact: str) -> Optional[str]:
        if exact in names:
            return exact
        return None

    @staticmethod
    def _bare(name: str) -> str:
        return name[: -len(SUFFIX)] if name.endswith(SUFFIX) else name

    @classmethod
    def _indexed_names(cls, names: Sequence[str], pattern: str) -> List[Tuple[int, str]]:
        compiled = re.compile(pattern)
        matched = []
        for name in names:
            match = compiled.match(cls._bare(name))
            if match:
                matched.append((int(match.group(1)), name))
        return sorted(matched, key=lambda item: item[0])

    @classmethod
    def _conv_cache_names(cls, names: Sequence[str], prefix: str) -> List[Tuple[int, str, str]]:
        compiled = re.compile(rf"{re.escape(prefix)}(?:_([A-Za-z]+))?_(\d+)$")
        matched = []
        for name in names:
            match = compiled.match(cls._bare(name))
            if not match:
                continue
            kind = match.group(1) or ""
            layer_idx = int(match.group(2))
            matched.append((layer_idx, kind, name))
        return sorted(
            matched,
            key=lambda item: (
                item[0],
                cls._CONV_KIND_ORDER.get(item[1], 100),
                item[1],
            ),
        )

    @staticmethod
    def _require_same_indices(
        label: str,
        expected_indices: Sequence[int],
        actual_items: Sequence[Tuple[int, str]],
    ) -> None:
        actual_indices = [layer_idx for layer_idx, _ in actual_items]
        if list(expected_indices) != actual_indices:
            raise RuntimeError(
                f"{label} layer indices mismatch: expected {list(expected_indices)}, " f"got {actual_indices}"
            )

    @classmethod
    def _split_outputs(cls, names: Sequence[str], pattern: str) -> Dict[int, List[str]]:
        compiled = re.compile(pattern)
        outputs: Dict[int, List[Tuple[int, str]]] = {}
        for name in names:
            match = compiled.match(cls._bare(name))
            if not match:
                continue
            layer_idx = int(match.group(1))
            step_idx = int(match.group(2))
            outputs.setdefault(layer_idx, []).append((step_idx, name))
        return {
            layer_idx: [name for _, name in sorted(step_items, key=lambda item: item[0])]
            for layer_idx, step_items in outputs.items()
        }

    @classmethod
    def _split_conv_outputs(cls, names: Sequence[str], prefix: str) -> Dict[Tuple[int, str], List[str]]:
        compiled = re.compile(rf"{re.escape(prefix)}(?:_([A-Za-z]+))?_(\d+)_(\d+)$")
        outputs: Dict[Tuple[int, str], List[Tuple[int, str]]] = {}
        for name in names:
            match = compiled.match(cls._bare(name))
            if not match:
                continue
            kind = match.group(1) or ""
            layer_idx = int(match.group(2))
            step_idx = int(match.group(3))
            outputs.setdefault((layer_idx, kind), []).append((step_idx, name))
        return {
            key: [name for _, name in sorted(step_items, key=lambda item: item[0])]
            for key, step_items in outputs.items()
        }


class MtpNames:
    def __init__(self, model):
        in_names = [model.get_input_name(index) for index in range(model.get_num_inputs())]
        out_names = [model.get_output_name(index) for index in range(model.get_num_outputs())]

        self.hidden_states_in = self._pick_any(in_names, ("hidden_states", "post_norm_hidden", "pre_norm_hidden"))
        self.input_embedding_in = self._pick_any(in_names, ("input_embedding", "next_token_embedding"))
        self.position_ids = tuple(name for name in in_names if "position_ids" in self._bare(name))
        self.past_seq_length = self._pick_scalar(model, in_names, ("past_seq", "valid_length"))
        self.current_input_length = self._pick_scalar(model, in_names, ("current_",))
        self.past_key_cache = self._pick_any(in_names, ("past_key_cache",))
        self.past_value_cache = self._pick_any(in_names, ("past_value_cache",))
        self.mtp_logits_out = self._pick_any(out_names, ("mtp_logits", "logits"))
        self.mtp_hidden_out = self._pick_any(
            out_names,
            ("mtp_hidden_states", "post_norm_out", "hidden_states", "post_norm_hidden"),
        )

    @staticmethod
    def _bare(name: str) -> str:
        return name[: -len(SUFFIX)] if name.endswith(SUFFIX) else name

    @classmethod
    def _pick_any(cls, names: Sequence[str], keywords: Sequence[str]) -> str:
        bare_map = {cls._bare(name): name for name in names}
        for keyword in keywords:
            for bare_name, full_name in bare_map.items():
                if keyword in bare_name:
                    return full_name
        raise RuntimeError(f"failed to find tensor with keywords {list(keywords)}, candidates={list(names)}")

    @classmethod
    def _pick_scalar(cls, model, names: Sequence[str], keywords: Sequence[str]) -> str:
        matches = []
        for name in names:
            if np.prod(model.get_dev_input(name).info.shape) != 1:
                continue
            bare_name = cls._bare(name)
            if any(keyword in bare_name for keyword in keywords):
                matches.append(name)
        if len(matches) == 1:
            return matches[0]
        raise RuntimeError(f"failed to find unique scalar tensor with keywords {list(keywords)}, candidates={matches}")


@dataclass
class SpecDecodeStats:
    mode: str = "mtp"
    rounds: int = 0
    total_tokens: int = 0
    draft_tokens: int = 0
    accepted: int = 0
    mtp_prefill_tokens: int = 0
    drafts_per_round: int = 0
    elapsed_s: float = 0.0

    @property
    def avg_accepted_per_round(self) -> float:
        return self.accepted / max(self.rounds, 1)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.draft_tokens, 1)

    def show_summary(self) -> None:
        print(
            f"[SpecDecode] mode={self.mode} "
            f"rounds={self.rounds} "
            f"total_tokens={self.total_tokens} "
            f"draft_tokens={self.draft_tokens} "
            f"accepted={self.accepted} "
            f"avg_accepted_per_round={self.avg_accepted_per_round:.2f} "
            f"acceptance_rate={self.acceptance_rate:.2%} "
            f"mtp_prefill_tokens={self.mtp_prefill_tokens} "
            f"drafts_per_round={self.drafts_per_round} "
            f"elapsed_s={self.elapsed_s:.4f}",
            flush=True,
        )


def show_ttft_breakdown(
    ttft_time: float,
    perf_tracker: InferencePerformanceTracker,
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
    if extra_timings:
        logger.success("  Debug Details (newly split from residual):")
        for label, value in extra_timings.items():
            if value > 0:
                logger.success(f"    {label}: {value:.3f} ms")
    if residual_ms > 0:
        logger.success(f"  Other/Untracked: {residual_ms:.3f} ms")
    logger.success(f"  Total TTFT: {ttft_ms:.3f} ms")


class HmQwenMTP:
    def __init__(
        self,
        prefill_path,
        prefill_mtp_path,
        decode_mtp_path,
        decode_verify_path,
        embedding_path,
        tokenizer_dir,
        ndevice=1,
        debug=False,
    ):
        self.perf_tracker = InferencePerformanceTracker()
        self.ndevice = ndevice
        self.debug = debug
        self.generated_ids: List[int] = []
        self.messages: List[Dict[str, str]] = []
        self.ttft_time = 0.0
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.ttft_debug_extra_timings: dict[str, float] = {}
        self.last_spec_stats = SpecDecodeStats()

        if self.ndevice == 1:
            weight_manager = tcim.runtime.WeightManager(0)
        elif self.ndevice == 2 and HOUMO_TARGET == "xh2":
            dev_manager = tcim.runtime.DevManager([0, 1], "Xh2HalBackend")
            weight_manager = tcim.runtime.WeightManager(dev_manager)
        else:
            raise ValueError("Unsupport device number!")

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self.prefill = tcim.runtime.load(prefill_path, option=tcim.runtime.Option(weight_manager))
        self.pn = PrefillNames(self.prefill)
        self.prefill_mtp = tcim.runtime.load(prefill_mtp_path, option=tcim.runtime.Option(weight_manager))
        self.pm = MtpNames(self.prefill_mtp)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)
        logger.info("prefill and prefill_mtp models loaded")

        verify_option = tcim.runtime.Option(weight_manager)
        verify_option.set_dummy_tensors(list(self.pn.kv_in))
        self.perf_tracker.perf_start(PERFTYPE.DECODE_LOAD_TIME)
        self.verify = tcim.runtime.load(decode_verify_path, option=verify_option)
        self.vn = PrefillNames(self.verify)
        self.mtp = tcim.runtime.load(decode_mtp_path, option=tcim.runtime.Option(weight_manager))
        self.mn = MtpNames(self.mtp)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_LOAD_TIME)
        logger.info("decode_verify and decode_mtp models loaded")

        self._link_verify_kv()

        activation_shape = tuple(self.prefill.get_dev_input(self.pn.activation).info.shape)
        self.batch, self.prefill_length, self.embedding_len = [int(dim) for dim in activation_shape]
        verify_shape = tuple(self.verify.get_dev_input(self.vn.activation).info.shape)
        self.verify_seq = int(verify_shape[1])
        self.block_size = self.verify_seq - 1
        self.context_max_length = max(int(dim) for dim in self.prefill.get_dev_input(self.pn.kv_in[0]).info.shape)
        self.num_recurrent_layers = len(self.pn.rec_state_in)
        self.context_length = 0

        self._ensure_mtp_cache_compatible()
        self._link_decode_mtp_cache_to_prefill_mtp()

        self.samplingmanager = SamplingManager(
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            repetition_penalty=args.repetition_penalty,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        embedding_weight = torch.load(embedding_path, map_location="cpu", weights_only=False)
        if isinstance(embedding_weight, dict):
            if "weight" not in embedding_weight:
                raise KeyError(f"Embedding state_dict at {embedding_path} does not contain 'weight'")
            embedding_tensor = embedding_weight["weight"]
        elif isinstance(embedding_weight, torch.nn.Embedding):
            embedding_tensor = embedding_weight.weight.data
        else:
            embedding_tensor = embedding_weight
        self.embedding_weight = embedding_tensor.reshape(-1, self.embedding_len).to(torch.float16)

        self.stop_ids = _as_list_eos(getattr(self.tokenizer, "eos_token_id", None))
        for token in ("<|im_end|>", "<|endoftext|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and int(token_id) >= 0:
                self.stop_ids.add(int(token_id))

        self.perf_tracker.reset_perf_time()

    def _ensure_mtp_cache_compatible(self) -> None:
        for prefill_name, decode_name in (
            (self.pm.past_key_cache, self.mn.past_key_cache),
            (self.pm.past_value_cache, self.mn.past_value_cache),
        ):
            prefill_info = self.prefill_mtp.get_dev_input(prefill_name).info
            decode_info = self.mtp.get_dev_input(decode_name).info
            if tuple(prefill_info.shape) != tuple(decode_info.shape):
                raise RuntimeError(
                    "prefill_mtp and decode_mtp cache shape differ: "
                    f"{prefill_name} {tuple(prefill_info.shape)} vs "
                    f"{decode_name} {tuple(decode_info.shape)}"
                )
            if np.dtype(prefill_info.dtype) != np.dtype(decode_info.dtype):
                raise RuntimeError(
                    "prefill_mtp and decode_mtp cache dtype differ: "
                    f"{prefill_name} {prefill_info.dtype} vs "
                    f"{decode_name} {decode_info.dtype}"
                )

    def _link_decode_mtp_cache_to_prefill_mtp(self) -> None:
        self.mtp.set_input(
            self.mn.past_key_cache,
            self.prefill_mtp.get_dev_input(self.pm.past_key_cache),
        )
        self.mtp.set_input(
            self.mn.past_value_cache,
            self.prefill_mtp.get_dev_input(self.pm.past_value_cache),
        )

    def _link_verify_kv(self) -> None:
        for name in self.pn.kv_in:
            self.verify.set_input(name, self.prefill.get_dev_input(name))
        if self.debug:
            logger.info("linked verify KV inputs to prefill KV buffers")

    @staticmethod
    def _bare_name(name: str) -> str:
        return name.removesuffix(SUFFIX)

    def _find_verify_split_recurrent_outputs(self, input_name: str) -> List[str]:
        return self.vn.split_rec_state_out_by_input.get(input_name, [])

    def _find_verify_split_conv_outputs(self, input_name: str) -> List[str]:
        return self.vn.split_conv_cache_out_by_input.get(input_name, [])

    def _cross_propagate_rec_to_verify(self) -> None:
        for conv_in in self.vn.conv_cache_in:
            key = self.vn.conv_cache_keys_by_input[conv_in]
            conv_out = self.pn.conv_cache_out_by_key.get(key)
            if conv_out is None:
                raise RuntimeError(f"missing prefill conv output for verify input {conv_in}")
            self.verify.set_input(conv_in, self.prefill.get_dev_output(conv_out))

        for rec_in in self.vn.rec_state_in:
            layer_idx = self.vn.rec_state_keys_by_input[rec_in]
            rec_out = self.pn.rec_state_out_by_layer.get(layer_idx)
            if rec_out is None:
                raise RuntimeError(f"missing prefill recurrent output for verify input {rec_in}")
            self.verify.set_input(rec_in, self.prefill.get_dev_output(rec_out))

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
            logger.info(f"Auto-reshaping input '{input_name}' from {array.shape} to {expected_shape}")
        return array.reshape(expected_shape)

    def set_model_input(self, runtime_model, input_name: str, value) -> None:
        runtime_model.set_input(
            input_name,
            self.adapt_input_to_model_shape(runtime_model, input_name, value),
        )

    def _embed_ids(self, token_ids, perf_type: PERFTYPE) -> np.ndarray:
        self.perf_tracker.perf_start(perf_type)
        embeddings = F.embedding(token_ids.long(), self.embedding_weight)
        output = embeddings.to(torch.float16).detach().cpu().numpy()
        self.perf_tracker.perf_end(perf_type)
        return output

    def _set_scalar_i32(self, model, name: str, value: int) -> None:
        shape = tuple(model.get_dev_input(name).info.shape)
        self.set_model_input(model, name, np.full(shape, int(value), dtype=np.int32))

    def _set_position_ids(self, model, names: Sequence[str], start: int, length: int) -> None:
        values = np.arange(start, start + length, dtype=np.int32)
        for name in names:
            shape = tuple(model.get_dev_input(name).info.shape)
            self.set_model_input(model, name, values.reshape(shape))

    def clear_cache(self) -> None:
        for name in self.pn.conv_cache_in + self.pn.rec_state_in:
            self.set_model_input(self.prefill, name, _zeros_like_input(self.prefill, name))
        self.prefill_mtp.set_input(
            self.pm.past_key_cache,
            _zeros_like_input(self.prefill_mtp, self.pm.past_key_cache),
        )
        self.prefill_mtp.set_input(
            self.pm.past_value_cache,
            _zeros_like_input(self.prefill_mtp, self.pm.past_value_cache),
        )
        self._link_decode_mtp_cache_to_prefill_mtp()

    def _propagate_recurrent_in_prefill(self) -> None:
        for conv_in in self.pn.conv_cache_in:
            conv_out = self.pn.conv_cache_out_by_input.get(conv_in)
            if conv_out is None:
                raise RuntimeError(f"missing prefill conv output for input {conv_in}")
            self.prefill.set_input(conv_in, self.prefill.get_dev_output(conv_out))

        for rec_in in self.pn.rec_state_in:
            rec_out = self.pn.rec_state_out_by_input.get(rec_in)
            if rec_out is None:
                raise RuntimeError(f"missing prefill recurrent output for input {rec_in}")
            self.prefill.set_input(rec_in, self.prefill.get_dev_output(rec_out))

    def _create_linear_attn_mask(self, fill_length: int, current_length: int) -> np.ndarray:
        mask = np.zeros((1, fill_length), dtype=np.float16)
        mask[0, :current_length] = 1.0
        return mask

    def _run_main_prefill_chunk(self, past_seq_len: int, chunk_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        token_ids = torch.as_tensor(chunk_ids.reshape(1, -1), dtype=torch.long)
        valid_len = int(token_ids.shape[1])
        inputs_embeds = self._embed_ids(token_ids, PERFTYPE.PREFILL_EMBED_TIME)
        if valid_len < self.prefill_length:
            pad = np.zeros(
                (1, self.prefill_length - valid_len, self.embedding_len),
                dtype=np.float16,
            )
            inputs_embeds = np.concatenate([inputs_embeds, pad], axis=1)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
        self.set_model_input(self.prefill, self.pn.activation, inputs_embeds)
        self._set_scalar_i32(self.prefill, self.pn.valid_length, past_seq_len)
        self._set_scalar_i32(self.prefill, self.pn.current_length, valid_len)
        self._set_position_ids(
            self.prefill,
            (
                self.pn.time_position_ids,
                self.pn.height_position_ids,
                self.pn.width_position_ids,
            ),
            past_seq_len,
            self.prefill_length,
        )
        if self.pn.linear_attn_mask:
            self.set_model_input(
                self.prefill,
                self.pn.linear_attn_mask,
                self._create_linear_attn_mask(self.prefill_length, valid_len),
            )
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
        prefill_start = time.time()
        self.prefill.run(sync=False)
        self.prefill.sync()
        self.prefill_time += time.time() - prefill_start
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
        logits = _to_numpy(self.prefill.get_output(self.pn.logits_out))[:, :valid_len, :].copy()
        hidden = _to_numpy(self.prefill.get_output(self.pn.hidden_out))[:, :valid_len, :].copy()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)
        self._propagate_recurrent_in_prefill()
        return logits, hidden

    def _prefill_mtp_chunk(self, hidden: np.ndarray, token_ids: np.ndarray, past_seq_len: int) -> None:
        valid_len = int(token_ids.shape[0])
        if valid_len <= 0:
            return
        target_len = int(self.prefill_mtp.get_dev_input(self.pm.hidden_states_in).info.shape[1])
        if valid_len > target_len:
            raise ValueError(f"MTP prefill chunk too long: {valid_len} > {target_len}")
        mtp_token_ids = torch.as_tensor(token_ids.reshape(1, -1), dtype=torch.long)
        input_embedding = self._embed_ids(mtp_token_ids, PERFTYPE.PREFILL_EMBED_TIME)
        if valid_len < target_len:
            hidden_pad = np.zeros((1, target_len - valid_len, self.embedding_len), dtype=np.float16)
            embed_pad = np.zeros_like(hidden_pad)
            hidden = np.concatenate([hidden.astype(np.float16), hidden_pad], axis=1)
            input_embedding = np.concatenate([input_embedding, embed_pad], axis=1)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
        self.set_model_input(self.prefill_mtp, self.pm.hidden_states_in, hidden.astype(np.float16))
        self.set_model_input(
            self.prefill_mtp,
            self.pm.input_embedding_in,
            input_embedding.astype(np.float16),
        )
        if self.pm.position_ids:
            self._set_position_ids(self.prefill_mtp, self.pm.position_ids, past_seq_len, target_len)
        self._set_scalar_i32(self.prefill_mtp, self.pm.past_seq_length, past_seq_len)
        self._set_scalar_i32(self.prefill_mtp, self.pm.current_input_length, valid_len)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
        self.prefill_mtp.run(sync=False)
        self.prefill_mtp.sync()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)

    def _run_mtp_step(self, hidden: np.ndarray, token_id: int, past_seq_len: int) -> Tuple[int, np.ndarray]:
        token_ids = torch.as_tensor([[int(token_id)]], dtype=torch.long)
        input_embedding = self._embed_ids(token_ids, PERFTYPE.DECODE_EMBED_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
        self.set_model_input(self.mtp, self.mn.hidden_states_in, hidden.astype(np.float16))
        self.set_model_input(self.mtp, self.mn.input_embedding_in, input_embedding.astype(np.float16))
        if self.mn.position_ids:
            self._set_position_ids(self.mtp, self.mn.position_ids, past_seq_len, 1)
        self._set_scalar_i32(self.mtp, self.mn.past_seq_length, past_seq_len)
        self._set_scalar_i32(self.mtp, self.mn.current_input_length, 1)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INFER_TIME)
        self.mtp.run(sync=False)
        self.mtp.sync()
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_OUTPUT_TIME)
        logits = _to_numpy(self.mtp.get_output(self.mn.mtp_logits_out))
        next_hidden = _to_numpy(self.mtp.get_output(self.mn.mtp_hidden_out)).copy()
        next_token = int(np.argmax(logits, axis=-1).reshape(-1)[-1])
        self.perf_tracker.perf_end(PERFTYPE.DECODE_OUTPUT_TIME)
        return next_token, next_hidden

    def _run_draft_mtp(
        self,
        current_token: int,
        last_hidden: np.ndarray,
        mtp_past_seq_len: int,
        num_drafts: int,
    ) -> List[int]:
        draft_tokens: List[int] = []
        hidden = last_hidden
        token = int(current_token)
        for offset in range(num_drafts):
            token, hidden = self._run_mtp_step(hidden, token, mtp_past_seq_len + offset)
            draft_tokens.append(token)
        return draft_tokens

    def _run_verify(
        self,
        past_seq_len: int,
        verify_tokens: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(verify_tokens) != self.verify_seq:
            raise ValueError(f"expected {self.verify_seq} verify tokens, got {len(verify_tokens)}")
        token_ids = torch.as_tensor([verify_tokens], dtype=torch.long)
        input_embedding = self._embed_ids(token_ids, PERFTYPE.DECODE_EMBED_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
        self.set_model_input(self.verify, self.vn.activation, input_embedding)
        self._set_scalar_i32(self.verify, self.vn.valid_length, past_seq_len)
        self._set_scalar_i32(self.verify, self.vn.current_length, self.verify_seq)
        self._set_position_ids(
            self.verify,
            (
                self.vn.time_position_ids,
                self.vn.height_position_ids,
                self.vn.width_position_ids,
            ),
            past_seq_len,
            self.verify_seq,
        )
        if self.vn.linear_attn_mask:
            self.set_model_input(
                self.verify,
                self.vn.linear_attn_mask,
                np.ones((1, self.verify_seq), dtype=np.float16),
            )
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INFER_TIME)
        verify_start = time.time()
        self.verify.run(sync=False)
        self.verify.sync()
        self.decode_time += time.time() - verify_start
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_OUTPUT_TIME)
        logits = _to_numpy(self.verify.get_output(self.vn.logits_out))
        hidden = _to_numpy(self.verify.get_output(self.vn.hidden_out))
        self.perf_tracker.perf_end(PERFTYPE.DECODE_OUTPUT_TIME)
        return logits, hidden

    def _commit_verify_linear_cache(self, accepted_steps: int) -> None:
        accept_pos = accepted_steps - 1
        for conv_in in self.vn.conv_cache_in:
            split_conv_outputs = self._find_verify_split_conv_outputs(conv_in)
            if split_conv_outputs:
                conv_out_name = split_conv_outputs[min(accept_pos, len(split_conv_outputs) - 1)]
                self.verify.set_input(conv_in, self.verify.get_dev_output(conv_out_name))
            else:
                conv_out_name = self.vn.conv_cache_out_by_input.get(conv_in)
                if conv_out_name is None:
                    raise RuntimeError(f"missing verify conv output for input {conv_in}")
                conv_out_array = _to_numpy(self.verify.get_output(conv_out_name))
                conv_kernel = int(self.verify.get_dev_input(conv_in).info.shape[-1])
                if conv_out_array.shape[-1] > conv_kernel:
                    start = min(accept_pos, conv_out_array.shape[-1] - conv_kernel)
                    conv_slice = conv_out_array[:, :, start : start + conv_kernel].copy()
                else:
                    conv_slice = conv_out_array.copy()
                self.verify.set_input(conv_in, conv_slice)

        for rec_in in self.vn.rec_state_in:
            rec_outputs = self._find_verify_split_recurrent_outputs(rec_in)
            if rec_outputs:
                rec_out_name = rec_outputs[min(accept_pos, len(rec_outputs) - 1)]
            else:
                rec_out_name = self.vn.rec_state_out_by_input.get(rec_in)
                if rec_out_name is None:
                    raise RuntimeError(f"missing verify recurrent output for input {rec_in}")
            self.verify.set_input(rec_in, self.verify.get_dev_output(rec_out_name))

    def _do_prefill(self, input_ids: np.ndarray) -> Tuple[int, int, np.ndarray, int]:
        self.clear_cache()
        total_prompt_len = int(input_ids.shape[0])
        past_seq_len = 0
        mtp_prefill_seq_len = 0
        mtp_pending_hidden: Optional[np.ndarray] = None
        last_logits: Optional[np.ndarray] = None
        last_hidden: Optional[np.ndarray] = None

        prefill_loop_round = math.ceil(total_prompt_len / self.prefill_length)
        for round_idx in range(prefill_loop_round):
            chunk_start = round_idx * self.prefill_length
            chunk = input_ids[chunk_start : chunk_start + self.prefill_length]
            valid_len = int(chunk.shape[0])
            logits, hidden_all = self._run_main_prefill_chunk(past_seq_len, chunk)
            last_logits = logits
            last_hidden = hidden_all[:, valid_len - 1 : valid_len, :].copy()

            hidden_parts = []
            token_parts = []
            if mtp_pending_hidden is not None:
                hidden_parts.append(mtp_pending_hidden)
                token_parts.append(chunk[:1])
            if valid_len > 1:
                hidden_parts.append(hidden_all[:, : valid_len - 1, :])
                token_parts.append(chunk[1:valid_len])
            if hidden_parts:
                mtp_hidden = np.concatenate(hidden_parts, axis=1)
                mtp_tokens = np.concatenate(token_parts, axis=0)
                self._prefill_mtp_chunk(mtp_hidden, mtp_tokens, mtp_prefill_seq_len)
                mtp_prefill_seq_len += int(mtp_hidden.shape[1])
            mtp_pending_hidden = last_hidden
            past_seq_len += valid_len

        if last_logits is None or last_hidden is None:
            raise RuntimeError("empty prompt is not supported")
        self._cross_propagate_rec_to_verify()
        next_token = int(np.argmax(last_logits[:, -1:, :], axis=-1).reshape(-1)[-1])
        return past_seq_len, next_token, last_hidden, mtp_prefill_seq_len

    def _build_text_input_ids(self, question: str) -> np.ndarray:
        if not self.messages:
            self.messages.append({"role": "system", "content": "You are a helpful assistant."})
        self.messages.append({"role": "user", "content": question})
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="np", add_special_tokens=False)
        return inputs["input_ids"].reshape(-1).astype(np.int64)

    def generate_from_ids(
        self,
        input_ids: np.ndarray,
        prefill_total_started: bool = False,
    ) -> Tuple[str, int, int]:
        if input_ids.size <= 0:
            raise ValueError("empty input_ids")
        if input_ids.size >= self.context_max_length:
            raise ValueError(f"prompt too long: {input_ids.size} >= {self.context_max_length}")

        self.generated_ids = []
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.ttft_debug_extra_timings = {}

        if not prefill_total_started:
            self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        start_time = time.time()
        past_seq_len, current_token, last_hidden, mtp_past_seq_len = self._do_prefill(input_ids)
        initial_mtp_prefill_tokens = mtp_past_seq_len
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)

        first_token_decode_start = time.time()
        prefill_response = self.tokenizer.decode([current_token], skip_special_tokens=True)
        self.ttft_debug_extra_timings["First Token Decode"] = (time.time() - first_token_decode_start) * 1000
        self.ttft_time = time.time() - start_time

        if current_token in self.stop_ids:
            self.perf_tracker.set_basic_info(
                batch_size=1,
                input_seq_length=int(input_ids.size),
                output_seq_length=0,
                num_images=0,
            )
            self.last_spec_stats = SpecDecodeStats(
                total_tokens=0,
                mtp_prefill_tokens=initial_mtp_prefill_tokens,
                drafts_per_round=self.block_size,
            )
            return "", int(input_ids.size), 0

        generated_ids: List[int] = [current_token]
        self.generated_ids = generated_ids
        chat_history_ids = input_ids.tolist() + generated_ids
        all_response = prefill_response
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)

        total_rounds = 0
        total_draft_tokens = 0
        total_accepted_tokens = 0
        decode_start = time.time()
        skip_tokens = 0
        slide_len = 10
        last_response = self.tokenizer.decode(chat_history_ids[-slide_len:])
        decode_response = ""

        def stream_token(token_id: int) -> bool:
            nonlocal all_response, last_response, skip_tokens, decode_response
            chat_history_ids.append(int(token_id))
            decode_response = self.tokenizer.decode(chat_history_ids[-(slide_len + 1) - skip_tokens :])[
                len(last_response) :
            ]
            if decode_response != "" and is_valid_char(ord(decode_response[-1])):
                print(decode_response, end="", flush=True)
                all_response += decode_response
                last_response = self.tokenizer.decode(chat_history_ids[-slide_len:])
                skip_tokens = 0
                return True
            skip_tokens += 1
            return False

        while True:
            if past_seq_len + self.verify_seq > self.context_max_length:
                print("\033[0m", flush=True)
                logger.info(f"context length reached: {past_seq_len} + {self.verify_seq} > {self.context_max_length}")
                break

            self.perf_tracker.perf_start(PERFTYPE.DECODE_TOTAL_TIME)
            total_rounds += 1
            draft_tokens = self._run_draft_mtp(
                current_token,
                last_hidden,
                mtp_past_seq_len,
                self.block_size,
            )
            total_draft_tokens += len(draft_tokens)

            initial_seq_len = past_seq_len
            verify_tokens = [current_token] + draft_tokens
            verify_logits, verify_hidden = self._run_verify(initial_seq_len, verify_tokens)

            accepted_count = 0
            for token_idx, draft_token in enumerate(draft_tokens):
                predicted = int(np.argmax(verify_logits[:, token_idx : token_idx + 1, :], axis=-1).reshape(-1)[0])
                if predicted != int(draft_token):
                    break
                accepted_count += 1

            total_accepted_tokens += accepted_count
            accepted_steps = accepted_count + 1
            self._commit_verify_linear_cache(accepted_steps)
            past_seq_len = initial_seq_len + accepted_steps

            eos_hit = False
            for token_idx in range(accepted_count):
                token = int(draft_tokens[token_idx])
                generated_ids.append(token)
                if token in self.stop_ids:
                    eos_hit = True
                    break
                stream_token(token)
            if eos_hit:
                self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)
                break

            self.perf_tracker.perf_start(PERFTYPE.DECODE_TOKEN_TIME)
            if accepted_count < len(draft_tokens):
                current_token = int(
                    np.argmax(
                        verify_logits[:, accepted_count : accepted_count + 1, :],
                        axis=-1,
                    ).reshape(
                        -1
                    )[0]
                )
            else:
                current_token = int(np.argmax(verify_logits[:, -1:, :], axis=-1).reshape(-1)[0])
            self.perf_tracker.perf_end(PERFTYPE.DECODE_TOKEN_TIME)

            last_hidden = verify_hidden[:, accepted_count : accepted_count + 1, :].copy()
            mtp_past_seq_len += accepted_steps

            if current_token in self.stop_ids:
                self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)
                break
            generated_ids.append(current_token)
            stream_token(current_token)
            if self.debug:
                logger.info(
                    f"round={total_rounds} accepted={accepted_count}/{len(draft_tokens)} "
                    f"past_seq_len={past_seq_len} mtp_past_seq_len={mtp_past_seq_len}"
                )
            self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)

        if skip_tokens > 0 and decode_response:
            print(decode_response, end="", flush=True)
            all_response += decode_response
        print("\033[0m", flush=True)

        output = all_response
        self.messages.append({"role": "assistant", "content": output})
        self.context_length = int(input_ids.size) + len(generated_ids)
        self.last_spec_stats = SpecDecodeStats(
            rounds=total_rounds,
            total_tokens=len(generated_ids),
            draft_tokens=total_draft_tokens,
            accepted=total_accepted_tokens,
            mtp_prefill_tokens=initial_mtp_prefill_tokens,
            drafts_per_round=self.block_size,
            elapsed_s=time.time() - decode_start,
        )
        self.perf_tracker.set_basic_info(
            batch_size=1,
            input_seq_length=int(input_ids.size),
            output_seq_length=max(len(generated_ids) - 1, 0),
            num_images=0,
        )
        return output, int(input_ids.size), len(generated_ids)

    def chat(self, question) -> Tuple[str, int, int]:
        self.generated_ids = []
        if not args.history:
            self.context_length = 0
            self.messages = []
        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOKEN_TIME)
        token_start = time.time()
        input_ids = self._build_text_input_ids(question)
        self.ttft_debug_extra_timings["Prompt Build"] = (time.time() - token_start) * 1000
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOKEN_TIME)

        if input_ids.size >= self.context_max_length:
            logger.error(f"Question long than {self.context_max_length}, please shorten it!")
            sys.exit(1)

        logger.success("response:")
        output, input_tokens, output_tokens = self.generate_from_ids(
            input_ids,
            prefill_total_started=True,
        )
        return output, input_tokens, output_tokens


def validate_paths(parsed_args: argparse.Namespace) -> None:
    for attr in (
        "prefill_path",
        "prefill_mtp_path",
        "decode_mtp_path",
        "decode_verify_path",
        "tokenizer_dir",
        "embedding_path",
    ):
        path = Path(getattr(parsed_args, attr))
        if not path.exists():
            raise FileNotFoundError(f"--{attr} does not exist: {path}")


if __name__ == "__main__":
    args = get_args()
    validate_paths(args)

    from tcim.test_utils.utils import DeviceLock

    def run_once() -> None:
        hmqwen = HmQwenMTP(
            args.prefill_path,
            args.prefill_mtp_path,
            args.decode_mtp_path,
            args.decode_verify_path,
            args.embedding_path,
            args.tokenizer_dir,
            args.ndevice,
            debug=args.debug,
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
                    question = "请介绍一下存算一体技术的优势"

                start_time = time.time()
                try:
                    hmqwen.chat(question)
                    _ = time.time() - start_time
                    if args.debug:
                        show_ttft_breakdown(
                            hmqwen.ttft_time,
                            hmqwen.perf_tracker,
                            extra_timings=hmqwen.ttft_debug_extra_timings,
                        )
                    hmqwen.perf_tracker.show_summary()
                    hmqwen.last_spec_stats.show_summary()
                    hmqwen.perf_tracker.reset_perf_time()
                except Exception as exc:
                    print(f"Error during chat: {exc}")
                    import traceback

                    traceback.print_exc()
                    if not args.it:
                        break
                    continue
                if not args.it:
                    break
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        except Exception as exc:
            print(f"Program execution failed: {exc}")

    if os.environ.get("HDPL_PLATFORM") == "ISIM":
        run_once()
    else:
        with contextlib.ExitStack() as stack:
            for dev_id in range(args.ndevice):
                stack.enter_context(DeviceLock("xh2", dev_id, "qwen3.5_hmm_mtp_demo"))
            run_once()
