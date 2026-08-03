# Copyright 2025 HOUMO AI
#
# File: gemma4.py
# Description:
#   Gemma4 demo
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
import base64
import warnings
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
try:
    from transformers import GemmaTokenizer
except ImportError as e:
    raise ImportError(
        "Transformers not available. Please install transformers >= 5.5.0"
    )
from .processor import XHGemma4Processor
import tcim_lite as tcim

RESET = "\x1b[0m"
GREY = "\x1b[98;20m"
GREEN = "\x1b[92;20m"
FUCHSIA = "\033[1;95m"
YELLOW = "\x1b[93;20m"
CYAN = "\x1b[96;20m"


class PerfStats:
    def __init__(self):
        self.times = {}
        self.counts = {}

    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - start) * 1000.0)

    def add(self, name: str, elapsed_ms: float):
        self.times[name] = self.times.get(name, 0.0) + elapsed_ms

    def inc(self, name: str):
        """Increment the iteration count for a stage."""
        self.counts[name] = self.counts.get(name, 0) + 1

    def get(self, name: str) -> float:
        return self.times.get(name, 0.0)

    def get_count(self, name: str) -> int:
        return self.counts.get(name, 0)

    def stage_total(self, stage: str) -> float:
        keys = ("preprocess", "embed", "h2d", "infer", "d2h")
        return sum(self.get(f"{stage}.{key}") for key in keys)

    @staticmethod
    def _fmt_ms(value: float) -> str:
        return f"{value:10.2f}" if value > 0 else "          -"

    def _print_load(self, name: str, key: str):
        value = self.get(key)
        if value > 0:
            print(f"{CYAN}  {name:<10}{YELLOW}{value:8.2f} ms{RESET}")

    def _print_stage_row(self, title: str, stage: str, count: int = 1):
        total = self.stage_total(stage)
        if total <= 0:
            return
        n = max(count, 1)
        preprocess = (self.get(f"{stage}.preprocess") or self.get(f"{stage}.embed")) / n
        h2d = self.get(f"{stage}.h2d") / n
        infer = self.get(f"{stage}.infer") / n
        d2h = self.get(f"{stage}.d2h") / n
        avg_total = total / n
        print(
            f"{CYAN}{title:<12}"
            f"{YELLOW}{self._fmt_ms(preprocess)}"
            f"{YELLOW}{self._fmt_ms(h2d)}"
            f"{YELLOW}{self._fmt_ms(infer)}"
            f"{YELLOW}{self._fmt_ms(d2h)}  "
            f"{YELLOW}{avg_total:10.2f}{RESET}"
        )

    def print_summary(
        self,
        input_tokens: int,
        output_tokens: int,
        has_image: bool,
        has_audio: bool,
        has_assistant: bool = False,
    ):
        vision_total = self.stage_total("vision") if has_image else 0.0
        audio_total = self.stage_total("audio") if has_audio else 0.0
        prefill_total = self.stage_total("prefill")
        ttft = vision_total + audio_total + prefill_total
        # Use only infer time (not embed/h2d/d2h) for throughput speed
        prefill_infer = self.get("prefill.infer")
        decode_infer = self.get("decode.infer")
        assistant_infer = self.get("assistant.infer") if has_assistant else 0.0
        prefill_speed = (
            input_tokens / (prefill_infer / 1000.0) if prefill_infer > 0 else 0.0
        )
        # MTP: output tokens come from both assistant (draft) and decode (verify),
        # so decode speed must account for the total infer time of both models.
        decode_infer_total = decode_infer + assistant_infer
        decode_speed = (
            output_tokens / (decode_infer_total / 1000.0)
            if decode_infer_total > 0 and output_tokens > 0
            else 0.0
        )
        total = self.get("total")

        print(f"{CYAN}{'─' * 60}{RESET}")
        print(f"{CYAN}  Performance Summary{RESET}")
        print(f"{CYAN}{'─' * 60}{RESET}")
        print(f"{CYAN}Model Load Times{RESET}")
        self._print_load("Prefill", "prefill.load")
        self._print_load("Decode", "decode.load")
        self._print_load("Vision", "vision.load")
        self._print_load("Audio", "audio.load")
        self._print_load("Assistant", "assistant.load")
        self._print_load("PLE", "ple.load")
        self._print_load("Embedding", "embedding.load")

        print(f"\n{CYAN}Processor{RESET}")
        processor_time = self.get("processor")
        if processor_time > 0:
            print(f"{CYAN}  {'apply_chat':<10}{YELLOW}{processor_time:8.2f} ms{RESET}")

        print(
            f"\n{CYAN}{'Stage':<12}{'Pre/Emb':>10}{'H2D':>10}"
            f"{'Infer':>10}{'D2H':>10}  {'Total(avg)':>10}{RESET}"
        )
        if has_image:
            self._print_stage_row("Vision", "vision")
        if has_audio:
            self._print_stage_row("Audio", "audio", self.get_count("audio.steps"))
        self._print_stage_row("Prefill", "prefill", self.get_count("prefill.steps"))
        if has_assistant:
            self._print_stage_row(
                "Asst", "assistant", self.get_count("assistant.steps")
            )
        self._print_stage_row("Decode", "decode", self.get_count("decode.steps"))

        print(f"\n{CYAN}Tokens & Speeds{RESET}")
        print(f"{CYAN}  Input tokens     {YELLOW}{input_tokens:8d}{RESET}")
        print(f"{CYAN}  Output tokens    {YELLOW}{output_tokens:8d}{RESET}")
        print(f"{CYAN}  TTFT             {YELLOW}{ttft:8.2f} ms{RESET}")
        print(
            f"{CYAN}  Prefill speed    {YELLOW}{prefill_speed:8.2f} tok/s{CYAN} (infer only){RESET}"
        )
        decode_note = "(decode+asst infer)" if has_assistant else "(infer only)"
        print(
            f"{CYAN}  Decode speed     {YELLOW}{decode_speed:8.2f} tok/s{CYAN} {decode_note}{RESET}"
        )
        if total > 0:
            overall_speed = (
                (input_tokens + output_tokens) / (total / 1000.0) if total > 0 else 0.0
            )
            print(f"{CYAN}  Total time       {YELLOW}{total:8.2f} ms{RESET}")
            print(f"{CYAN}  Overall speed    {YELLOW}{overall_speed:8.2f} tok/s{RESET}")
        print(f"{CYAN}{'─' * 60}{RESET}")


def is_valid_char(cp):
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x0041 <= cp <= 0x005A
        or 0x0061 <= cp <= 0x007A
    )


class Gemma4(object):
    def __init__(
        self,
        prefill_path,
        decode_path,
        embedding_path,
        tokenizer_dir,
        vit_path=None,
        assistant_path=None,
        max_new_tokens=2048,
        devices=0,
        verbose=False,
    ):
        self.verbose = verbose
        self.sliding_window = 1024
        self.sliding_cache_valid_length = 0
        self.accepted_count_input_name = None
        self.max_new_tokens = max_new_tokens
        self.visual_bidirectional_attention = True
        if isinstance(devices, int):
            devices = [devices]
        self.devices = devices
        self.perf = PerfStats()

        self.processor = self._load_tokenizer(tokenizer_dir)
        self.tokenizer: GemmaTokenizer = self.processor.tokenizer
        self.pad_token_id = (
            0
            if not hasattr(self.tokenizer, "pad_token_id")
            else self.tokenizer.pad_token_id
        )
        self.image_token_id = self.processor.image_token_id
        self.video_token_id = self.processor.video_token_id
        self.audio_token_id = self.processor.audio_token_id
        self.eos_ids = [1, 106]

        backend_name = "Xh2HalBackend"
        dm = tcim.runtime.DevManager(devices, backend_name)
        self.wm = tcim.runtime.WeightManager(dm)

        self.prefill, self.decode = self._load_llm(prefill_path, decode_path)
        self.prefill_input_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        self.llm_embed_dim = self.decode.get_input_info(
            self.decode.get_input_name(0)
        ).shape[2]
        self.decode_input_len = self.decode.get_input_info(
            self.decode.get_input_name(0)
        ).shape[1]
        self.decode_sliding_attention_mask_width = (
            self._get_decode_sliding_attention_mask_width()
        )
        self.context_max_length = self._get_context_max_length()
        self._print(
            f"Prefill loaded: prefill_chunk_len={self.prefill_input_len}, embed_dim={self.llm_embed_dim}, context_max_length={self.context_max_length}"
        )

        self.vit = None
        if vit_path and os.path.isfile(vit_path):
            self.vit = self._load_vision(vit_path)

        self.audio = None
        self.PLE = None

        self.assistant = None
        if assistant_path and os.path.isfile(assistant_path):
            self.assistant = self._load_assistant(assistant_path)
        self.num_draft_tokens = self.decode_input_len - 1

        self.embedding = self._load_embedding(embedding_path)

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _get_context_max_length(self):
        cache_names = [
            self.prefill.get_input_name(i)
            for i in range(self.prefill.get_num_inputs())
            if "cache" in self.prefill.get_input_name(i).lower()
        ]
        context_max_length = self.prefill.get_input_info(cache_names[-1]).shape[2]
        return context_max_length

    def _get_decode_sliding_attention_mask_width(self):
        return self.decode.get_input_info(self.decode.get_input_name(3)).shape[3]

    @staticmethod
    def _load_tokenizer(tokenizer_dir):
        processor = XHGemma4Processor.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )
        return processor

    def _load_vision(self, vit_path):
        wm = self.wm
        if len(self.devices) > 1:
            wm = tcim.runtime.WeightManager(self.devices[0])
        with self.perf.track("vision.load"):
            vit = tcim.runtime.load(vit_path, option=tcim.runtime.Option(wm))
        self._log_model_io(vit, "ViT")
        return vit

    def _load_audio(self, audio_path):
        raise NotImplementedError("Audio processing is not implemented")

    def _load_llm(self, prefill_path, decode_path):
        with self.perf.track("prefill.load"):
            prefill = tcim.runtime.load(
                prefill_path, option=tcim.runtime.Option(self.wm)
            )
        self._log_model_io(prefill, "Prefill")
        cache_names = [
            prefill.get_input_name(i)
            for i in range(prefill.get_num_inputs())
            if "cache" in prefill.get_input_name(i).lower()
        ]
        opt = tcim.runtime.Option(self.wm)
        opt.set_dummy_tensors(cache_names)
        with self.perf.track("decode.load"):
            decode = tcim.runtime.load(decode_path, option=opt)
        self._log_model_io(decode, "Decode")

        for name in cache_names:
            info = prefill.get_input_info(name)
            prefill.set_input(name, np.zeros(info.shape, dtype=np.dtype(info.dtype)))
            decode.set_input(name, prefill.get_dev_input(name))

        for idx in range(decode.get_num_inputs()):
            name = decode.get_input_name(idx)
            if name == "accepted_count" or name.startswith("accepted_count."):
                self.accepted_count_input_name = name
                self._print(f"Decode accepted_count input enabled: {name}")
                break
        return prefill, decode

    def _load_assistant(self, assistant_path):
        last_kvcache_names = self._get_last_kvcache_names()
        dummy_tensors = [
            "shared_key_cache_sliding",
            "shared_value_cache_sliding",
            "shared_key_cache_full",
            "shared_value_cache_full",
        ]
        opt = tcim.runtime.Option(self.wm)
        opt.set_dummy_tensors(dummy_tensors)
        self._print(f"Loading MTP draft model from {assistant_path}")
        with self.perf.track("assistant.load"):
            assistant = tcim.runtime.load(assistant_path, option=opt)
        self._log_model_io(assistant, "Assistant")
        for idx, name in enumerate(dummy_tensors):
            self._print(f"Bind {last_kvcache_names[idx]} ==> {name}")
            assistant.set_input(
                name, self.prefill.get_dev_input(last_kvcache_names[idx])
            )
        return assistant

    def _load_embedding(self, embedding_path):
        with self.perf.track("embedding.load"):
            saved = torch.load(embedding_path, map_location="cpu", weights_only=True)
            embedding = nn.Embedding(
                saved["weight"].shape[0],
                saved["weight"].shape[1],
                padding_idx=self.pad_token_id,
                dtype=torch.float16,
            )
            embedding.load_state_dict(saved, strict=False)
        return embedding

    def _get_last_kvcache_names(self):
        kcache_names = [
            self.prefill.get_input_name(i)
            for i in range(self.prefill.get_num_inputs())
            if "kcache" in self.prefill.get_input_name(i).lower()
        ]
        vcache_names = [
            self.prefill.get_input_name(i)
            for i in range(self.prefill.get_num_inputs())
            if "vcache" in self.prefill.get_input_name(i).lower()
        ]
        last_kcache_local, last_kcache_global = kcache_names[-2], kcache_names[-1]
        last_vcache_local, last_vcache_global = vcache_names[-2], vcache_names[-1]
        return [
            last_kcache_local,
            last_vcache_local,
            last_kcache_global,
            last_vcache_global,
        ]

    def _log_model_io(self, model, model_name):
        """Log input/output info for a model, marking KV-cache inputs."""
        n_in = model.get_num_inputs()
        n_out = model.get_num_outputs()
        self._print(f"[{model_name}] {n_in} inputs, {n_out} outputs:")
        for i in range(n_in):
            name = model.get_input_name(i)
            if "cache" in name.lower():
                continue
            info = model.get_input_info(name)
            self._print(
                f"  in[{i}]: {name} shape={info.shape} dtype={np.dtype(info.dtype).name}"
            )
        for i in range(n_out):
            name = model.get_output_name(i)
            info = model.get_output_info(name)
            self._print(
                f"  out[{i}]: {name} shape={info.shape} dtype={np.dtype(info.dtype).name}"
            )

    def _build_embeddings(
        self, input_ids: torch.Tensor, inputs: Dict[str, torch.Tensor]
    ):
        img_mask: torch.Tensor = input_ids == self.image_token_id
        audio_mask: torch.Tensor = input_ids == self.audio_token_id
        llm_ids: torch.Tensor = input_ids.clone()
        llm_ids[img_mask] = self.pad_token_id
        llm_ids[audio_mask] = self.pad_token_id
        embeds: torch.Tensor = self.embedding(llm_ids)

        if (
            img_mask.any()
            and self.vit is not None
            and inputs.get("pixel_values") is not None
        ):
            img_emb: torch.Tensor = self._run_vision(
                inputs
            )  # [image_soft_token, embed_dim]
            embeds[img_mask] = img_emb

        if (
            audio_mask.any()
            and self.audio is not None
            and inputs.get("input_features") is not None
        ):
            audio_emb: torch.Tensor = self._run_audio(
                inputs
            )  # [audio_soft_token, embed_dim]
            embeds[audio_mask] = audio_emb

        return embeds, llm_ids

    @staticmethod
    def _aligned(size: int, align: int) -> int:
        return ((size + align - 1) // align) * align

    def _build_masks(
        self, cur_len, past_len, mask_len=None, mm_types: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len = mask_len
        neg = torch.tensor(torch.finfo(torch.float16).min, dtype=torch.float16)
        global_ctx = self.context_max_length

        global_mask = torch.full((1, 1, q_len, global_ctx), neg, dtype=torch.float16)
        valid_k = min(global_ctx, max(1, past_len + cur_len))
        for q in range(q_len):
            if q < cur_len:
                global_mask[0, 0, q, : min(valid_k, past_len + q + 1)] = 0
            else:
                global_mask[0, 0, q, 0] = 0

        sw = self.sliding_window
        slide_ctx = (
            global_ctx
            if sw is None
            else min(global_ctx, self._aligned(sw + q_len - 1, 16))
        )
        clamped_past = min(past_len, sw - 1) if sw is not None and sw > 0 else past_len
        local_mask = torch.full((1, 1, q_len, slide_ctx), neg, dtype=torch.float16)
        for q in range(q_len):
            if q < cur_len:
                causal_end = min(slide_ctx, clamped_past + q + 1)
                sw_start = max(0, clamped_past + q - sw + 1) if sw is not None else 0
                local_mask[0, 0, q, sw_start:causal_end] = 0
            else:
                local_mask[0, 0, q, 0] = 0

        if (
            self.visual_bidirectional_attention
            and mm_types is not None
            and mm_types.numel() > 0
        ):
            mm = mm_types[0, :cur_len] if mm_types.dim() == 2 else mm_types[:cur_len]
            is_mm = (mm == 1) | (mm == 2)
            cache_offset = max(0, past_len - clamped_past)
            group_start = None
            for idx in range(cur_len):
                if bool(is_mm[idx]) and group_start is None:
                    group_start = idx
                if group_start is not None and (
                    idx == cur_len - 1 or not bool(is_mm[idx + 1])
                ):
                    group_end = idx + 1
                    abs_start, abs_end = past_len + group_start, past_len + group_end
                    global_mask[0, 0, group_start:group_end, abs_start:abs_end] = 0
                    c_start = max(0, abs_start - cache_offset)
                    c_end = min(slide_ctx, abs_end - cache_offset)
                    if c_start < slide_ctx and c_end > 0:
                        local_mask[0, 0, group_start:group_end, c_start:c_end] = 0
                    group_start = None

        return global_mask, local_mask

    def _build_draft_masks(self, past_len) -> tuple[torch.Tensor, torch.Tensor]:
        width = self.assistant.get_dev_input(
            self.assistant.get_input_name(3)
        ).info.shape[-1]
        l_mask = np.full((1, 1, 1, width), np.finfo(np.float16).min, dtype=np.float16)
        valid = min(width, max(1, int(self.sliding_cache_valid_length or past_len)))
        end = valid
        start = max(0, end - self.sliding_window)
        l_mask[0, 0, 0, start:end] = 0.0
        return l_mask

    def _run_vision(self, inputs):
        pixel_values: torch.Tensor = inputs["pixel_values"]
        attention_mask: torch.Tensor = inputs["visual_attention_mask"]
        pooling_matrix: torch.Tensor = inputs["pooling_matrix"]
        pixel_position_ids: torch.Tensor = inputs["pixel_position_ids"]
        image_soft_token_count = inputs.get("image_soft_token_count")
        counts = (
            torch.as_tensor(image_soft_token_count).flatten()
            if image_soft_token_count is not None
            else None
        )
        if counts is not None and len(counts) != pixel_values.shape[0]:
            raise ValueError(
                "image_soft_token_count does not match vision input batch size"
            )

        outputs = []
        for index in range(pixel_values.shape[0]):
            with self.perf.track("vision.h2d"):
                self.vit.set_input(
                    "pixel_values",
                    pixel_values[index : index + 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                )
                self.vit.set_input(
                    "attention_mask",
                    attention_mask[index : index + 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                )
                self.vit.set_input(
                    "pooling_matrix",
                    pooling_matrix[index : index + 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                )
                self.vit.set_input(
                    "pixel_position_ids",
                    pixel_position_ids[index : index + 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32),
                )

            with self.perf.track("vision.infer"):
                self.vit.run()
                self.vit.sync()

            with self.perf.track("vision.d2h"):
                out = torch.from_numpy(
                    self.vit.get_output(self.vit.get_output_name(0)).numpy()
                )
                count = int(counts[index]) if counts is not None else out.shape[1]
                outputs.append(out[0, :count, :])

        return torch.cat(outputs, dim=0)

    def _run_audio(self, inputs):
        raise NotImplementedError("Audio processing is not implemented")

    def _prefill(
        self,
        llm_ids: torch.Tensor,
        embeds: torch.Tensor,
        mm_types: torch.Tensor = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        input_len = embeds.shape[1]
        steps = math.ceil(input_len / self.prefill_input_len)
        cur_len = 0
        for step in range(steps):
            start = step * self.prefill_input_len
            end = min((step + 1) * self.prefill_input_len, input_len)
            cur_len = end - start
            sub_emb = embeds[:, start:end]
            sub_llm_ids = llm_ids[:, start:end]

            with self.perf.track("prefill.embed"):
                if cur_len < self.prefill_input_len:
                    padded_ids = torch.from_numpy(
                        np.array(
                            [[self.pad_token_id] * (self.prefill_input_len - cur_len)]
                        )
                    )
                    sub_emb = torch.cat([sub_emb, self.embedding(padded_ids)], dim=1)
                    sub_llm_ids = torch.cat([sub_llm_ids, padded_ids], dim=1)

                pl_embeds = None
                if self.PLE is not None:
                    pl_embeds: torch.Tensor = self.PLE(sub_llm_ids).view(
                        -1,
                        self.prefill_input_len,
                        self.num_hidden_layers,
                        self.hidden_size_per_layer_input,
                    )
                chunk_mm = mm_types[:, start:end][0] if mm_types is not None else None

                _, l_mask = self._build_masks(
                    cur_len, start, self.prefill_input_len, chunk_mm
                )

                # Prepare numpy arrays for set_input
                sub_emb_np = sub_emb.detach().cpu().numpy().astype(np.float16)
                l_mask_np = l_mask.detach().cpu().numpy().astype(np.float16)
                pl_embeds_np = (
                    pl_embeds.detach().cpu().numpy().astype(np.float16)
                    if pl_embeds is not None
                    else None
                )

            with self.perf.track("prefill.h2d"):
                self.prefill.set_input(self.prefill.get_input_name(0), sub_emb_np)
                self.prefill.set_input(
                    self.prefill.get_input_name(1), np.array([start], dtype="int32")
                )
                self.prefill.set_input(
                    self.prefill.get_input_name(2), np.array([cur_len], dtype="int32")
                )
                self.prefill.set_input(self.prefill.get_input_name(3), l_mask_np)
                if pl_embeds_np is not None:
                    self.prefill.set_input(self.prefill.get_input_name(4), pl_embeds_np)

            with self.perf.track("prefill.infer"):
                self.prefill.run()
                self.prefill.sync()
                self.perf.inc("prefill.steps")

        with self.perf.track("prefill.d2h"):
            logits = (
                self.prefill.get_output(self.prefill.get_output_name(0))
                .numpy()
                .astype(np.float32)
            )
            valid_len = min(logits.shape[1], cur_len)
            next_ids = logits[0, valid_len - 1 : valid_len, :].argmax(-1)
            hidden_states = None
            if self.assistant is not None:
                hidden_states = self.prefill.get_output(
                    self.prefill.get_output_name(1)
                ).numpy()
                self.sliding_cache_valid_length = min(
                    self.decode_sliding_attention_mask_width, input_len
                )
        return next_ids, hidden_states

    def _decode_step(
        self, token_ids: list, past_len: int, accepted_count: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        input_ids = token_ids.copy()
        cur_len = len(input_ids)
        if cur_len < self.decode_input_len:
            input_ids += [self.pad_token_id] * (self.decode_input_len - cur_len)
        input_ids = torch.from_numpy(np.array([input_ids]))

        with self.perf.track("decode.embed"):
            embeds: torch.Tensor = self.embedding(input_ids)
            if self.PLE is not None:
                pl_embeds: torch.Tensor = self.PLE(input_ids).view(
                    -1,
                    self.decode_input_len,
                    self.num_hidden_layers,
                    self.hidden_size_per_layer_input,
                )
                pl_embeds_np = pl_embeds.detach().cpu().numpy().astype(np.float16)

            _, l_mask = self._build_masks(cur_len, past_len, self.decode_input_len)

            # Prepare numpy arrays for set_input
            embeds_np = embeds.detach().cpu().numpy().astype(np.float16)
            l_mask_np = l_mask.detach().cpu().numpy().astype(np.float16)

        with self.perf.track("decode.h2d"):
            self.decode.set_input(self.decode.get_input_name(0), embeds_np)
            self.decode.set_input(
                self.decode.get_input_name(1), np.array([past_len], dtype="int32")
            )
            self.decode.set_input(
                self.decode.get_input_name(2), np.array([cur_len], dtype="int32")
            )
            self.decode.set_input(self.decode.get_input_name(3), l_mask_np)
            if self.PLE is not None:
                self.decode.set_input(self.decode.get_input_name(4), pl_embeds_np)
            if self.accepted_count_input_name is not None:
                self.decode.set_input(
                    self.accepted_count_input_name,
                    np.array([accepted_count], dtype="int32"),
                )

        with self.perf.track("decode.infer"):
            self.decode.run()
            self.decode.sync()

        with self.perf.track("decode.d2h"):
            logits: np.ndarray = (
                self.decode.get_output(self.decode.get_output_name(0))
                .numpy()
                .astype(np.float32)
            )
            hidden_states = None
            if self.assistant is not None:
                hidden_states = self.decode.get_output(
                    self.decode.get_output_name(1)
                ).numpy()
        return logits, hidden_states

    def _assistant_step(
        self, next_id: int, past_len: int, pos_idx: int, hidden_states: np.ndarray
    ) -> tuple[int, np.ndarray]:
        with self.perf.track("assistant.embed"):
            embeds = self.embedding(torch.from_numpy(np.array([[next_id]])))
            embeds = torch.cat([embeds, torch.from_numpy(hidden_states)], dim=2)
            l_mask = self._build_draft_masks(past_len)
            embeds_np = embeds.detach().cpu().numpy().astype(np.float16)
            l_mask_np = l_mask.astype(np.float16)

        with self.perf.track("assistant.h2d"):
            self.assistant.set_input(self.assistant.get_input_name(0), embeds_np)
            self.assistant.set_input(
                self.assistant.get_input_name(1),
                np.array([pos_idx - 1], dtype=np.int32),
            )
            self.assistant.set_input(
                self.assistant.get_input_name(2), np.array([1], dtype=np.int32)
            )
            self.assistant.set_input(self.assistant.get_input_name(3), l_mask_np)

        with self.perf.track("assistant.infer"):
            self.assistant.run()
            self.assistant.sync()

        with self.perf.track("assistant.d2h"):
            logits: np.ndarray = (
                self.assistant.get_output(self.assistant.get_output_name(0))
                .numpy()
                .astype(np.float32)
            )
            hidden_states = self.assistant.get_output(
                self.assistant.get_output_name(1)
            ).numpy()
            next_ids = logits.argmax(-1)
        return next_ids[0][0], hidden_states

    def _assistant_loop(
        self, next_id: int, past_len: int, hidden_states: np.ndarray
    ) -> tuple[int, np.ndarray]:
        draft_tokens = list()
        for step in range(self.num_draft_tokens):
            next_id, hidden_states = self._assistant_step(
                next_id, past_len, past_len + step, hidden_states
            )
            self.perf.inc("assistant.steps")
            draft_tokens.append(int(next_id))
            if next_id in self.eos_ids:
                break
        return draft_tokens, hidden_states

    def _decode_loop(
        self,
        next_ids: np.ndarray,
        input_ids: torch.Tensor,
        hidden_states: np.ndarray = None,
    ) -> tuple[str, int]:
        self._print(f"{CYAN}{'─' * 60}{RESET}")
        self._print(f"{CYAN}  Response:{RESET}")
        self._print(f"{CYAN}{'─' * 60}{RESET}")
        self._print(
            FUCHSIA + self.tokenizer.decode(next_ids) + RESET, end="", flush=True
        )  # first token
        history = input_ids[0].tolist() + next_ids.tolist()
        generated_ids = next_ids.tolist()
        past_len = input_ids.shape[1]
        decode_count = 0
        stop_decode = False
        slide = 10
        skip = 0
        last_text = self.tokenizer.decode(history[-slide:])
        next_id = next_ids.tolist()[-1]
        last_committed_tokens = 0
        total_committed_tokens = 0
        total_drafted = 0
        total_accepted = 0
        while (
            past_len < self.context_max_length
            and decode_count < self.max_new_tokens
            and not stop_decode
        ):
            token_ids = [next_ids.tolist()[-1]]
            if self.assistant is not None:
                draft_tokens, hidden_states = self._assistant_loop(
                    next_id, past_len, hidden_states
                )
                token_ids.extend(draft_tokens)
            logits, hidden_states = self._decode_step(
                token_ids, past_len, last_committed_tokens
            )
            self.perf.inc("decode.steps")

            if self.assistant is None:
                valid_len = min(logits.shape[1], len(token_ids))
                logits = logits[0, valid_len - 1 : valid_len, :]
                next_ids = logits.argmax(-1)
            else:
                accepted_count = 0
                for idx, draft_token in enumerate(draft_tokens):
                    target_token = int(logits[0, idx, :].argmax(-1))
                    if target_token != int(draft_token):
                        break
                    accepted_count += 1
                total_drafted += len(draft_tokens)
                total_accepted += accepted_count
                last_token = int(logits[0, accepted_count, :].argmax(-1))
                next_ids = np.array(
                    draft_tokens[:accepted_count] + [last_token], dtype=np.int64
                )
                next_id = int(next_ids[-1])
                hidden_states = hidden_states[:, accepted_count : accepted_count + 1, :]
                last_committed_tokens = accepted_count + 1
                total_committed_tokens += last_committed_tokens
                self.sliding_cache_valid_length = min(
                    self.decode_sliding_attention_mask_width,
                    self.sliding_cache_valid_length + last_committed_tokens,
                )

            token_ids = next_ids.tolist()
            for token_id in token_ids:
                if token_id in self.eos_ids:
                    stop_decode = True
                    break
                history.append(token_id)
                generated_ids.append(token_id)
                cur_text = self.tokenizer.decode(history[-(slide + 1) - skip :])
                text = cur_text[len(last_text) :]
                if text and is_valid_char(ord(text[-1])):
                    self._print(FUCHSIA + text + RESET, end="", flush=True)
                    last_text = self.tokenizer.decode(history[-slide:])
                    skip = 0
                else:
                    skip += 1
                past_len += 1
                decode_count += 1

        # flush remaining buffered text (e.g. trailing punctuation)
        if skip > 0:
            cur_text = self.tokenizer.decode(history[-skip:])
            self._print(FUCHSIA + cur_text + RESET, flush=True)
        self._print()  # newline after response
        # draft acceptance stats
        if self.assistant is not None and total_drafted > 0:
            rate = total_accepted / total_drafted * 100
            self._print(f"{CYAN}{'─' * 60}{RESET}")
            self._print(
                f"{CYAN}  Draft Acceptance: {YELLOW}{total_accepted}{CYAN}/{YELLOW}{total_drafted}{CYAN} ({YELLOW}{rate:.1f}%{CYAN}){RESET}"
            )
            self._print(f"{CYAN}  Output Tokens: {YELLOW}{decode_count}{RESET}")
            self._print(f"{CYAN}{'─' * 60}{RESET}")
        response_text = self.tokenizer.decode(generated_ids)
        return response_text, decode_count

    @staticmethod
    def _load_image(image_data):
        from PIL import Image

        if isinstance(image_data, str) and image_data.startswith("data:image/"):
            header, separator, payload = image_data.partition(",")
            if not separator or ";base64" not in header:
                raise ValueError("Invalid base64 image data URI")
            image_data = BytesIO(base64.b64decode(payload, validate=True))

        return Image.open(image_data).convert("RGB")

    @staticmethod
    def _load_audio_input(audio_path):
        import torchaudio

        waveform, sampling_rate = torchaudio.load(audio_path)
        if sampling_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sampling_rate, 16000)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        return waveform.numpy()

    def chat(
        self,
        question=None,
        image_path=None,
        audio_path=None,
        enable_thinking=False,
        content=None,
        messages=None,
    ):
        chat_start = time.perf_counter()

        if messages is None:
            if content is None:
                if image_path and audio_path:
                    q_text = question or "请详细描述这张图片和音频的内容。"
                elif image_path and not audio_path:
                    q_text = question or "请详细描述这张图片的内容。"
                elif audio_path and not image_path:
                    q_text = question or "请详细描述这个音频的内容。"
                else:
                    q_text = question or "你好，请介绍一下你自己。"

                content = []
                if image_path is not None:
                    content.append({"type": "image", "image": image_path})
                if audio_path is not None:
                    content.append({"type": "audio", "audio": audio_path})
                content.append({"type": "text", "text": q_text})
            else:
                q_text = "\n".join(
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                )
            messages = [{"role": "user", "content": content}]
        else:
            q_text = "\n".join(
                item.get("text", "")
                for message in messages
                for item in (
                    [{"type": "text", "text": message.get("content", "")}]
                    if isinstance(message.get("content", ""), str)
                    else message.get("content", [])
                )
                if item.get("type", "text") == "text"
            )

        self._print(f"{CYAN}{'─' * 60}{RESET}")
        self._print(f"{CYAN}  Question:{RESET}")
        self._print(f"{GREEN}  {q_text}{RESET}")
        self._print(f"{CYAN}{'─' * 60}{RESET}")

        processed_messages = []
        has_image = False
        has_audio = False
        for message in messages:
            message_content = message.get("content", "")
            if isinstance(message_content, str):
                message_content = [{"type": "text", "text": message_content}]

            media_parts = []
            text_parts = []
            for item in message_content:
                item_type = item.get("type", "text")
                if item_type == "image":
                    if self.vit is None:
                        continue
                    with self.perf.track("vision.preprocess"):
                        image = self._load_image(item.get("image"))
                    media_parts.append({"type": "image", "image": image})
                    has_image = True
                elif item_type == "audio":
                    if self.audio is None:
                        continue
                    with self.perf.track("audio.preprocess"):
                        waveform = self._load_audio_input(item.get("audio"))
                    media_parts.append(
                        {
                            "type": "audio",
                            "audio": waveform,
                            "sampling_rate": 16000,
                        }
                    )
                    has_audio = True
                elif item_type == "text":
                    text_parts.append(
                        {"type": "text", "text": item.get("text", "")}
                    )
                else:
                    raise ValueError(
                        f"Gemma4 does not support content type {item_type!r}"
                    )
            processed_messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": media_parts + text_parts,
                }
            )

        with self.perf.track("processor"):
            inputs = self.processor.apply_chat_template(
                processed_messages, enable_thinking=enable_thinking
            )

        input_ids: torch.Tensor = inputs["input_ids"]
        input_len = input_ids.shape[-1]
        if input_len >= self.context_max_length:
            raise ValueError(
                f"Input length {input_len} exceeds maximum length {self.context_max_length}"
            )

        embeds, llm_ids = self._build_embeddings(input_ids, inputs)

        mm_types = inputs.get("mm_token_type_ids")
        next_ids, hidden_states = self._prefill(llm_ids, embeds, mm_types=mm_types)
        self._print(f"Prefill done, first token: {self.tokenizer.decode(next_ids)}")
        response_text, output_tokens = self._decode_loop(
            next_ids, input_ids, hidden_states
        )

        # Record total wall time and print performance summary
        chat_total = (time.perf_counter() - chat_start) * 1000.0
        self.perf.add("total", chat_total)
        if self.verbose:
            self.perf.print_summary(
                input_len,
                output_tokens,
                has_image=has_image,
                has_audio=has_audio,
                has_assistant=self.assistant is not None,
            )
        return response_text


class Gemma4E(Gemma4):
    def __init__(
        self,
        prefill_path,
        decode_path,
        embedding_path,
        tokenizer_dir,
        PLE_path,
        vit_path=None,
        audio_path=None,
        assistant_path=None,
        max_new_tokens=2048,
        devices=0,
        verbose=False,
    ):
        super().__init__(
            prefill_path,
            decode_path,
            embedding_path,
            tokenizer_dir,
            vit_path,
            assistant_path,
            max_new_tokens,
            devices,
            verbose,
        )
        self.sliding_window = 512
        self.visual_bidirectional_attention = False

        self.audio = None
        if audio_path and os.path.isfile(audio_path):
            self.audio = self._load_audio(audio_path)

        self.num_hidden_layers = self.prefill.get_input_info(
            self.prefill.get_input_name(4)
        ).shape[2]
        self.hidden_size_per_layer_input = self.prefill.get_input_info(
            self.prefill.get_input_name(4)
        ).shape[3]
        self.PLE = self._load_PLE(PLE_path)

    def _load_PLE(self, PLE_path):
        with self.perf.track("ple.load"):
            saved = torch.load(PLE_path, map_location="cpu", weights_only=True)
            weight = saved["state_dict"]["embed_tokens_per_layer.weight"]
            embedding = nn.Embedding(
                weight.shape[0],
                weight.shape[1],
                padding_idx=self.pad_token_id,
                dtype=torch.float16,
            )
            embedding.weight.data.copy_(weight)
        return embedding

    def _load_audio(self, audio_path):
        wm = self.wm
        if len(self.devices) > 1:
            wm = tcim.runtime.WeightManager(self.devices[0])
        with self.perf.track("audio.load"):
            audio = tcim.runtime.load(audio_path, option=tcim.runtime.Option(wm))
        self._log_model_io(audio, "Audio")
        self.audio_feature_length = audio.get_input_info(audio.get_input_name(1)).shape[
            1
        ]
        self.audio_feature_size = audio.get_input_info(audio.get_input_name(0)).shape[2]
        self.processor.config.audio_feature_length = self.audio_feature_length
        self.processor.config.audio_chunk_overlap = 8
        return audio

    def _run_audio(self, inputs):
        input_features_mask = inputs["input_features_mask"]
        if not isinstance(input_features_mask, torch.Tensor):
            input_features_mask = torch.as_tensor(input_features_mask)
        valid_frames = int(input_features_mask.to(torch.int64).sum().item())
        chunk_ranges = inputs.get("audio_chunk_output_ranges")

        if chunk_ranges is None:
            audio_embeds, audio_embeds_mask = self._run_audio_once(inputs)
            outputs = [
                embeds[mask.to(torch.bool)]
                for embeds, mask in zip(audio_embeds, audio_embeds_mask)
            ]
            audio_out = torch.cat(outputs, dim=0)
            self._print(f"Audio output: valid_frames={valid_frames}")
            return audio_out

        ranges = torch.as_tensor(chunk_ranges, dtype=torch.long)
        input_indices = inputs.get("audio_chunk_input_indices")
        if input_indices is None:
            input_indices = torch.zeros(len(ranges), dtype=torch.long)
        else:
            input_indices = torch.as_tensor(input_indices, dtype=torch.long)
            if len(input_indices) != len(ranges):
                raise ValueError(
                    "audio_chunk_input_indices does not match audio chunks"
                )
        audio_embeds = None
        outputs = [
            []
            for _ in range(
                int(input_indices.max().item()) + 1 if len(input_indices) else 0
            )
        ]
        for chunk_idx, (keep_start, keep_end) in enumerate(ranges.tolist()):
            chunk_inputs = self._slice_audio_inputs(inputs, chunk_idx)
            audio_embeds, audio_embeds_mask = self._run_audio_once(chunk_inputs)
            chunk_out = audio_embeds[0][audio_embeds_mask[0].to(torch.bool)]
            input_index = int(input_indices[chunk_idx].item())
            outputs[input_index].append(chunk_out[int(keep_start) : int(keep_end)])

        ordered_outputs = [
            chunk_output for input_outputs in outputs for chunk_output in input_outputs
        ]
        audio_out = (
            torch.cat(ordered_outputs, dim=0)
            if ordered_outputs
            else torch.empty(0, audio_embeds.shape[-1])
        )
        stride = inputs.get("audio_chunk_stride")
        overlap = inputs.get("audio_chunk_overlap")
        stride_value = (
            int(torch.as_tensor(stride).flatten()[0].item())
            if stride is not None
            else None
        )
        overlap_value = (
            int(torch.as_tensor(overlap).flatten()[0].item())
            if overlap is not None
            else None
        )
        self._print(
            f"Audio chunks: {len(ranges)}, stride={stride_value}, overlap={overlap_value}, "
            f"output={audio_out.shape} (valid_frames={valid_frames})"
        )
        return audio_out

    def _run_audio_once(self, chunk_inputs):
        audio_input_aliases = {"attention_mask": "audio_attention_mask"}
        with self.perf.track("audio.h2d"):
            for i in range(self.audio.get_num_inputs()):
                name = self.audio.get_input_name(i)
                bare_name = name.removesuffix(".hmcc.format")
                input_key = (
                    bare_name
                    if bare_name in chunk_inputs
                    else audio_input_aliases.get(bare_name)
                )
                if input_key not in chunk_inputs:
                    raise KeyError(
                        f"Audio input {name!r} is not found in processor inputs"
                    )
                value = chunk_inputs[input_key]
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                self.audio.set_input(
                    name, value.detach().cpu().numpy().astype(np.float16)
                )

        with self.perf.track("audio.infer"):
            self.audio.run()
            self.audio.sync()
            self.perf.inc("audio.steps")

        with self.perf.track("audio.d2h"):
            audio_embeds = torch.from_numpy(
                self.audio.get_output(self.audio.get_output_name(0)).numpy()
            )
            audio_embeds_mask = torch.from_numpy(
                self.audio.get_output(self.audio.get_output_name(1)).numpy()
            )
        return audio_embeds, audio_embeds_mask

    def _slice_audio_inputs(self, inputs, chunk_idx):
        chunk_inputs = dict(inputs)
        for key in ("input_features", "input_features_mask", "audio_attention_mask"):
            value = inputs.get(key)
            if hasattr(value, "shape") and value.shape[0] > chunk_idx:
                chunk_inputs[key] = value[chunk_idx : chunk_idx + 1]
        return chunk_inputs


API_NAME = "HmGemma4"
_GEMMA4_SIZES = ("26b-a4b", "31b")
_GEMMA4E_SIZES = ("e2b", "e4b")


@dataclass(frozen=True)
class Gemma4Artifacts:
    class_name: str
    prefill_path: str
    decode_path: str
    embedding_path: str
    tokenizer_dir: str
    ple_path: Optional[str]
    vit_path: Optional[str]
    audio_path: Optional[str]
    assistant_path: Optional[str]
    max_new_tokens: int
    devices: List[int]
    enable_thinking: bool
    verbose: bool


def select_gemma4_class_name(model_size: str) -> str:
    normalized_size = model_size.strip().lower()
    if normalized_size in _GEMMA4_SIZES:
        return "Gemma4"
    if normalized_size in _GEMMA4E_SIZES:
        return "Gemma4E"
    supported = ", ".join(_GEMMA4_SIZES + _GEMMA4E_SIZES)
    raise ValueError(
        f"Unsupported Gemma4 model size {model_size!r}. Supported sizes: {supported}"
    )


def parse_devices(value: Any) -> List[int]:
    if isinstance(value, int):
        devices = [value]
    elif isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
        if not all(parts):
            raise ValueError(f"Invalid Gemma4 devices value: {value!r}")
        try:
            devices = [int(item) for item in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid Gemma4 devices value: {value!r}") from exc
    elif (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(item, int) for item in value)
    ):
        devices = list(value)
    else:
        raise ValueError(f"Invalid Gemma4 devices value: {value!r}")

    if any(device < 0 for device in devices):
        raise ValueError(f"Invalid Gemma4 devices value: {value!r}")
    return devices


def _optional_path(model_args: Dict[str, Any], key: str, default: Path):
    explicit = model_args.get(key)
    if explicit is not None:
        return os.path.normpath(str(explicit))
    return str(default) if default.exists() else None


def resolve_gemma4_artifacts(
    model_path: str,
    model_size: str,
    backend: str,
    model_args: Dict[str, Any],
) -> Gemma4Artifacts:
    if backend != "hmm":
        raise NotImplementedError(
            f"Backend {backend!r} is recognized but is not implemented for "
            "model 'gemma4'. Supported executable backends: hmm"
        )

    normalized_size = model_size.strip().lower()
    root = Path(os.path.normpath(model_path))
    devices = parse_devices(model_args.get("devices", 0))
    extension = ".hmms" if len(devices) > 1 else ".hmm"
    prefix = f"gemma4-{normalized_size}"
    class_name = select_gemma4_class_name(normalized_size)
    verbose = model_args.get("verbose", False)
    if not isinstance(verbose, bool):
        raise ValueError("Gemma4 verbose must be a boolean")

    tokenizer_dir = os.path.normpath(
        str(model_args.get("tokenizer_dir", root / "hmquant" / "hf_config"))
    )
    embedding_path = os.path.normpath(
        str(model_args.get("embedding_path", root / "hmquant" / "quant_embedding.pt"))
    )
    prefill_path = os.path.normpath(
        str(model_args.get("prefill_path", root / f"{prefix}_prefill{extension}"))
    )
    decode_path = os.path.normpath(
        str(model_args.get("decode_path", root / f"{prefix}_decode{extension}"))
    )
    ple_path = None
    if class_name == "Gemma4E":
        ple_path = os.path.normpath(
            str(
                model_args.get(
                    "PLE_path",
                    root / "hmquant" / "per_layer_input_embedding.pt",
                )
            )
        )

    required = {
        "tokenizer_dir": tokenizer_dir,
        "embedding_path": embedding_path,
        "prefill_path": prefill_path,
        "decode_path": decode_path,
    }
    if ple_path is not None:
        required["PLE_path"] = ple_path
    missing = [
        f"{name}={path}" for name, path in required.items() if not Path(path).exists()
    ]
    if missing:
        raise ValueError("Missing required Gemma4 artifacts: " + ", ".join(missing))

    return Gemma4Artifacts(
        class_name=class_name,
        prefill_path=prefill_path,
        decode_path=decode_path,
        embedding_path=embedding_path,
        tokenizer_dir=tokenizer_dir,
        ple_path=ple_path,
        vit_path=_optional_path(model_args, "vit_path", root / f"{prefix}_visual.hmm"),
        audio_path=_optional_path(
            model_args, "audio_path", root / f"{prefix}_audio.hmm"
        ),
        assistant_path=_optional_path(
            model_args,
            "assistant_path",
            root / f"{prefix}_assistant{extension}",
        ),
        max_new_tokens=int(model_args.get("max_new_tokens", 2048)),
        devices=devices,
        enable_thinking=bool(model_args.get("enable_thinking", False)),
        verbose=verbose,
    )


def _message_parts(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    result = []
    for message in messages:
        contents = message.content
        if isinstance(contents, str):
            content_parts = [{"type": "text", "text": contents}]
        else:
            media_parts = []
            text_parts = []
            for content in contents:
                content_type = content.type
                if content_type == "text":
                    text_parts.append({"type": "text", "text": content.text})
                elif content_type == "image":
                    media_parts.append({"type": "image", "image": content.image})
                elif content_type == "audio":
                    media_parts.append({"type": "audio", "audio": content.audio})
                else:
                    raise ValueError(
                        f"Gemma4 does not support EvalScope content type {content_type!r}"
                    )
            content_parts = media_parts + text_parts
        result.append({"role": message.role, "content": content_parts})

    return result


@register_model_api(name=API_NAME)
class HmGemma4(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Dict[str, Any],
    ) -> None:
        super().__init__(model_name, base_url, api_key, config)
        model_size = model_args.pop("model_size", None)
        backend = model_args.pop("backend", None)
        if not model_size:
            raise ValueError("`model_size` is required for HmGemma4")
        if not backend:
            raise ValueError("`backend` is required for HmGemma4")

        artifacts = resolve_gemma4_artifacts(
            model_name, model_size, backend, model_args
        )
        runtime_class = Gemma4E if artifacts.class_name == "Gemma4E" else Gemma4
        runtime_args = {
            "prefill_path": artifacts.prefill_path,
            "decode_path": artifacts.decode_path,
            "embedding_path": artifacts.embedding_path,
            "tokenizer_dir": artifacts.tokenizer_dir,
            "vit_path": artifacts.vit_path,
            "assistant_path": artifacts.assistant_path,
            "max_new_tokens": artifacts.max_new_tokens,
            "devices": artifacts.devices,
            "verbose": artifacts.verbose,
        }
        if artifacts.class_name == "Gemma4E":
            runtime_args["PLE_path"] = artifacts.ple_path
            runtime_args["audio_path"] = artifacts.audio_path
        self.model = runtime_class(**runtime_args)
        self.enable_thinking = artifacts.enable_thinking
        normalized_path = os.path.normpath(model_name)
        self.model_name = os.path.basename(normalized_path) or API_NAME

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if tools:
            raise NotImplementedError("HmGemma4 does not support tools")
        messages = _message_parts(input)
        response = self.model.chat(
            messages=messages,
            enable_thinking=self.enable_thinking,
        )
        return ModelOutput.from_content(model=self.model_name, content=response)
