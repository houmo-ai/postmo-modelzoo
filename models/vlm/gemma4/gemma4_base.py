# Copyright 2025 HOUMO AI
#
# File: gemma4_base.py
# Description:
#   Gemma4 Base Class for shared functionality.
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
# fmt: off
import os
import sys
import math
import time
import numpy as np
import torch
from PIL import Image
from loguru import logger
try:
    from transformers import GemmaTokenizer
except ImportError as e:
    logger.error(f"Transformers not available: {e}, Please install transformers >= 5.5.0")
    exit(-1)
from gemma4_processor import XHGemma4Processor
import tcim_lite as tcim
from hmatc.utils.perf_infomations import InferencePerformanceTracker, PERFTYPE

MAX_SOFT_TOKENS = 280


def is_valid_char(cp):
    return 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A


class Gemma4Base:
    """Base class for Gemma4 inference (E2B and MoE)."""

    # Subclasses override these
    sliding_window = 512
    audio_enabled = False
    perf_tracker = InferencePerformanceTracker()

    def _init_common(self, devices):
        if isinstance(devices, int):
            devices = [devices]
        self.devices = devices
        self.image_token_id = 258880
        self.pad_token_id = 0

    @staticmethod
    def _log_model_io(model, model_name):
        """Log input/output info for a model, marking KV-cache inputs."""
        n_in = model.get_num_inputs()
        n_out = model.get_num_outputs()
        logger.info(f"[{model_name}] {n_in} inputs, {n_out} outputs:")
        for i in range(n_in):
            name = model.get_input_name(i)
            if "cache" in name.lower():
                continue
            info = model.get_input_info(name)
            logger.info(f"  in[{i}]: {name} shape={info.shape} dtype={np.dtype(info.dtype).name}")
        for i in range(n_out):
            name = model.get_output_name(i)
            info = model.get_output_info(name)
            logger.info(f"  out[{i}]: {name} shape={info.shape} dtype={np.dtype(info.dtype).name}")

    def _load_vision(self, vit_path, devices, backend_name):
        if vit_path and os.path.isfile(vit_path):
            dmv = tcim.runtime.DevManager(devices, backend_name)
            wmv = tcim.runtime.WeightManager(dmv)
            self.perf_tracker.perf_start(PERFTYPE.VISION_LOAD_TIME)
            self.vit = tcim.runtime.load(vit_path, option=tcim.runtime.Option(wmv))
            self.perf_tracker.perf_end(PERFTYPE.VISION_LOAD_TIME)
            self._log_model_io(self.vit, "vit")
            vit_in_shape = self.vit.get_input_info(self.vit.get_input_name(0)).shape
            vit_out_shape = self.vit.get_output_info(self.vit.get_output_name(0)).shape
            self.vit_num_patches = vit_in_shape[1]
            self.vit_num_tokens = vit_out_shape[1] if len(vit_out_shape) == 3 else vit_out_shape[0]
            self.vit_patch_dim = vit_in_shape[2]
            logger.info(f"Vision: patches={self.vit_num_patches}, tokens={self.vit_num_tokens}")
        else:
            self.vit = None
            logger.warning("Vision model not loaded, text-only mode")

    def _load_llm(self, prefill_path, decode_path, devices, backend_name):
        dm = tcim.runtime.DevManager(devices, backend_name)
        wm = tcim.runtime.WeightManager(dm)
        logger.info(f"Loading prefill model from {prefill_path}")
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self.prefill = tcim.runtime.load(prefill_path, option=tcim.runtime.Option(wm))
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)
        self._log_model_io(self.prefill, "prefill")
        # Subclass reads specific input indices for prefill_len, embed_dim, etc.
        self._read_prefill_info()

        # Decode (share KV caches with prefill)
        cache_names = [self.prefill.get_input_name(i) for i in range(self.prefill.get_num_inputs()) if "cache" in self.prefill.get_input_name(i).lower()]
        opt = tcim.runtime.Option(wm)
        opt.set_dummy_tensors(cache_names)
        logger.info(f"Loading decode model from {decode_path}")
        self.perf_tracker.perf_start(PERFTYPE.DECODE_LOAD_TIME)
        self.decode = tcim.runtime.load(decode_path, option=opt)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_LOAD_TIME)
        self._log_model_io(self.decode, "decode")
        self._read_decode_info()
        logger.info(f"Decode loaded: len={self.decode_len}")

        self.context_max_length = self.prefill.get_input_info(cache_names[-1]).shape[2]
        logger.info(f"Prefill loaded: len={self.prefill_len}, embed_dim={self.embed_dim}, context_max_length={self.context_max_length}")

        for name in cache_names:
            self.decode.set_input(name, self.prefill.get_dev_input(name))

    def _load_tokenizer(self, tokenizer_dir):
        tokenizer = GemmaTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        processor = XHGemma4Processor.from_pretrained(tokenizer_dir, trust_remote_code=True)
        return tokenizer, processor

    # Subclasses override to read model-specific input layout
    def _read_prefill_info(self):
        raise NotImplementedError

    def _read_decode_info(self):
        raise NotImplementedError

    @staticmethod
    def _aligned(size: int, align: int) -> int:
        return ((size + align - 1) // align) * align

    def _build_masks(self, cur_len, past_len, mask_len=None, mm_types=None):
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
        slide_ctx = global_ctx if sw is None else min(global_ctx, self._aligned(sw + q_len - 1, 16))
        clamped_past = min(past_len, sw - 1) if sw is not None and sw > 0 else past_len
        local_mask = torch.full((1, 1, q_len, slide_ctx), neg, dtype=torch.float16)
        for q in range(q_len):
            if q < cur_len:
                causal_end = min(slide_ctx, clamped_past + q + 1)
                sw_start = max(0, clamped_past + q - sw + 1) if sw is not None else 0
                local_mask[0, 0, q, sw_start:causal_end] = 0
            else:
                local_mask[0, 0, q, 0] = 0

        if mm_types is not None and mm_types.numel() > 0:
            mm = mm_types[0, :cur_len] if mm_types.dim() == 2 else mm_types[:cur_len]
            is_mm = (mm == 1) | (mm == 2)
            if self.audio_enabled:
                is_mm = is_mm | (mm == 3)
            cache_offset = max(0, past_len - clamped_past)
            group_start = None
            for idx in range(cur_len):
                if bool(is_mm[idx]) and group_start is None:
                    group_start = idx
                if group_start is not None and (idx == cur_len - 1 or not bool(is_mm[idx + 1])):
                    group_end = idx + 1
                    abs_start, abs_end = past_len + group_start, past_len + group_end
                    global_mask[0, 0, group_start:group_end, abs_start:abs_end] = 0
                    c_start = max(0, abs_start - cache_offset)
                    c_end = min(slide_ctx, abs_end - cache_offset)
                    if c_start < slide_ctx and c_end > 0:
                        local_mask[0, 0, group_start:group_end, c_start:c_end] = 0
                    group_start = None

        return global_mask.numpy(), local_mask.numpy()

    def _fit_model_input(self, model, value: torch.Tensor, name: str) -> np.ndarray:
        info = model.get_input_info(name)
        expected_shape = tuple(info.shape)
        arr = value.detach().cpu().numpy().astype(np.dtype(info.dtype))
        if arr.shape == expected_shape:
            return arr

        fitted = np.zeros(expected_shape, dtype=arr.dtype)
        slices = tuple(slice(0, min(src, dst)) for src, dst in zip(arr.shape, expected_shape))
        fitted[slices] = arr[slices]
        return fitted

    def _run_vision(self, inputs: dict) -> torch.Tensor:
        if self.vit is None:
            raise RuntimeError("Vision model not loaded")
        if inputs.get("pixel_values") is None:
            raise RuntimeError("pixel_values not found in processor inputs")

        vit_input_aliases = {"attention_mask": "visual_attention_mask"}

        self.perf_tracker.perf_start(PERFTYPE.VISION_INPUT_TIME)
        for i in range(self.vit.get_num_inputs()):
            name = self.vit.get_input_name(i)
            bare_name = name.removesuffix(".hmcc.format")
            input_key = bare_name if bare_name in inputs else vit_input_aliases.get(bare_name)
            if input_key not in inputs:
                raise KeyError(f"VIT input {name!r} is not found in processor inputs")
            value = inputs[input_key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            if bare_name == "pixel_values":
                value = value.half()
            self.vit.set_input(name, self._fit_model_input(self.vit, value, name))
        self.perf_tracker.perf_end(PERFTYPE.VISION_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.VISION_INFER_TIME)
        self.vit.run()
        self.vit.sync()
        self.perf_tracker.perf_end(PERFTYPE.VISION_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.VISION_OUTPUT_TIME)
        out = torch.from_numpy(self.vit.get_output(self.vit.get_output_name(0)).numpy())
        image_soft_token_count = inputs.get("image_soft_token_count")
        if image_soft_token_count is not None:
            counts = torch.as_tensor(image_soft_token_count, dtype=torch.long).flatten().tolist()
            if out.dim() == 3:
                out = torch.cat([out[i, : int(count), :] for i, count in enumerate(counts)], dim=0)
            elif out.dim() == 2 and len(counts) == 1:
                out = out[: int(counts[0]), :]
            else:
                out = out.squeeze(0)
        else:
            out = out.squeeze(0)
        self.perf_tracker.perf_end(PERFTYPE.VISION_OUTPUT_TIME)
        return out

    @staticmethod
    def _flatten_features(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            return features
        return features.reshape(-1, features.shape[-1])

    def _scatter_features(self, embeds: torch.Tensor, input_ids: torch.Tensor, token_id: int, features: torch.Tensor, feature_name: str) -> torch.Tensor:
        flat_features = self._flatten_features(features).to(embeds.device, embeds.dtype)
        token_positions = (input_ids == token_id).nonzero(as_tuple=False)
        if token_positions.numel() == 0:
            if flat_features.shape[0] != 0:
                raise ValueError(
                    f"Received {feature_name} features for token id {token_id}, but prompt does not contain that token."
                )
            return embeds
        if token_positions.shape[0] != flat_features.shape[0]:
            raise ValueError(
                f"{feature_name} features and token count do not match: "
                f"tokens={token_positions.shape[0]}, features={flat_features.shape[0]}"
            )

        updated = embeds.clone()
        updated[token_positions[:, 0], token_positions[:, 1]] = flat_features
        return updated

    def _decode_loop(self, first_token_id, input_ids, input_len):
        eos_ids = {self.tokenizer.eos_token_id} if isinstance(self.tokenizer.eos_token_id, int) else set(self.tokenizer.eos_token_id)
        eos_ids.add(106)
        logger.success("response:")
        print(f"\033[1;95m{self.tokenizer.decode(first_token_id[0])}", end="", flush=True)

        history = input_ids[0].tolist() + [first_token_id[0][0]]
        past_len = input_len
        step = 0
        slide = 10
        skip = 0
        last_resp = self.tokenizer.decode(history[-slide:])
        t0 = time.time()

        while past_len < self.context_max_length and step < self.max_new_tokens:
            tok_id = first_token_id[0][0]
            first_token_id = self._decode_step(tok_id, past_len)
            tok_id = first_token_id[0][0]
            if tok_id in eos_ids:
                break
            history.append(tok_id)
            resp = self.tokenizer.decode(history[-(slide + 1) - skip :])[len(last_resp) :]
            if resp and is_valid_char(ord(resp[-1])):
                print(resp, end="", flush=True)
                last_resp = self.tokenizer.decode(history[-slide:])
                skip = 0
            else:
                skip += 1
            past_len += 1
            step += 1

        print(f"\033[0m")
        logger.info(f"Decode: {step} tokens in {time.time() - t0:.2f}s")
        return step

    def _decode_step(self, tok_id, past_len):
        raise NotImplementedError

    def _build_embeddings(self, input_ids, inputs):
        raise NotImplementedError

    def _prefill(self, embeds, input_len, **kwargs):
        raise NotImplementedError

    def chat(self, question="", image_path=None, audio_path=None):
        raise NotImplementedError
