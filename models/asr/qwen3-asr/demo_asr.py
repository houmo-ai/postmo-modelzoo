#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo_asr.py
# Description:
#   Optimized Qwen3 ASR Inference Demo - Python script for running qwen3-asr
# model on HOUMO AI device.
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
import os
import re
import time
import argparse
import numpy as np
import librosa
from typing import List, Optional
from loguru import logger
import tcim_lite as tcim
import torch
from transformers import AutoConfig
from qwen_asr.core.transformers_backend import (
    Qwen3ASRProcessor,
)
from hmatc.python.get_hm_devices import get_hm_devices
from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
ASR_TEXT_PATTERN = re.compile(r"(?<=<asr_text>)[\s\S]*")
# Common punctuation characters to preserve in output
PUNCTUATION_CHARS = set("，。！？；：" "''（）【】《》、·…—" ",.!?;:\"'()[]<>-" " ")


def get_default_processor_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "qwen3-asr")
    model_size = model_config.get("model_size", "0.6b")
    return f"{model_name}-{model_size}"


def is_valid_char(cp):
    """Check if a code point is a valid Chinese, English character or punctuation."""
    # Chinese characters
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x20000 and cp <= 0x2A6DF)
        or (cp >= 0x2A700 and cp <= 0x2B73F)
        or (cp >= 0x2B740 and cp <= 0x2B81F)
        or (cp >= 0x2B820 and cp <= 0x2CEAF)
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
    ):
        return True
    # English letters
    if (0x0041 <= cp and cp <= 0x005A) or (0x0061 <= cp and cp <= 0x007A):
        return True
    # Numbers
    if 0x0030 <= cp and cp <= 0x0039:
        return True
    # Common punctuation (both Chinese and English)
    if chr(cp) in PUNCTUATION_CHARS:
        return True
    return False


def filter_valid_chars(text: str) -> str:
    """Filter text to keep only valid Chinese, English characters and common punctuation."""
    return "".join(c for c in text if is_valid_char(ord(c)))


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

    def process_logits(self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None) -> np.ndarray:
        processed_logits = logits.astype(np.float32, copy=True)
        processed_logits = self.apply_repetition_penalty(processed_logits, previous_tokens)
        probs = self.apply_top_k(processed_logits)
        probs = self.apply_top_p(probs)
        probs = self.apply_temperature(probs)
        return probs

    def sample(self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None) -> int:
        probs = np.asarray(logits)
        while probs.ndim > 1:
            probs = probs[0]

        probs = self.process_logits(probs, previous_tokens)
        if np.all(probs == 0):
            probs = np.ones_like(probs) / len(probs)

        return int(np.argmax(probs))


class StreamingOutput:
    """Manages sliding window streaming output for token decoding.

    The sliding window approach ensures stable text output by keeping
    the last few tokens in a buffer before emitting, avoiding partial
    or corrupted token decoding results. Only valid Chinese, English
    characters and common punctuation are emitted.
    """

    def __init__(self, tokenizer, slide_len: int = 10):
        self.tokenizer = tokenizer
        self.slide_len = slide_len
        self.emitted_response = ""

    def emit_stable(self, generated_ids: list) -> bool:
        """
        Emit stable text from generated tokens using sliding window.

        Args:
            generated_ids: List of generated token IDs

        Returns:
            True if content was emitted, False otherwise
        """
        if len(generated_ids) <= self.slide_len:
            return False

        stable_ids = generated_ids[: -self.slide_len]
        stable_response = self.tokenizer.decode(stable_ids, skip_special_tokens=True)

        match = ASR_TEXT_PATTERN.search(stable_response)
        if not match:
            return False

        stable_response = match.group()
        # Filter to keep only valid characters
        stable_response = filter_valid_chars(stable_response)
        new_content = stable_response[len(self.emitted_response) :]

        if new_content:
            print(new_content, end="", flush=True)
            self.emitted_response = stable_response
            return True

        return False

    def emit_final(self, generated_ids: list) -> str:
        """
        Emit remaining tokens after decode loop completes.

        Args:
            generated_ids: Full list of generated token IDs

        Returns:
            The final decoded result text
        """
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        match = ASR_TEXT_PATTERN.search(result)

        if match:
            final_text = match.group()
            # Filter to keep only valid characters
            final_text = filter_valid_chars(final_text)
            new_content = final_text[len(self.emitted_response) :]
            if new_content:
                print(new_content, end="", flush=True)
            return final_text

        return result


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
        "--processor_dir",
        dest="processor_dir",
        type=str,
        default=None,
        help="processor dir",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="../../../data/audio/audio.mp3",
    )
    parser.add_argument(
        "--encode_path",
        dest="encode_path",
        type=str,
        default=None,
        help="houmo encode model path",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default=None,
        help="houmo decode model path",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=None,
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"),
        help="houmo embedding weight path",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number, only xh2 support",
    )
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})

    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    if args.processor_dir is None:
        args.processor_dir = get_default_processor_dir(model_config)
    model_prefix = f"{args.model_name}-{args.model_size}"
    if args.encode_path is None:
        args.encode_path = os.path.join("output", HOUMO_TARGET, f"{model_prefix}_encode.hmm")
    if args.decode_path is None:
        args.decode_path = os.path.join("output", HOUMO_TARGET, f"{model_prefix}_decode.hmm")
    if args.prefill_path is None:
        args.prefill_path = os.path.join("output", HOUMO_TARGET, f"{model_prefix}_prefill.hmm")
    if args.ndevice > 1:
        if args.decode_path.endswith(".hmm"):
            args.decode_path = args.decode_path.replace(".hmm", ".hmms")
        if args.prefill_path.endswith(".hmm"):
            args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
    return args


class Qwen3Asr:
    def __init__(
        self,
        encode_path,
        decode_path,
        prefill_path,
        processor_dir,
        embedding_path,
        ndevice=1,
    ):
        self.device = torch.device("cpu")
        self.ndevice = ndevice
        dev_manager = tcim.runtime.DevManager(get_hm_devices(self.ndevice), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option1 = tcim.runtime.Option(weight_manager)
        self.encode = tcim.runtime.load(encode_path, option=option1)
        logger.info("encode model loaded")
        option2 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option2)
        logger.info("prefill model loaded")

        option3 = tcim.runtime.Option(weight_manager)
        self.nblocks = self.get_nblocks()
        dummy_tensor_names = [f"model_layers_{i}_self_attn_kcache_input" for i in range(self.nblocks)]
        dummy_tensor_names += [f"model_layers_{i}_self_attn_vcache_input" for i in range(self.nblocks)]
        option3.set_dummy_tensors(dummy_tensor_names)
        self.decode = tcim.runtime.load(decode_path, option=option3)
        logger.info("decode model loaded")

        self.max_new_tokens = self.prefill.get_input_info(self.prefill.get_input_name(3)).shape[2]
        self.max_prefill = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[1]

        self.processor = Qwen3ASRProcessor.from_pretrained(processor_dir, fix_mistral_regex=True)
        self.tokenizer = self.processor.tokenizer
        self.config = AutoConfig.from_pretrained(processor_dir, trust_remote_code=True)
        self.hidden_size = self.config.thinker_config.text_config.hidden_size

        if "<|audio_pad|>" in self.tokenizer.get_vocab():
            self.audio_pad_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        else:
            self.audio_pad_id = self.tokenizer.encode("<|audio_pad|>", add_special_tokens=False)[0]

        embedding_tensor = torch.load(embedding_path, map_location=self.device)
        if HOUMO_TARGET == "xh2":
            embedding_tensor = embedding_tensor["weight"]
        if embedding_tensor.dtype != torch.float16:
            embedding_tensor = embedding_tensor.to(torch.float16)
        # Store as numpy for direct use in inference
        self.embedding_weight = np.ascontiguousarray(embedding_tensor.cpu().numpy())

        # Reuse hot-path buffers to reduce per-loop allocations.
        self.prefill_embeds = np.zeros((1, self.max_prefill, self.hidden_size), dtype=np.float16)
        self.prefill_valid_length_np = np.array([0], dtype=np.int32)
        self.prefill_current_length_np = np.array([0], dtype=np.int32)
        self.decode_valid_length_np = np.array([0], dtype=np.int32)
        self.decode_current_length_np = np.array([1], dtype=np.int32)

        # Link KV caches between prefill and decode
        for i in range(3, 2 * self.nblocks + 3):
            cache = self.prefill.get_dev_input(self.prefill.get_input_name(i))
            self.decode.set_dev_input(self.decode.get_input_name(i), cache)

        self.all_features_len = 0
        self.max_feature_one_loop = self.encode.get_input_info(self.encode.get_input_name(0)).shape[2]
        self.loop_count = 0
        self.sampling_manager = SamplingManager(top_k=None, top_p=1.0, repetition_penalty=1.0)

    def run_encode(self, inputs):
        encode_prep_start = time.perf_counter()
        # Convert input features to float32 numpy array
        input_features = inputs["input_features"].float().numpy()
        # Calculate feature lengths and convert to int
        origin_feature_lens = input_features.shape[2]

        pad_width = (0, self.max_feature_one_loop - origin_feature_lens)
        input_features = np.pad(
            input_features,
            ((0, 0), (0, 0), pad_width),
            mode="constant",
            constant_values=0.0,
        )

        encode_prep_time = time.perf_counter() - encode_prep_start

        encode_infer_start = time.perf_counter()
        self.encode.set_input(self.encode.get_input_name(0), input_features)
        self.encode.set_input(
            self.encode.get_input_name(1),
            np.array([origin_feature_lens], dtype=np.int32),
        )

        self.encode.run()
        self.encode.sync()
        encode_infer_time = time.perf_counter() - encode_infer_start

        encode_output_start = time.perf_counter()
        outputs = self.encode.get_output(self.encode.get_output_name(0))
        # Ensure outputs is numpy array
        if hasattr(outputs, "numpy"):
            outputs = outputs.numpy()
        encode_output_time = time.perf_counter() - encode_output_start

        return (
            outputs,
            origin_feature_lens,
            encode_prep_time,
            encode_infer_time,
            encode_output_time,
        )

    def run_decode(self, next_token_id, L):
        """Run decode loop with streaming output."""
        # Use numpy arrays directly to avoid repeated conversions
        self.decode_valid_length_np[0] = L
        valid_length_np = self.decode_valid_length_np
        current_length_np = self.decode_current_length_np

        generated_ids = [next_token_id]
        eos_token_id = self.tokenizer.eos_token_id

        decode_start = time.perf_counter()
        tokens_generated = 0
        total_decode_time = 0

        # Use StreamingOutput for cleaner streaming logic
        streaming = StreamingOutput(self.tokenizer, slide_len=10)

        for _ in range(self.max_new_tokens):
            # Direct numpy embedding lookup - avoid torch conversion
            token_id = generated_ids[-1]
            next_embed = self.embedding_weight[token_id : token_id + 1, :]
            next_embed = next_embed.reshape(1, 1, -1)

            self.decode.set_input(self.decode.get_input_name(0), next_embed)
            self.decode.set_input(self.decode.get_input_name(1), valid_length_np)
            self.decode.set_input(self.decode.get_input_name(2), current_length_np)

            t0 = time.perf_counter()
            self.decode.run()
            self.decode.sync()
            total_decode_time += time.perf_counter() - t0

            # Process output directly with numpy
            decode_outputs = self.decode.get_output(self.decode.get_output_name(0))
            if hasattr(decode_outputs, "numpy"):
                decode_outputs = decode_outputs.numpy()
            next_id = self.sampling_manager.sample(decode_outputs, generated_ids)
            generated_ids.append(next_id)

            # Update lengths
            valid_length_np[0] += 1
            tokens_generated += 1

            # Emit stable text using sliding window
            streaming.emit_stable(generated_ids)

            # Stop on EOS
            if next_id == eos_token_id:
                break

        decode_time = time.perf_counter() - decode_start

        # Emit remaining tokens
        result = streaming.emit_final(generated_ids)

        return result, tokens_generated, decode_time, total_decode_time

    def run_prefill(self, origin_feature_lens, inputs, audio_embeds):
        prefill_prep_start = time.perf_counter()
        T_out = self._get_feat_extract_output_lengths(origin_feature_lens)
        audio_embeds = audio_embeds[:, :T_out, :]

        if audio_embeds.ndim == 2:
            audio_embeds = audio_embeds[np.newaxis, :, :]
        if audio_embeds.dtype != np.float16:
            audio_embeds = audio_embeds.astype(np.float16, copy=False)

        text_input_ids = inputs["input_ids"].cpu().numpy()
        # Direct numpy embedding lookup
        text_embeds = self.embedding_weight[text_input_ids]

        prefill_embeds = self.prefill_embeds
        prefill_embeds.fill(0)
        pad_indices = np.where(text_input_ids == self.audio_pad_id)[1]

        if len(pad_indices) > 0:
            start_idx = int(pad_indices[0])
            end_idx = int(pad_indices[-1])

            cursor = 0
            left_len = min(start_idx, self.max_prefill)
            if left_len > 0:
                prefill_embeds[:, :left_len, :] = text_embeds[:, :left_len, :]
                cursor += left_len

            remain = self.max_prefill - cursor
            if remain > 0:
                audio_len = min(audio_embeds.shape[1], remain)
                if audio_len > 0:
                    prefill_embeds[:, cursor : cursor + audio_len, :] = audio_embeds[:, :audio_len, :]
                    cursor += audio_len

            remain = self.max_prefill - cursor
            if remain > 0:
                tail_start = end_idx + 1
                tail_avail = max(text_embeds.shape[1] - tail_start, 0)
                tail_len = min(tail_avail, remain)
                if tail_len > 0:
                    prefill_embeds[:, cursor : cursor + tail_len, :] = text_embeds[
                        :, tail_start : tail_start + tail_len, :
                    ]
                    cursor += tail_len
            L = cursor
        else:
            L = min(text_embeds.shape[1], self.max_prefill)
            if L > 0:
                prefill_embeds[:, :L, :] = text_embeds[:, :L, :]

        self.prefill_current_length_np[0] = L

        self.prefill.set_input(self.prefill.get_input_name(0), prefill_embeds)
        self.prefill.set_input(self.prefill.get_input_name(1), self.prefill_valid_length_np)
        self.prefill.set_input(self.prefill.get_input_name(2), self.prefill_current_length_np)
        prefill_prep_time = time.perf_counter() - prefill_prep_start

        prefill_infer_start = time.perf_counter()
        self.prefill.run()
        self.prefill.sync()
        prefill_infer_time = time.perf_counter() - prefill_infer_start

        prefill_output_start = time.perf_counter()
        last_hidden_state = self.prefill.get_output(self.prefill.get_output_name(0))
        if hasattr(last_hidden_state, "numpy"):
            last_hidden_state = last_hidden_state.numpy()
        prefill_output_time = time.perf_counter() - prefill_output_start

        return (
            last_hidden_state,
            L,
            prefill_prep_time,
            prefill_infer_time,
            prefill_output_time,
        )

    def get_nblocks(self):
        """Calculate number of transformer blocks from input tensor names."""
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return int(output_lengths)

    def run(self, audio_path):
        # Load audio
        import torchaudio

        total_start = time.perf_counter()
        audio_load_start = time.perf_counter()
        if os.path.exists(audio_path):
            waveform, orig_sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            if orig_sr != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=16000)
            audio = waveform.numpy()
            sr = 16000
        else:
            logger.error(f"Audio file {audio_path} does not exist.")
            return
        audio_duration = len(audio) / sr  # audio duration in seconds
        audio_load_time = time.perf_counter() - audio_load_start
        # Prepare inputs
        prep_start = time.perf_counter()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "audio", "audio": "placeholder"}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        all_inputs = self.processor(text=prompt, audio=audio, return_tensors="pt", padding=True)
        all_inputs = all_inputs.to(self.device)

        self.all_features_lens = all_inputs["input_features"].shape[2]
        self.loop_count = (self.all_features_lens + self.max_feature_one_loop - 1) // self.max_feature_one_loop

        prep_time = time.perf_counter() - prep_start

        # Accumulate performance statistics
        total_encode_prep_time = 0.0
        total_encode_infer_time = 0.0
        total_encode_output_time = 0.0
        total_prefill_prep_time = 0.0
        total_prefill_infer_time = 0.0
        total_prefill_output_time = 0.0
        total_decode_time = 0.0
        total_decode_infer_time = 0.0
        total_tokens_generated = 0
        ttft_time = 0.0

        # Run encode
        for loop_idx in range(self.loop_count):
            start_idx = loop_idx * self.max_feature_one_loop
            end_idx = min(start_idx + self.max_feature_one_loop, self.all_features_lens)
            inputs_features = all_inputs["input_features"][:, :, start_idx:end_idx]
            inputs = {
                "input_features": inputs_features,
                "input_ids": all_inputs["input_ids"],
            }
            (
                outputs,
                origin_feature_lens,
                encode_prep_time,
                encode_infer_time,
                encode_output_time,
            ) = self.run_encode(inputs)
            audio_embeds = outputs
            (
                last_hidden_state,
                L,
                prefill_prep_time,
                prefill_infer_time,
                prefill_output_time,
            ) = self.run_prefill(origin_feature_lens, inputs, audio_embeds)

            next_token_id = self.sampling_manager.sample(last_hidden_state)
            if loop_idx == 0:
                ttft_time = time.perf_counter() - total_start

            (
                result,
                tokens_generated,
                decode_time,
                decode_infer_time,
            ) = self.run_decode(next_token_id, L)

            # Accumulate timing
            total_encode_prep_time += encode_prep_time
            total_encode_infer_time += encode_infer_time
            total_encode_output_time += encode_output_time
            total_prefill_prep_time += prefill_prep_time
            total_prefill_infer_time += prefill_infer_time
            total_prefill_output_time += prefill_output_time
            total_decode_time += decode_time
            total_decode_infer_time += decode_infer_time
            total_tokens_generated += tokens_generated

        # Print newline after streaming output
        print()

        # Calculate total time
        total_time = time.perf_counter() - total_start
        infer_time = total_time  # Total inference time in seconds
        # Calculate RTF (Real Time Factor)
        rtf = infer_time / audio_duration

        # Calculate average decode time per token
        avg_decode_time = total_decode_infer_time / total_tokens_generated if total_tokens_generated > 0 else 0

        # Performance statistics
        logger.success("=" * 60)
        logger.success("Performance Statistics:")
        logger.success("=" * 60)
        logger.success(f"Audio duration: {audio_duration * 1000:.2f} ms ({audio_duration:.2f} s)")
        logger.success(f"Audio loading time: {audio_load_time * 1000:.3f} ms")
        logger.success(f"Input preparation time: {prep_time * 1000:.3f} ms")
        logger.success(f"Loop count: {self.loop_count}")
        logger.success("-" * 60)
        logger.success(f"Total Encode time: {total_encode_infer_time * 1000:.3f} ms")
        logger.success(f"  - Encode infer time: {total_encode_infer_time * 1000:.3f} ms")
        logger.success("-" * 60)
        logger.success(f"Total Prefill time: {total_prefill_infer_time * 1000:.3f} ms")
        logger.success(f"  - Prefill infer time: {total_prefill_infer_time * 1000:.3f} ms")
        logger.success("-" * 60)
        logger.success(f"Output {total_tokens_generated} tokens")
        logger.success(f"Total Decode time: {total_decode_infer_time * 1000:.3f} ms")
        if total_decode_infer_time > 0:
            logger.success(f"Decode Speed: {total_tokens_generated / total_decode_infer_time:.2f} tokens/s")
        logger.success("-" * 60)
        logger.success(f"TTFT (Time to First Token): {ttft_time * 1000:.3f} ms")
        logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
        if total_time > 0:
            logger.success(
                f"E2E TPS (End-to-End Tokens Per Second): {total_tokens_generated / total_time:.2f} tokens/s"
            )
        logger.success(f"RTF (Real Time Factor): {rtf:.2f}")
        logger.success("=" * 60)

        return result


if __name__ == "__main__":
    args = get_args()

    qwen3asr = Qwen3Asr(
        args.encode_path,
        args.decode_path,
        args.prefill_path,
        args.processor_dir,
        args.embedding_path,
        args.ndevice,
    )
    qwen3asr.run(args.audio)
