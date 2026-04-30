#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   Whisper ASR Inference Demo - Python script for running Whisper
# automatic speech recognition on HOUMO AI device.
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
import os
import time
import numpy as np
import argparse
import soundfile as sf
from typing import List, Optional
from loguru import logger

import torch
from transformers import WhisperProcessor
from datasets import load_dataset

import tcim_lite as tcim
from hmatc.python.get_hm_devices import get_hm_devices

HOUMO_TARGET = os.getenv("HOUMO_TARGET")


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
        if top_k <= 0:
            return probs

        top_k_indices = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.ones_like(probs, dtype=bool)
        mask[top_k_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = -np.inf
        return filtered_probs

    def apply_top_p(self, probs: np.ndarray) -> np.ndarray:
        if self.top_p >= 1.0:
            return probs

        sorted_indices = np.argsort(probs)[::-1]
        sorted_logits = probs[sorted_indices]
        max_logit = np.max(sorted_logits)
        sorted_probs = np.exp(sorted_logits - max_logit)
        sorted_probs /= np.sum(sorted_probs)
        cumulative_probs = np.cumsum(sorted_probs)

        cutoff_indices = np.where(cumulative_probs >= self.top_p)[0]
        if len(cutoff_indices) > 0:
            cutoff_index = cutoff_indices[0]
            cutoff_index = max(cutoff_index, self.min_tokens_to_keep - 1)
            selected_indices = sorted_indices[: cutoff_index + 1]
        else:
            selected_indices = sorted_indices

        mask = np.ones_like(probs, dtype=bool)
        mask[selected_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = -np.inf
        return filtered_probs

    def process_logits(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        processed_logits = logits.copy()
        processed_logits = self.apply_repetition_penalty(
            processed_logits, previous_tokens
        )
        processed_logits = self.apply_top_k(processed_logits)
        processed_logits = self.apply_top_p(processed_logits)
        processed_logits = self.apply_temperature(processed_logits)
        return processed_logits

    def sample(
        self, logits: torch.Tensor, previous_tokens: Optional[List[int]] = None
    ) -> torch.Tensor:
        logits_np = logits[0].detach().cpu().numpy()
        processed_logits = self.process_logits(logits_np, previous_tokens)
        sampled_index = int(np.argmax(processed_logits, axis=-1))
        return torch.tensor([sampled_index], device=logits.device)


def is_valid_char(cp):
    return cp != 0xFFFD and cp > 0x001F


lang_to_id = [
    50327,
    50334,
    50272,
    50350,
    50304,
    50355,
    50330,
    50292,
    50302,
    50347,
    50309,
    50315,
    50270,
    50283,
    50297,
    50285,
    50261,
    50281,
    50259,
    50262,
    50307,
    50310,
    50300,
    50277,
    50338,
    50265,
    50319,
    50333,
    50352,
    50354,
    50279,
    50276,
    50291,
    50339,
    50286,
    50312,
    50275,
    50311,
    50274,
    50266,
    50356,
    50329,
    50316,
    50323,
    50306,
    50264,
    50294,
    50345,
    50353,
    50336,
    50293,
    50301,
    50349,
    50295,
    50308,
    50296,
    50314,
    50320,
    50282,
    50343,
    50346,
    50313,
    50271,
    50342,
    50288,
    50328,
    50321,
    50269,
    50340,
    50267,
    50284,
    50263,
    50344,
    50332,
    50322,
    50298,
    50305,
    50324,
    50326,
    50317,
    50303,
    50357,
    50273,
    50318,
    50287,
    50299,
    50331,
    50289,
    50341,
    50348,
    50268,
    50351,
    50280,
    50290,
    50337,
    50278,
    50335,
    50325,
    50260,
]


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processor_dir",
        dest="processor_dir",
        type=str,
        default="whisper-medium",
        help="processor dir",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="../../../data/audio/audio.mp3",
    )
    parser.add_argument(
        "--encoder_path",
        dest="encoder_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "whisper_encode.hmm"),
        help="houmo encoder model path",
    )
    parser.add_argument(
        "--decoder_path",
        dest="decoder_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "whisper_decode.hmm"),
        help="houmo decoder model path",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "whisper_prefill.hmm"),
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--chunk_size",
        dest="chunk_size",
        type=int,
        default=30,
        help="chunk size of audio",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="language code (e.g. 'zh', 'en') or 'auto' for language detection",
    )
    args = parser.parse_args()
    return args


class HmWhisper:
    def __init__(self, encoder_path, decoder_path, prefill_path):
        super().__init__()
        dev_manager = tcim.runtime.DevManager(get_hm_devices(), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option1 = tcim.runtime.Option(weight_manager)
        self.encoder = tcim.runtime.load(encoder_path, option=option1)
        logger.info("encoder model loaded")
        option2 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option2)
        logger.info("prefill model loaded")
        option3 = tcim.runtime.Option(weight_manager)
        self.decoder = tcim.runtime.load(decoder_path, option=option3)
        logger.info("decoder model loaded")

    def run_encoder(self, input_features):
        input_features = input_features.numpy()
        self.encoder.set_input(self.encoder.get_input_name(0), input_features)
        self.encoder.run()
        self.encoder.sync()
        outputs = []
        for i in range(self.encoder.get_num_outputs()):
            outputs.append(
                torch.tensor(
                    self.encoder.get_output(self.encoder.get_output_name(i)).numpy()
                )
            )
        return outputs

    def run_decoder(self, inputs):
        for input_name, input_data in inputs.items():
            self.decoder.set_input(input_name, input_data.numpy())
        self.decoder.run()
        self.decoder.sync()
        outputs = torch.tensor(
            self.decoder.get_output(self.decoder.get_output_name(0)).numpy()
        )
        return outputs

    def run_prefill(
        self,
        inputs,
    ):
        for input_name, input_data in inputs.items():
            self.prefill.set_input(input_name, input_data.numpy())
        self.prefill.run()
        self.prefill.sync()
        outputs = torch.tensor(
            self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        )
        return outputs

    def get_input_names(self, model_type):
        if model_type == "encoder":
            self.model = self.encoder
        elif model_type == "decoder":
            self.model = self.decoder
        elif model_type == "prefill":
            self.model = self.prefill
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        return input_names


def asr(
    hmwhisper,
    processor,
    input_features,
    sampling_manager,
    state=None,
    slide_len=10,
    language="auto",
):
    """
    Args:
        state: Dictionary containing cross-chunk continuous state:
            - last_response: Last decoded text
            - skip_tokens: Number of tokens to skip
    Returns:
        (prefill_ids_len, all_ids_len, ttft_time, total_time, audio_duration, state)
    """
    sot_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    lang_id = processor.tokenizer.convert_tokens_to_ids("<|zh|>")
    transcribe_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notime_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    eos_id = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if state is None:
        # First call, initialize state
        skip_tokens = 0
        last_response = ""
    else:
        # Use the passed state (only maintain the continuity of text decoding)
        skip_tokens = state["skip_tokens"]
        last_response = state["last_response"]

    start_time = time.time()
    detect_ids = torch.tensor([[sot_id]])  # [1,1]
    default_decoder_ids = torch.tensor([[sot_id, 0, transcribe_id, notime_id]])  # [1,1]
    cache_position = torch.tensor([[0]])
    cache_position_prefill = torch.tensor([[0, 1, 2, 3]])
    cnt = 3

    # detect language  input_features  detect_ids => [1,51865]
    detect_encoder_out = hmwhisper.run_encoder(input_features.half())
    mask_atten = torch.zeros(([1, 16, 1, 1024]), dtype=torch.float16)
    mask_atten[:, :, :, 1:] = -65504.0

    enc_seq_len = detect_encoder_out[0].shape[2]
    encoder_attention_mask = torch.zeros(
        (1, 1, 1, enc_seq_len), device="cpu", dtype=torch.float16
    )

    decoder_input_names = hmwhisper.get_input_names("decoder")
    decoder_detext_inputs = {
        decoder_input_names[0]: detect_ids.to(torch.int32),
        decoder_input_names[1]: cache_position.to(torch.int32),
        decoder_input_names[2]: torch.tensor([0]).to(torch.int32),
        decoder_input_names[3]: torch.tensor([1]).to(torch.int32),
        decoder_input_names[4]: mask_atten,
        decoder_input_names[5]: encoder_attention_mask.to(torch.float16),
    }

    k_cache = [
        torch.ones([1, 16, 1024, 64], dtype=torch.float16) * (-65504) for i in range(24)
    ]
    v_cache = [
        torch.ones([1, 16, 1024, 64], dtype=torch.float16) * (-65504) for i in range(24)
    ]

    for data_detect, k_data_cache in zip(decoder_input_names[6:30], k_cache):
        decoder_detext_inputs[data_detect] = k_data_cache
    for data_detect, v_data_cache in zip(decoder_input_names[30:54], v_cache):
        decoder_detext_inputs[data_detect] = v_data_cache

    k_list = []
    for i in range(24):
        k_list.append(detect_encoder_out[2 * i])

    v_list = []
    for i in range(24):
        v_list.append(detect_encoder_out[2 * i + 1])

    for data_detect, k_data in zip(decoder_input_names[54:78], k_list):
        decoder_detext_inputs[data_detect] = k_data

    for data_detect, v_data in zip(decoder_input_names[78:102], v_list):
        decoder_detext_inputs[data_detect] = v_data

    logits = hmwhisper.run_decoder(decoder_detext_inputs)

    if language == "auto" or language is None:
        non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
        non_lang_mask[0, list(lang_to_id)] = False
        logits[:, :, non_lang_mask[0]] = -np.inf
        lang_ids = logits.argmax(-1)

        non_lang_mask = torch.ones_like(logits, dtype=torch.bool)
        non_lang_mask[0, 0, list(lang_to_id)] = False
        logits[:, :, non_lang_mask[0][0]] = -np.inf
        lang_ids = logits.argmax(-1)
        print(f"Detected language id: {lang_ids.item()}")
    else:
        # Convert language string (e.g. 'zh') to token id
        token_str = f"<|{language}|>"
        lang_token_id = processor.tokenizer.convert_tokens_to_ids(token_str)
        if lang_token_id is None or lang_token_id == processor.tokenizer.unk_token_id:
            logger.warning(
                f"Unknown language code '{language}', falling back to auto detection."
            )
            lang_ids = logits.argmax(-1)
            print(f"Detected language id: {lang_ids.item()}")
        else:
            lang_ids = torch.tensor([[lang_token_id]])
            print(f"Forced language id: {lang_token_id} for '{language}'")

    default_decoder_ids[0, 1] = lang_ids  # [[50258, 50259, 50359, 50363]] # 34.5197

    mask_atten = torch.full((1, 16, 4, 1024), -65504.0, dtype=torch.float16)
    for i in range(4):
        mask_atten[:, :, i, : i + 1] = 0.0

    prefill_input_names = hmwhisper.get_input_names("prefill")
    prefill_inputs = {
        prefill_input_names[0]: default_decoder_ids.to(torch.int32),
        prefill_input_names[1]: cache_position_prefill.to(torch.int32),
        prefill_input_names[2]: torch.tensor([0]).to(torch.int32),
        prefill_input_names[3]: torch.tensor([4]).to(torch.int32),
        prefill_input_names[4]: mask_atten,
        prefill_input_names[5]: encoder_attention_mask.to(torch.float16),
    }

    for i in range(96):
        cache = hmwhisper.decoder.get_dev_input(hmwhisper.decoder.get_input_name(i + 6))
        hmwhisper.prefill.set_dev_input(hmwhisper.prefill.get_input_name(i + 6), cache)
    logits = hmwhisper.run_prefill(prefill_inputs)

    next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32)

    # Fix: Initialize last_response (containing only prompt tokens at this point) before appending new tokens
    # This prepares it for sliding window slicing
    last_response = processor.decode(default_decoder_ids[0][-slide_len:])

    next_tokens = sampling_manager.sample(
        next_token_logits, default_decoder_ids[0].tolist()
    )
    default_decoder_ids = torch.cat([default_decoder_ids, next_tokens[:, None]], dim=-1)

    prefill_ids_len = default_decoder_ids.shape[1]
    ttft_time = time.time() - start_time

    # Preset output color
    print("\033[1;95m", end="", flush=True)

    # Ensure the first token correctly applies the UTF-8 truncation/sliding check
    decode_response = processor.decode(
        default_decoder_ids[0][-(slide_len + 1) - skip_tokens :]
    )[len(last_response) :]

    if (
        decode_response != ""
        and is_valid_char(ord(decode_response[-1]))
        and next_tokens.item() != eos_id
    ):
        print(decode_response, end="", flush=True)
        last_response = processor.decode(default_decoder_ids[0][-slide_len:])
        skip_tokens = 0
    else:
        skip_tokens += 1
    decode_response = "" if next_tokens.item() == eos_id else decode_response

    # Restart the timer, only measure the decoding phase
    start_time = time.time()

    for i in range(48):
        cache = hmwhisper.prefill.get_dev_output(
            hmwhisper.prefill.get_output_name(i + 1)
        )
        hmwhisper.decoder.set_dev_input(hmwhisper.decoder.get_input_name(i + 6), cache)
        hmwhisper.decoder.set_dev_output(
            hmwhisper.decoder.get_output_name(i + 1), cache
        )

    while default_decoder_ids.shape[1] < 448 and next_tokens.item() != eos_id:
        cnt += 1
        mask_atten = torch.zeros(([1, 16, 1, 1024]), dtype=torch.float16)
        if cnt + 1 < 1024:
            mask_atten[:, :, :, cnt + 1 :] = -65504.0

        prefill_inputs[prefill_input_names[0]] = next_tokens.unsqueeze(0).to(
            torch.int32
        )
        prefill_inputs[prefill_input_names[1]] = torch.tensor([[cnt]]).to(torch.int32)
        prefill_inputs[prefill_input_names[2]] = torch.tensor([cnt]).to(torch.int32)
        prefill_inputs[prefill_input_names[3]] = torch.tensor([1]).to(torch.int32)
        prefill_inputs[prefill_input_names[4]] = mask_atten
        prefill_inputs[prefill_input_names[5]] = encoder_attention_mask.to(
            torch.float16
        )

        logits = hmwhisper.run_decoder(prefill_inputs)
        next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32)
        next_tokens = sampling_manager.sample(
            next_token_logits, default_decoder_ids[0].tolist()
        )
        default_decoder_ids = torch.cat(
            [default_decoder_ids, next_tokens[:, None]], dim=-1
        )

        decode_response = processor.decode(
            default_decoder_ids[0][-(slide_len + 1) - skip_tokens :]
        )[len(last_response) :]

        if (
            decode_response != ""
            and is_valid_char(ord(decode_response[-1]))
            and next_tokens.item() != eos_id
        ):
            print(decode_response, end="", flush=True)
            last_response = processor.decode(default_decoder_ids[0][-slide_len:])
            skip_tokens = 0
        else:
            skip_tokens += 1
        decode_response = "" if next_tokens.item() == eos_id else decode_response

    # Save state for the next chunk (only maintain the continuity of text decoding)
    state = {
        "last_response": last_response,
        "skip_tokens": skip_tokens,
    }
    return (
        prefill_ids_len,
        default_decoder_ids.shape[1],
        ttft_time,
        (time.time() - start_time),
        len(input_features[0][0]) * 0.02,  # audio duration in seconds, 20ms per frame
        state,
    )


if __name__ == "__main__":
    args = get_args()

    # init houmo whisper model
    if HOUMO_TARGET == "xh2":
        hmwhisper = HmWhisper(
            args.encoder_path,
            args.decoder_path,
            args.prefill_path,
        )
    else:
        raise ValueError("Unsupport houmo target!")

    processor = WhisperProcessor.from_pretrained(args.processor_dir)
    sampling_manager = SamplingManager(top_k=None, top_p=1.0, repetition_penalty=1.1)

    sample, sr = sf.read(args.audio)
    if sr != 16000:
        import librosa

        sample = librosa.resample(sample, orig_sr=sr, target_sr=16000)
        logger.info(f"Resampled audio from {sr}Hz to 16000Hz")
        sr = 16000
    total_samples = len(sample)
    chunk_size = int(args.chunk_size * sr)
    total_samples = len(sample)
    chunks = []

    for start in range(0, total_samples, chunk_size):
        end = start + chunk_size
        chunk = sample[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")
        chunks.append(chunk)
    chunks_array = np.array(chunks)

    logger.success("transcription:")
    total_ttft_time = 0.0
    total_decode_time = 0.0
    total_tokens = 0
    total_prefill_tokens = 0
    total_audio_duration = 0.0
    state = None  # initial state is None, first chunk will initialize it
    for i, chunk in enumerate(chunks):
        input_features = processor(
            chunk, sampling_rate=sr, return_tensors="pt"
        ).input_features
        prefill_ids_len, all_ids_len, ttft_time, total_time, audio_duration, state = (
            asr(
                hmwhisper,
                processor,
                input_features,
                sampling_manager,
                state,
                language=args.language,
            )
        )
        total_ttft_time += ttft_time
        total_decode_time += total_time
        total_tokens += all_ids_len
        total_prefill_tokens += prefill_ids_len
        total_audio_duration += audio_duration

    # reset color and print newline
    print("\033[0m")

    # run whisper model

    e2e_time = total_decode_time + total_ttft_time  # end-to-end total time

    logger.success(
        f"Output {total_tokens} tokens, Decode Cost {total_decode_time*1000:.3f} ms"
    )
    logger.success(
        f"Decode Speed: {(total_tokens - total_prefill_tokens) / total_decode_time:.2f} tokens/s"
    )
    logger.success(f"TTFT (Time to First Token): {total_ttft_time * 1000:.3f} ms")
    logger.success(
        f"TPOT (Time Per Output Token): {total_decode_time * 1000 / (total_tokens - total_prefill_tokens):.3f} ms/token"
    )
    logger.success(f"E2E Latency (End-to-End Latency): {e2e_time:.3f} seconds")
    logger.success(
        f"E2E TPS (End-to-End Tokens Per Second): {total_tokens / e2e_time:.2f} tokens/s"
    )
    logger.success(
        f"RTF (Real-Time Factor): {e2e_time / total_audio_duration:.4f} (lower is better, <1 means real-time)"
    )
