#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Whisper-Turbo ASR Inference Demo - Python script for running Whisper
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
import argparse
import os
import time
import torch
import torchaudio
import numpy as np
import tcim_lite as tcim
from loguru import logger
import librosa
from typing import List, Optional
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from hmatc.python.get_hm_devices import get_hm_devices
from hmatc.utils.utils import first_not_none, get_model_configs

MAX_GEN_LEN = 448
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(FILE_DIR, "config.yaml")

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


def get_default_tokenizer_path(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "whisper")
    model_size = model_config.get("model_size", "medium")
    return f"{model_name}-{model_size}"


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

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

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

    def process_logits(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        processed_logits = logits.copy()
        # 1. apply repetition penalty
        processed_logits = self.apply_repetition_penalty(
            processed_logits, previous_tokens
        )

        # 2. apply softmax
        # not using softmax in case of long time cost
        probs = processed_logits
        # probs = self.softmax(processed_logits)

        # 3. apply top-k
        probs = self.apply_top_k(probs)

        # 4. apply top-p
        probs = self.apply_top_p(probs)

        # 5. apply temperature
        probs = self.apply_temperature(probs)
        return probs

    def sample(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> int:
        logits = logits[0].numpy()
        if HOUMO_TARGET == "xh2":
            logits = logits[0]
        probs = self.process_logits(logits, previous_tokens)
        if np.all(probs == 0):
            probs = np.ones_like(probs) / len(probs)

        # sampled_index = np.random.choice(len(probs), p=probs)
        sampled_index = probs.argmax(-1)

        return np.array([[sampled_index]])

    def get_processed_probs(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        return self.process_logits(logits, previous_tokens)


class HmWhisper:
    def __init__(self, encode_path, decode_path, prefill_path, ndevice=1):
        super().__init__()
        dev_manager = tcim.runtime.DevManager(get_hm_devices(ndevice), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option1 = tcim.runtime.Option(weight_manager)
        self.encode = tcim.runtime.load(encode_path, option=option1)
        logger.info("encode model loaded")
        option2 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option2)
        logger.info("prefill model loaded")
        option3 = tcim.runtime.Option(weight_manager)
        self.decode = tcim.runtime.load(decode_path, option=option3)
        logger.info("decode model loaded")
        self.encode_time = 0.0
        self.decode_time = 0.0
        self.prefill_time = 0.0
        self.num_head = self.prefill.get_input_info(
            self.prefill.get_input_name(4)
        ).shape[1]
        self.cache_max_len = self.prefill.get_input_info(
            self.prefill.get_input_name(4)
        ).shape[3]
        self.num_decode_layers, self.base_idx = self.get_num_decode_layers()

    def get_num_decode_layers(self):
        """Calculate number of transformer blocks from input tensor names."""
        count = 0
        base_idx = 0
        for i in range(self.prefill.get_num_inputs()):
            input_name = self.prefill.get_input_name(i)
            if "k_cache" in input_name or "v_cache" in input_name:
                count += 1
            if input_name == "k_cache_0":

                base_idx = i
        return count // 2, base_idx

    def run_encode(self, input_features):
        input_features = input_features.numpy()
        self.encode.set_input(self.encode.get_input_name(0), input_features)
        start_time = time.time()
        self.encode.run()
        self.encode.sync()
        self.encode_time += time.time() - start_time
        outputs = []
        for i in range(self.encode.get_num_outputs()):
            output = self.encode.get_output(self.encode.get_output_name(i)).numpy()
            outputs.append(torch.tensor(output))
        return outputs

    def run_decode(self, inputs):
        for input_name, input_data in inputs.items():
            self.decode.set_input(input_name, input_data.numpy())
        start_time = time.time()
        self.decode.run()
        self.decode.sync()
        self.decode_time += time.time() - start_time
        outputs = [
            torch.tensor(self.decode.get_output(self.decode.get_output_name(0)).numpy())
        ]
        return outputs

    def run_prefill(
        self,
        inputs,
    ):
        for input_name, input_data in inputs.items():
            self.prefill.set_input(input_name, input_data.numpy())
        start_time = time.time()
        self.prefill.run()
        self.prefill.sync()
        self.prefill_time += time.time() - start_time
        outputs = [
            torch.tensor(
                self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
            )
        ]
        return outputs

    def get_input_names(self, model_type):
        if model_type == "encode":
            self.model = self.encode
        elif model_type == "decode":
            self.model = self.decode
        elif model_type == "prefill":
            self.model = self.prefill
        input_names = []
        for i in range(self.model.get_num_inputs()):
            input_names.append(self.model.get_input_name(i))
        return input_names

    def clear_time(self):
        self.encode_time = 0.0
        self.decode_time = 0.0
        self.prefill_time = 0.0


def is_valid_char(cp):
    return cp != 0xFFFD and cp > 0x001F


def asr(whisper, processor, audio_array, language="auto", language_id=None):

    # 1. prepare config
    num_heads = whisper.num_head
    num_decode_layers = whisper.num_decode_layers
    generated_ids = []
    sampling_manager = SamplingManager(top_k=None, top_p=1.0, repetition_penalty=1.1)
    start_time = time.time()

    # get prompt Tokens ID
    sot_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    default_lang_id = processor.tokenizer.convert_tokens_to_ids("<|zh|>")
    transcribe_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notime_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    eos_id = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")

    input_features = processor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features.half()
    enc_out = whisper.run_encode(input_features)

    # Dynamically obtain sequence length
    enc_seq_len = enc_out[0].shape[2]  # 1500

    # [k0, k1, k2, k3, v0, v1, v2, v3]
    k_list = enc_out[:num_decode_layers]
    v_list = enc_out[num_decode_layers:]

    dec_names = whisper.get_input_names("decode")
    encode_attention_mask = torch.zeros((1, 1, 1, enc_seq_len), dtype=torch.float16)
    base_idx = whisper.base_idx

    def detect_language_id():
        detect_inputs = {
            dec_names[0]: torch.tensor([[sot_id]], dtype=torch.int32),
            dec_names[1]: torch.tensor([[0]], dtype=torch.int32),
            dec_names[2]: torch.tensor([0], dtype=torch.int32),
            dec_names[3]: torch.tensor([1], dtype=torch.int32),
            dec_names[4]: torch.zeros(
                (1, num_heads, 1, whisper.cache_max_len), dtype=torch.float16
            ),
            dec_names[5]: encode_attention_mask,
        }

        for i in range(2 * num_decode_layers):
            cache_name = dec_names[base_idx + i]
            cache_shape = whisper.decode.get_input_info(cache_name).shape
            detect_inputs[cache_name] = torch.ones(cache_shape, dtype=torch.float16) * (
                -65504.0
            )

        for l in range(num_decode_layers):
            detect_inputs[dec_names[base_idx + 2 * num_decode_layers + l]] = k_list[l]
            detect_inputs[dec_names[base_idx + 3 * num_decode_layers + l]] = v_list[l]

        for input_name, input_data in detect_inputs.items():
            whisper.decode.set_input(input_name, input_data.numpy())
        whisper.decode.run()
        whisper.decode.sync()
        detect_logits = torch.tensor(
            whisper.decode.get_output(whisper.decode.get_output_name(0)).numpy()
        )[:, -1, :].to(torch.float32)
        lang_logits = detect_logits.clone()
        non_lang_mask = torch.ones_like(lang_logits, dtype=torch.bool)
        non_lang_mask[0, lang_to_id] = False
        lang_logits[non_lang_mask] = -np.inf
        if torch.isneginf(lang_logits).all():
            return default_lang_id
        return int(lang_logits.argmax(-1).item())

    if language_id is not None:
        lang_id = language_id
    elif language == "auto" or language is None:
        lang_id = detect_language_id()
        logger.info(f"Detected language id: {lang_id}")
    else:
        token_str = f"<|{language.lower()}|>"
        lang_id = processor.tokenizer.convert_tokens_to_ids(token_str)
        if lang_id is None or lang_id == processor.tokenizer.unk_token_id:
            logger.warning(
                f"Unknown language code '{language}', falling back to auto detection."
            )
            lang_id = detect_language_id()
            logger.info(f"Detected language id: {lang_id}")
        else:
            logger.info(f"Forced language id: {lang_id} for '{language}'")

    prompt_tokens = [sot_id, lang_id, transcribe_id, notime_id]

    logits = None
    step = 0

    # === step1 : prefill ===

    # [sot_id, lang_id, transcribe_id, notime_id]
    input_ids = torch.tensor([prompt_tokens], dtype=torch.int32)
    prompt_len = len(prompt_tokens)  # 4
    # 2. Position IDs: [1, 4] -> [[0, 1, 2, 3]]
    position_ids = torch.arange(prompt_len, dtype=torch.int32).unsqueeze(0)
    # build self-attention Mask
    mask_atten = torch.full(
        (1, num_heads, prompt_len, whisper.cache_max_len), -65504.0, dtype=torch.float16
    )

    # add mask
    for i in range(prompt_len):
        mask_atten[:, :, i, : i + 1] = 0.0

    # This block of code is creating a dictionary named `inputs` that contains various tensors used as
    # input for the model during the Prefill stage of the decoding process. Here's a breakdown of each
    # key-value pair in the `inputs` dictionary:
    inputs = {
        dec_names[0]: input_ids,
        dec_names[1]: position_ids,
        dec_names[2]: torch.tensor(
            [0], dtype=torch.int32
        ),  # past_key_values_length = 0
        dec_names[3]: torch.tensor(
            [prompt_len], dtype=torch.int32
        ),  # current_sequence_length = 4
        dec_names[4]: mask_atten,
        dec_names[5]: encode_attention_mask,
    }

    for l in range(num_decode_layers):
        inputs[dec_names[base_idx + 2 * num_decode_layers + l]] = k_list[l]
        inputs[dec_names[base_idx + 3 * num_decode_layers + l]] = v_list[l]

    # run Prefill
    for i in range(2 * num_decode_layers):
        cache = whisper.decode.get_dev_input(
            whisper.decode.get_input_name(i + whisper.base_idx)
        )
        whisper.prefill.set_dev_input(
            whisper.prefill.get_input_name(i + whisper.base_idx), cache
        )
    out = whisper.run_prefill(inputs)

    logits = out[0]

    step += prompt_len

    for i in range(2 * num_decode_layers):
        cache = whisper.prefill.get_dev_output(whisper.prefill.get_output_name(i + 1))
        whisper.decode.set_dev_input(
            whisper.decode.get_input_name(i + whisper.base_idx), cache
        )
        whisper.decode.set_dev_output(whisper.decode.get_output_name(i + 1), cache)

    for i in range(2 * num_decode_layers):
        enc_out_kv = whisper.prefill.get_dev_input(
            whisper.prefill.get_input_name(base_idx + 2 * num_decode_layers + i)
        )
        whisper.decode.set_dev_input(
            whisper.decode.get_input_name(base_idx + 2 * num_decode_layers + i),
            enc_out_kv,
        )

    # === step 2: Decode ===
    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    generated_ids.append(next_token)

    ttft_time = time.time() - start_time
    print("\033[1;95m", end="", flush=True)

    slide_len = 10
    skip_tokens = 0
    last_response = ""
    # 使用 tensor 存储已生成的 token，与 demo_mix.py 保持一致
    generated_token_ids = torch.tensor([[next_token]])

    while step < MAX_GEN_LEN:
        # 增量解码：只解码滑动窗口内的 token
        decode_response = processor.decode(
            generated_token_ids[0][-(slide_len + 1) - skip_tokens :]
        )[len(last_response) :]

        # 只有当解码结果不为空且最后一个字符是有效字符时才打印
        if (
            decode_response != ""
            and is_valid_char(ord(decode_response[-1]))
            and next_token != eos_id
        ):
            print(decode_response, end="", flush=True)
            last_response = processor.decode(generated_token_ids[0][-slide_len:])
            skip_tokens = 0
        else:
            skip_tokens += 1

        if next_token == eos_id:
            break

        input_ids = torch.tensor([[next_token]], dtype=torch.int32)

        mask_atten = torch.zeros(
            (1, num_heads, 1, whisper.cache_max_len), dtype=torch.float16
        )
        if step + 1 < whisper.cache_max_len:
            mask_atten[:, :, :, step + 1 :] = -65504

        inputs = {
            dec_names[0]: input_ids,
            dec_names[1]: torch.tensor([[step]], dtype=torch.int32),
            dec_names[2]: torch.tensor([step], dtype=torch.int32),
            dec_names[3]: torch.tensor([1], dtype=torch.int32),
            dec_names[4]: mask_atten,
            dec_names[5]: encode_attention_mask,
        }

        out = whisper.run_decode(inputs)
        logits = out[0]

        next_token_arr = sampling_manager.sample(logits, generated_ids)
        next_token = int(next_token_arr[0, 0])
        generated_ids.append(next_token)
        generated_token_ids = torch.cat(
            [generated_token_ids, torch.tensor([[next_token]])], dim=-1
        )
        step += 1
    decoded_text = processor.decode(generated_token_ids[0], skip_special_tokens=True)
    return max(len(generated_ids) - 1, 0), ttft_time, decoded_text, lang_id


if __name__ == "__main__":
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
        "--tokenizer_path",
        type=str,
        default=None,
        help="Whisper tokenizer path",
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
        "--audio",
        type=str,
        default="../../../data/audio/audio.mp3",
    )
    parser.add_argument(
        "--chunk_size",
        type=float,
        default=30.0,
        help="audio chunk size in seconds",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="language code (e.g. 'zh', 'en') or 'auto' for language detection",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number",
    )
    args = parser.parse_args()

    if args.chunk_size <= 0 or args.chunk_size > 30:
        parser.error("--chunk_size must be greater than 0 and less than or equal to 30")
    if args.ndevice is not None and args.ndevice != 1:
        parser.error("Only supports ndevice=1.")

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    args.tokenizer_path = first_not_none(
        args.tokenizer_path, get_default_tokenizer_path(model_config)
    )
    model_prefix = f"{args.model_name}-{args.model_size}"
    if args.encode_path is None:
        args.encode_path = os.path.join(
            "output", HOUMO_TARGET, f"{model_prefix}_encode.hmm"
        )
    if args.decode_path is None:
        args.decode_path = os.path.join(
            "output", HOUMO_TARGET, f"{model_prefix}_decode.hmm"
        )
    if args.prefill_path is None:
        args.prefill_path = os.path.join(
            "output", HOUMO_TARGET, f"{model_prefix}_prefill.hmm"
        )

    whisper = HmWhisper(
        args.encode_path, args.decode_path, args.prefill_path, args.ndevice
    )
    processor = WhisperProcessor.from_pretrained(args.tokenizer_path)
    results = ""
    output_tokens = 0
    total_start = time.time()
    audio_load_start = time.time()
    waveform, sr = torchaudio.load(args.audio)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        logger.info(f"Resampled audio from {sr}Hz to 16000Hz")
        sr = 16000
    audio_array = waveform.numpy()
    audio_load_time = time.time() - audio_load_start
    chunk_size = int(args.chunk_size * sr)
    total_samples = len(audio_array)
    chunks = []

    for start in range(0, total_samples, chunk_size):
        end = start + chunk_size
        chunk = audio_array[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")
        chunks.append(chunk)
    chunks_array = np.array(chunks)

    logger.success("transcription:")
    language_id = None
    for i, chunk in enumerate(chunks):
        output_token, current_ttft, decoded_text, language_id = asr(
            whisper, processor, chunk, language=args.language, language_id=language_id
        )
        results += decoded_text
        output_tokens += output_token
        ttft_time = current_ttft + audio_load_time if i == 0 else ttft_time

    # reset color and print newline
    print("\033[0m")

    e2e_time = time.time() - total_start
    total_audio_duration = len(audio_array) / sr

    logger.success("=" * 60)
    logger.success("Performance Statistics:")
    logger.success("=" * 60)
    logger.success(
        f"Audio duration: {total_audio_duration * 1000:.2f} ms ({total_audio_duration:.2f} s)"
    )
    logger.success(f"Loop count: {len(chunks)}")
    logger.success("-" * 60)
    logger.success(
        f"Total Encode time: {whisper.encode_time * 1000:.3f} ms"
    )
    logger.success("-" * 60)
    logger.success(
        f"Total Prefill time: {whisper.prefill_time * 1000:.3f} ms"
    )
    logger.success("-" * 60)
    logger.success(
        f"Output {output_tokens} tokens"
    )
    logger.success(
        f"Total Decode time: {whisper.decode_time * 1000:.3f} ms"
    )
    if whisper.decode_time > 0:
        logger.success(
            f"Decode Speed: {output_tokens / whisper.decode_time:.2f} tokens/s"
        )
    logger.success("-" * 60)
    logger.success(f"TTFT (Time to First Token): {ttft_time * 1000:.3f} ms")
    logger.success(f"E2E Latency (End-to-End Latency): {e2e_time:.3f} seconds")
    if e2e_time > 0:
        logger.success(
            f"E2E TPS (End-to-End Tokens Per Second): {output_tokens / e2e_time:.2f} tokens/s"
        )
    logger.success(f"RTF (Real-Time Factor): {e2e_time / total_audio_duration:.2f}")
    logger.success("=" * 60)
    whisper.clear_time()
