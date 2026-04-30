#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description: Demo script for running end-to-end inference with exported GLM-ASR HMM models via tcim_lite
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
import re
import argparse
import time
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from loguru import logger
import tcim_lite as tcim
from hmatc.python.get_hm_devices import get_hm_devices

TARGET_TYPE = torch.float16
HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def is_valid_stream_char(cp: int) -> bool:
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x20000 and cp <= 0x2A6DF)
        or (cp >= 0x2A700 and cp <= 0x2B73F)
        or (cp >= 0x2B740 and cp <= 0x2B81F)
        or (cp >= 0x2B820 and cp <= 0x2CEAF)
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
        or (0x0030 <= cp and cp <= 0x0039)
        or (0x0041 <= cp and cp <= 0x005A)
        or (0x0061 <= cp and cp <= 0x007A)
    ):
        return True

    return chr(cp) in {
        " ",
        "\n",
        ".",
        ",",
        "!",
        "?",
        ":",
        ";",
        "'",
        '"',
        "-",
        "_",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "，",
        "。",
        "！",
        "？",
        "：",
        "；",
        "、",
    }


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser(description="GLM-ASR Tcim Inference Demo")
    parser.add_argument(
        "--processor_path",
        type=str,
        default="glm-asr-nano-2512",
        help="Processor/config path or Hub ID for loading processor/config",
    )
    parser.add_argument(
        "--encoder_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "glm-asr_encode.hmm"),
        help="houmo encoder model path",
    )
    parser.add_argument(
        "--prefill_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "glm-asr_prefill.hmm"),
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--decode_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "glm-asr_decode.hmm"),
        help="houmo decode model path",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"),
        help="houmo embedding weight path",
    )
    parser.add_argument(
        "--ndevice",
        type=int,
        default=1,
        choices=[1, 2],
        help="device number, only xh2 support",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="../../../data/audio/audio.mp3",
        help="Path to audio file (.wav, .mp3, .flac, etc.)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate",
    )
    return parser.parse_args()


class HmGLM_ASR:
    """End-to-end GLM-ASR inference pipeline using Tcim Lite (HMM models)."""

    def __init__(self, args):
        self.ndevice = args.ndevice
        dev_manager = tcim.runtime.DevManager(get_hm_devices(self.ndevice), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)

        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        option3 = tcim.runtime.Option(weight_manager)

        # Load Encoder
        logger.info(f"Loading encoder from: {args.encoder_path}")
        self.encoder_sess = tcim.runtime.load(args.encoder_path, option=option1)
        logger.info("encoder model loaded")

        # Load Prefill
        logger.info(f"Loading prefill from: {args.prefill_path}")
        self.prefill_sess = tcim.runtime.load(args.prefill_path, option=option2)
        logger.info("prefill model loaded")

        # Setup KV Cache Dummy Tensors for Memory Reuse
        self.num_hidden_layers = self._get_nblocks()
        dummy_tensor_names = [
            f"model_layers_{i}_self_attn_kcache_input"
            for i in range(self.num_hidden_layers)
        ]
        dummy_tensor_names += [
            f"model_layers_{i}_self_attn_vcache_input"
            for i in range(self.num_hidden_layers)
        ]
        option2.set_dummy_tensors(dummy_tensor_names)

        # Load Decode
        logger.info(f"Loading decode from: {args.decode_path}")
        self.decode_sess = tcim.runtime.load(args.decode_path, option=option3)
        logger.info("decode model loaded")

        # Get configurations from prefill inputs
        self.max_prefill = self.prefill_sess.get_input_info("input_embeds").shape[1]
        self.hidden_size = self.prefill_sess.get_input_info("input_embeds").shape[2]

        # Bind KV caches between prefill & decode
        for name in dummy_tensor_names:
            cache = self.prefill_sess.get_input(name)
            self.decode_sess.set_input(name, cache)

        # Pre-set some unvarying components in decode to save time if needed
        # (Though valid/current lengths update each step)

        # Load HF assets
        logger.info(f"Loading processor and tokenizer from: {args.processor_path}")
        self.processor = AutoProcessor.from_pretrained(
            args.processor_path, trust_remote_code=True
        )
        self.config = AutoConfig.from_pretrained(
            args.processor_path, trust_remote_code=True
        )
        self.text_config = self.config.text_config

        self.audio_token_id = getattr(self.config, "audio_token_id", 59260)
        self.eos_token_ids = getattr(
            self.text_config, "eos_token_id", [59246, 59253, 59255]
        )
        if not isinstance(self.eos_token_ids, list):
            self.eos_token_ids = [self.eos_token_ids]

        # Token embedding
        logger.info(f"Loading token embedding from: {args.embedding_path}")
        w = torch.load(args.embedding_path, map_location="cpu", weights_only=True)[
            "weight"
        ]
        self.embed_tokens = nn.Embedding(*w.shape).eval()
        self.embed_tokens.weight.data.copy_(w.float())

        # Timing metrics
        self.encoder_time = 0
        self.prefill_time = 0
        self.decode_time = 0
        self.ttft_time = 0

    def _get_nblocks(self):
        input_names = []
        for i in range(self.prefill_sess.get_num_inputs()):
            input_names.append(self.prefill_sess.get_input_name(i))
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def _compute_audio_output_length(
        self, input_features_mask: torch.Tensor
    ) -> torch.Tensor:
        audio_lengths = input_features_mask.sum(-1)
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            audio_lengths = (
                audio_lengths + 2 * padding - (kernel_size - 1) - 1
            ) // stride + 1
        merge_factor = 4
        post_lengths = (audio_lengths - merge_factor) // merge_factor + 1
        return post_lengths

    def _run_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        # Pad to fixed length 3000 if necessary
        feat_len = input_features.shape[2]
        if feat_len < 3000:
            input_features = torch.nn.functional.pad(
                input_features, (0, 3000 - feat_len), value=0.0
            )

        encoder_input_name = self.encoder_sess.get_input_name(0)
        self.encoder_sess.set_input(
            encoder_input_name, input_features.numpy().astype(np.float16)
        )

        start_time = time.time()
        self.encoder_sess.run()
        self.encoder_sess.sync()
        self.encoder_time += time.time() - start_time

        encoder_output = self.encoder_sess.get_output(
            self.encoder_sess.get_output_name(0)
        )
        return torch.tensor(encoder_output.numpy())

    def _run_prefill(self, embeds: torch.Tensor, L: int) -> int:
        prefill_embeds = torch.zeros(
            (1, self.max_prefill, self.hidden_size), dtype=torch.float16
        )
        prefill_embeds[:, :L, :] = embeds[:, :L, :].half()

        self.prefill_sess.set_input(
            "input_embeds", prefill_embeds.detach().numpy().astype(np.float16)
        )
        self.prefill_sess.set_input("valid_length", np.array([0], dtype=np.int32))
        self.prefill_sess.set_input("current_length", np.array([L], dtype=np.int32))

        start_time = time.time()
        self.prefill_sess.run()
        self.prefill_sess.sync()
        self.prefill_time += time.time() - start_time

        prefill_output = self.prefill_sess.get_output(
            self.prefill_sess.get_output_name(0)
        )
        next_id = np.argmax(prefill_output.numpy(), axis=-1).item()
        return next_id

    def _run_decode(self, token_id: int, valid_length_val: int) -> int:
        token_tensor = torch.tensor([[token_id]])
        next_embed = self.embed_tokens(token_tensor).half()

        self.decode_sess.set_input(
            "input_embeds", next_embed.detach().numpy().astype(np.float16)
        )
        self.decode_sess.set_input(
            "valid_length", np.array([valid_length_val], dtype=np.int32)
        )
        self.decode_sess.set_input("current_length", np.array([1], dtype=np.int32))

        start_time = time.time()
        self.decode_sess.run()
        self.decode_sess.sync()
        self.decode_time += time.time() - start_time

        decode_output = self.decode_sess.get_output(self.decode_sess.get_output_name(0))
        next_id = np.argmax(decode_output.numpy(), axis=-1).item()
        return next_id

    def _decode_token_ids(self, token_ids: List[int]) -> str:
        if not token_ids:
            return ""

        decoded = self.processor.batch_decode([token_ids], skip_special_tokens=True)
        if isinstance(decoded, list):
            return decoded[0] if decoded else ""
        return decoded

    def _stream_window_text(
        self,
        generated_ids: List[int],
        slide_len: int,
        skip_tokens: int,
        last_window_text: str,
        streamed_text: str,
    ) -> Tuple[int, str, str]:
        window_size = slide_len + 1 + skip_tokens
        candidate_text = self._decode_token_ids(generated_ids[-window_size:])

        if candidate_text.startswith(last_window_text):
            emit_text = candidate_text[len(last_window_text) :]
        else:
            emit_text = candidate_text

        if (
            emit_text
            and "�" not in emit_text
            and is_valid_stream_char(ord(emit_text[-1]))
        ):
            print("\033[1;95m{}".format(emit_text), end="", flush=True)
            streamed_text += emit_text
            last_window_text = self._decode_token_ids(generated_ids[-slide_len:])
            skip_tokens = 0
        else:
            skip_tokens += 1

        return skip_tokens, last_window_text, streamed_text

    def _run_inference(self, inputs: dict, max_new_tokens: int) -> Tuple[str, int]:
        input_features = inputs["input_features"].float()
        input_features_mask = inputs["input_features_mask"]
        input_ids = inputs["input_ids"]

        logger.info("Running Encoder...")
        audio_embeds = self._run_encoder(input_features)

        T_out = self._compute_audio_output_length(input_features_mask).item()
        audio_embeds = audio_embeds[:, : int(T_out), :]
        if audio_embeds.dim() == 2:
            audio_embeds = audio_embeds.unsqueeze(0)

        # Feature Fusion
        text_embeds = self.embed_tokens(input_ids)
        pad_indices = (input_ids == self.audio_token_id).nonzero(as_tuple=True)[1]

        if len(pad_indices) > 0:
            start_idx = pad_indices[0].item()
            end_idx = pad_indices[-1].item()
            final_inputs_embeds = torch.cat(
                [
                    text_embeds[:, :start_idx, :],
                    audio_embeds.to(text_embeds.dtype),
                    text_embeds[:, end_idx + 1 :, :],
                ],
                dim=1,
            )
        else:
            final_inputs_embeds = text_embeds

        seq_len = final_inputs_embeds.shape[1]
        L = min(seq_len, self.max_prefill)

        logger.info("Running Llm...")
        ttft_start = time.time()
        next_token_id = self._run_prefill(final_inputs_embeds, L)
        self.ttft_time += time.time() - ttft_start
        generated_ids = [next_token_id]

        slide_len = 10
        skip_tokens = 0
        last_window_text = ""
        streamed_text = ""
        skip_tokens, last_window_text, streamed_text = self._stream_window_text(
            generated_ids,
            slide_len,
            skip_tokens,
            last_window_text,
            streamed_text,
        )

        valid_length = L
        for _ in range(max_new_tokens):
            if generated_ids[-1] in self.eos_token_ids:
                break

            next_id = self._run_decode(generated_ids[-1], valid_length)
            generated_ids.append(next_id)
            valid_length += 1

            skip_tokens, last_window_text, streamed_text = self._stream_window_text(
                generated_ids,
                slide_len,
                skip_tokens,
                last_window_text,
                streamed_text,
            )

        result_text = self._decode_token_ids(generated_ids)
        if result_text.startswith(streamed_text):
            remaining_text = result_text[len(streamed_text) :]
        else:
            remaining_text = result_text

        if remaining_text:
            print("\033[1;95m{}".format(remaining_text), end="", flush=True)

        print("\033[0m")
        return result_text, valid_length - L

    def transcribe(self, audio_input, max_new_tokens: int = 2048) -> Tuple[str, int]:
        import librosa

        if isinstance(audio_input, str):
            audio_array, _ = librosa.load(
                audio_input,
                sr=self.processor.feature_extractor.sampling_rate,
                mono=True,
            )
        else:
            audio_array = audio_input

        sr = self.processor.feature_extractor.sampling_rate
        chunk_size = int(sr * 30.0)
        n_samples = len(audio_array)
        n_chunks = max(1, (n_samples + chunk_size - 1) // chunk_size)

        total_tokens = 0
        results = []
        for i in range(n_chunks):
            chunk = audio_array[i * chunk_size : (i + 1) * chunk_size]
            logger.info(f"Processing Chunk {i + 1}/{n_chunks} ({len(chunk) / sr:.1f}s)")
            inputs = self.processor.apply_transcription_request(chunk)

            res_str, tokens = self._run_inference(inputs, max_new_tokens)
            results.append(res_str)
            total_tokens += tokens

        return " ".join(filter(None, results)), total_tokens


def show_statictic_info(inference: HmGLM_ASR, output_tokens: int):
    logger.success(f"Encoder Time: {inference.encoder_time * 1000:.3f} ms")
    logger.success(f"Prefill Cost: {inference.prefill_time * 1000:.3f} ms")
    logger.success(f"Decode Cost: {inference.decode_time * 1000:.3f} ms")
    logger.success(
        f"Output {output_tokens} tokens, Decode Speed: {output_tokens / max(inference.decode_time, 1e-5):.2f} tokens/s"
    )
    logger.success(f"TTFT (Time to First Token): {inference.ttft_time * 1000:.3f} ms")
    logger.success(
        f"TPOT (Time Per Output Token): {inference.decode_time * 1000 / max(output_tokens, 1):.3f} ms/token"
    )
    logger.success(
        f"E2E Latency: {(inference.ttft_time + inference.decode_time):.3f} seconds"
    )


def main(args):
    logger.info("Initializing GLM-ASR Inference via Tcim...")
    inference = HmGLM_ASR(args)

    result, out_tokens = inference.transcribe(
        args.audio, max_new_tokens=args.max_new_tokens
    )

    match = re.search(r"(?<=<asr_text>)[\s\S]*", result)
    if match:
        result = match.group().strip()

    print("\n" + "=" * 60)
    print(f"Transcription Result:")
    print("=" * 60)
    print(result)
    print("=" * 60)

    show_statictic_info(inference, out_tokens)


if __name__ == "__main__":
    args = get_args()
    main(args)
