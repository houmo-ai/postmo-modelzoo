#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo_new.py
# Description:
#   Optimized Qwen3 ASR Inference Demo - Minimized tensor conversions,
#   removed unused variables, and enabled performance statistics.
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
import time
import argparse
import numpy as np
import librosa
from loguru import logger

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from xhquant.api import HMONNXInference as InferenceEngine
from qwen_asr.core.transformers_backend import (
    Qwen3ASRProcessor,
)

import tcim_lite as tcim

HOUMO_TARGET = os.getenv("HOUMO_TARGET")


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processor_dir",
        dest="processor_dir",
        type=str,
        default="Qwen3-ASR-0.6B",
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
        default=os.path.join("output", HOUMO_TARGET, "Qwen3-ASR-1.7B_Encoder_xh2a_w8a8_sefp.onnx"),
        help="houmo encode model path",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3_asr_decode.hmm"),
        help="houmo decode model path",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3_asr_prefill.hmm"),
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"),
        help="houmo embedding weight path",
    )
    return parser.parse_args()


class Qwen3Asr:
    def __init__(self, encode_path, decode_path, prefill_path, processor_dir, embedding_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encode = InferenceEngine(str(encode_path))
        self.encode.to(str(self.device))
        weight_manager = tcim.runtime.WeightManager(0)
        logger.info("encode model loaded")
        option2 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option2)
        logger.info("prefill model loaded")

        option3 = tcim.runtime.Option(weight_manager)
        self.nblocks = self.get_nblocks()
        dummy_tensor_names = [
            f"model_layers_{i}_self_attn_kcache_input" for i in range(self.nblocks)
        ]
        dummy_tensor_names += [
            f"model_layers_{i}_self_attn_vcache_input" for i in range(self.nblocks)
        ]
        option3.set_dummy_tensors(dummy_tensor_names)
        self.decode = tcim.runtime.load(decode_path, option=option3)
        logger.info("decode model loaded")

        self.max_new_tokens = self.prefill.get_input_info(
            self.prefill.get_input_name(3)
        ).shape[2]
        self.max_prefill = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]

        self.processor = Qwen3ASRProcessor.from_pretrained(processor_dir, fix_mistral_regex=True)
        self.config = AutoConfig.from_pretrained(processor_dir, trust_remote_code=True)

        self.embedding_weight = torch.load(embedding_path, map_location=self.device)
        if HOUMO_TARGET == "xh2":
            self.embedding_weight = self.embedding_weight["weight"].float()
        logger.info(f"embedding weight shape: {self.embedding_weight.shape}")

        # Link KV caches between prefill and decode
        for i in range(3, 2 * self.nblocks + 3):
            cache = self.prefill.get_input(self.prefill.get_input_name(i))
            self.decode.set_input(self.decode.get_input_name(i), cache)

    def run_encode(self, inputs):
        inputs['input_features'] = inputs['input_features'].float()
        origin_feature_lens = inputs['feature_attention_mask'].sum(dim=-1).to(torch.int32)

        pad_width = (0, 3000 - inputs['input_features'].shape[2])

        inputs['input_features'] = F.pad(
            inputs['input_features'],
            pad_width,
            mode='constant',
            value=0.0
        )

        inputs['feature_attention_mask'] = F.pad(
            inputs['feature_attention_mask'],
            pad_width,
            mode='constant',
            value=0
        )

        logger.info(f"encoder input shape: {inputs['input_features'].shape}, feature_attention_mask shape: {inputs['feature_attention_mask'].shape}")

        if not os.path.exists("outputs.pt"):
            outputs = self.encode.run({
                "input_features": inputs['input_features'].to(torch.float16),
                "feature_lens": origin_feature_lens,
            })
            torch.save(outputs.to("cpu"), "outputs.pt")
        else:
            outputs = torch.load("outputs.pt", map_location=torch.device('cpu')).to(self.device)

        return outputs, origin_feature_lens

    def run_decode(self, next_token_id, L):
        # Use numpy arrays directly to avoid repeated conversions
        valid_length_np = np.array([L], dtype=np.int32)
        current_length_np = np.array([1], dtype=np.int32)  # Fixed: should be 1, not L

        generated_ids = [next_token_id]
        eos_token_id = self.processor.tokenizer.eos_token_id

        decode_start = time.perf_counter()
        tokens_generated = 0
        total_decode_time = 0
        
        for _ in range(self.max_new_tokens):
            # Use numpy array for token embedding lookup
            token_tensor = torch.tensor([[generated_ids[-1]]], device=self.device)
            next_embed = F.embedding(token_tensor, self.embedding_weight)

            self.decode.set_input(self.decode.get_input_name(0), next_embed.numpy())
            self.decode.set_input(self.decode.get_input_name(1), valid_length_np)
            self.decode.set_input(self.decode.get_input_name(2), current_length_np)

            t0 = time.time()
            self.decode.run()
            self.decode.sync()
            t1 = time.time() - t0
            total_decode_time += t1
            logger.info(f"Loop {_} Single Decode time: {t1 * 1000:.3f} ms")
            # Process output directly with numpy
            decode_outputs = self.decode.get_output(self.decode.get_output_name(0)).numpy()
            next_id = int(np.argmax(decode_outputs, axis=-1).item())
            generated_ids.append(next_id)

            # Update lengths
            valid_length_np[0] += 1
            tokens_generated += 1

            # Stop on EOS
            if next_id == eos_token_id:
                break

        decode_time = time.perf_counter() - decode_start
        logger.info(f"total Decode time: {total_decode_time * 1000:.3f} ms")
        result = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        match = re.search(r'(?<=<asr_text>)[\s\S]*', result)
        if match:
            print(match.group())

        return result, tokens_generated, decode_time

    def run_prefill(self, origin_feature_lens, inputs, audio_embeds):
        T_out = self._get_feat_extract_output_lengths(origin_feature_lens).item()
        audio_embeds = audio_embeds[:, :T_out, :]

        logger.info(f"origin_feature_lens: {origin_feature_lens}, T_out: {T_out}, audio_embeds shape after trim: {audio_embeds.shape}")

        if audio_embeds.dim() == 2:
            audio_embeds = audio_embeds.unsqueeze(0)

        text_input_ids = inputs['input_ids']
        text_embeds = F.embedding(text_input_ids, self.embedding_weight)

        tokenizer = self.processor.tokenizer
        if "<|audio_pad|>" in tokenizer.get_vocab():
            audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        else:
            audio_pad_id = tokenizer.encode("<|audio_pad|>", add_special_tokens=False)[0]

        pad_indices = (text_input_ids == audio_pad_id).nonzero(as_tuple=True)[1]

        if len(pad_indices) > 0:
            start_idx = pad_indices[0].item()
            end_idx = pad_indices[-1].item()
            final_inputs_embeds = torch.cat([
                text_embeds[:, :start_idx, :],
                audio_embeds,
                text_embeds[:, end_idx+1:, :]
            ], dim=1)
        else:
            final_inputs_embeds = text_embeds

        text_config = self.config.thinker_config.text_config
        num_layers = text_config.num_hidden_layers
        num_kv_heads = text_config.num_key_value_heads
        hidden_size = text_config.hidden_size
        head_dim = text_config.head_dim
        cache_len = 2048

        seq_len = final_inputs_embeds.shape[1]
        L = min(seq_len, self.max_prefill)

        prefill_embeds = torch.zeros((1, self.max_prefill, hidden_size), dtype=torch.float16, device=self.device)
        prefill_embeds[:, :L, :] = final_inputs_embeds[:, :L, :].to(torch.float16).to(self.device)

        valid_length_np = np.array([0], dtype=np.int32)
        current_length_np = np.array([L], dtype=np.int32)

        # Move to CPU before numpy() - fix for GPU runtime
        if "cuda" in str(self.device):
            prefill_embeds = prefill_embeds.cpu()

        self.prefill.set_input(self.prefill.get_input_name(0), prefill_embeds.numpy())
        self.prefill.set_input(self.prefill.get_input_name(1), valid_length_np)
        self.prefill.set_input(self.prefill.get_input_name(2), current_length_np)
        t0 = time.time()
        self.prefill.run()
        self.prefill.sync()
        t1 = time.time() - t0
        logger.info(f"Prefill time: {t1 * 1000:.3f} ms.")
        last_hidden_state = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        return last_hidden_state, L

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
        return output_lengths

    def run(self, audio_path):
        total_start = time.perf_counter()

        # Load audio
        audio_load_start = time.perf_counter()
        if os.path.exists(audio_path):
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
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
        inputs = self.processor(text=prompt, audio=audio, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        prep_time = time.perf_counter() - prep_start

        # Run encode
        encode_start = time.perf_counter()
        outputs, origin_feature_lens = self.run_encode(inputs)
        audio_embeds = outputs.to(self.device)
        encode_time = time.perf_counter() - encode_start
        logger.info(f"audio_embeds shape: {audio_embeds.shape}")

        # Run prefill
        prefill_start = time.perf_counter()
        last_hidden_state, L = self.run_prefill(origin_feature_lens, inputs, audio_embeds)
        prefill_time = time.perf_counter() - prefill_start

        # Get first token
        first_token_start = time.perf_counter()
        next_token_id = int(np.argmax(last_hidden_state, axis=-1).item())
        first_token_time = time.perf_counter() - first_token_start

        # Calculate TTFT (Time To First Token)
        ttft_time = encode_time + prefill_time + first_token_time

        # Run decode
        result, tokens_generated, decode_time = self.run_decode(next_token_id, L)

        # Calculate total time
        total_time = time.perf_counter() - total_start
        infer_time = total_time  # Total inference time in seconds
        prefill_ids_len = L
        all_ids_len = prefill_ids_len + tokens_generated

        # Calculate RTF (Real Time Factor)
        rtf = infer_time / audio_duration

        # Performance statistics
        logger.success("=" * 60)
        logger.success("Performance Statistics:")
        logger.success("=" * 60)
        logger.success(f"Audio duration: {audio_duration * 1000:.2f} ms ({audio_duration:.2f} s)")
        logger.success(f"Audio loading time: {audio_load_time * 1000:.3f} ms")
        logger.success(f"Input preparation time: {prep_time * 1000:.3f} ms")
        logger.success(f"Encode time: {encode_time * 1000:.3f} ms")
        logger.success(f"Prefill time: {prefill_time * 1000:.3f} ms")
        logger.success(f"First token selection time: {first_token_time * 1000:.3f} ms")
        logger.success(f"Output {all_ids_len} tokens ({prefill_ids_len} prefill + {tokens_generated} generated)")
        logger.success(f"TTFT (Time to First Token): {ttft_time * 1000:.3f} ms")
        logger.success(f"Decode Cost: {decode_time * 1000:.3f} ms")
        logger.success(f"Decode Speed: {tokens_generated / decode_time:.2f} tokens/s")
        if tokens_generated > 0:
            logger.success(f"TPOT (Time Per Output Token): {decode_time * 1000 / tokens_generated:.3f} ms/token")
        logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
        logger.success(f"E2E TPS (End-to-End Tokens Per Second): {all_ids_len / total_time:.2f} tokens/s")
        logger.success(f"RTF (Real Time Factor): {rtf:.2f}")
        logger.success("=" * 60)

        return result


if __name__ == "__main__":
    args = get_args()

    if HOUMO_TARGET == "xh2":
        qwen3asr = Qwen3Asr(
            args.encode_path,
            args.decode_path,
            args.prefill_path,
            args.processor_dir,
            args.embedding_path
        )
    else:
        raise ValueError("Unsupported houmo target!")

    qwen3asr.run(args.audio)
