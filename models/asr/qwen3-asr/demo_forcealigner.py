#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo_forcealigner.py
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
from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor
from hmatc.python.get_hm_devices import get_hm_devices
import tcim_lite as tcim

HOUMO_TARGET = os.getenv("HOUMO_TARGET")


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processor_dir",
        dest="processor_dir",
        type=str,
        default="Qwen3-ForcedAligner-0.6B",
        help="processor dir",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="../../../data/audio/61-70968-0000.wav",
    )
    parser.add_argument(
        "--encode_path",
        dest="encode_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3_forcealigner_encode.hmm"),
        help="houmo encode model path",
    )
    parser.add_argument(
        "--text",
        dest="text",
        type=str,
        default="HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND THE CURTAIN ON THE LEFT",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3_forcealigner_prefill.hmm"),
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
        "--language",
        dest="language",
        type=str,
        default="Chinese",
        help="language of the text",
    )
    return parser.parse_args()


class Qwen3ForceAligner:
    def __init__(self, encode_path, prefill_path, processor_dir, embedding_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dev_manager = tcim.runtime.DevManager(get_hm_devices(), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option1 = tcim.runtime.Option(weight_manager)
        self.encode = tcim.runtime.load(encode_path, option=option1)
        logger.info("encode model loaded")
        option2 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option2)
        logger.info("prefill model loaded")

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

        self.aligner_processor = Qwen3ForceAlignProcessor()
        self.sample_rate = 16000
        self.max_audio_len = 3000
    def run_encode(self, inputs):
        input_features = inputs["input_features"]
        feature_attention_mask = inputs["feature_attention_mask"]
        input_ids = inputs["input_ids"]
        origin_feature_lens = feature_attention_mask.sum(dim=-1).to(torch.int32)

        pad_width = (0, self.max_audio_len - input_features.shape[2])

        input_features = torch.nn.functional.pad(
            input_features,
            pad_width,
            mode="constant",
            value=0.0
        )

        self.encode.set_input(self.encode.get_input_name(0), input_features.to(torch.float16).numpy())
        self.encode.set_input(self.encode.get_input_name(1), origin_feature_lens.numpy())
        
        self.encode.run()
        self.encode.sync()

        outputs = self.encode.get_output(self.encode.get_output_name(0)).numpy()

        outputs = torch.from_numpy(outputs)

        return outputs, origin_feature_lens

    def run_prefill(self, origin_feature_lens, inputs, audio_embeds):
        T_out = self._get_feat_extract_output_lengths(origin_feature_lens).item()
        audio_embeds = audio_embeds[:, :T_out, :]

        text_input_ids = inputs['input_ids']
        text_embeds = F.embedding(text_input_ids, self.embedding_weight)
        tokenizer = self.processor.tokenizer
        if "<|audio_pad|>" in tokenizer.get_vocab():
            audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        else:
            audio_pad_id = tokenizer.encode("<|audio_pad|>", add_special_tokens=False)[0]

        pad_indices = (text_input_ids == audio_pad_id).nonzero(as_tuple=True)[1]

        start_idx = pad_indices[0].item()
        end_idx = pad_indices[-1].item()

        assert audio_embeds.shape[1] == (end_idx - start_idx + 1)
        
        inputs_embeds = torch.cat(
            [
                text_embeds[:, :start_idx],
                audio_embeds,
                text_embeds[:, end_idx + 1 :]
            ],
            dim=1
        )

        text_config = self.config.thinker_config.text_config

        num_layers = text_config.num_hidden_layers
        num_kv_heads = text_config.num_key_value_heads
        head_dim = text_config.head_dim
        hidden_size = text_config.hidden_size


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
        cache_len = self.max_new_tokens

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

        self.prefill.run()
        self.prefill.sync()

        prefill_outputs = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        return prefill_outputs, text_input_ids
    def _get_feat_extract_output_lengths(self, input_lengths):
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1

        output_lengths = (
            ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
            + (input_lengths // 100) * 13
        )

        return output_lengths

    def to_torch(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        return x.to(self.device)
    def load_audio(self, path):

        audio, sr = librosa.load(path, sr=None, mono=False)

        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        audio = audio.astype(np.float32)

        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak

        return audio
    def run(self, audio_path, text, language):
        total_start = time.perf_counter()

        # Load audio
        audio_load_start = time.perf_counter()
        if os.path.exists(audio_path):
            audio = self.load_audio(audio_path)
        else:
            logger.error(f"Audio file {audio_path} does not exist.")
            return
        audio_duration = len(audio) / self.sample_rate
        audio_load_time = time.perf_counter() - audio_load_start

        # Prepare inputs
        prep_start = time.perf_counter()
        word_list, aligner_input_text = self.aligner_processor.encode_timestamp(
            text,
            language
        )

        inputs = self.processor(
            text=[aligner_input_text],
            audio=[audio],
            return_tensors="pt",
            padding=True
        )

        inputs = inputs.to(self.device)
        prep_time = time.perf_counter() - prep_start

        # Run encode
        encode_start = time.perf_counter()
        encoder_outputs, origin_feature_lens = self.run_encode(inputs)
        audio_embeds = encoder_outputs
        encode_time = time.perf_counter() - encode_start

        if isinstance(audio_embeds, np.ndarray):
            audio_embeds = torch.from_numpy(audio_embeds)
        
        audio_embeds = audio_embeds.to(self.device).to(torch.float16)
        # Run prefill
        prefill_start = time.perf_counter()
        prefill_outputs, input_ids = self.run_prefill(origin_feature_lens, inputs, audio_embeds)
        prefill_time = time.perf_counter() - prefill_start

        for i, out in enumerate(prefill_outputs):
            t = self.to_torch(out)

        logits = None
        for out in prefill_outputs:
            t = self.to_torch(out)
            # [B,T,V]
            if t.ndim == 3:
                logits = t
                break
            # [T,V]
            if t.ndim == 2:
                logits = t.unsqueeze(0)
                break
        
        if logits is None:
            raise RuntimeError("No suitable logits found from prefill outputs.")

        output_ids = logits.argmax(dim=-1)     # [1, T]

        if input_ids.shape[1] != output_ids.shape[1]:
            print(f"WARNING: input_ids len={input_ids.shape[1]} != output_ids len={output_ids.shape[1]}")
            T = min(input_ids.shape[1], output_ids.shape[1])
            input_ids_cut = input_ids[:, :T]
            output_ids_cut = output_ids[:, :T]
        else:
            input_ids_cut = input_ids
            output_ids_cut = output_ids

        timestamp_token_id = self.config.timestamp_token_id
        timestamp_segment_time = self.config.timestamp_segment_time
        masked_output_id = output_ids_cut[input_ids_cut == timestamp_token_id]
        timestamp_ms = (masked_output_id * timestamp_segment_time).cpu().numpy()

        timestamp_output = self.aligner_processor.parse_timestamp(word_list, timestamp_ms)
        for it in timestamp_output:
            it["start_time"] = round(it["start_time"] / 1000.0, 3)
            it["end_time"] = round(it["end_time"] / 1000.0, 3)

        logger.success("\n输出:")
        for w in timestamp_output:
            logger.success(f"{w['text']:10s} {w['start_time']:6.3f}  {w['end_time']:6.3f}")

        total_time = time.perf_counter() - total_start

        # Performance statistics
        logger.success("=" * 60)
        logger.success("Performance Statistics:")
        logger.success("=" * 60)
        logger.success(f"Audio duration: {audio_duration * 1000:.2f} ms ({audio_duration:.2f} s)")
        logger.success(f"Audio loading time: {audio_load_time * 1000:.3f} ms")
        logger.success(f"Input preparation time: {prep_time * 1000:.3f} ms")
        logger.success(f"Encode time: {encode_time * 1000:.3f} ms")
        logger.success(f"Prefill time: {prefill_time * 1000:.3f} ms")
        logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
        logger.success("=" * 60)
        return


if __name__ == "__main__":
    args = get_args()

    if HOUMO_TARGET == "xh2":
        qwen3forcealigner = Qwen3ForceAligner(
            args.encode_path,
            args.prefill_path,
            args.processor_dir,
            args.embedding_path
        )
    else:
        raise ValueError("Unsupported houmo target!")

    qwen3forcealigner.run(args.audio, args.text, args.language)
