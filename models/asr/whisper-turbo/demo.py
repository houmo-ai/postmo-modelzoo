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
import sys
import torch
import tcim_lite as tcim
from loguru import logger
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MAX_GEN_LEN = 448
CACHE_MAX_LEN = 1280
HOUMO_TARGET = os.getenv("HOUMO_TARGET")


class HmWhisper:
    def __init__(self, encoder_path, decoder_path, prefill_path):
        super().__init__()
        weight_manager = tcim.runtime.WeightManager(0)
        option1 = tcim.runtime.Option(weight_manager)
        self.encoder = tcim.runtime.load(encoder_path, option=option1)
        logger.info("encoder model loaded")
        option2 = tcim.runtime.Option(weight_manager)
        self.decoder = tcim.runtime.load(decoder_path, option=option2)
        logger.info("decoder model loaded")
        option3 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option3)
        logger.info("prefill model loaded")
        self.encoder_time = 0.0
        self.decoder_time = 0.0
        self.prefill_time = 0.0

    def run_encoder(self, input_features):
        input_features = input_features.numpy()
        self.encoder.set_input(self.encoder.get_input_name(0), input_features)
        start_time = time.time()
        self.encoder.run()
        self.encoder.sync()
        self.encoder_time += time.time() - start_time
        outputs = []
        for i in range(self.encoder.get_num_outputs()):
            output = self.encoder.get_output(self.encoder.get_output_name(i)).numpy()
            outputs.append(torch.tensor(output))
        return outputs

    def run_decoder(self, inputs):
        for input_name, input_data in inputs.items():
            self.decoder.set_input(input_name, input_data.numpy())
        start_time = time.time()
        self.decoder.run()
        self.decoder.sync()
        self.decoder_time += time.time() - start_time
        outputs = []
        for i in range(6):
            outputs.append(
                torch.tensor(
                    self.decoder.get_output(self.decoder.get_output_name(i)).numpy()
                )
            )
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
        outputs = []
        for i in range(6):
            output = self.prefill.get_output(self.prefill.get_output_name(i)).numpy()
            outputs.append(torch.tensor(output))
        return outputs

    def get_input_names(self, model_type):
        if model_type == "encoder":
            self.model = self.encoder
        elif model_type == "decoder":
            self.model = self.decoder
        elif model_type == "prefill":
            self.model = self.prefill
        input_names = []
        for i in range(self.model.get_num_inputs()):
            input_names.append(self.model.get_input_name(i))
        return input_names

    def clear_time(self):
        self.encoder_time = 0.0
        self.decoder_time = 0.0
        self.prefill_time = 0.0


def stream_print(text):
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    sys.stdout.write("\033[1;95m{}".format(text))
    sys.stdout.flush()


# def show_statics():


def asr(whisper, processor, model_config, audio_array):

    # 1. prepare config
    num_heads = 20
    head_dim = 64
    num_decode_layers = 4
    start_time = time.time()

    # get prompt Tokens ID
    sot_id = model_config.decoder_start_token_id
    lang_id = processor.tokenizer.convert_tokens_to_ids("<|zh|>")
    transcribe_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notime_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    eos_id = model_config.eos_token_id

    prompt_tokens = [sot_id, lang_id, transcribe_id, notime_id]

    input_features = processor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features.half()
    enc_out = whisper.run_encoder(input_features)

    # Dynamically obtain sequence length
    enc_seq_len = enc_out[0].shape[2]  # 1500

    # [k0, v0, k1, v1, k2, v2, k3, v3]
    k_list = enc_out[0::2]
    v_list = enc_out[1::2]

    dec_names = whisper.get_input_names("decoder")
    encoder_attention_mask = torch.zeros((1, 1, 1, enc_seq_len), dtype=torch.float16)

    all_tokens_list = []

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
        (1, num_heads, prompt_len, CACHE_MAX_LEN), -65504.0, dtype=torch.float16
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
        dec_names[5]: encoder_attention_mask,
    }

    base_idx = 6
    for l in range(num_decode_layers):
        inputs[dec_names[base_idx + num_decode_layers * 2 + l]] = k_list[l]
        inputs[dec_names[base_idx + num_decode_layers * 3 + l]] = v_list[l]

    # run Prefill
    for i in range(8):
        cache = whisper.decoder.get_dev_input(whisper.decoder.get_input_name(i + 6))
        whisper.prefill.set_dev_input(whisper.prefill.get_input_name(i + 6), cache)
    out = whisper.run_prefill(inputs)

    logits = out[0]

    step += prompt_len

    for i in range(8):
        cache = whisper.prefill.get_dev_output(whisper.prefill.get_output_name(i + 1))
        whisper.decoder.set_dev_input(whisper.decoder.get_input_name(i + 6), cache)
        whisper.decoder.set_dev_output(whisper.decoder.get_output_name(i + 1), cache)

    # === step 2: Decode ===
    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    logger.success("transcription:\n")
    ttft_time = time.time() - start_time
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    while step < MAX_GEN_LEN:
        all_tokens_list.append(next_token)

        current_tensor = torch.tensor([all_tokens_list])
        decoded_text = processor.batch_decode(current_tensor, skip_special_tokens=True)[
            0
        ]
        stream_print(decoded_text)

        if next_token == eos_id:
            break

        input_ids = torch.tensor([[next_token]], dtype=torch.int32)

        mask_atten = torch.zeros(
            ([1, num_heads, 1, CACHE_MAX_LEN]), dtype=torch.float16
        )
        if step + 1 < CACHE_MAX_LEN:
            mask_atten[:, :, :, step + 1 :] = -65504

        inputs = {
            dec_names[0]: input_ids,
            dec_names[1]: torch.tensor([[step]], dtype=torch.int32),
            dec_names[2]: torch.tensor([step], dtype=torch.int32),
            dec_names[3]: torch.tensor([1], dtype=torch.int32),
            dec_names[4]: mask_atten,
            dec_names[5]: encoder_attention_mask,
        }

        base_idx = 6
        for l in range(num_decode_layers):
            inputs[dec_names[base_idx + num_decode_layers * 2 + l]] = k_list[l]
            inputs[dec_names[base_idx + num_decode_layers * 3 + l]] = v_list[l]

        out = whisper.run_decoder(inputs)
        logits = out[0]
        k_cache = out[1 : 1 + num_decode_layers]
        v_cache = out[1 + num_decode_layers : 1 + 2 * num_decode_layers]

        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        step += 1
    print("\033[0m")
    return len(decoded_text), ttft_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="Whisper-large-v3-turbo")
    parser.add_argument(
        "--encoder_path",
        dest="encoder_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "whisper_encoder.hmm"),
        help="houmo encoder model path",
    )
    parser.add_argument(
        "--decoder_path",
        dest="decoder_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "whisper_decoder.hmm"),
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
        "--audio",
        type=str,
        default="../../../data/audio/audio.mp3",
    )
    args = parser.parse_args()

if __name__ == "__main__":
    whisper = HmWhisper(args.encoder_path, args.decoder_path, args.prefill_path)
    processor = WhisperProcessor.from_pretrained(args.tokenizer_path)
    model_config = WhisperForConditionalGeneration.from_pretrained(
        args.tokenizer_path
    ).config
    audio_array, _ = sf.read(args.audio)
    output_tokens, ttft_time = asr(whisper, processor, model_config, audio_array)
    logger.success(
        f"Output {output_tokens} tokens, Decode Cost {whisper.decoder_time*1000:.3f} ms"
    )
    logger.success(f"Decode Speed: { output_tokens/ whisper.decoder_time:.2f} tokens/s")
    logger.success(f"TTFT (Time to First Token): {ttft_time * 1000:.3f} ms")
    logger.success(
        f"TPOT (Time Per Output Token): {whisper.decoder_time * 1000 / output_tokens:.3f} ms/token"
    )
    logger.success(
        f"E2E Latency (End-to-End Latency): {(ttft_time + whisper.decoder_time):.3f} seconds"
    )
    logger.success(
        f"E2E TPS (End-to-End Tokens Per Second): {output_tokens / (ttft_time + whisper.decoder_time):.2f} tokens/s"
    )
    whisper.clear_time()
