#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: perf.py
# Description:
#   Qwen3 Perf test Demo - Python script for running Qwen3
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
import re
import sys
import math
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from loguru import logger

import tcim_lite as tcim

from hmatc.utils.perf_infomations import InferencePerformanceTracker, InferenceMetrics, PERFTYPE

HOUMO_TARGET = os.getenv("HOUMO_TARGET")


def generate_random_digit_string(length=1000):
    random_digits = [str(random.randint(0, 9)) for _ in range(length)]
    return "".join(random_digits)


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default="qwen3-14b",
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
        default=os.path.join("output", HOUMO_TARGET, "qwen3_prefill.hmm"),
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3_decode.hmm"),
        help="houmo decode model path",
    )
    parser.add_argument(
        "--isl",
        dest="isl",
        type=int,
        default=1024,
        help="input seq length",
    )
    parser.add_argument(
        "--osl",
        dest="osl",
        type=int,
        default=1024,
        help="output seq length",
    )
    args = parser.parse_args()
    return args


class HmQwenXh2:
    def __init__(
        self, prefill_path, decode_path, embedding_path, tokenizer_dir, isl, osl
    ):
        self.perf_tracker = InferencePerformanceTracker()
        weight_manager = tcim.runtime.WeightManager(0)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self.prefill = tcim.runtime.load(prefill_path, option=option1)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_LOAD_TIME)
        self.decode = tcim.runtime.load(decode_path, option=option2)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_LOAD_TIME)

        self.nblocks = self.get_nblocks()
        dummy_tensor_names = [
            f"model_layers_{i}_self_attn_kcache_input" for i in range(self.nblocks)
        ]
        dummy_tensor_names += [
            f"model_layers_{i}_self_attn_vcache_input" for i in range(self.nblocks)
        ]
        option1.set_dummy_tensors(dummy_tensor_names)
        self.prefill_length = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        self.embedding_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]
        self.context_max_length = self.decode.get_input_info(
            self.decode.get_input_name(3)
        ).shape[2]
        self.batch = self.decode.get_input_info(self.decode.get_input_name(0)).shape[0]
        self.next_ids = [0] * self.batch
        self.current_echo_lens = [0] * self.batch

        # Set decode input
        for b in range(self.batch):
            index = 2 if b == 0 else 2 * self.nblocks * b + 3 + 2 * b - 1
            current_length_input = np.array([1]).astype("int32")
            self.decode.set_input(
                self.decode.get_input_name(index), current_length_input
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )

        # Load embedding weights
        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=True
        )["weight"]
        self.embedding_weight = embedding_weight.reshape(-1, self.embedding_len)

        if isl + osl >= self.context_max_length:
            logger.error(
                f"Context length exceeds maximum limit {self.context_max_length}, please reduce input/output sequence length!"
            )
            sys.exit(1)
        self.perf_tracker.reset_perf_time()

    def get_nblocks(self):
        input_names = []
        for i in range(self.decode.get_num_inputs()):
            input_names.append(self.decode.get_input_name(i))
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def preprocess_prefill(self, isl):
        text = generate_random_digit_string(isl)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOKEN_TIME)
        input = self.tokenizer(text, return_tensors="pt")
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOKEN_TIME)
        all_input_id = input["input_ids"]
        return all_input_id

    def run_prefill(self, b, all_input_id):
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)

        decode_input_index_start = 2 * self.nblocks * b + 3 + 2 * b if b > 0 else 3
        decode_input_index_finish = 2 * self.nblocks * (b + 1) + 3 + b * 2
        prefill_input_index = 3

        for i in range(decode_input_index_start, decode_input_index_finish):
            cache = self.decode.get_dev_input(self.decode.get_input_name(i))
            self.prefill.set_input(
                self.prefill.get_input_name(prefill_input_index), cache
            )
            prefill_input_index += 1

        input_echo_len = all_input_id.numel()
        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)

        for round in range(prefill_loop_round):
            valid_length = round * self.prefill_length
            if round == prefill_loop_round - 1:
                current_length = input_echo_len - round * self.prefill_length
                input_ids = all_input_id[
                    :, round * self.prefill_length : input_echo_len
                ]
            else:
                current_length = self.prefill_length
                input_ids = all_input_id[
                    :, round * self.prefill_length : (round + 1) * self.prefill_length
                ]

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
            inputs_embeds = F.embedding(input_ids, self.embedding_weight)
            effective_length = input_ids.size(-1)
            # Padding alignment for prefill (required by model input shape)
            _pad_embeds = torch.zeros(
                1,
                self.prefill_length - effective_length,
                inputs_embeds.size(-1),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(
                1, self.prefill_length, self.embedding_len
            )
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)

            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")

            input_name = self.prefill.get_input_name(0)
            valid_length_name = self.prefill.get_input_name(1)
            current_length_name = self.prefill.get_input_name(2)
            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
            self.prefill.set_input(input_name, input_data.float().numpy())
            self.prefill.set_input(valid_length_name, valid_length_data)
            self.prefill.set_input(current_length_name, current_length_data)
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
            self.prefill.run()
            self.prefill.sync()
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
        input_data = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)

        next_id = input_data.argmax(-1)[0]
        next_id = torch.from_numpy(next_id)

        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)

        return input_echo_len, next_id

    def run_decode(self, input_datas):
        self.perf_tracker.perf_start(PERFTYPE.DECODE_TOTAL_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
        input_name = self.decode.get_input_name(0)

        input_datas_np = (
            np.concatenate(input_datas, axis=0) if self.batch > 1 else input_datas[0]
        )
        self.decode.set_input(input_name, input_datas_np)

        for b in range(self.batch):
            valid_length_index = 1 if b == 0 else 2 * self.nblocks * b + 3 + 2 * b - 2
            valid_length_data = np.array(self.context_lengths[b]).astype("int32")
            self.decode.set_input(
                self.decode.get_input_name(valid_length_index), valid_length_data
            )
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INFER_TIME)
        self.decode.run()
        self.decode.sync()
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_OUTPUT_TIME)
        output_data = (
            self.decode.get_output(self.decode.get_output_name(0)).to_host().numpy()
        )
        self.perf_tracker.perf_end(PERFTYPE.DECODE_OUTPUT_TIME)

        # Retrieve next_token_id (DECODE_TOTAL_TIME ends here)
        next_ids = [output_data[b].argmax(-1) for b in range(self.batch)]

        self.perf_tracker.perf_start(PERFTYPE.DECODE_TOKEN_TIME)
        for b in range(self.batch):
            token_text = self.tokenizer.decode(next_ids[b])
            message = [{"role": "assistant", "content": token_text}]
            self.tokenizer.apply_chat_template(message, tokenize=False)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_TOKEN_TIME)

        self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)

        return output_data

    def chat(self, isl, osl):
        input_echo_lens = []

        logger.info("Total prefill token count: {}".format(isl))
        for b in range(self.batch):
            all_input_id = self.preprocess_prefill(isl)
            input_echo_len, next_id = self.run_prefill(b, all_input_id)
            self.next_ids[b] = next_id
            self.current_echo_lens[b] = input_echo_len
            input_echo_lens.append(input_echo_len)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
        input_datas = []
        for b in range(self.batch):
            next_id = self.next_ids[b].unsqueeze(0)
            # Convert token id to embedding (no padding for decode stage)
            inputs_embeds = F.embedding(next_id, self.embedding_weight)

            embed_data = inputs_embeds.reshape(1, 1, -1).float().numpy()
            input_datas.append(embed_data)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)

        self.context_lengths = self.current_echo_lens
        count = 0
        logger.info("Total decode token count: {}".format(osl))

        while count < osl - 1:
            output_data = self.run_decode(input_datas)

            # Process output and record DECODE_EMBED_TIME
            self.next_ids = [output_data[b].argmax(-1) for b in range(self.batch)]
            self.next_ids = [torch.from_numpy(next_id) for next_id in self.next_ids]

            input_datas = []
            self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
            for b in range(self.batch):
                self.context_lengths[b] += 1
                next_id = torch.from_numpy(np.asarray(self.next_ids[b])).unsqueeze(0)

                inputs_embeds = F.embedding(next_id, self.embedding_weight)
                embed_data = inputs_embeds.reshape(1, 1, -1).float().numpy()
                input_datas.append(embed_data)
            self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)

            count += 1

        self.perf_tracker.set_basic_info(
            self.batch,
            sum(input_echo_lens),
            count,
        )


if __name__ == "__main__":
    args = get_args()
    if HOUMO_TARGET == "xh2":
        # Initialize with isl and osl parameters
        hmqwen = HmQwenXh2(
            args.prefill_path,
            args.decode_path,
            args.embedding_path,
            args.tokenizer_dir,
            args.isl,
            args.osl,
        )

    hmqwen.chat(args.isl, args.osl)
    hmqwen.perf_tracker.show_summary()
