#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Qwen3 Inference Demo - Python script for running Qwen3
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
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from loguru import logger

import tcim_lite as tcim

from hmatc.utils.perf_infomations import InferencePerformanceTracker, InferenceMetrics, PERFTYPE
from hmatc.python.get_hm_devices import get_hm_devices

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", "Only support HOUMO_TARGET: xh2."


def is_valid_char(cp):
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x20000 and cp <= 0x2A6DF)
        or (cp >= 0x2A700 and cp <= 0x2B73F)
        or (cp >= 0x2B740 and cp <= 0x2B81F)
        or (cp >= 0x2B820 and cp <= 0x2CEAF)
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
        or (0x0041 <= cp and cp <= 0x005A)
        or (0x0061 <= cp and cp <= 0x007A)
    ):
        return True

    return False


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default="qwen3-30b-a3b",
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
        "--ndevice",
        dest="ndevice",
        type=int,
        default=1,
        choices=[1, 2],
        help="device number, only xh2 support",
    )
    parser.add_argument(
        "--question",
        dest="question",
        type=str,
        default="请介绍一下存算一体技术的优势",
        help="question to ask",
    )
    args = parser.parse_args()
    if args.ndevice > 1:
        args.prefill_path = (
            args.prefill_path.replace(".hmm", ".hmms")
            if args.prefill_path.endswith(".hmm")
            else args.prefill_path
        )
        args.decode_path = (
            args.decode_path.replace(".hmm", ".hmms")
            if args.decode_path.endswith(".hmm")
            else args.decode_path
        )
    return args


class HmQwenXh2:

    def __init__(
        self, prefill_path, decode_path, embedding_path, tokenizer_dir, ndevice
    ):
        self.perf_tracker = InferencePerformanceTracker()
        self.ndevice = ndevice
        dev_manager = tcim.runtime.DevManager(get_hm_devices(self.ndevice), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self.prefill = tcim.runtime.load(prefill_path, option=option1)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)

        logger.info("prefill model loaded")
        self.nblocks = self.get_nblocks()
        dummy_tensor_names = [
            f"model_layers_{i}_self_attn_kcache_input" for i in range(self.nblocks)
        ]
        dummy_tensor_names += [
            f"model_layers_{i}_self_attn_vcache_input" for i in range(self.nblocks)
        ]
        option2.set_dummy_tensors(dummy_tensor_names)
        self.perf_tracker.perf_start(PERFTYPE.DECODE_LOAD_TIME)
        self.decode = tcim.runtime.load(decode_path, option=option2)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_LOAD_TIME)
        logger.info("decode model loaded")
        self.prefill_length = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        self.embedding_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]
        self.context_max_length = self.decode.get_input_info(
            self.decode.get_input_name(4)
        ).shape[2]
        self.batch = self.decode.get_input_info(self.decode.get_input_name(0)).shape[0]
        self.pre_input_num = 4 if self.prefill.get_num_inputs() > 99 else 3
        for i in range(self.pre_input_num, 2 * self.nblocks + self.pre_input_num):
            cache = self.prefill.get_input(self.prefill.get_input_name(i))
            self.decode.set_input(self.decode.get_input_name(i), cache)
        # set decode input
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode.get_input_name(2)
        self.decode.set_input(decode_current_length_name, current_length_input_1)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )
        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=True
        )["weight"]
        self.embedding_weight = embedding_weight.reshape(-1, self.embedding_len)
        self.perf_tracker.reset_perf_time()

    def get_nblocks(self):
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def chat(self, question):
        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOKEN_TIME)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": question,
            },
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        text = self.tokenizer.batch_decode(inputs.input_ids)[0]
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOKEN_TIME)

        if input_echo_len >= self.context_max_length:
            logger.error(
                f"Question long than {self.context_max_length}, please shorten it!"
            )
            sys.exit(1)

        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        for round in range(prefill_loop_round):
            valid_length = round * self.prefill_length
            if round == prefill_loop_round - 1:
                current_length = input_echo_len - round * self.prefill_length
                input_ids = all_input_ids[
                    :, round * self.prefill_length : input_echo_len
                ]
            else:
                current_length = self.prefill_length
                input_ids = all_input_ids[
                    :, round * self.prefill_length : (round + 1) * self.prefill_length
                ]

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
            inputs_embeds = F.embedding(input_ids, self.embedding_weight)
            effective_length = input_ids.size(-1)
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
            position_id = torch.arange(
                valid_length, valid_length + self.prefill_length, dtype=torch.long
            )
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)

            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")
            input_name = self.prefill.get_input_name(0)
            valid_length_name = self.prefill.get_input_name(1)
            current_length_name = self.prefill.get_input_name(2)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
            self.prefill.set_input(input_name, input_data.numpy())
            self.prefill.set_input(valid_length_name, valid_length_data)
            self.prefill.set_input(current_length_name, current_length_data)
            if self.pre_input_num > 3:
                position_id_data = np.expand_dims(np.array(position_id), axis=0).astype(
                    "int32"
                )
                position_id_name = self.prefill.get_input_name(3)
                self.prefill.set_input(position_id_name, position_id_data)
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
            self.prefill.run()
            self.prefill.sync()
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
        input_data = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)

        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)

        next_id = input_data.argmax(-1)[0]
        prefill_response = self.tokenizer.decode(next_id)
        chat_history_ids = all_input_ids[0]
        next_id = torch.from_numpy(next_id)

        chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
        input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(
            1, 1, -1
        )
        self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)

        all_response = prefill_response
        context_length = input_echo_len
        logger.success("response:")
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)

        decode_count = 0
        skip_tokens = 0
        slide_len = 10  # sliding window length for decode
        last_response = self.tokenizer.decode(chat_history_ids.tolist()[-slide_len:])

        while True:
            if context_length >= self.context_max_length:
                logger.info(
                    f"context length greater than {self.context_max_length}, break!"
                )
                break

            self.perf_tracker.perf_start(PERFTYPE.DECODE_TOTAL_TIME)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
            input_name = self.decode.get_input_name(0)
            valid_length_name = self.decode.get_input_name(1)
            self.decode.set_input(input_name, input_data.numpy())
            valid_length_data = np.array(context_length).astype("int32")
            self.decode.set_input(valid_length_name, valid_length_data)
            if self.pre_input_num > 3:
                position_id_data = np.array([[context_length + 1]]).astype("int32")
                position_id_name = self.decode.get_input_name(3)
                self.decode.set_input(position_id_name, position_id_data)
            self.perf_tracker.perf_end(PERFTYPE.DECODE_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_INFER_TIME)
            self.decode.run()
            self.decode.sync()
            self.perf_tracker.perf_end(PERFTYPE.DECODE_INFER_TIME)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_OUTPUT_TIME)
            input_data = self.decode.get_output(self.decode.get_output_name(0)).numpy()
            self.perf_tracker.perf_end(PERFTYPE.DECODE_OUTPUT_TIME)

            decode_count += 1

            next_id = input_data.astype(np.float32).argmax(-1)[0]
            next_id = torch.from_numpy(next_id)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_TOKEN_TIME)

            if next_id == self.tokenizer.eos_token_id:
                if "decode_response" in locals():
                    print(decode_response, end="", flush=True)
                    all_response += decode_response
                self.perf_tracker.perf_end(PERFTYPE.DECODE_TOKEN_TIME)
                self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)
                break

            chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)
            decode_response = self.tokenizer.decode(
                chat_history_ids.tolist()[-(slide_len + 1) - skip_tokens :]
            )[len(last_response) :]
            self.perf_tracker.perf_end(PERFTYPE.DECODE_TOKEN_TIME)

            self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)

            # Validate and print decoded text (outside timing scope)
            if decode_response != "" and is_valid_char(ord(decode_response[-1])):
                print(decode_response, end="", flush=True)
                all_response += decode_response
                last_response = self.tokenizer.decode(
                    chat_history_ids.tolist()[-slide_len:]
                )
                skip_tokens = 0
            else:
                skip_tokens += 1

            self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
            input_data = F.embedding(
                next_id.unsqueeze(0), self.embedding_weight
            ).reshape(1, 1, -1)
            self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)

            context_length = context_length + 1

        print("\033[0m")

        # Set basic performance metrics for reporting
        self.perf_tracker.set_basic_info(
            batch_size=1,
            input_seq_length=input_echo_len,
            output_seq_length=decode_count,
        )


if __name__ == "__main__":

    args = get_args()
    question = args.question

    hmqwen = HmQwenXh2(
        args.prefill_path,
        args.decode_path,
        args.embedding_path,
        args.tokenizer_dir,
        args.ndevice,
    )
    hmqwen.chat(question)
    hmqwen.perf_tracker.show_summary()
