#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo_multibatch.py
# Description:
#   Qwen3 Inference Demo - Python script for running Qwen3
# automatic speech recognition on HOUMO AI device.
#   Example for Qwen3-8B 16K 4-batch model.
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
from hmatc.python.get_hm_devices import get_hm_devices

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


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
        default="qwen3-8b",
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
        default=os.path.join("output", HOUMO_TARGET, "qwen3-8b_prefill.hmm"),
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3-8b_decode.hmm"),
        help="houmo decode model path",
    )
    parser.add_argument(
        "--forbid_flush",
        dest="forbid_flush",
        action="store_true",
        help="forbid flush print n batch",
    )
    args = parser.parse_args()
    return args


class HmQwen:
    def __init__(
        self,
        prefill_path,
        decode_path,
        embedding_path,
        tokenizer_dir,
        forbid_flush=True,
    ):
        dev_manager = tcim.runtime.DevManager(get_hm_devices(), "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option1)
        logger.info("prefill model loaded")
        self.decode = tcim.runtime.load(decode_path, option=option2)
        logger.info("decode model loaded")
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

        self.flush = not forbid_flush
        self.next_ids = [0] * self.batch
        self.current_questions = [""] * self.batch
        self.current_echo_lens = [0] * self.batch
        self.current_responses = [""] * self.batch
        self.decode_break = [False] * self.batch
        self.chat_history_ids = [[]] * self.batch
        self.colors = [
            {"color": "\033[94m"},
            {"color": "\033[92m"},
            {"color": "\033[91m"},
            {"color": "\033[95m"},
            {"color": "\033[93m"},
            {"color": "\033[96m"},
            {"color": "\033[33m"},
            {"color": "\033[35m"},
        ]
        # set decode input
        for b in range(self.batch):
            index = 2 if b == 0 else 2 * self.nblocks * b + 3 + 2 * b - 1
            current_length_input = np.array([1]).astype("int32")
            self.decode.set_input(
                self.decode.get_input_name(index), current_length_input
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )
        embedding_weight = torch.load(embedding_path, map_location="cpu")
        if HOUMO_TARGET == "xh2":
            embedding_weight = embedding_weight["weight"]
        self.embedding_weight = embedding_weight.reshape(-1, self.embedding_len)

    def get_nblocks(self):
        input_names = []
        for i in range(self.decode.get_num_inputs()):
            input_names.append(self.decode.get_input_name(i))
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def preprocess_prefill(self, question):
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input = self.tokenizer(text, return_tensors="pt")
        all_input_id = input["input_ids"]
        return all_input_id

    def show_response(self):
        if self.flush:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            print("=== HMQwen Multi-batch Inference Demo ===")
            for i in range(self.batch):
                color = self.colors[i]
                status = "✓" if self.decode_break[i] else "●"
                print(
                    f"{color['color']}quesion: {self.current_questions[i]} {status}:\033[0m"
                )
                print(f"  {self.current_responses[i]}")

    def run_prefill(self, b, all_input_id):
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
        if input_echo_len >= self.context_max_length:
            logger.error(
                f"Question long than {self.context_max_length}, please shorten it!"
            )
            sys.exit(1)
        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        prefill_start_time = time.time()
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
            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")

            input_name = self.prefill.get_input_name(0)
            valid_length_name = self.prefill.get_input_name(1)
            current_length_name = self.prefill.get_input_name(2)
            self.prefill.set_input(input_name, input_data.float().numpy())
            self.prefill.set_input(valid_length_name, valid_length_data)
            self.prefill.set_input(current_length_name, current_length_data)
            self.prefill.run()
            self.prefill.sync()

        input_data = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        prefill_time = time.time() - prefill_start_time
        next_id = input_data.argmax(-1)[0]
        prefill_response = self.tokenizer.decode(next_id.tolist())
        next_id = torch.from_numpy(next_id)
        self.chat_history_ids[b] = torch.cat(
            [all_input_id.reshape(-1), next_id], dim=-1
        )
        return prefill_time, input_echo_len, prefill_response, next_id

    def run_decode(self, input_datas):
        input_name = self.decode.get_input_name(0)
        input_datas = np.array(input_datas)
        decode_start_time = time.time()
        self.decode.set_input(input_name, input_datas)
        for b in range(self.batch):
            valid_length_index = 1 if b == 0 else 2 * self.nblocks * b + 3 + 2 * b - 2
            valid_length_data = np.array(self.context_lengths[b]).astype("int32")
            self.decode.set_input(
                self.decode.get_input_name(valid_length_index), valid_length_data
            )
        self.decode.run()
        self.decode.sync()
        input_datas = (
            self.decode.get_output(self.decode.get_output_name(0)).to_host().numpy()
        )
        decode_time = time.time() - decode_start_time
        return decode_time, input_datas

    def chat(self, questions):
        max_decode_count = 0
        all_questions = []
        all_responses = []
        input_echo_lens = []
        all_prefill_time = 0
        all_decode_time = 0
        for b in range(self.batch):
            question = questions.pop(0)
            self.current_questions[b] = question
            all_input_id = self.preprocess_prefill(question)
            prefill_time, input_echo_len, prefill_response, next_id = self.run_prefill(
                b, all_input_id
            )
            all_prefill_time += prefill_time
            self.next_ids[b] = next_id
            self.current_responses[b] = prefill_response
            self.current_echo_lens[b] = input_echo_len
        slide_len = 10  # sliding window length for decode
        skip_tokens = [0] * self.batch
        decode_responses = [""] * self.batch
        last_responses = [
            self.tokenizer.decode(
                self.chat_history_ids[b].tolist()[-slide_len:],
            )
            for b in range(self.batch)
        ]
        input_datas = [
            F.embedding(
                self.next_ids[b].unsqueeze(0),
                self.embedding_weight,
            )
            .reshape(1, -1)
            .float()
            .numpy()
            for b in range(self.batch)
        ]
        self.context_lengths = self.current_echo_lens
        self.show_response()

        while not all(self.decode_break[: self.batch]):
            decode_time, input_datas = self.run_decode(np.array(input_datas))
            max_decode_count += 1
            all_decode_time += decode_time
            self.next_ids = [input_datas[b].argmax(-1) for b in range(self.batch)]
            self.next_ids = [torch.from_numpy(next_id) for next_id in self.next_ids]
            for b in range(self.batch):
                if self.context_lengths[b] + 1 >= self.context_max_length:
                    self.decode_break[b] = True
                if self.next_ids[b] == self.tokenizer.eos_token_id:
                    self.current_responses[b] += decode_responses[b]
                    self.decode_break[b] = True
                else:
                    self.context_lengths[b] += 1
                if not self.decode_break[b]:
                    self.chat_history_ids[b] = torch.cat(
                        [self.chat_history_ids[b], self.next_ids[b]], dim=-1
                    )
                    decode_responses[b] = self.tokenizer.decode(
                        self.chat_history_ids[b].tolist()[
                            -(slide_len + 1) - skip_tokens[b] :
                        ],
                    )[len(last_responses[b]) :]
                else:
                    decode_responses[b] = ""
                if decode_responses[b] != "" and is_valid_char(
                    ord(decode_responses[b][-1])
                ):
                    self.current_responses[b] += decode_responses[b]
                    last_responses[b] = self.tokenizer.decode(
                        self.chat_history_ids[b].tolist()[-slide_len:],
                    )
                    skip_tokens[b] = 0
                else:
                    skip_tokens[b] += 1
            self.show_response()
            for b in range(self.batch):
                if self.decode_break[b] == True and len(questions) != 0:
                    question = questions.pop(0)
                    all_questions.append(self.current_questions[b])
                    all_responses.append(self.current_responses[b])
                    input_echo_lens.append(self.current_echo_lens[b])
                    self.current_questions[b] = question
                    all_input_id = self.preprocess_prefill(question)
                    prefill_time, input_echo_len, prefill_response, next_id = (
                        self.run_prefill(b, all_input_id)
                    )
                    all_prefill_time += prefill_time
                    self.next_ids[b] = next_id
                    self.current_responses[b] = prefill_response
                    self.current_echo_lens[b] = input_echo_len
                    self.context_lengths[b] = input_echo_len
                    self.decode_break[b] = False
                    skip_tokens[b] = 0
                    last_responses[b] = self.tokenizer.decode(
                        self.chat_history_ids[b].tolist()[-slide_len:]
                    )

            input_datas = []
            for b in range(self.batch):
                next_id = torch.from_numpy(np.array(self.next_ids[b]))
                input_datas.append(
                    F.embedding(
                        next_id.unsqueeze(
                            0,
                        ),
                        self.embedding_weight,
                    )
                    .reshape(1, -1)
                    .float()
                    .numpy(),
                )
        sys.stdout.write("\033[0m")
        sys.stdout.flush()
        for b in range(self.batch):
            all_questions.append(self.current_questions[b])
            all_responses.append(self.current_responses[b])
            input_echo_lens.append(self.current_echo_lens[b])
        return (
            self.batch,
            all_questions,
            all_responses,
            sum(input_echo_lens),
            max_decode_count * self.batch,
            all_prefill_time,
            all_decode_time,
        )


if __name__ == "__main__":
    args = get_args()
    hmqwen = HmQwen(
        args.prefill_path,
        args.decode_path,
        args.embedding_path,
        args.tokenizer_dir,
        forbid_flush=args.forbid_flush,
    )
    questions = [
        "1+1=?",
        "你好",
        "写一个冷笑话",
        "请介绍一下时间晶体",
        "请介绍一下存算一体技术的优势",
    ]
    start_time = time.time()
    (
        batch,
        questions,
        responses,
        input_tokens,
        output_tokens,
        prefill_time,
        decode_time,
    ) = hmqwen.chat(questions)
    for i in range(len(questions)):
        print(f"\033[35mQuestion{i}:")
        print(f"\033[96m{questions[i]}")
        print(f"\033[35mResponse{i}:")
        print(f"\033[96m{responses[i]}")

    total_time = time.time() - start_time

    logger.success(
        f"Batch: {batch}, Total Input: {input_tokens} tokens, Output {output_tokens + batch} tokens, Prefill Cost: {prefill_time:.3f} seconds, Decode Cost: {decode_time:.3f} seconds"
    )
    logger.success(
        f"Prefill Speed: {input_tokens / prefill_time:.2f} tokens/s; Decode Speed: {(output_tokens) / decode_time:.2f} tokens/s"
    )
    logger.success(f"TTFT (Time to First Token): {prefill_time * 1000:.3f} ms")
    logger.success(
        f"TPOT (Time Per Output Token): {decode_time * 1000 / (output_tokens):.3f} ms/token"
    )
    logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
    logger.success(
        f"E2E TPS (End-to-End Tokens Per Second): {(output_tokens + batch) / total_time:.2f} tokens/s"
    )
