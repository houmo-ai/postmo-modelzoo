#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Qwen3-Embedding Demo - Python script for running Qwen3-Embedding
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

HOUMO_TARGET = os.getenv("HOUMO_TARGET")


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default="qwen3-embedding-8b",
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
        default=os.path.join("output", HOUMO_TARGET, "qwen3-embedding_prefill.hmm"),
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--context_max_length",
        dest="context_max_length",
        type=int,
        default=2048,
        help="context max length",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=1,
        choices=[1, 2],
        help="device number, only xh2 support",
    )
    args = parser.parse_args()
    if args.ndevice > 1:
        args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
    return args


def show_statistics(input_tokens, prefill_time, total_time):
    logger.success(
        f"Total Input: {input_tokens} tokens, Prefill Cost {prefill_time*1000:.3f} ms"
    )
    logger.success(f"Prefill Speed: {input_tokens / prefill_time:.2f} tokens/s")
    logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")


class HmQwen:
    """Main class for Qwen model inference with Houmo backend and performance tracking."""

    def __init__(self, prefill_path, embedding_path, tokenizer_dir, ndevice=1):
        self.ndevice = ndevice
        # Initialize device and weight manager based on device count
        if self.ndevice == 1:
            weight_manager = tcim.runtime.WeightManager(0)
        elif self.ndevice == 2 and HOUMO_TARGET == "xh2":
            dev_manager = tcim.runtime.DevManager([0, 1], "Xh2HalBackend")
            weight_manager = tcim.runtime.WeightManager(dev_manager)
        else:
            raise ValueError(
                "Unsupported device number! Only 1 or 2 devices are supported for xh2"
            )

        option1 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option1)
        logger.info("Prefill model loaded successfully")
        self.nblocks = self.get_nblocks()

        # Get model dimension information from input metadata
        self.prefill_length = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        self.embedding_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]
        self.batch = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[
            0
        ]

        # Load tokenizer and embedding weights
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, padding_side="left"
        )
        embedding_weight = torch.load(embedding_path, map_location="cpu")
        if HOUMO_TARGET == "xh2":
            embedding_weight = embedding_weight["weight"]
        self.embedding_weight = embedding_weight.reshape(-1, self.embedding_len).float()
        self.context_length = 0
        self.prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
        self.context_max_length = args.context_max_length

    def get_nblocks(self):
        """Calculate number of transformer blocks from input tensor names."""
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def encode(self, question, prompt_name="document"):
        self.generated_ids = []
        self.context_length = 0
        self.prefill_time = 0
        start_time = time.time()
        logger.success(prompt_name + ":")
        print("\033[1;95m{}\033[0m".format(question))

        if prompt_name == "query":
            text = self.prompt + question
        else:
            text = question

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.context_max_length,
            return_tensors="pt",
        )
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()

        if input_echo_len >= self.context_max_length:
            logger.error(
                f"Input sequence length ({input_echo_len}) exceeds maximum context length ({self.context_max_length}), please shorten your question!"
            )
            sys.exit(1)

        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        for round in range(prefill_loop_round):
            valid_length = round * self.prefill_length + self.context_length
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

            self.prefill.set_input(input_name, input_data.numpy())
            self.prefill.set_input(valid_length_name, valid_length_data)
            self.prefill.set_input(current_length_name, current_length_data)
            self.prefill.run()
            self.prefill.sync()

        output_data = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        embeddings = last_token_pool(
            torch.tensor(output_data), inputs["attention_mask"]
        )
        prefill_time = time.time() - start_time
        return embeddings, input_echo_len, prefill_time


if __name__ == "__main__":
    # Parse command line arguments and initialize model
    args = get_args()
    hmqwen = HmQwen(
        args.prefill_path,
        args.embedding_path,
        args.tokenizer_dir,
        args.ndevice,
    )

    embeddings = []
    start_time = time.time()
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    prefill_times = 0
    input_tokens = 0
    for query in queries:
        output_data, input_token, prefill_time = hmqwen.encode(
            query, prompt_name="query"
        )
        embeddings.append(output_data)
        input_tokens += input_token
        prefill_times += prefill_time

    for document in documents:
        output_data, input_token, prefill_time = hmqwen.encode(document)
        embeddings.append(output_data)
        input_tokens += input_token
        prefill_times += prefill_time

    embeddings = torch.stack(embeddings).squeeze()
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = embeddings[:2] @ embeddings[2:].T
    total_time = time.time() - start_time
    logger.success("scores:")
    print("\033[1;95m{}".format(scores.tolist()))
    show_statistics(input_tokens, prefill_time, total_time)
