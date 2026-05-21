#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Qwen3-Reranker Inference Demo - Python script for running Qwen3-Reranker
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
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from loguru import logger

import tcim_lite as tcim
from hmatc.python.get_hm_devices import get_hm_devices
from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_tokenizer_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "qwen3-reranker")
    model_size = model_config.get("model_size", "8b")
    return f"{model_name}-{model_size}"


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output


def process_inputs(pairs, tokenizer):
    max_length = 8192
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(
        inputs, padding=True, return_tensors="pt", max_length=max_length
    )
    return inputs


def compute_logits(batch_scores):
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


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
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default=None,
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
        default=None,
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number",
    )
    args = parser.parse_args()
    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    if args.tokenizer_dir is None:
        args.tokenizer_dir = get_default_tokenizer_dir(model_config)
    if args.prefill_path is None:
        args.prefill_path = os.path.join(
            "output",
            HOUMO_TARGET,
            f"{args.model_name}-{args.model_size}_prefill.hmm",
        )
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

    def __init__(self, prefill_path, embedding_path, ndevice=1):
        self.ndevice = ndevice
        dev_manager = tcim.runtime.DevManager(
            get_hm_devices(self.ndevice), "Xh2HalBackend"
        )
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option1 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option1)
        logger.info("prefill model loaded")
        self.nblocks = self.get_nblocks()

        self.prefill_length = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        self.embedding_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]
        self.context_max_length = self.prefill.get_input_info(
            self.prefill.get_input_name(3)
        ).shape[2]

        embedding_weight = torch.load(embedding_path, map_location="cpu")
        if HOUMO_TARGET == "xh2":
            embedding_weight = embedding_weight["weight"]
        self.embedding_weight = embedding_weight.reshape(-1, self.embedding_len).float()
        self.context_length = 0

    def get_nblocks(self):
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def chat(self, all_input_ids):
        self.generated_ids = []
        self.context_length = 0
        self.prefill_time = 0
        start_time = time.time()
        input_echo_len = all_input_ids.numel()
        if input_echo_len >= self.context_max_length:
            logger.error(
                f"Question long than {self.context_max_length}, please shorten it!"
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
            prefill_start = time.time()
            self.prefill.run()
            self.prefill.sync()
            self.prefill_time += time.time() - prefill_start

        output_data = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        prefill_time = time.time() - start_time

        print("\033[0m")

        return output_data, input_echo_len, prefill_time


if __name__ == "__main__":
    args = get_args()
    hmqwen = HmQwen(
        args.prefill_path,
        args.embedding_path,
        args.ndevice,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, trust_remote_code=True, padding_side="left"
    )

    task = "Given a web search query, retrieve relevant passages that answer the query"

    start_time = time.time()
    queries = [
        "Explain gravity",
        # "What is the capital of China?",
    ]

    documents = [
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        # "The capital of China is Beijing.",
    ]

    pairs = [
        format_instruction(task, query, doc) for query, doc in zip(queries, documents)
    ]
    inputs = process_inputs(pairs, tokenizer)
    output_data, input_tokens, prefill_time = hmqwen.chat(
        inputs["input_ids"][0].unsqueeze(0)
    )
    batch_scores = torch.stack([torch.from_numpy(output_data)]).squeeze().unsqueeze(0)

    # Tokenize the input text
    scores = compute_logits(batch_scores)
    total_time = time.time() - start_time

    logger.success("scores:")
    print("\033[1;95m{}".format(scores))
    show_statistics(input_tokens, prefill_time, total_time)
