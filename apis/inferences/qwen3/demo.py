#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   HmQwenXh2 Model Inference Script - A script for running inference with the Houmo AI Qwen3-XH2 model using TCIM runtime.
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
import sys
import math
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from typing_extensions import override

from qwen_demo_utils import show_statistics, is_valid_char
from qwen_base import HmQwenBase
from sampling_manager import SamplingManager

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the script."""
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
        default=os.path.join("hmquant", "quant_embedding.pt"),
        help="houmo embedding weight path",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default="qwen3_prefill.hmm",
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default="qwen3_decode.hmm",
        help="houmo decode model path",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=1,
        choices=[1, 2],
        help="device number",
    )
    parser.add_argument(
        "--repetition_penalty",
        dest="repetition_penalty",
        type=float,
        default=1.0,
        help="sampling repetition_penalty",
    )
    parser.add_argument(
        "--topk",
        dest="topk",
        type=int,
        default=None,
        help="sampling top-k",
    )
    parser.add_argument(
        "--topp",
        dest="topp",
        type=float,
        default=1.0,
        help="sampling top-p",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=1.0,
        help="sampling temperature",
    )
    parser.add_argument(
        "--power_opt",
        action="store_true",
        default=False,
        help="power consumption optimization",
    )

    args = parser.parse_args()
    if args.ndevice > 1:
        args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        args.decode_path = args.decode_path.replace(".hmm", ".hmms")
    return args


class HmQwenXh2(HmQwenBase):
    """
    Houmo Qwen XH2 Model implementation that inherits from HmQwenBase.
    This class handles both prefill and decode phases of text generation.
    """

    def __init__(
        self,
        prefill_path,
        decode_path,
        embedding_path,
        tokenizer_dir,
        ndevice,
        **kwargs,
    ):
        """
        Initialize the HmQwenXh2 model with specified paths and parameters.

        Args:
            prefill_path (str): Path to the prefill model file
            decode_path (str): Path to the decode model file
            embedding_path (str): Path to the embedding weights file
            tokenizer_dir (str): Directory containing tokenizer files
            ndevice (int): Number of devices to use
            **kwargs: Additional keyword arguments for sampling parameters
        """
        super().__init__(
            prefill_path=prefill_path,
            decode_path=decode_path,
            embedding_path=embedding_path,
            tokenizer_dir=tokenizer_dir,
            ndevice=ndevice,
        )

        # Initialize sampling manager with provided parameters
        self.samplingmanager = SamplingManager(
            temperature=kwargs.get("temperature", 1.0),
            top_k=kwargs.get("topk", None),
            top_p=kwargs.get("topp", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        )

        # Default system prompt for the conversation
        self.default_prompt = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    @override
    def chat(self, question):
        """
        Perform a complete chat interaction with the model.

        Args:
            question (str): The user's input question

        Returns:
            tuple: (complete response string, input token count, output token count)
        """
        # Reset tracking variables for this chat session
        self.generated_ids = []
        self.context_length = 0
        self.prefill_time = 0
        self.decode_time = 0
        self.ttft_time = 0

        # Log the input question with color formatting
        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))

        start_time = time.time()
        # Format the conversation with chat template
        messages = self.default_prompt + [{"role": "user", "content": question}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        # Tokenize the formatted text
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        # text = self.tokenizer.batch_decode(inputs.input_ids)[0]

        # Extract input token IDs and calculate lengths
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()
        if input_echo_len >= self.context_max_length:
            logger.error(
                f"Question long than {self.context_max_length}, please shorten it!"
            )
            sys.exit(1)

        # Process input in chunks for prefill phase
        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        for round_idx in range(prefill_loop_round):
            # Calculate valid length and current chunk size
            valid_length = round_idx * self.prefill_length + self.context_length
            if round_idx == prefill_loop_round - 1:
                # Last chunk: process remaining tokens
                current_length = input_echo_len - round_idx * self.prefill_length
                input_ids = all_input_ids[
                    :, round_idx * self.prefill_length : input_echo_len
                ]
            else:
                current_length = self.prefill_length
                input_ids = all_input_ids[
                    :,
                    round_idx
                    * self.prefill_length : (round_idx + 1)
                    * self.prefill_length,
                ]

            # Get embeddings for the current chunk
            inputs_embeds = F.embedding(input_ids, self.embedding_weight)
            effective_length = input_ids.size(-1)
            # Create padding embeddings for incomplete chunks
            _pad_embeds = torch.zeros(
                1,
                self.prefill_length - effective_length,
                inputs_embeds.size(-1),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            # Concatenate actual embeddings with padding
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(
                1, self.prefill_length, self.embedding_len
            )
            # Prepare additional input data
            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")

            # Set inputs to prefill model and run inference
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

        # Get the first generated token after prefill
        input_data = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        next_id = input_data.argmax(-1)[0]
        prefill_response = self.tokenizer.decode(next_id)
        logger.success("response:")
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)
        self.ttft_time = time.time() - start_time

        chat_history_ids = all_input_ids[0]
        next_id = torch.from_numpy(next_id)
        self.generated_ids.append(next_id)
        chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)

        input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(
            1, 1, -1
        )
        all_response = prefill_response
        self.context_length += input_echo_len

        # Start the decoding loop
        decode_count = 0
        skip_tokens = 0
        slide_len = 10  # sliding window length for decode
        last_response = self.tokenizer.decode(chat_history_ids.tolist()[-slide_len:])

        while True:
            if self.context_length >= self.context_max_length:
                logger.info(
                    f"context length greater than {self.context_max_length}, break!"
                )
                break

            # Set inputs to decode model and run inference
            input_name = self.decode.get_input_name(0)
            valid_length_name = self.decode.get_input_name(1)
            self.decode.set_input(input_name, input_data.numpy())
            valid_length_data = np.array(self.context_length).astype("int32")
            self.decode.set_input(valid_length_name, valid_length_data)

            decode_start = time.time()
            self.decode.run()
            self.decode.sync()
            self.decode_time += time.time() - decode_start

            # Get output from decode model
            input_data = self.decode.get_output(self.decode.get_output_name(0)).numpy()
            decode_count += 1
            # Sample the next token ID
            next_id = (
                self.samplingmanager.sample(input_data, self.generated_ids)
                .squeeze()
                .reshape(1)
            )
            next_id = torch.from_numpy(next_id)
            # Check if end-of-sequence token was generated
            if next_id == self.tokenizer.eos_token_id:
                print(decode_response, end="", flush=True)
                all_response += decode_response
                break

            chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)
            self.generated_ids.append(next_id)
            decode_response = self.tokenizer.decode(
                chat_history_ids.tolist()[-(slide_len + 1) - skip_tokens :]
            )[len(last_response) :]
            # Print response only if it contains valid characters
            if decode_response != "" and is_valid_char(ord(decode_response[-1])):
                print(decode_response, end="", flush=True)
                all_response += decode_response
                last_response = self.tokenizer.decode(
                    chat_history_ids.tolist()[-slide_len:]
                )
                skip_tokens = 0
            else:
                skip_tokens += 1

            # Prepare input for next iteration
            input_data = F.embedding(
                next_id.unsqueeze(0), self.embedding_weight
            ).reshape(1, 1, -1)
            self.context_length = self.context_length + 1

        print("\033[0m")

        return all_response, input_echo_len, decode_count + 1


if __name__ == "__main__":
    args = get_args()
    enable_power_opt = args.power_opt

    hmqwen = HmQwenXh2(
        args.prefill_path,
        args.decode_path,
        args.embedding_path,
        args.tokenizer_dir,
        args.ndevice,
        repetition_penalty=args.repetition_penalty,
        topk=args.topk,
        topp=args.topp,
        temperature=args.temperature,
    )

    question = "请介绍一下存算一体技术的优势" 

    dev = None
    pre_h = None
    post_h = None

    if enable_power_opt:
        import gc
        import tcim_lite
        hw = tcim_lite.dev_ctrl

        def pre_reset(device_id:int):
            global hmqwen
            if hmqwen is not None:
                hmqwen = None
                gc.collect()
                # Running twice is a defensive memory cleanup practice, not redundant logic.
                gc.collect()

        def post_reset(device_id:int):
            global hmqwen
            hmqwen = HmQwenXh2(
                args.prefill_path, args.decode_path, args.embedding_path,
                args.tokenizer_dir, args.ndevice,
                repetition_penalty=args.repetition_penalty,
                topk=args.topk, topp=args.topp, temperature=args.temperature,
            )

        dev = hw.HalDeviceFactory.create("Xh2aHalBackend")
        pre_h = dev.register_pre_reset_callback(0, pre_reset)
        post_h = dev.register_post_reset_callback(0, post_reset)

    start_time = time.time()
    try:
        response, input_tokens, output_tokens = hmqwen.chat(question)
        total_time = time.time() - start_time
        show_statistics(
            input_tokens,
            output_tokens,
            hmqwen.ttft_time,
            hmqwen.prefill_time,
            hmqwen.decode_time,
            total_time,
        )


        if enable_power_opt:
            # Set DVFS mode to ON_DEMAND for power optimization and log device statistics
            status, mode = dev.get_dvfs_mode(0)
            if status == 0:
                status  = dev.set_dvfs_mode(0, hw.DvfsMode.ON_DEMAND)
                if status != 0:
                    logger.error(f"Failed to set DVFS mode, status: {status}")
            else:
                logger.error(f"Failed to get DVFS mode, status: {status}")

            # Log IPU frequency, memory info, and utilization for power demonstration
            status, freq = dev.get_ipu_frequency(0)
            status, mem_info = dev.get_mem_info(0)
            status,util = dev.get_ipu_util_rate(0)


            print(f"IPU frequency: {freq} MHz,mem_total: {mem_info.mem_total},    mem_avail: {mem_info.mem_avail},mem_used: {mem_info.mem_used},IPU utilization: {util} %")
            dev.ipu_reset(0)

    except Exception as e:
        logger.error(f"Error occurred during the chat: {str(e)}")

    finally:
        if enable_power_opt:
            dev.unregister_reset_callback(pre_h)
            dev.unregister_reset_callback(post_h)
