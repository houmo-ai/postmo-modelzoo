# Copyright (c) 2025 HOUMO AI
#
# File: prefill_pp_decode_pp.py
# Description:
#   Qwen3 Pipeline Parallel Inference Implementation - Implements pipeline parallel inference
#   for the Qwen3 model with multi-chip support using prefill and decode stages.
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

import multiprocessing as mp
import queue
import time
import random
from typing import Any
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
        "--model_path",
        dest="model_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
        help="split hmm model path",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=2,
        choices=[2, 4],
        help="device number, only xh2 support",
    )
    parser.add_argument(
        "--input_tokens",
        dest="input_tokens",
        type=int,
        default=30720,
        help="input_tokens len for perfill",
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
    args = parser.parse_args()
    return args


class SamplingManager:
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        min_tokens_to_keep: int = 1,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.min_tokens_to_keep = min_tokens_to_keep

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            raise ValueError("Temperature must larger than 0")

        return logits / self.temperature

    def apply_repetition_penalty(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        if self.repetition_penalty == 1.0 or not previous_tokens:
            return logits

        adjusted_logits = logits.copy()
        for token_id in set(previous_tokens):
            if 0 <= token_id < len(logits):
                if logits[token_id] < 0:
                    adjusted_logits[token_id] = (
                        logits[token_id] * self.repetition_penalty
                    )
                else:
                    adjusted_logits[token_id] = (
                        logits[token_id] / self.repetition_penalty
                    )

        return adjusted_logits

    def apply_top_k(self, probs: np.ndarray) -> np.ndarray:
        if self.top_k is None or self.top_k <= 0:
            return probs

        top_k = min(self.top_k, len(probs))

        if top_k <= 0:
            return probs

        top_k_indices = np.argpartition(probs, -top_k)[-top_k:]

        mask = np.ones_like(probs, dtype=bool)
        mask[top_k_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0

        if np.sum(filtered_probs) > 0:
            normalized_probs = filtered_probs / np.sum(filtered_probs)
        else:
            normalized_probs = np.ones_like(probs) / len(probs)

        return normalized_probs

    def apply_top_p(self, probs: np.ndarray) -> np.ndarray:
        if self.top_p >= 1.0:
            return probs

        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        cumulative_probs = np.cumsum(sorted_probs)

        cutoff_indices = np.where(cumulative_probs >= self.top_p)[0]

        if len(cutoff_indices) > 0:
            cutoff_index = cutoff_indices[0]
            if cutoff_index < self.min_tokens_to_keep - 1:
                cutoff_index = self.min_tokens_to_keep - 1

            selected_indices = sorted_indices[: cutoff_index + 1]
        else:
            selected_indices = sorted_indices

        mask = np.ones_like(probs, dtype=bool)
        mask[selected_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0

        if np.sum(filtered_probs) > 0:
            normalized_probs = filtered_probs / np.sum(filtered_probs)
        else:
            normalized_probs = np.ones_like(probs) / len(probs)

        return normalized_probs

    def process_logits(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        processed_logits = logits.copy()
        # 1. apply repetition penalty
        processed_logits = self.apply_repetition_penalty(
            processed_logits, previous_tokens
        )

        # 2. apply softmax
        # not using softmax in case of long time cost
        probs = processed_logits
        # probs = self.softmax(processed_logits)

        # 3. apply top-k
        probs = self.apply_top_k(probs)

        # 4. apply top-p
        probs = self.apply_top_p(probs)

        # 5. apply temperature
        probs = self.apply_temperature(probs)
        return probs

    def sample(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> int:
        logits = logits[0]
        if HOUMO_TARGET == "xh2":
            logits = logits[0]
        probs = self.process_logits(logits, previous_tokens)
        if np.all(probs == 0):
            probs = np.ones_like(probs) / len(probs)

        # sampled_index = np.random.choice(len(probs), p=probs)
        sampled_index = probs.argmax(-1)

        return np.array([[sampled_index]])

    def get_processed_probs(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        return self.process_logits(logits, previous_tokens)


def gen_params(device_id: int, args):
    model_path = args.model_path
    prefill_path = os.path.join(
        model_path, f"qwen3-pipeline-8b_prefill_part{device_id}.hmm"
    )
    decode_path = os.path.join(
        model_path, f"qwen3-pipeline-8b_decode_part{device_id}.hmm"
    )
    return {
        "device_id": device_id,
        "prefill_path": prefill_path,
        "decode_path": decode_path,
        "tokenizer_dir": args.tokenizer_dir,
        "embedding_path": args.embedding_path,
        "temperature": args.temperature,
        "top_k": args.topk,
        "top_p": args.topp,
        "repetition_penalty": args.repetition_penalty,
    }


def calculate_append_len(tokenizer, input_tokens):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Hello, ",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    all_input_ids = inputs["input_ids"]
    input_echo_len = all_input_ids.numel()

    return input_tokens - input_echo_len


def producer(
    model_params: dict,
    out_queue: mp.Queue,
    in_queue: mp.Queue,
    question_len: int,
    barrier: mp.Barrier,
):
    """producer init"""
    weight_manager = tcim.runtime.WeightManager(model_params["device_id"])
    option_prefill = tcim.runtime.Option(weight_manager)
    option_decode = tcim.runtime.Option(weight_manager)
    # load model
    prefill_model = tcim.runtime.load(
        model_params["prefill_path"], option=option_prefill
    )
    logger.info(f"Prefill Stage{model_params['device_id']} load model successfully")

    prefill_length = prefill_model.get_input_info(
        prefill_model.get_input_name(0)
    ).shape[1]
    embedding_len = prefill_model.get_input_info(prefill_model.get_input_name(0)).shape[
        2
    ]
    context_max_length = prefill_model.get_input_info(
        prefill_model.get_input_name(3)
    ).shape[2]
    batch = prefill_model.get_input_info(prefill_model.get_input_name(0)).shape[0]

    # Load decode
    dummy_tensor_names = []
    total_layer_num = prefill_model.get_num_inputs()
    for i in range(total_layer_num):
        layer_name = prefill_model.get_input_name(i)
        if "cache" in layer_name:
            dummy_tensor_names.append(layer_name)
        else:
            continue
    option_decode.set_dummy_tensors(dummy_tensor_names)
    decode_model = tcim.runtime.load(model_params["decode_path"], option=option_decode)
    logger.info(f"Decode Stage{model_params['device_id']} load model successfully")

    # Load tokenizer and embedding weights
    tokenizer = AutoTokenizer.from_pretrained(
        model_params["tokenizer_dir"], trust_remote_code=True
    )
    embedding_weight = torch.load(model_params["embedding_path"], map_location="cpu")
    embedding_weight = embedding_weight["weight"]
    embedding_weight = embedding_weight.reshape(-1, embedding_len).float()
    context_length = 0

    logger.success("User question:")
    append_len = calculate_append_len(tokenizer, question_len)
    print("\033[1;95m{}\033[0m".format(f"Hello, '1' *{append_len}"))
    question = "1" * append_len

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Hello, " + question,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    all_input_ids = inputs["input_ids"]
    input_echo_len = all_input_ids.numel()
    logger.info(f"Prefill input tokens length: {input_echo_len}")
    # Validate input length against maximum context length
    if input_echo_len >= context_max_length:
        logger.error(
            f"Input sequence length ({input_echo_len}) exceeds maximum context length ({context_max_length}), please shorten your question!"
        )
        sys.exit(1)

    # Process prefill in chunks if input length exceeds prefill length
    total_time = 0
    prefill_loop_round = math.ceil(input_echo_len / prefill_length)
    barrier.wait()
    for round in range(prefill_loop_round):
        try:
            valid_length = round * prefill_length + context_length
            if round == prefill_loop_round - 1:
                current_length = input_echo_len - round * prefill_length
                input_ids = all_input_ids[:, round * prefill_length : input_echo_len]
            else:
                current_length = prefill_length
                input_ids = all_input_ids[
                    :, round * prefill_length : (round + 1) * prefill_length
                ]

            inputs_embeds = F.embedding(input_ids, embedding_weight)
            effective_length = input_ids.size(-1)
            _pad_embeds = torch.zeros(
                1,
                prefill_length - effective_length,
                inputs_embeds.size(-1),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(
                1, prefill_length, embedding_len
            )

            # Prepare length parameters for prefill input
            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")

            input_name = prefill_model.get_input_name(0)
            valid_length_name = prefill_model.get_input_name(1)
            current_length_name = prefill_model.get_input_name(2)

            prefill_model.set_input(input_name, input_data.numpy())
            prefill_model.set_input(valid_length_name, valid_length_data)
            prefill_model.set_input(current_length_name, current_length_data)

            t0 = time.time()
            prefill_model.run()
            prefill_model.sync()
            t1 = time.time()
            total_time += t1 - t0

            output_name = prefill_model.get_output_name(0)
            output_data = prefill_model.get_output(output_name).numpy()

            data = (output_data, valid_length_data, current_length_data, t0, t1)
            out_queue.put(data)
        except queue.Full:
            logger.error(
                f"[Prefill Stage {model_params['device_id']}] Queue Full, Drop data"
            )

    out_queue.put("END")
    prefill_end_data = (all_input_ids.numpy(), context_max_length, embedding_len)
    out_queue.put(prefill_end_data)
    logger.info(
        f"[Prefill Stage {model_params['device_id']}] Process Finished, time: {total_time:.2f} s."
    )

    # kvcache share
    for i in range(total_layer_num):
        layer_name = prefill_model.get_input_name(i)
        if "cache" in layer_name:
            cache = prefill_model.get_dev_input(layer_name).to_host(True).numpy()
            decode_model.set_input(layer_name, cache)
    total_time = 0
    # decode stage
    while True:
        try:
            data = in_queue.get()
            if data == "END":
                out_queue.put("END", timeout=1)
                break

            input_name = decode_model.get_input_name(0)
            valid_length_name = decode_model.get_input_name(1)
            current_length_name = decode_model.get_input_name(2)
            output_data, valid_length_data, current_length_data, _, _ = data

            decode_model.set_input(input_name, output_data)
            decode_model.set_input(valid_length_name, valid_length_data)
            decode_model.set_input(current_length_name, current_length_data)

            t0 = time.time()
            decode_model.run()
            decode_model.sync()
            t1 = time.time()
            total_time += t1 - t0
            output_data = decode_model.get_output(
                decode_model.get_output_name(0)
            ).numpy()
            result = (output_data, valid_length_data, current_length_data, t0, t1)

            try:
                out_queue.put(result)
            except queue.Full:
                logger.error(
                    f"[Prefill Stage {model_params['device_id']}] Queue Full, Drop data"
                )

        except queue.Empty:
            continue

    logger.info(
        f"[Decode Stage {model_params['device_id']}] Process Finished, time: {total_time:.2f} s."
    )


def consumer(
    model_params: dict, in_queue: mp.Queue, out_queue: mp.Queue, barrier: mp.Barrier
):
    """consumer init"""
    weight_manager = tcim.runtime.WeightManager(model_params["device_id"])
    option_prefill = tcim.runtime.Option(weight_manager)
    option_decode = tcim.runtime.Option(weight_manager)
    # load model
    prefill_model = tcim.runtime.load(
        model_params["prefill_path"], option=option_prefill
    )
    logger.info(f"Prefill Stage{model_params['device_id']} load model successfully")

    # Load decode
    dummy_tensor_names = []
    total_layer_num = prefill_model.get_num_inputs()
    for i in range(total_layer_num):
        layer_name = prefill_model.get_input_name(i)
        if "cache" in layer_name:
            dummy_tensor_names.append(layer_name)
        else:
            continue
    option_decode.set_dummy_tensors(dummy_tensor_names)
    decode_model = tcim.runtime.load(model_params["decode_path"], option=option_decode)
    logger.info(f"Decode Stage{model_params['device_id']} load model successfully")

    total_time = 0
    barrier.wait()
    while True:
        try:
            data = in_queue.get()
            if data == "END":
                out_queue.put("END", timeout=1)
                break

            input_name = prefill_model.get_input_name(0)
            valid_length_name = prefill_model.get_input_name(1)
            current_length_name = prefill_model.get_input_name(2)
            output_data, valid_length_data, current_length_data, t0, _ = data
            prefill_model.set_input(input_name, output_data)
            prefill_model.set_input(valid_length_name, valid_length_data)
            prefill_model.set_input(current_length_name, current_length_data)

            t1 = time.time()
            prefill_model.run()
            prefill_model.sync()
            t2 = time.time()
            total_time += t2 - t1
            output_data = prefill_model.get_output(
                prefill_model.get_output_name(0)
            ).numpy()
            result = (output_data, valid_length_data, current_length_data, t0, t2)

            try:
                out_queue.put(result)
            except queue.Full:
                logger.error(
                    f"[Prefill Stage {model_params['device_id']}] Queue Full, Drop data"
                )

        except queue.Empty:
            continue

    logger.info(
        f"[Prefill Stage {model_params['device_id']}] Process Finished, time: {total_time:.2f} s."
    )

    prefill_end_data = in_queue.get()
    out_queue.put(prefill_end_data)

    # kvcache share
    for i in range(total_layer_num):
        layer_name = prefill_model.get_input_name(i)
        if "cache" in layer_name:
            cache = prefill_model.get_dev_input(layer_name).to_host(True).numpy()
            decode_model.set_input(layer_name, cache)
    # decode stage
    total_time = 0
    while True:
        try:
            data = in_queue.get()
            if data == "END":
                out_queue.put("END", timeout=1)
                break

            input_name = decode_model.get_input_name(0)
            valid_length_name = decode_model.get_input_name(1)
            current_length_name = decode_model.get_input_name(2)
            output_data, valid_length_data, current_length_data, t0, _ = data
            decode_model.set_input(input_name, output_data)
            decode_model.set_input(valid_length_name, valid_length_data)
            decode_model.set_input(current_length_name, current_length_data)

            t1 = time.time()
            decode_model.run()
            decode_model.sync()
            t2 = time.time()
            total_time += t2 - t1
            output_data = decode_model.get_output(
                decode_model.get_output_name(0)
            ).numpy()
            result = (output_data, valid_length_data, current_length_data, t0, t2)

            try:
                out_queue.put(result)
            except queue.Full:
                logger.error(
                    f"[Prefill Stage {model_params['device_id']}] Queue Full, Drop data"
                )

        except queue.Empty:
            continue

    logger.info(
        f"[Decode Stage {model_params['device_id']}] Process Finished, time: {total_time:.2f} s."
    )


def result_shower(
    model_params: dict, in_queue: mp.Queue, out_queue: mp.Queue, barrier: mp.Barrier
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_params["tokenizer_dir"], trust_remote_code=True
    )
    embedding_weight = torch.load(model_params["embedding_path"], map_location="cpu")
    embedding_weight = embedding_weight["weight"]
    samplingmanager = SamplingManager(
        temperature=model_params["temperature"],
        top_k=model_params["top_k"],
        top_p=model_params["top_p"],
        repetition_penalty=model_params["repetition_penalty"],
    )
    prefill_count = 0
    start_time = 0
    end_time = 0
    context_length = 0
    pre_data = None
    barrier.wait()
    while True:
        try:
            data = in_queue.get()
            if data == "END":
                end_time = pre_data[4]
                break
            if prefill_count == 0:
                start_time = data[3]

            prefill_count += 1
            pre_data = data

        except queue.Empty:
            continue
    next_id = pre_data[0].argmax(-1)[0]
    logger.info(
        f"[Result] total Run Prefill {prefill_count} times, total_time : {end_time - start_time:.2f} s, output id = {next_id}"
    )

    prefill_end_data = in_queue.get()
    all_input_ids = prefill_end_data[0]
    all_input_ids = torch.from_numpy(all_input_ids)
    prefill_response = tokenizer.decode(next_id)
    context_max_length = prefill_end_data[1]
    embedding_len = prefill_end_data[2]
    embedding_weight = embedding_weight.reshape(-1, embedding_len).float()
    logger.success("Model response:")
    print("\033[1;95m{}".format(prefill_response), end="", flush=True)

    chat_history_ids = all_input_ids[0]
    generated_ids = []
    next_id = torch.from_numpy(next_id)
    generated_ids.append(next_id)
    chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)

    all_response = prefill_response
    input_echo_len = all_input_ids.numel()
    context_length += input_echo_len

    # Initialize decode stage variables
    decode_count = 0
    skip_tokens = 0
    slide_len = 10  # Sliding window length for decode stage token decoding
    last_response = tokenizer.decode(chat_history_ids.tolist()[-slide_len:])

    start_time = 0
    end_time = 0
    # Decode loop for generating subsequent tokens
    while True:
        # Stop generation if context length exceeds maximum limit
        if context_length >= context_max_length:
            logger.info(
                f"Context length ({context_length}) exceeds maximum limit ({context_max_length}), stopping generation!"
            )
            break

        input_data = F.embedding(next_id.unsqueeze(0), embedding_weight).reshape(
            1, 1, -1
        )

        valid_length_data = np.array(context_length).astype("int32")
        current_length_data = np.array([1]).astype("int32")

        out_queue.put(
            (
                input_data.numpy(),
                valid_length_data,
                current_length_data,
                start_time,
                end_time,
            )
        )

        decode_data = in_queue.get()
        output_data, valid_length_data, current_length_data, t0, t2 = decode_data
        if start_time == 0:
            start_time = t0
        # Get next token id (sampling from logits)
        decode_next_id = samplingmanager.sample(output_data, generated_ids)
        decode_next_id = torch.from_numpy(decode_next_id[0])

        decode_count += 1

        # Check for end-of-sequence token
        if decode_next_id == tokenizer.eos_token_id:
            if "decode_response" in locals():
                print(decode_response, end="", flush=True)
                all_response += decode_response
            break

        # Update chat history with new token
        chat_history_ids = torch.cat([chat_history_ids, decode_next_id], dim=-1)
        generated_ids.append(decode_next_id)

        decode_response = tokenizer.decode(
            chat_history_ids.tolist()[-(slide_len + 1) - skip_tokens :]
        )[len(last_response) :]

        # Validate and print decoded text (outside timing scope)
        if decode_response != "" and is_valid_char(ord(decode_response[-1])):
            print(decode_response, end="", flush=True)
            all_response += decode_response
            last_response = tokenizer.decode(chat_history_ids.tolist()[-slide_len:])
            skip_tokens = 0
        else:
            skip_tokens += 1

        # Prepare for next iteration
        next_id = decode_next_id
        context_length += 1
    out_queue.put("END")
    end_time = t2
    print("\033[0m")
    logger.info(
        f"[Result] total Run Decode {decode_count} times, total_time : {end_time - start_time:.2f} s"
    )
    logger.info(
        f"Decode Model average time {(end_time - start_time) * 1000 / decode_count :.2f} ms/token"
    )


def main():
    args = get_args()
    if args.ndevice > tcim.runtime.get_device_num():
        logger.error(
            f"Device number {args.ndevice} exceeds available device count {tcim.runtime.get_device_num()}"
        )
        sys.exit(1)
    logger.info("MultiChip pipeline parallel run demo Start!")
    barrier = mp.Barrier(args.ndevice + 1)
    model_params_list = [gen_params(k, args) for k in range(args.ndevice)]

    queue_list = [mp.Queue(maxsize=5) for _ in range(args.ndevice + 1)]
    question_len = args.input_tokens

    threads = []
    for i in range(args.ndevice + 1):
        if i == 0:
            producer_process = mp.Process(
                target=producer,
                args=(
                    model_params_list[i],
                    queue_list[i],
                    queue_list[args.ndevice],
                    question_len,
                    barrier,
                ),
                name=f"producer_process_{i}",
            )
            threads.append(producer_process)
        else:
            if i == args.ndevice:
                decode_process = mp.Process(
                    target=result_shower,
                    args=(
                        model_params_list[0],
                        queue_list[i - 1],
                        queue_list[i],
                        barrier,
                    ),
                    name=f"decode_process_{i}",
                )
                threads.append(decode_process)
            else:
                consumer_process = mp.Process(
                    target=consumer,
                    args=(
                        model_params_list[i],
                        queue_list[i - 1],
                        queue_list[i],
                        barrier,
                    ),
                    name=f"consumer_process_{i}",
                )
                threads.append(consumer_process)

    for _, thread in enumerate(threads):
        thread.start()

    for _, thread in enumerate(threads):
        thread.join()

    logger.info("MultiChip pipeline parallel run demo finished!")


if __name__ == "__main__":
    main()
