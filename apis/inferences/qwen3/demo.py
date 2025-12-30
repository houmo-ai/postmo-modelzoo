#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
HmQwenXh2 Model Inference Script - A script for running inference with the Houmo AI Qwen3-XH2 model using TCIM runtime.

Copyright (c) 2025 HOUMOAI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SPDX-License-Identifier: Apache-2.0
"""
import os
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
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

TOKENIZER_PATH = "qwen3-8b"
EMBEDDING_PATH = "hmquant/quant_embedding.pt"


def is_valid_char(cp):
    """
    Check if a Unicode character is a valid Chinese character or English letter.

    Valid character ranges:
    - CJK Unified Ideographs (4E00-9FFF)
    - CJK Unified Ideographs Extension A (3400-4DBF)
    - CJK Unified Ideographs Extension B (20000-2A6DF)
    - CJK Unified Ideographs Extension C (2A700-2B73F)
    - CJK Unified Ideographs Extension D (2B740-2B81F)
    - CJK Unified Ideographs Extension E (2B820-2CEAF)
    - CJK Compatibility Ideographs (F900-FAFF)
    - CJK Compatibility Ideographs Supplement (2F800-2FA1F)
    - English uppercase letters (A-Z)
    - English lowercase letters (a-z)

    Args:
        cp (int): Unicode code point of the character to check.

    Returns:
        bool: True if the character is valid, False otherwise.
    """
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)       # CJK Unified Ideographs
        or (cp >= 0x3400 and cp <= 0x4DBF)     # CJK Extension A
        or (cp >= 0x20000 and cp <= 0x2A6DF)   # CJK Extension B
        or (cp >= 0x2A700 and cp <= 0x2B73F)   # CJK Extension C
        or (cp >= 0x2B740 and cp <= 0x2B81F)   # CJK Extension D
        or (cp >= 0x2B820 and cp <= 0x2CEAF)   # CJK Extension E
        or (cp >= 0xF900 and cp <= 0xFAFF)     # CJK Compatibility Ideographs
        or (cp >= 0x2F800 and cp <= 0x2FA1F)   # CJK Compatibility Ideographs Supplement
        or (0x0041 <= cp and cp <= 0x005A)     # English uppercase letters
        or (0x0061 <= cp and cp <= 0x007A)     # English lowercase letters
    ):
        return True

    return False


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments containing model directory, prefill length,
        decode length, and number of blocks.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default="./",
        help='Directory containing Houmo model files',
    )
    parser.add_argument(
        '--prefill',
        dest='prefill_length',
        type=int,
        default=256,
        help='Maximum length for prefill step',
    )
    parser.add_argument(
        '--decode',
        dest='decode_length',
        type=int,
        default=8192,
        help='Maximum context length for decode step (includes prefill and generated tokens)',
    )
    parser.add_argument(
        '--nblocks',
        dest='nblocks',
        type=int,
        default=36,
        help='Number of transformer blocks in the model',
    )
    args = parser.parse_args()
    return args


class HmQwenXh2:
    """
    HmQwenXh2 is a class for running inference with the Houmo AI Qwen3-XH2 model using TCIM runtime.

    This class handles both prefill and decode stages of inference, manages KV cache,
    and provides a chat interface for interacting with the model.

    Attributes:
        batch (int): Batch size for inference.
        prefill_length (int): Maximum length for the prefill step.
        decode_length (int): Maximum context length allowed (prefill + generated tokens).
        nblocks (int): Number of transformer blocks in the model.
        prefill: TCIM runtime prefill model instance.
        decode: TCIM runtime decode model instance.
        tokenizer: Hugging Face tokenizer for text processing.
        embedding_weight: Quantized embedding weights.
    """

    def __init__(self, model_dir, prefill_length, decode_length, batch=1, nblocks=28):
        """
        Initialize the HmQwenXh2 model instance.

        Args:
            model_dir (str): Directory containing model files (prefill.hmm and decode.hmm).
            prefill_length (int): Maximum length for prefill step.
            decode_length (int): Maximum context length for decode step.
            batch (int, optional): Batch size. Defaults to 1.
            nblocks (int, optional): Number of transformer blocks. Defaults to 28.
        """
        self.batch = batch
        self.prefill_length = prefill_length
        self.decode_length = decode_length
        self.nblocks = nblocks

        # Initialize weight manager and options for TCIM runtime
        weight_manager = tcim.runtime.WeightManager(0)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)

        # Set dummy tensors for decode model options
        dummy_tensor_names = [
            f'model_layers_{i}_self_attn_kcache_input' for i in range(nblocks)
        ]
        dummy_tensor_names += [
            f'model_layers_{i}_self_attn_vcache_input' for i in range(nblocks)
        ]
        option2.set_dummy_tensors(dummy_tensor_names)

        # Load prefill and decode models from file
        self.prefill = tcim.runtime.load(
            os.path.join(model_dir, "qwen3_prefill.hmm"), option=option1
        )
        print("prefill model loaded.")
        self.decode = tcim.runtime.load(
            os.path.join(model_dir, "qwen3_decode.hmm"), option=option2
        )
        print("decode model loaded.")

        # Transfer KV cache inputs from prefill to decode model
        for i in range(3, 2 * nblocks + 3):
            cache = self.prefill.get_input(self.prefill.get_input_name(i))
            self.decode.set_input(self.decode.get_input_name(i), cache)

        # Initialize decode current length input
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode.get_input_name(2)
        self.decode.set_input(decode_current_length_name, current_length_input_1)

        # Load tokenizer and quantized embedding weights
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH, trust_remote_code=True
        )
        embedding_weight = torch.load(
            EMBEDDING_PATH, map_location="cpu", weights_only=True
        )['weight']
        embedding_weight = embedding_weight.float()
        self.embedding_weight = embedding_weight.reshape(-1, 4096)

    def chat(self, question):
        """
        Generate a response to a given question using the model.

        Args:
            question (str): The user's question.

        Returns:
            tuple: A tuple containing:
                - all_response (str): The complete generated response.
                - total_tokens (int): Total number of generated tokens (including prefill output).
                - prefill_time (float): Time taken for prefill step in seconds.
                - decode_time (float): Time taken for decode step in seconds.
        """
        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))
        start_time = time.time()

        # Prepare chat messages and apply chat template
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()

        # Check if question is too long
        if input_echo_len >= self.decode_length:
            logger.error(f"Question longer than {self.decode_length} tokens, please shorten it!")
            return f"Question longer than {self.decode_length} tokens, please shorten it!", 0, 0, 0

        # Determine number of prefill rounds based on prefill length
        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        for round in range(prefill_loop_round):
            valid_length = round * self.prefill_length

            # Get current batch of input ids
            if round == prefill_loop_round - 1:
                current_length = input_echo_len - round * self.prefill_length
                input_ids = all_input_ids[:, round * self.prefill_length : input_echo_len]
            else:
                current_length = self.prefill_length
                input_ids = all_input_ids[:, round * self.prefill_length : (round + 1) * self.prefill_length]

            # Generate input embeddings and pad to prefill length
            inputs_embeds = F.embedding(input_ids, self.embedding_weight)
            effective_length = input_ids.size(-1)
            pad_embeds = torch.zeros(
                1,
                self.prefill_length - effective_length,
                inputs_embeds.size(-1),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            input_data = torch.cat([inputs_embeds, pad_embeds], dim=1).reshape(
                1, self.prefill_length, 4096
            )

            # Prepare prefill model inputs
            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")

            # Set prefill model inputs
            input_name = self.prefill.get_input_name(0)
            valid_length_name = self.prefill.get_input_name(1)
            current_length_name = self.prefill.get_input_name(2)
            self.prefill.set_input(input_name, input_data.float().numpy())
            self.prefill.set_input(valid_length_name, valid_length_data)
            self.prefill.set_input(current_length_name, current_length_data)

            # Run prefill and synchronize
            self.prefill.run()
            self.prefill.sync()

        # Get first token from prefill output and decode it
        input_data = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        next_id = input_data.argmax(-1)[0]
        prefill_response = self.tokenizer.decode(next_id)
        prefill_time = time.time() - start_time

        # Initialize chat history and context tracking
        chat_history_ids = all_input_ids[0]
        next_id = torch.from_numpy(next_id)
        chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)

        # Generate embedding for the next token
        input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(
            1, 1, -1
        )

        all_response = prefill_response
        context_length = input_echo_len

        logger.success("response:")
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)

        decode_count = 0
        skip_tokens = 0
        slide_len = 10  # Sliding window length for decoding
        last_response = self.tokenizer.decode(chat_history_ids.tolist()[-slide_len:])

        start_time = time.time()
        while True:
            # Check if maximum context length is reached
            if context_length >= self.decode_length:
                logger.info(f"Context length greater than {self.decode_length}, breaking!")
                break

            # Set decode model inputs
            input_name = self.decode.get_input_name(0)
            valid_length_name = self.decode.get_input_name(1)
            self.decode.set_input(input_name, input_data.float().numpy())
            valid_length_data = np.array(context_length - 1).astype("int32")
            self.decode.set_input(valid_length_name, valid_length_data)

            # Run decode and synchronize
            self.decode.run()
            self.decode.sync()
            input_data = self.decode.get_output(self.decode.get_output_name(0)).numpy()
            decode_count += 1

            # Get next token id and check if it's EOS
            next_id = input_data.astype(np.float32).argmax(-1)[0]
            next_id = torch.from_numpy(next_id)
            if next_id == self.tokenizer.eos_token_id:
                print(decode_response, end="", flush=True)
                all_response += decode_response
                break

            # Update chat history
            chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)

            # Decode response with sliding window
            decode_response = self.tokenizer.decode(
                chat_history_ids.tolist()[-(slide_len + 1) - skip_tokens :]
            )[len(last_response) :]

            # Print and accumulate valid responses
            if decode_response != '' and is_valid_char(ord(decode_response[-1])):
                print(decode_response, end="", flush=True)
                all_response += decode_response
                last_response = self.tokenizer.decode(
                    chat_history_ids.tolist()[-slide_len:]
                )
                skip_tokens = 0
            else:
                skip_tokens += 1

            # Generate embedding for next iteration
            input_data = F.embedding(
                next_id.unsqueeze(0), self.embedding_weight
            ).reshape(1, 1, -1)
            context_length += 1

        decode_time = time.time() - start_time
        print("\033[0m")

        total_tokens = decode_count + 1  # +1 for prefill output token
        return all_response, total_tokens, prefill_time, decode_time


if __name__ == "__main__":
    args = get_args()
    hmqwen = HmQwenXh2(
        args.model_dir,
        args.prefill_length,
        args.decode_length,
        nblocks=args.nblocks,
    )
    question = "请介绍一下存算一体技术的优势"

    start_time = time.time()
    response, tokens, prefill_time, decode_time = hmqwen.chat(question)
    total_time = time.time() - start_time

    logger.success(f"Total: {tokens} tokens, cost {total_time:.3f} s")
    logger.success(
        f"Prefill time: {prefill_time * 1000:.3f} ms, throughput: {1 / prefill_time:.2f} tokens/s"
    )
    if tokens > 1:
        decode_latency = decode_time * 1000 / (tokens - 1)
        logger.success(
            f"Decode average time: {decode_latency:.3f} ms, throughput: {1000 / decode_latency:.2f} tokens/s"
        )
    res_latency = total_time * 1000 / tokens
    logger.success(
        f"End-to-end average time: {res_latency:.3f} ms, throughput: {1000 / res_latency:.2f} tokens/s"
    )