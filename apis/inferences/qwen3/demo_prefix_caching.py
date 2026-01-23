#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025 HOUMO AI
#
# File: demo_prefix_caching.py
# Description:
#   Prefix Caching Demo for Qwen3 - Demonstrates KV cache prefix caching functionality
#   for improved inference performance by reusing previously computed key-value pairs.
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
import math
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from typing import Dict, Tuple, List
from typing_extensions import override

from qwen_base import HmQwenBase
from sampling_manager import SamplingManager
from qwen_demo_utils import show_statistics, is_valid_char


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default="qwen3-8b",
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=os.path.join("hmquant", "quant_embedding.pt"),
        help="Path to houmo embedding weights",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default="qwen3_prefill.hmm",
        help="Path to houmo prefill model",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default="qwen3_decode.hmm",
        help="Path to houmo decode model",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of devices",
    )
    parser.add_argument(
        "--repetition_penalty",
        dest="repetition_penalty",
        type=float,
        default=1.0,
        help="Sampling repetition penalty parameter",
    )
    parser.add_argument(
        "--topk",
        dest="topk",
        type=int,
        default=None,
        help="Sampling top-k parameter",
    )
    parser.add_argument(
        "--topp",
        dest="topp",
        type=float,
        default=1.0,
        help="Sampling top-p parameter",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=1.0,
        help="Sampling temperature parameter",
    )

    args = parser.parse_args()
    if args.ndevice > 1:
        args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        args.decode_path = args.decode_path.replace(".hmm", ".hmms")

    return args


class LLMPrefixTrie:
    """
    Trie structure for caching LLM prefix KV caches.
    Nodes store token IDs hierarchically with each node containing:
    - Child mappings (token_id → child nodes)
    - Prefix length (number of tokens in the prefix)
    """

    class TrieNode:
        def __init__(self):
            # Maps token_id to child nodes
            self.children: Dict[int, LLMPrefixTrie.TrieNode] = {}
            # Token length of the prefix up to this node
            self.prefix_len: int = 0
            # Indicates if this node represents the end of a complete prefix
            self.is_end: bool = False

    def __init__(self):
        self.root = self.TrieNode()

    def insert(self, prefix_token_ids: List[int]):
        """
        Insert a prefix into the trie, storing its token ID list.

        Args:
            prefix_token_ids: List of token IDs representing the prefix
        """
        current_node = self.root
        for idx, token_id in enumerate(prefix_token_ids):
            if token_id not in current_node.children:
                current_node.children.clear()
                current_node.children[token_id] = self.TrieNode()
            current_node = current_node.children[token_id]
            current_node.prefix_len = idx + 1

        current_node.is_end = True
        logger.success(
            f"✅ Successfully inserted prefix cache, token length: {current_node.prefix_len}"
        )

    def find_longest_matched(self, token_ids: List[int]) -> int:
        """
        Find the longest cached prefix matching the input token IDs.

        Args:
            token_ids: List of token IDs to match against cached prefixes

        Returns:
            matched prefix_length
        """
        current_node = self.root
        matched_len = 0

        for token_id in token_ids:
            if token_id not in current_node.children or current_node.is_end:
                matched_len = current_node.prefix_len
                current_node.children.clear()
                current_node.is_end = True
                break
            current_node = current_node.children[token_id]

        if matched_len:
            logger.info(
                f"🔍 Found longest matching prefix, match length: {matched_len}"
            )
        else:
            logger.info("❌ No matching cached prefix found")

        return matched_len


class HmQwenPrefix(HmQwenBase):
    """
    Qwen3 model implementation with prefix caching capabilities.
    Extends the base Qwen implementation to support KV cache reuse for improved efficiency.
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
        Initialize the Qwen model with prefix caching.

        Args:
            prefill_path: Path to prefill model
            decode_path: Path to decode model
            embedding_path: Path to embedding weights
            tokenizer_dir: Path to tokenizer
            ndevice: Number of devices to use
            **kwargs: Additional parameters like sampling settings
        """
        super().__init__(
            prefill_path=prefill_path,
            decode_path=decode_path,
            embedding_path=embedding_path,
            tokenizer_dir=tokenizer_dir,
            ndevice=ndevice,
        )
        self.samplingmanager = SamplingManager(
            temperature=kwargs.get("temperature", 1.0),
            top_k=kwargs.get("topk", None),
            top_p=kwargs.get("topp", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        )
        self.default_prompt = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.prefix_trie = LLMPrefixTrie()

    def get_prefix_kv_cache(self) -> Dict[str, torch.Tensor]:
        """
        Get the KV cache from the prefill model.

        Returns:
            Dictionary mapping input names to their tensor values for KV cache
        """
        prefill_input_names = self.get_input_names("prefill")
        prefix_cache = {}
        for input_name in prefill_input_names:
            if "kcache" not in input_name and "vcache" not in input_name:
                continue
            prefix_cache[input_name] = torch.tensor(
                self.prefill.get_dev_input(input_name).to_host().numpy()
            )

        return prefix_cache

    def load_prefix_cache(self, prefix_cache, token_ids):
        """
        Load the provided KV cache and insert the token IDs into the prefix trie.

        Args:
            prefix_cache: Dictionary of KV cache tensors to load
            token_ids: List of token IDs corresponding to the cached prefix
        """
        self.prefix_trie.insert(token_ids)

        for input_name, input_data in prefix_cache.items():
            self.prefill.set_input(input_name, input_data.numpy())

    def _generate_message(self, question: str, background_prompts: list):
        """
        Generate a message to be sent to the model

        Args:
            question: The question to be asked
            background_prompts: A list of background prompts to be used in the message
        """
        if background_prompts:
            messages = background_prompts + [{"role": "user", "content": question}]
        else:
            messages = self.default_prompt + [{"role": "user", "content": question}]

        return messages

    @override
    def chat(self, question: str, background_prompts: list = []):
        """
        Perform a chat interaction with the model using prefix caching.

        Args:
            question: User's question to respond to
            background_prompts: List of background context messages

        Returns:
            A tuple of (prefill_count, decode_count, token_ids)
        """
        # Initialize tracking variables for generation metrics
        self.generated_ids = []
        self.prefill_time = 0
        self.decode_time = 0
        self.ttft_time = 0

        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))

        start_time = time.time()
        messages = self._generate_message(question, background_prompts)
        # Apply chat template formatting to convert messages to text format suitable for the model
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        # Tokenize the formatted text
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()
        # Check if the input exceeds the maximum context length
        if input_echo_len >= self.context_max_length:
            logger.error(
                f"Question long than {self.context_max_length}, please shorten it!"
            )
            return {}, 0, 0

        # Store the total number of tokens in the original input
        prefill_count = input_echo_len
        token_ids = all_input_ids.flatten().tolist()
        # Find the longest matching prefix in the cache to avoid recomputation
        matched_len = self.prefix_trie.find_longest_matched(token_ids)
        # Calculate how many tokens still need to be processed (excluding the cached prefix)
        input_echo_len = input_echo_len - matched_len

        # Process the input in chunks of size up to prefill_length
        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        for round_idx in range(prefill_loop_round):
            # Calculate the valid length for the current chunk
            valid_length = round_idx * self.prefill_length + matched_len
            start_idx = matched_len + round_idx * self.prefill_length
            if round_idx == prefill_loop_round - 1:
                # Last round processing - handle remaining tokens
                current_length = input_echo_len - round_idx * self.prefill_length
                end_idx = prefill_count
            else:
                # Regular round - process a full chunk of prefill_length
                current_length = self.prefill_length
                end_idx = (round_idx + 1) * self.prefill_length
            # Extract the token IDs for the current chunk
            input_ids = all_input_ids[:, start_idx:end_idx]

            # Compute embeddings for the current input tokens
            inputs_embeds = F.embedding(input_ids, self.embedding_weight)
            effective_length = input_ids.size(-1)
            # Create padding embeddings to fill up to the prefill_length
            _pad_embeds = torch.zeros(
                1,
                self.prefill_length - effective_length,
                inputs_embeds.size(-1),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

            # Concatenate actual embeddings with padding, reshape for model input
            input_data = (
                torch.cat([inputs_embeds, _pad_embeds], dim=1)
                .reshape(1, self.prefill_length, self.embedding_len)
                .numpy()
            )
            # Prepare length metadata for the model
            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")
            # Set the prepared data as input to the prefill model
            self.prefill.set_input(self.prefill_input_name, input_data)
            self.prefill.set_input(self.prefill_valid_length_name, valid_length_data)
            self.prefill.set_input(
                self.prefill_current_length_name, current_length_data
            )

            # Run the prefill operation and measure execution time
            prefill_start = time.time()
            self.prefill.run()
            self.prefill.sync()
            self.prefill_time += time.time() - prefill_start

        # Extract the output from the prefill stage
        prefill_outputs = self.prefill.get_output(
            self.prefill.get_output_name(0)
        ).numpy()
        # Select the next token ID based on the highest probability
        next_id = prefill_outputs.argmax(-1)[0]
        # Decode the predicted token ID back to text
        prefill_response = self.tokenizer.decode(next_id)
        logger.success("response:")
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)

        self.ttft_time = time.time() - start_time
        # Initialize the history of generated token IDs with the original input
        chat_history_ids = all_input_ids[0]
        next_id = torch.from_numpy(next_id)
        self.generated_ids.append(next_id)
        # Extend the chat history with the new token
        chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)
        all_response = prefill_response
        # Set the context length to the count of tokens from the original input
        self.context_length = prefill_count

        # Initialize counters for decoding loop
        decode_count = 0
        skip_tokens = 0
        slide_len = 10  # sliding window length for decode
        # Get the last few tokens for comparison purposes
        last_response = self.tokenizer.decode(chat_history_ids.tolist()[-slide_len:])

        # Main generation loop - continue generating tokens until EOS or max context reached
        while True:
            if self.context_length >= self.context_max_length:
                logger.warning(
                    f"context length greater than {self.context_max_length}, break!"
                )
                break

            # Prepare input for the decode step (single token embedding)
            input_data = (
                F.embedding(next_id.unsqueeze(0), self.embedding_weight)
                .reshape(1, 1, -1)
                .numpy()
            )
            valid_length_data = np.array(self.context_length).astype("int32")
            self.decode.set_input(self.decode_input_name, input_data)
            self.decode.set_input(self.decode_valid_length_name, valid_length_data)

            # Run the decode operation and track execution time
            decode_start = time.time()
            self.decode.run()
            self.decode.sync()
            self.decode_time += time.time() - decode_start

            decode_count += 1
            # Get the output probabilities from the decode model
            decode_outputs = self.decode.get_output(
                self.decode.get_output_name(0)
            ).numpy()
            # Sample the next token ID using the configured sampling strategy
            next_id = (
                self.samplingmanager.sample(decode_outputs, self.generated_ids)
                .squeeze()
                .reshape(1)
            )
            next_id = torch.from_numpy(next_id)
            # Check if the generated token is the end-of-sequence token
            if next_id == self.tokenizer.eos_token_id:
                print(decode_response, end="", flush=True)
                all_response += decode_response
                break

            # Extend the chat history with the newly generated token
            chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)
            self.generated_ids.append(next_id)
            # Decode a segment of recent tokens to form the response
            decode_response = self.tokenizer.decode(
                chat_history_ids.tolist()[-(slide_len + 1) - skip_tokens :]
            )[len(last_response) :]
            if decode_response != "" and is_valid_char(ord(decode_response[-1])):
                print(decode_response, end="", flush=True)
                all_response += decode_response
                # Update last response for next iteration
                last_response = self.tokenizer.decode(
                    chat_history_ids.tolist()[-slide_len:]
                )
                skip_tokens = 0
            else:
                skip_tokens += 1
            # Increase the context length for the next iteration
            self.context_length += 1
        print("\033[0m")

        # Update prefix tokens in trie - cache the full conversation history for future use
        self.prefix_trie.insert(chat_history_ids.tolist())

        # Return statistics about the generation process
        return prefill_count, (decode_count + 1), chat_history_ids.tolist()


def start_chat(
    hmqwen: HmQwenPrefix, question_list: list, background_prompts: list
) -> Tuple[bool, list]:
    """
    Execute a chat session with multiple questions.

    Args:
        hmqwen: Instance of HmQwenPrefix model
        question_list: List of questions to ask
        background_prompts: Background context for the conversation

    Returns:
        A tuple of (success flag, list of generated token IDs)
    """
    try:
        for question in question_list:
            start_time = time.time()
            try:
                prefill_tokens, decode_tokens, token_ids = hmqwen.chat(
                    question, background_prompts
                )
                total_time = time.time() - start_time
                show_statistics(
                    prefill_tokens,
                    decode_tokens,
                    hmqwen.ttft_time,
                    hmqwen.prefill_time,
                    hmqwen.decode_time,
                    total_time,
                )
            except Exception as e:
                logger.error(f"Error occurred during the chat: {str(e)}")
                return False, []
    except KeyboardInterrupt:
        logger.error("The program was interrupted by the user.")
        return False, []
    except Exception as e:
        logger.error(f"Error occurred during execution: {str(e)}")
        return False, []

    return True, token_ids


if __name__ == "__main__":
    args = get_args()

    # Define background prompts for conversation context
    background_prompts = [
        {
            "role": "system",
            "content": "基于以下背景信息和多轮对话，完成最后的推理任务："
            "【背景信息】某电商平台的订单处理规则如下："
            "1. 订单状态分为：待付款、待发货、待收货、已完成、已取消、退款中、已退款；"
            "2. 付款规则：待付款订单超过24小时未付款自动取消，支持支付宝、微信、银行卡三种支付方式；"
            "3. 发货规则：待发货订单需在付款后48小时内发货，预售订单可延迟至7天内发货，偏远地区（新疆、西藏、青海、内蒙古）额外延迟2天；"
            "4. 收货规则：待收货订单发货后15天未确认收货，系统自动确认；"
            "5. 退款规则："
            "   - 已付款未发货订单：支持全额退款，审核时间2小时；"
            "   - 已发货未确认收货订单：需用户退回商品，商家验收后72小时内退款；"
            "   - 已完成订单：仅售后问题（质量问题、错发漏发）支持退款，需提供质检报告，审核时间3个工作日；"
            "6. 优先级规则：VIP用户（等级1-5）的订单处理优先级高于普通用户，VIP等级越高优先级越高；"
            "7. 异常订单：同一用户24小时内下单超过10单、单笔订单金额超过5万元，需人工审核，审核时间12小时。"
            "【多轮对话】",
        },
        {
            "role": "user",
            "content": "我是VIP3用户，昨天10点下了3个订单，分别是："
            "- 订单1：普通商品，金额899元，付款时间昨天11点，收货地址是西藏拉萨；"
            "- 订单2：预售商品，金额1599元，付款时间昨天12点，收货地址是北京朝阳区；"
            "- 订单3：奢侈品，金额5.2万元，付款时间昨天15点，收货地址是上海浦东新区。",
        },
        {
            "role": "system",
            "content": "已记录，订单1和2已进入待发货状态，订单3触发异常订单规则，正在人工审核。",
        },
        {
            "role": "user",
            "content": "今天9点我发现订单1还没发货，订单3的审核结果还没出来，订单2的预售发货时间写的是7天，这符合规则吗？",
        },
        {
            "role": "system",
            "content": "订单1因收货地址是西藏，发货时间可延迟2天，目前仍在规则范围内；订单3的人工审核还在12小时有效期内，预计今天21点出结果；订单2的预售发货时间符合7天规则。",
        },
        {
            "role": "user",
            "content": "如果我现在申请订单1的退款，是否符合规则？退款流程和时间是怎样的？",
        },
        {
            "role": "system",
            "content": "订单1已付款且未发货，符合全额退款规则，审核时间2小时，审核通过后即时到账。",
        },
        {
            "role": "user",
            "content": "订单3如果审核通过，发货时间是多久？如果我之后申请退款，流程会有变化吗？",
        },
    ]

    # Baseline question for initial model warmup
    baseline_question = ["你是谁？"]
    question_list = [
        "回答用户A最后的两个问题，需严格遵循背景中的所有规则。",
        "订单3若审核不通过，会进入什么状态？原因是什么？",
        # "假设订单1在今天12点发货，用户A最晚需要在什么时间确认收货？若未确认，系统会在什么时间自动确认？",
        # "若用户A的VIP等级升级为5级，订单处理优先级会有什么变化？对发货/审核时间是否有影响？",
    ]

    hmqwen_cache = HmQwenPrefix(
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
    _, cached_tokens = start_chat(hmqwen_cache, baseline_question, background_prompts)
    prefix_cache = hmqwen_cache.get_prefix_kv_cache()
    del hmqwen_cache

    # Initialize new model instance and load the prefix cache
    hmqwen = HmQwenPrefix(
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
    hmqwen.load_prefix_cache(prefix_cache, cached_tokens)
    start_chat(hmqwen, question_list, background_prompts)
