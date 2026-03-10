#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Qwen3 Speculative Inference Demo - Python script for running Qwen3
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
        default=os.path.join(
            "output", HOUMO_TARGET, "draft", "hmquant", "quant_embedding.pt"
        ),
        help="houmo embedding weight path",
    )
    parser.add_argument(
        "--verify_embedding_path",
        dest="verify_embedding_path",
        type=str,
        default=os.path.join(
            "output", HOUMO_TARGET, "target", "hmquant", "quant_embedding.pt"
        ),
        help="houmo embedding weight path",
    )
    parser.add_argument(
        "--draft_prefill_path",
        dest="draft_prefill_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3_prefill_draft.hmm"),
        help="houmo draft prefill model path",
    )
    parser.add_argument(
        "--draft_decode_path",
        dest="draft_decode_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3_decode_draft.hmm"),
        help="houmo draft decode model path",
    )
    parser.add_argument(
        "--verify_path",
        dest="verify_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3_verify.hmm"),
        help="houmo verify model path",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3_prefill.hmm"),
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--max_step",
        dest="max_step",
        type=int,
        default=5,
        help="max_step",
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
        "--it",
        dest="it",
        action="store_true",
        help="interactive mode",
    )
    parser.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        help="fast mode",
    )
    args = parser.parse_args()
    if args.ndevice > 1:
        args.draft_prefill_path = args.draft_prefill_path.replace(".hmm", ".hmms")
        args.draft_decode_path = args.draft_decode_path.replace(".hmm", ".hmms")
        args.verify_path = args.verify_path.replace(".hmm", ".hmms")
        args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
    return args


def cal_accept(data, verify_steps):
    mean = sum(data) / len(data)
    acceptance_rate = sum(data) / (sum(verify_steps) - len(data))
    average_acceptance_num = sum(data) / len(data)
    squared_diffs = [(x - mean) ** 2 for x in data]
    population_variance_manual = sum(squared_diffs) / len(data)
    return acceptance_rate, average_acceptance_num, population_variance_manual


def show_statistics(hmqwen, input_tokens, output_tokens, total_time):
    ttft_time = hmqwen.ttft_time
    prefill_time = hmqwen.prefill_time
    decode_time = hmqwen.decode_time
    acceptance_rate, average_acceptance_num, population_variance_manual = cal_accept(
        hmqwen.accept_nums,
        hmqwen.verify_steps,
    )
    draft_speed = hmqwen.draft_decode_count / (decode_time - hmqwen.verify_time)
    verify_speed = sum(hmqwen.verify_steps) / (hmqwen.verify_time)
    verify_round_time = hmqwen.verify_time / hmqwen.verify_round
    verify_generate_token_speed = (
        hmqwen.verify_round * (average_acceptance_num + 1) / (hmqwen.verify_time)
    )
    decode_speed = (output_tokens - 1) / decode_time
    speedup = decode_speed / (hmqwen.verify_round / hmqwen.verify_time)
    logger.success(
        f"Total Input: {input_tokens} tokens, Output {output_tokens} tokens, Prefill Cost {prefill_time*1000:.3f} ms, Decode Cost {decode_time*1000:.3f} ms"
    )
    logger.success(
        f"Prefill Speed: {input_tokens / prefill_time:.2f} tokens/s; Decode Speed: {decode_speed:.2f} tokens/s"
    )
    logger.success(
        f"Draft Decode Speed: {draft_speed:.2f} tokens/s; Verify Speed: {verify_speed:.2f} tokens/s; Average Verify Accept Speed: {verify_generate_token_speed:.2f} tokens/s"
    )
    logger.success(f"TTFT (Time to First Token): {ttft_time * 1000:.3f} ms")
    logger.success(
        f"TPOT (Time Per Output Token): {decode_time * 1000 / (output_tokens - 1):.3f} ms/token"
    )
    logger.success(f"Time Per Verify Round: {(verify_round_time * 1000):.3f} ms/round")
    logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
    logger.success(
        f"E2E TPS (End-to-End Tokens Per Second): {output_tokens / total_time:.2f} tokens/s"
    )
    logger.success(f"SpeedUp: {speedup:.3f}x")
    logger.success(f"Acceptance Rate: {acceptance_rate * 100:.3f} %")
    logger.success(f"Average Acceptance Num: {average_acceptance_num:.3f} ")
    logger.success(f"Population Variance Manual: {population_variance_manual:.3f}")


def is_top2_unvalid(arr):
    partitioned_indices = np.argpartition(arr, -2)
    top2_indices_unordered = partitioned_indices[-2:]
    top2_indices = top2_indices_unordered[np.argsort(arr[top2_indices_unordered])][::-1]
    top2_values = arr[top2_indices]
    if (np.abs(top2_values[1] - top2_values[0])) < 1:
        return True
    else:
        return False


def sample(probs: torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    if idx_next.item() == 0:
        raise RuntimeError
    return idx_next


# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float("-inf")
    return logits


def norm_logits(
    logits: torch.Tensor, temperature: float, top_k: float, top_p: float
) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


class HmQwen:

    def __init__(self, args):
        # init weight manager
        self.ndevice = args.ndevice
        if self.ndevice == 1:
            weight_manager = tcim.runtime.WeightManager(0)
        elif self.ndevice == 2 and HOUMO_TARGET == "xh2":
            dev_manager = tcim.runtime.DevManager([0, 1], "Xh2HalBackend")
            weight_manager = tcim.runtime.WeightManager(dev_manager)
        else:
            raise ValueError("Unsupport device number!")
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        option3 = tcim.runtime.Option(weight_manager)
        option4 = tcim.runtime.Option(weight_manager)

        # load draft model
        self.prefill = tcim.runtime.load(args.draft_prefill_path, option=option1)
        logger.info("prefill draft model loaded")
        dummy_tensor_names = []
        input_names = self.get_input_names(self.prefill)
        for input_name in input_names:
            if "model_layers" in input_name:
                dummy_tensor_names.append(input_name)
        option2.set_dummy_tensors(dummy_tensor_names)
        self.decode = tcim.runtime.load(args.draft_decode_path, option=option2)
        logger.info("decode draft model loaded")
        self.prefill_length = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        for input_name in input_names:
            if "model_layers" in input_name:
                cache = self.prefill.get_input(input_name)
                self.decode.set_input(input_name, cache)

        # load verify model
        self.prefill_verify = tcim.runtime.load(args.prefill_path, option=option3)
        logger.info("prefill verify model loaded")
        dummy_tensor_names = []
        input_names = self.get_input_names(self.prefill_verify)
        for input_name in input_names:
            if "model_layers" in input_name:
                dummy_tensor_names.append(input_name)
        option4.set_dummy_tensors(dummy_tensor_names)
        self.verify = tcim.runtime.load(args.verify_path, option=option4)
        logger.info("verify model loaded")
        for input_name in input_names:
            if "model_layers" in input_name:
                cache = self.prefill_verify.get_input(input_name)
                self.verify.set_input(input_name, cache)

        # set decode input
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode.get_input_name(2)
        self.decode.set_input(decode_current_length_name, current_length_input_1)

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir, trust_remote_code=True
        )

        # init embedding weight
        self.embedding_weight = torch.load(args.embedding_path, map_location="cpu")[
            "weight"
        ].float()
        self.verify_embedding_weight = torch.load(
            args.verify_embedding_path, map_location="cpu"
        )["weight"].float()

        self.context_length = 0
        self.verify_max_step = self.verify.get_input_info(
            self.verify.get_input_name(0)
        ).shape[1]
        self.max_step = min(args.max_step, self.verify_max_step)
        self.context_max_length = min(
            self.decode.get_input_info(self.decode.get_input_name(3)).shape[2],
            self.verify.get_input_info(self.verify.get_input_name(3)).shape[2],
        )
        self.chat_history_ids = []
        self.slide_len = 10
        self.fast = args.fast

    def get_input_names(self, model):
        input_names = []
        for i in range(model.get_num_inputs()):
            input_names.append(model.get_input_name(i))
        return input_names

    def _run_prefill(self, prefill, embedding_weight, all_input_ids):
        input_echo_len = all_input_ids.numel()
        if input_echo_len >= self.context_max_length:
            logger.error(
                f"Question long than {self.context_max_length}, please shorten it!"
            )
            sys.exit(1)

        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        input_name = prefill.get_input_name(0)
        valid_length_name = prefill.get_input_name(1)
        current_length_name = prefill.get_input_name(2)

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
            inputs_embeds = F.embedding(input_ids, embedding_weight)
            effective_length = input_ids.size(-1)
            _pad_embeds = torch.zeros(
                1,
                self.prefill_length - effective_length,
                inputs_embeds.size(-1),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(
                1, self.prefill_length, embedding_weight.shape[-1]
            )
            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")
            prefill.set_input(input_name, input_data.numpy())
            prefill.set_input(valid_length_name, valid_length_data)
            prefill.set_input(current_length_name, current_length_data)
            prefill_start = time.time()
            prefill.run()
            prefill.sync()
            self.prefill_time += time.time() - prefill_start

        input_data = prefill.get_output(prefill.get_output_name(0)).numpy()
        next_id = input_data.argmax(-1)[0]
        self.chat_history_ids = all_input_ids[0]
        self.chat_history_ids = torch.cat(
            [self.chat_history_ids, torch.from_numpy(next_id)], dim=-1
        )
        self.last_response = self.tokenizer.decode(
            self.chat_history_ids.tolist()[-self.slide_len :]
        )

        return next_id

    def _run_decode(self, next_id):
        input_name = self.decode.get_input_name(0)
        valid_length_name = self.decode.get_input_name(1)
        current_length_name = self.decode.get_input_name(2)

        input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(
            1, 1, -1
        )
        valid_length_data = np.array(self.context_length).astype("int32")
        current_length_data = np.array(1).astype("int32")
        self.decode.set_input(input_name, input_data.numpy())
        self.decode.set_input(valid_length_name, valid_length_data)
        self.decode.set_input(current_length_name, current_length_data)
        decode_start = time.time()
        self.decode.run()
        self.decode.sync()
        self.decode_time += time.time() - decode_start
        output_data = self.decode.get_output(self.decode.get_output_name(0)).numpy()
        return output_data

    def _run_verify(self, draft_ids):
        verify_input_name = self.verify.get_input_name(0)
        verify_valid_length_name = self.verify.get_input_name(1)
        verify_current_length_name = self.verify.get_input_name(2)
        ids_embedding = []
        self.verify_steps.append(self.step)
        for id in draft_ids:
            id_embedding = F.embedding(
                id.unsqueeze(0), self.verify_embedding_weight
            ).reshape(1, 1, -1)
            ids_embedding.append(id_embedding)
        verify_input_data = torch.stack(ids_embedding, dim=0).squeeze().unsqueeze(0)
        padding_needed = self.verify_max_step - self.step
        pad_amount = (0, 0, 0, padding_needed, 0, 0)
        verify_input_data = F.pad(
            verify_input_data, pad_amount, mode="constant", value=0
        ).numpy()
        verify_valid_length_data = np.array(self.verify_context_length).astype("int32")
        verify_current_length_data = np.array(self.step).astype("int32")
        self.verify.set_input(verify_valid_length_name, verify_valid_length_data)
        self.verify.set_input(verify_current_length_name, verify_current_length_data)
        self.verify.set_input(verify_input_name, verify_input_data)
        decode_start = time.time()
        self.verify.run()
        self.verify.sync()
        self.decode_time += time.time() - decode_start
        self.verify_time += time.time() - decode_start
        self.verify_round += 1
        verify_output_data = self.verify.get_output(
            self.verify.get_output_name(0)
        ).numpy()
        return verify_output_data

    def _slide_print(self, decode_id):
        self.chat_history_ids = torch.cat(
            [self.chat_history_ids, torch.tensor([decode_id])], dim=-1
        )
        decode_response = self.tokenizer.decode(
            self.chat_history_ids.tolist()[-(self.slide_len + 1) - self.skip_tokens :]
        )[len(self.last_response) :]
        if decode_response != "" and is_valid_char(ord(decode_response[-1])):
            print(decode_response, end="", flush=True)
            self.all_response += decode_response
            self.last_response = self.tokenizer.decode(
                self.chat_history_ids.tolist()[-self.slide_len :]
            )
            self.skip_tokens = 0
        else:
            self.skip_tokens += 1

    def classic_mode(self, next_id, temperature=0.6, top_k=1, top_p=0):
        decode_count = 0
        draft_ids = []
        draft_logits = []

        is_accept_all = False
        is_decode_finish = False
        self.verify_context_length = self.context_length
        while True:
            # draft decode stage0
            draft_ids.append(next_id)
            self.step = self.max_step
            for i in range(self.max_step - 1):
                if self.context_length >= self.context_max_length:
                    logger.info(
                        f"context length greater than {self.context_max_length}, break!"
                    )
                    is_decode_finish = True
                    break

                output_data = self._run_decode(next_id)
                next_id = output_data.argmax(-1)[0]
                next_id = torch.from_numpy(next_id)
                draft_ids.append(next_id)
                draft_logits.append(output_data)
                self.context_length = self.context_length + 1

            # verify stage
            if not is_decode_finish:
                verify_output_data = self._run_verify(draft_ids)
                verified_ids = np.argmax(verify_output_data, axis=2).flatten()
                is_accept_all = True
                to_break = False
                for i in range(self.step - 1):
                    if to_break:
                        break
                    decode_count += 1
                    draft_logit = torch.tensor(
                        draft_logits[i][0][0], requires_grad=False
                    ).unsqueeze(0)
                    verify_logit = torch.tensor(
                        verify_output_data[0][i], requires_grad=False
                    ).unsqueeze(0)
                    draft_distrib = norm_logits(
                        draft_logit, temperature=temperature, top_k=top_k, top_p=top_p
                    ).squeeze(0)
                    verify_distrib = norm_logits(
                        verify_logit, temperature=temperature, top_k=top_k, top_p=top_p
                    ).squeeze(0)

                    # sampling from the draft
                    draft_id = sample(draft_distrib)
                    p = verify_distrib.numpy()[draft_id]
                    q = draft_distrib.numpy()[draft_id]
                    u = np.random.rand(1)[0]
                    if p < q and u > (p / q):
                        # reject and resample
                        new_verify_distrib = torch.maximum(
                            torch.zeros_like(verify_distrib, requires_grad=False),
                            verify_distrib - draft_distrib,
                        )
                        new_verify_distrib = (
                            new_verify_distrib
                            / new_verify_distrib.sum(dim=0, keepdim=True)
                        )
                        next_id = sample(new_verify_distrib)

                        self.accept_nums.append(i)
                        self.context_length = self.context_length - (self.step - i - 2)
                        self.verify_context_length = self.context_length
                        is_accept_all = False
                        to_break = True
                    else:
                        # othereise, accept the draft token
                        next_id = draft_id

                    self._slide_print(next_id)
                    if next_id == self.tokenizer.eos_token_id:
                        is_decode_finish = True
                        self.accept_nums.append(i)
                        break

            # draft decode stage1
            if is_accept_all and not is_decode_finish:
                self.accept_nums.append(self.step - 1)
                if verified_ids[self.step - 1] == self.tokenizer.eos_token_id:
                    is_decode_finish = True
                    break
                _ = self._run_decode(next_id)
                self.draft_decode_count += 1
                self.context_length = self.context_length + 1

                verify_logit = torch.tensor(
                    verify_output_data[0, -1, :], requires_grad=False
                ).unsqueeze(0)
                verify_distrib = norm_logits(
                    verify_logit, temperature=temperature, top_k=top_k, top_p=top_p
                ).squeeze(0)
                next_id = sample(verify_distrib)
                # next_id = torch.tensor([next_id])

                self.verify_context_length = self.context_length
                decode_count += 1
                self._slide_print(next_id)
                if next_id == self.tokenizer.eos_token_id:
                    is_decode_finish = True
            if is_decode_finish:
                break
            draft_ids.clear()
            draft_logits.clear()
        print("\033[0m")
        return decode_count

    def fast_mode(self, next_id):
        decode_count = 0
        draft_ids = []
        draft_logits = []

        is_accept_all = False
        is_decode_finish = False
        self.verify_context_length = self.context_length
        while True:
            # draft decode stage0
            draft_ids.append(next_id)
            is_start_verify = False
            self.step = self.max_step
            for i in range(self.max_step - 1):
                if self.context_length >= self.context_max_length:
                    logger.info(
                        f"context length greater than {self.context_max_length}, break!"
                    )
                    is_decode_finish = True
                    break

                output_data = self._run_decode(next_id)
                next_id = output_data.argmax(-1)[0]

                is_start_verify = is_top2_unvalid(output_data[0][0])
                next_id = torch.from_numpy(next_id)
                draft_ids.append(next_id)
                draft_logits.append(output_data)
                self.context_length = self.context_length + 1
                if is_start_verify:
                    self.step = i + 2
                    break

            # verify stage
            if not is_decode_finish:
                verify_output_data = self._run_verify(draft_ids)
                verified_ids = np.argmax(verify_output_data, axis=2).flatten()
                is_accept_all = True
                for i in range(self.step - 1):
                    decode_count += 1
                    if verified_ids[i] == self.tokenizer.eos_token_id:
                        is_decode_finish = True
                        self.accept_nums.append(i)
                        break
                    self._slide_print(verified_ids[i])
                    if verified_ids[i] != draft_ids[i + 1]:
                        self.accept_nums.append(i)
                        self.context_length = self.context_length - self.step + i + 2
                        self.verify_context_length = self.context_length
                        next_id = torch.tensor([verified_ids[i]])
                        is_accept_all = False
                        break

            # draft decode stage1
            if is_accept_all and not is_decode_finish:
                self.accept_nums.append(self.step - 1)
                if verified_ids[self.step - 1] == self.tokenizer.eos_token_id:
                    is_decode_finish = True
                    break
                _ = self._run_decode(next_id)
                self.draft_decode_count += 1
                self.context_length = self.context_length + 1
                next_id = torch.tensor([verified_ids[self.step - 1]])
                self.verify_context_length = self.context_length
                decode_count += 1
                self._slide_print(verified_ids[self.step - 1])
            if is_decode_finish:
                break
            draft_ids.clear()
        print("\033[0m")
        return decode_count

    def chat(self, question, temperature=0.6, top_k=1, top_p=0):
        self.context_length = 0
        self.prefill_time = 0
        self.decode_time = 0
        self.ttft_time = 0
        self.verify_time = 0
        self.accept_nums = []
        self.verify_steps = []
        self.verify_round = 0
        self.draft_decode_count = 0
        self.skip_tokens = 0
        self.slide_len = 10

        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": question,
            },
        ]
        start_time = time.time()
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()
        next_id = self._run_prefill(self.prefill, self.embedding_weight, all_input_ids)
        next_id = self._run_prefill(
            self.prefill_verify, self.verify_embedding_weight, all_input_ids
        )
        self.ttft_time += time.time() - start_time

        # get prefill output
        prefill_response = self.tokenizer.decode(next_id)
        logger.success("response:")
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)
        next_id = torch.from_numpy(next_id)
        self.all_response = prefill_response
        self.context_length += input_echo_len

        if self.fast:
            decode_count = self.fast_mode(next_id)
        else:
            decode_count = self.classic_mode(
                next_id, temperature=temperature, top_k=top_k, top_p=top_p
            )

        return input_echo_len, decode_count + 1


if __name__ == "__main__":

    args = get_args()
    hmqwen = HmQwen(args)

    if args.it:
        from prompt_toolkit import prompt
    try:
        while True:
            if args.it:
                try:
                    question = prompt("Input your instruction here: ").strip()
                    if question.lower() in ("stop", "exit", "quit"):
                        break
                    if not question:
                        print("Input cannot be empty, please try again.")
                        continue
                except (EOFError, KeyboardInterrupt):
                    print("\nProgram ended")
                    break
            else:
                question = "请介绍一下存算一体技术的优势"

            try:
                start_time = time.time()
                input_tokens, output_tokens = hmqwen.chat(question)
                total_time = time.time() - start_time
                show_statistics(
                    hmqwen,
                    input_tokens,
                    output_tokens,
                    total_time,
                )
            except Exception as e:
                print(f"Error occurred during chat: {e}")
                if not args.it:
                    break
                continue
            if not args.it:
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Error occurred during program execution: {e}")
