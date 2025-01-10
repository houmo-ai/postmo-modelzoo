#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import torch
import traceback
import math
import numpy as np
from loguru import logger
import torch.nn.functional as F
from transformers import AutoTokenizer
import time
import threading
import queue
import argparse

import tcim_lite as tcim


TOKENIZER_PATH = "qwen2-7b-instruct-hf"
EMBEDDING_PATH = 'quant_embedding.pt'


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prefill',
        dest='prefill_length',
        type=int,
        default=128,
        help='prefill max length',
    )
    parser.add_argument(
        '--decode',
        dest='decode_length',
        type=int,
        default=4096,
        help='decode max length',
    )
    parser.add_argument(
        '--nblocks',
        dest='nblocks',
        type=int,
        default=28,
        help='block number',
    )
    args = parser.parse_args()
    return args


class HmQwen:

    def __init__(self, prefill_length, decode_length, batch=1, nblocks=28):
        self.batch = batch
        self.prefill_length = prefill_length
        self.decode_length = decode_length
        weight_manager = tcim.runtime.WeightManager(0)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        dummy_tensor_names = [f'model_layers_{i}_self_attn_kcache_input' for i in range(nblocks)]
        dummy_tensor_names += [f'model_layers_{i}_self_attn_vcache_input' for i in range(nblocks)]
        option2.set_dummy_tensors(dummy_tensor_names)
        self.prefill_part1_model = tcim.runtime.load("qwen2_prefill_part1.hmm", option = option1)
        self.prefill_part2_model = tcim.runtime.load("qwen2_prefill_part2.hmm", option = option1)
        self.prefill_head_model = tcim.runtime.load("qwen2_prefill_head.hmm", option = option1)
        self.decode_part1_model = tcim.runtime.load("qwen2_decode_part1.hmm", option = option2)
        self.decode_part2_model = tcim.runtime.load("qwen2_decode_part2.hmm", option = option2)
        self.decode_head_model = tcim.runtime.load("qwen2_decode_head.hmm", option = option1)
        self.stream = tcim.runtime.Stream(0)
        self.prefill_part1_model.set_stream(self.stream)
        self.prefill_part2_model.set_stream(self.stream)
        self.prefill_head_model.set_stream(self.stream)
        self.decode_part1_model.set_stream(self.stream)
        self.decode_part2_model.set_stream(self.stream)
        self.decode_head_model.set_stream(self.stream)
        # set kvcache input
        for i in range(nblocks//2):
            kcache = self.prefill_part1_model.get_input(f'model_layers_{i}_self_attn_kcache_input')
            self.decode_part1_model.set_input(f'model_layers_{i}_self_attn_kcache_input', kcache)
            vcache = self.prefill_part1_model.get_input(f'model_layers_{i}_self_attn_vcache_input')
            self.decode_part1_model.set_input(f'model_layers_{i}_self_attn_vcache_input', vcache)
        for i in range(nblocks//2, nblocks):
            kcache = self.prefill_part2_model.get_input(f'model_layers_{i}_self_attn_kcache_input')
            self.decode_part2_model.set_input(f'model_layers_{i}_self_attn_kcache_input', kcache)
            vcache = self.prefill_part2_model.get_input(f'model_layers_{i}_self_attn_vcache_input')
            self.decode_part2_model.set_input(f'model_layers_{i}_self_attn_vcache_input', vcache)
        # set decode input
        current_length_input_1 = np.array([1]).astype("int16")
        self.decode_part1_model.set_input("current_length", current_length_input_1)
        self.decode_part2_model.set_input("current_length", current_length_input_1)
        decode_part1_output = self.decode_part1_model.get_output(f"model_layers_{nblocks//2-1}_resadd2", tcim.runtime.Device.HDPL)
        self.decode_part2_model.set_input(f"model_layers_{nblocks//2-1}_resadd2", decode_part1_output)
        decode_part2_output = self.decode_part2_model.get_output(f'model_layers_{nblocks-1}_resadd2', tcim.runtime.Device.HDPL)
        self.decode_head_model.set_input(f"model_layers_{nblocks-1}_resadd2", decode_part2_output)

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        embedding_weight = torch.load(EMBEDDING_PATH, map_location="cpu")
        self.embedding_weight = embedding_weight.reshape(-1, 3584)

    def chat(self, question, nblocks=28):
        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))
        start_time = time.time()
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": question,}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()
        if input_echo_len >= self.decode_length:
            logger.error(f"Question long than {self.decode_length}, please shorten it!")
            return f"Question long than {self.decode_length}, please shorten it!"

        prefill_part1_output = self.prefill_part1_model.get_output(f"model_layers_{nblocks//2-1}_resadd2", tcim.runtime.Device.HDPL)
        self.prefill_part2_model.set_input(f"model_layers_{nblocks//2-1}_resadd2", prefill_part1_output)

        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        for round in range(prefill_loop_round):
            valid_length = round * self.prefill_length
            if round == prefill_loop_round - 1:
                current_length = input_echo_len - round * self.prefill_length
                gather_index = current_length - 1
                input_ids = all_input_ids[:, round * self.prefill_length: input_echo_len]
            else:
                current_length = self.prefill_length
                input_ids = all_input_ids[:, round * self.prefill_length: (round + 1) * self.prefill_length]
            inputs_embeds = F.embedding(input_ids, self.embedding_weight)
            effective_length = input_ids.size(-1)
            _pad_embeds = torch.zeros(1, self.prefill_length - effective_length, inputs_embeds.size(-1),
                                    dtype=inputs_embeds.dtype, device=inputs_embeds.device)
             # [256, 1, 3584] ==> [4, 64, 3584]
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(4, self.prefill_length // 4, 3584)
            valid_length_data = np.array([valid_length]).astype("int16")
            current_length_data = np.array([current_length]).astype("int16")
            self.prefill_part1_model.set_input("input_1", input_data.numpy())
            self.prefill_part1_model.set_input("valid_length", valid_length_data)
            self.prefill_part1_model.set_input("current_length", current_length_data)
            self.prefill_part2_model.set_input("valid_length", valid_length_data)
            self.prefill_part2_model.set_input("current_length", current_length_data)
            self.prefill_part1_model.run()
            self.prefill_part2_model.run()
            self.prefill_part2_model.sync()

        prefill_part2_output = self.prefill_part2_model.get_output(f"model_layers_{nblocks-1}_resadd2", tcim.runtime.Device.HDPL)
        self.prefill_head_model.set_input(f"model_layers_{nblocks-1}_resadd2", prefill_part2_output)
        seq_length_data = np.array([gather_index]).astype("int16")
        self.prefill_head_model.set_input("current_length", seq_length_data)
        self.prefill_head_model.run()
        self.prefill_head_model.sync()
        input_data = self.prefill_head_model.get_output("lm_head_add_list_0").numpy()
        next_id = input_data.argmax(-1)
        prefill_response = self.tokenizer.decode(next_id.tolist())
        prefill_time = time.time() - start_time

        next_id = torch.from_numpy(next_id)
        input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(1, 1, -1)
        all_response = prefill_response
        context_length = input_echo_len

        decode_count = 0
        logger.success("response:")
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)
        start_time = time.time()
        while True:
            if context_length > self.decode_length:
                logger.info(f"context length greater than {self.decode_length}, break!")
                break

            self.decode_part1_model.set_input("input_1", input_data.numpy())
            valid_length_data = np.array([context_length - 1]).astype("int16")
            self.decode_part1_model.set_input("valid_length", valid_length_data)
            self.decode_part2_model.set_input("valid_length", valid_length_data)
            self.decode_part1_model.run()
            self.decode_part2_model.run()
            self.decode_head_model.run()
            self.decode_head_model.sync()
            input_data = self.decode_head_model.get_output("lm_head_add_list_0").numpy()
            decode_count += 1

            next_id = input_data.argmax(-1)
            decode_response = self.tokenizer.decode(next_id.tolist()[0])
            if decode_response == self.tokenizer.eos_token:
                break

            next_id = torch.from_numpy(next_id)
            input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(1, 1, -1)
            all_response = all_response + decode_response
            context_length = context_length + 1
            print(decode_response, end="", flush=True)

        decode_time = time.time() - start_time
        self.stream.yield_()
        print("\033[0m")

        return all_response, decode_count + 1, prefill_time, decode_time


if __name__ == "__main__":

    args = get_args()
    hmqwen = HmQwen(args.prefill_length, args.decode_length, nblocks=args.nblocks)
    question = "请介绍一下存算一体技术的优势"

    start_time = time.time()
    response, tokens, prefill_time, decode_time = hmqwen.chat(question, nblocks=args.nblocks)
    total_time = time.time() - start_time

    logger.success("total: {} tokens, cost {:.3f} s".format(tokens, total_time))
    logger.success("prefill time: {:.3f} ms, {:.2f} tokens/s".format(prefill_time * 1000, 1 / prefill_time))
    decode_latency = decode_time * 1000 / (tokens - 1)
    logger.success("decode average time: {:.3f} ms, {:.2f} tokens/s".format(decode_latency, 1000 / decode_latency))
    res_latency = total_time * 1000 / tokens
    logger.success("end2end average time: {:.3f} ms, {:.2f} tokens/s".format(res_latency, 1000 / res_latency))
