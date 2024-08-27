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
import tcim


TOKENIZER_PATH = "qwen1.5-7b-chat-hf"
EMBEDDING_PATH = os.path.join('output', os.getenv('HOUMO_TARGET', ''), 'result', 'quant_embedding.pt')

class HmQwen:

    def __init__(self):
        weight_manager = tcim.runtime.create_weight_manager()
        self.prefill_part1_model = tcim.runtime.load("qwen_prefill_part1.hmm", weight_manager=weight_manager)
        self.prefill_part2_model = tcim.runtime.load("qwen_prefill_part2.hmm", weight_manager=weight_manager)
        self.prefill_head_model = tcim.runtime.load("qwen_prefill_head.hmm", weight_manager=weight_manager)
        self.decode_part1_model = tcim.runtime.load("qwen_decode_part1.hmm", weight_manager=weight_manager)
        self.decode_part2_model = tcim.runtime.load("qwen_decode_part2.hmm", weight_manager=weight_manager)
        self.decode_head_model = tcim.runtime.load("qwen_decode_head.hmm", weight_manager=weight_manager)
        self.stream = tcim.runtime.Stream()
        self.decode_part1_model.set_stream(self.stream)
        self.decode_part2_model.set_stream(self.stream)
        self.prefill_part1_model.set_stream(self.stream)
        self.prefill_part2_model.set_stream(self.stream)
        self.decode_head_model.set_stream(self.stream)
        self.prefill_head_model.set_stream(self.stream)
        self.qwen1_5tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        embedding_weight = torch.load(EMBEDDING_PATH, map_location="cpu")
        self.embedding_weight = embedding_weight.reshape(-1, 4096)

    def chat(self, question, prefill_length=256, decode_length=1024):
        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))
        start_time = time.time()
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": question,}
        ]
        text = self.qwen1_5tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.qwen1_5tokenizer(text, return_tensors="pt")
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()
        if input_echo_len >= 2048:
            logger.error("Question too long, please shorten it!")
            return "Question too long, please shorten it!"
        
        prefill_part1_output = self.prefill_part1_model.get_dev_output("model_layers_15_resadd2")
        self.prefill_part2_model.set_input("model_layers_15_resadd2", prefill_part1_output)

        prefill_loop_round = math.ceil(input_echo_len / 256)
        for round in range(prefill_loop_round):
            valid_length = round * 256
            if round == prefill_loop_round - 1:
                current_length = input_echo_len - round * 256
                gather_index = current_length - 1
                input_ids = all_input_ids[:, round * 256: input_echo_len]
            else:
                current_length = 256
                input_ids = all_input_ids[:, round * 256: (round + 1) * 256]
            inputs_embeds = F.embedding(input_ids, self.embedding_weight)
            effective_length = input_ids.size(-1)
            _pad_embeds = torch.zeros(1, prefill_length - effective_length, inputs_embeds.size(-1),
                                    dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(4, 64, 4096) # [256, 1, 4096] ==> [4, 64, 4096]
            valid_length_data = np.array([valid_length]).astype("int16")
            current_length_data = np.array([current_length]).astype("int16")
            self.prefill_part1_model.set_input("input_1", input_data)
            self.prefill_part1_model.set_input("valid_length", valid_length_data)
            self.prefill_part1_model.set_input("current_length", current_length_data)
            self.prefill_part1_model.run()
            self.prefill_part2_model.set_input("valid_length", valid_length_data)
            self.prefill_part2_model.set_input("current_length", current_length_data)
            self.prefill_part2_model.run()
            self.prefill_part2_model.sync()

        prefill_part2_output = self.prefill_part2_model.get_dev_output("model_layers_31_resadd2")
        self.prefill_head_model.set_input("layers31_resadd2", prefill_part2_output)
        seq_length_data = np.array([gather_index]).astype("int16")
        self.prefill_head_model.set_input("current_length", seq_length_data)
        self.prefill_head_model.run()
        self.prefill_head_model.sync()
        input_data = self.prefill_head_model.get_output("output", True)
        next_id = input_data.argmax(-1)
        prefill_response = self.qwen1_5tokenizer.decode(next_id.tolist())
        prefill_time = time.time() - start_time

        next_id = torch.from_numpy(next_id)
        input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(1, 1, -1)
        all_response = prefill_response
        context_length = input_echo_len
        # set decode input
        input_data1 = np.array([context_length - 2]).astype("int16")
        self.decode_part1_model.set_input("valid_length", input_data1)
        self.decode_part2_model.set_input("valid_length", input_data1)
        input_data2 = np.array([1]).astype("int16")
        self.decode_part1_model.set_input("current_length", input_data2)
        self.decode_part2_model.set_input("current_length", input_data2)
        # valid_length_input = self.decode_model.get_dev_input("valid_length")
        # valid_length_input.set_stream(self.stream)
        # set decode_head input
        decode_part1_output = self.decode_part1_model.get_dev_output("model_layers_15_resadd2")
        self.decode_part2_model.set_input("model_layers_15_resadd2", decode_part1_output)
        decode_part2_output = self.decode_part2_model.get_dev_output("model_layers_31_resadd2")
        self.decode_head_model.set_input("layers31_resadd2", decode_part2_output)
        decode_count = 0
        logger.success("response:")
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)
        start_time = time.time()
        while True:
            if context_length > decode_length:
                logger.info(f"context length greater than", decode_length, "break!")
                break

            self.decode_part1_model.set_input("input_1", input_data)
            valid_length_data = np.array([context_length - 1]).astype("int16")
            self.decode_part1_model.set_input("valid_length", valid_length_data)
            self.decode_part2_model.set_input("valid_length", valid_length_data)
            # valid_length_input.add(1)
            self.decode_part1_model.run()
            self.decode_part2_model.run()
            self.decode_head_model.run()
            self.decode_head_model.sync()
            input_data = self.decode_head_model.get_output("output", True)
            decode_count += 1

            next_id = input_data.argmax(-1)
            decode_response = self.qwen1_5tokenizer.decode(next_id.tolist())
            if decode_response == self.qwen1_5tokenizer.eos_token:
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
    hmqwen = HmQwen()
    question = "请介绍一下存算一体技术的优势"

    start_time = time.time()
    response, tokens, prefill_time, decode_time = hmqwen.chat(question)
    total_time = time.time() - start_time

    logger.success("total: {} tokens, cost {:.3f} s".format(tokens, total_time))
    logger.success("prefill time: {:.3f} ms, {:.2f} tokens/s".format(prefill_time * 1000, 1 / prefill_time))
    decode_latency = decode_time * 1000 / (tokens - 1)
    logger.success("decode average time: {:.3f} ms, {:.2f} tokens/s".format(decode_latency, 1000 / decode_latency))
    res_latency = total_time * 1000 / tokens
    logger.success("end2end average time: {:.3f} ms, {:.2f} tokens/s".format(res_latency, 1000 / res_latency))

    del hmqwen
