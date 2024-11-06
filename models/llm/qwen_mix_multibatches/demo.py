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
import threading
import queue
import argparse


TOKENIZER_PATH = "qwen1.5-7b-chat-hf"
EMBEDDING_PATH = os.path.join('output', os.getenv('HOUMO_TARGET', ''), 'result', 'qwen15_quant_embedding.pt')
MONITER_ID = 0 # Indicate which session's results to view in real-time


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prefill',
        dest='prefill_length',
        type=int,
        default=256,
        help='prefill max length',
    )
    parser.add_argument(
        '--decode',
        dest='decode_length',
        type=int,
        default=4096,
        help='decode max length',
    )
    args = parser.parse_args()
    return args


class HmQwen:

    def __init__(self, prefill_length, decode_length, batch):
        self.batch = batch
        self.prefill_length = prefill_length
        self.decode_length = decode_length
        reuse_inputs = [i for i in range(3, 3+32*batch)]
        weight_manager = tcim.runtime.create_weight_manager(0)
        self.prefill_models = []
        for i in range(batch):
            stream = tcim.runtime.Stream(0)
            prefill_model = {}
            prefill_model["part1"] = tcim.runtime.load('qwen_prefill_part1.hmm', weight_manager=weight_manager)
            prefill_model["part2"] = tcim.runtime.load('qwen_prefill_part2.hmm', weight_manager=weight_manager)
            prefill_model["head"] = tcim.runtime.load("qwen_prefill_head.hmm", weight_manager=weight_manager)
            prefill_model["part1"].set_stream(stream)
            prefill_model["part2"].set_stream(stream)
            prefill_model["head"].set_stream(stream)
            self.prefill_models.append(prefill_model)
        self.decode_part1_model = tcim.runtime.load("qwen_decode_part1.hmm", weight_manager=weight_manager, reuse_inputs=reuse_inputs)
        self.decode_part2_model = tcim.runtime.load("qwen_decode_part2.hmm", weight_manager=weight_manager, reuse_inputs=reuse_inputs)
        self.decode_head_model = tcim.runtime.load("qwen_decode_head.hmm", weight_manager=weight_manager)
        self.stream = tcim.runtime.Stream(0)
        self.decode_part1_model.set_stream(self.stream)
        self.decode_part2_model.set_stream(self.stream)
        self.decode_head_model.set_stream(self.stream)
        # set kvcache input
        for b in range(batch):
            for i in range(16):
                kcache = self.prefill_models[b]["part1"].get_dev_input(f'model_layers_{i}_self_attn_kcache_input')
                self.decode_part1_model.set_input(f'model_layers_{i}_self_attn_kcache_input_batch{b}', kcache)
                vcache = self.prefill_models[b]["part1"].get_dev_input(f'model_layers_{i}_self_attn_vcache_input')
                self.decode_part1_model.set_input(f'model_layers_{i}_self_attn_vcache_input_batch{b}', vcache)
            for i in range(16, 32):
                kcache = self.prefill_models[b]["part2"].get_dev_input(f'model_layers_{i}_self_attn_kcache_input')
                self.decode_part2_model.set_input(f'model_layers_{i}_self_attn_kcache_input_batch{b}', kcache)
                vcache = self.prefill_models[b]["part2"].get_dev_input(f'model_layers_{i}_self_attn_vcache_input')
                self.decode_part2_model.set_input(f'model_layers_{i}_self_attn_vcache_input_batch{b}', vcache)
        # set decode input
        current_length_input_1 = np.array([1 for i in range(batch)]).astype("int16")
        self.decode_part1_model.set_input("current_length", current_length_input_1)
        self.decode_part2_model.set_input("current_length", current_length_input_1)
        decode_part1_output = self.decode_part1_model.get_dev_output("model_layers_15_resadd2")
        self.decode_part2_model.set_input("model_layers_15_resadd2", decode_part1_output)
        decode_part2_output = self.decode_part2_model.get_dev_output('model_layers_31_resadd2')
        self.decode_head_model.set_input("reshape", decode_part2_output)
        # current_length_input_0 = np.array([0 for i in range(batch)]).astype('int16')
        # self.decode_head_model.set_input('current_length', current_length_input_0)

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        embedding_weight = torch.load(EMBEDDING_PATH, map_location="cpu")
        self.embedding_weight = embedding_weight.reshape(-1, 4096)
        self.decode_input_datas = np.ones((batch, 1, 4096))
        self.decode_valid_length = np.ones(batch)
        self.task_queue = queue.Queue()
        task = {}
        task["owner"] = None
        self.task_list = [task for i in range(batch)]

    def prefill(self, i, inputs):
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()
        if input_echo_len >= self.decode_length:
            logger.error(f"Question long than {self.decode_length}, please shorten it!")
            return f"Question long than {self.decode_length}, please shorten it!"

        prefill_part1_output = self.prefill_models[i]["part1"].get_dev_output("model_layers_15_resadd2")
        self.prefill_models[i]["part2"].set_input("model_layers_15_resadd2", prefill_part1_output)

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
            # [256, 1, 4096] ==> [4, 64, 4096]
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(4, self.prefill_length // 4, 4096)
            valid_length_data = np.array([valid_length]).astype("int16")
            current_length_data = np.array([current_length]).astype("int16")
            self.prefill_models[i]["part1"].set_input("input_1", np.array(input_data).astype("int16"))
            self.prefill_models[i]["part1"].set_input("valid_length", valid_length_data)
            self.prefill_models[i]["part1"].set_input("current_length", current_length_data)
            self.prefill_models[i]["part2"].set_input("valid_length", valid_length_data)
            self.prefill_models[i]["part2"].set_input("current_length", current_length_data)
            self.prefill_models[i]["part1"].run()
            self.prefill_models[i]["part2"].run()
            self.prefill_models[i]["part2"].sync()

        prefill_part2_output = self.prefill_models[i]["part2"].get_dev_output("model_layers_31_resadd2")
        self.prefill_models[i]["head"].set_input("model_layers_31_resadd2", prefill_part2_output)
        seq_length_data = np.array([gather_index]).astype("int16")
        self.prefill_models[i]["head"].set_input("current_length", seq_length_data)
        self.prefill_models[i]["head"].run()
        self.prefill_models[i]["head"].sync()
        input_data = self.prefill_models[i]["head"].get_output("lm_head_add_list_0", True)
        self.prefill_models[i]["head"].stream.yield_()
        next_id = input_data.argmax(-1)
        prefill_response = self.tokenizer.decode(next_id.tolist())
        next_id = torch.from_numpy(next_id)
        prefill_out = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(1, 1, -1)
        return prefill_response, np.array(prefill_out), input_echo_len

    def decode(self):
        for i, task in enumerate(self.task_list):
            if task["owner"] is not None:
                if task["context_length"] > self.decode_length:
                    logger.info(f"context length greater than {self.decode_length}, break!")
                    task["owner"].answer(task["response"], task["decode_tokens"], task["prefill_time"], task["decode_time"])
                    task["owner"] = None
                    if i == MONITER_ID:
                        print("\033[0m\n", flush=True)

                self.decode_input_datas[i] = task["input_data"]
                self.decode_valid_length[i] = np.array(task["context_length"] - 1)

        start_time = time.time()
        input_datas = self.decode_input_datas.astype("int16")
        self.decode_part1_model.set_input("input_1", input_datas)
        valid_length_data = self.decode_valid_length.astype("int16")
        self.decode_part1_model.set_input("valid_length", valid_length_data)
        self.decode_part2_model.set_input("valid_length", valid_length_data)
        self.decode_part1_model.run()
        self.decode_part2_model.run()
        self.decode_head_model.run()
        self.decode_head_model.sync()
        input_datas = self.decode_head_model.get_output("lm_head_add_list_0", True)
        self.decode_head_model.stream.yield_()

        for i, task in enumerate(self.task_list):
            if task["owner"] is not None:
                next_id = input_datas[i].argmax(-1)
                decode_response = self.tokenizer.decode(next_id)
                task["context_length"] += 1
                task["decode_tokens"] += 1
                task["decode_time"] += (time.time() - start_time)
                if decode_response == self.tokenizer.eos_token:
                    task["owner"].answer(task["response"], task["decode_tokens"], task["prefill_time"], task["decode_time"])
                    task["owner"] = None
                    if i == MONITER_ID:
                        print("\033[0m\n", flush=True)
                    continue
                task["response"] += decode_response
                if i == MONITER_ID:
                    print("\033[1;95m{}".format(decode_response), end="", flush=True)
                next_id = torch.from_numpy(np.array(next_id))
                input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(1, 1, -1)
                task["input_data"] = np.array(input_data)

    def run(self):
        self.prefill_time = 0
        self.decode_time = 0
        self.prefill_tokens = 0
        self.decode_tokens = 0
        while True:
            if not self.task_queue.empty():
                for i, task in enumerate(self.task_list):
                    if task["owner"] is None:
                        t = self.task_queue.get()
                        messages = [
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                            {"role": "user", "content": t["question"],}
                        ]
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        inputs = self.tokenizer(text, return_tensors="pt")
                        prefill_start = time.time()
                        response, prefill_out, input_echo_len = self.prefill(i, inputs)
                        prefill_time = time.time() - prefill_start
                        self.prefill_time += prefill_time
                        self.prefill_tokens += 1
                        t["input_data"] = prefill_out
                        t["context_length"] = input_echo_len
                        t["decode_tokens"] = 0
                        t["prefill_time"] = prefill_time
                        t["decode_time"] = 0
                        t["response"] = response
                        self.task_list[i] = t
                        if i == MONITER_ID:
                            logger.success("question:")
                            print("\033[1;95m{}\033[0m".format(t["question"]))
                            logger.success("response:")
                            print("\033[1;95m{}".format(response), end="", flush=True)
                        break # The questions will fill the entire task queue if not break
            else:
                finished = True
                for i, task in enumerate(self.task_list):
                    if task["owner"] is not None:
                        finished = False
                        break
                if finished:
                    break
            decode_start = time.time()
            self.decode()
            decode_time = time.time() - decode_start
            self.decode_time += decode_time
            self.decode_tokens += self.batch

    def add_task(self, owner, question):
        task = {}
        task["owner"] = owner
        task["question"] = question
        self.task_queue.put(task)


class Session:

    def __init__(self, hmqwen):
        self.qwen = hmqwen

    def ask(self, question):
        self.question = question
        self.start_time = time.time()
        self.qwen.add_task(self, question)

    def answer(self, response, tokens, prefill_time, decode_time):
        self.total_time = time.time() - self.start_time
        self.response = response
        self.tokens = tokens
        self.prefill_time = prefill_time
        self.decode_time = decode_time

    def show(self):
        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(self.question))
        logger.success("response:")
        print("\033[1;95m{}\033[0m".format(self.response))

        logger.success("total: {} tokens, cost {:.3f} s".format(self.tokens, self.total_time))
        logger.success("prefill time: {:.3f} ms, {:.2f} tokens/s".format(self.prefill_time * 1000, 1 / self.prefill_time))
        decode_latency = self.decode_time * 1000 / (self.tokens - 1)
        logger.success("decode average time: {:.3f} ms, {:.2f} tokens/s".format(decode_latency, 1000 / decode_latency))
        res_latency = self.total_time * 1000 / self.tokens
        logger.success("end2end average time: {:.3f} ms, {:.2f} tokens/s\n".format(res_latency, 1000 / res_latency))


if __name__ == "__main__":

    args = get_args()
    hmqwen = HmQwen(args.prefill_length, args.decode_length, batch = 4)

    questions = [
        "请介绍一下存算一体技术的优势。",
        "你是谁？",
        "1+1=?",
        "Introduce yourself in 200 words.",
        "人类的本质是什么?",
        "意识是大脑的产物还是独立于大脑存在的实体?",
    ]

    sessions = []

    for q in questions:
        session = Session(hmqwen)
        session.ask(q)
        sessions.append(session)

    hmqwen.run()

    for i, s in enumerate(sessions):
        logger.success(f"[session{i+1}]:")
        s.show()

    total_time = hmqwen.prefill_time + hmqwen.decode_time
    total_tokens = hmqwen.prefill_tokens + hmqwen.decode_tokens
    prefill_latency = hmqwen.prefill_time * 1000 / hmqwen.prefill_tokens
    decode_latency = hmqwen.decode_time * 1000 / hmqwen.decode_tokens
    all_latency = total_time * 1000 / total_tokens
    logger.success("[summary]:")
    logger.success("total: {} tokens, cost {:.3f} s".format(total_tokens, total_time))
    logger.success("prefill average time: {:.3f} ms, {:.2f} tokens/s".format(prefill_latency, 1000 / prefill_latency))
    logger.success("decode average time: {:.3f} ms, {:.2f} tokens/s".format(decode_latency, 1000 / decode_latency))
    logger.success("end2end average time: {:.3f} ms, {:.2f} tokens/s\n".format(all_latency, 1000 / all_latency))
