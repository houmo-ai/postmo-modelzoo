#!/usr/bin/python3
# -*- coding: utf-8 -*-
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


TOKENIZER_PATH = "qwen3-8b"
HOUMO_TARGET = os.getenv('HOUMO_TARGET')
EMBEDDING_PATH = os.path.join('output', HOUMO_TARGET, 'hmquant', 'quant_embedding.pt')


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
        '--model_dir',
        dest='model_dir',
        type=str,
        default=os.path.join('output', HOUMO_TARGET),
        help='houmo model dir',
    )
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
        default=8192,
        help='decode max length',
    )
    parser.add_argument(
        '--forbid_flush',
        dest='forbid_flush',
        action="store_true",
        help='forbid flush print n batch',
    )
    parser.add_argument(
        '--nblocks',
        dest='nblocks',
        type=int,
        default=36,
        help='block number',
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default='2',
        help='batch size',
    )
    args = parser.parse_args()
    return args


class HmQwenXh2:
    def __init__(self, model_dir, prefill_length, decode_length, batch=1, nblocks=28, forbid_flush=True):
        self.batch = batch
        self.prefill_length = prefill_length
        self.decode_length = decode_length
        self.nblocks = nblocks
        self.flush = not forbid_flush
        self.next_ids = [0] * self.batch
        self.current_questions = [""] * self.batch
        self.current_echo_lens = [0] * self.batch
        self.current_responses = [""] * self.batch
        self.decode_break = [False] * self.batch
        self.chat_history_ids = [[]] * self.batch
        self.colors = [
            {"color": "\033[94m"},
            {"color": "\033[92m"},
            {"color": "\033[91m"},
            {"color": "\033[95m"},
            {"color": "\033[93m"},
            {"color": "\033[96m"},
            {"color": "\033[33m"},
            {"color": "\033[35m"}
        ]
        weight_manager = tcim.runtime.WeightManager(0)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        dummy_tensor_names = [
            f'model_layers_{i}_self_attn_kcache_input' for i in range(self.nblocks)
        ]
        dummy_tensor_names += [
            f'model_layers_{i}_self_attn_vcache_input' for i in range(self.nblocks)
        ]
        option1.set_dummy_tensors(dummy_tensor_names)
        # load prefill model
        self.prefill = tcim.runtime.load(os.path.join( model_dir, 'qwen3_prefill.hmm'), option=option1)
        # load decode model
        self.decode = tcim.runtime.load(
            os.path.join(
                model_dir, 'qwen3_decode.hmm',
            ), option=option2,
        )
        # set decode input
        for b in range(self.batch):
            index = 2 if b == 0 else 2 * self.nblocks * b + 3 + 2 * b - 1
            current_length_input = np.array([1]).astype('int32')
            self.decode.set_input(self.decode.get_input_name(index), current_length_input)

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        embedding_weight = torch.load(EMBEDDING_PATH, map_location="cpu", weights_only=True)['weight']
        self.embedding_weight = embedding_weight.reshape(-1, 4096)

    def preprocess_prefill(self, question):
        message = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': question},
        ]
        text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
        )
        input = self.tokenizer(text, return_tensors='pt')
        all_input_id = input['input_ids']
        return all_input_id

    def show_response(self):
        if self.flush:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            print("=== 后摩多batch模型推理展示 ===")
            for i in range(self.batch):
                color = self.colors[i]
                status = "✓" if self.decode_break[i] else "●"
                print(f"{color['color']}{"quesion: "}{self.current_questions[i]} {status}:{'\033[0m'}")
                print(f"  {self.current_responses[i]}")

    def run_prefill(self, b, all_input_id):
        decode_input_index_start = 2 * self.nblocks * b + 3 + 2 * b  if b > 0 else 3
        decode_input_index_finish = 2 * self.nblocks * (b + 1) + 3 + b * 2
        prefill_input_index = 3
        for i in range(decode_input_index_start, decode_input_index_finish):
            cache = self.decode.get_dev_input(self.decode.get_input_name(i))
            self.prefill.set_input(self.prefill.get_input_name(prefill_input_index), cache)
            prefill_input_index += 1
        input_echo_len = all_input_id.numel()
        if input_echo_len >= self.decode_length:
            logger.error(f"Question long than {self.decode_length}, please shorten it!")
            return f"Question long than {self.decode_length}, please shorten it!"
        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        prefill_start_time = time.time()
        for round in range(prefill_loop_round):
            valid_length = round * self.prefill_length
            if round == prefill_loop_round - 1:
                current_length = input_echo_len - round * self.prefill_length
                input_ids = all_input_id[:, round * self.prefill_length: input_echo_len]
            else:
                current_length = self.prefill_length
                input_ids = all_input_id[:, round * self.prefill_length: (round + 1) * self.prefill_length]
            inputs_embeds = F.embedding(input_ids, self.embedding_weight)
            effective_length = input_ids.size(-1)
            _pad_embeds = torch.zeros(1, self.prefill_length - effective_length, inputs_embeds.size(-1),
                                    dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(1, self.prefill_length, 4096)
            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")
            input_name = self.prefill.get_input_name(0)
            valid_length_name = self.prefill.get_input_name(1)
            current_length_name = self.prefill.get_input_name(2)
            self.prefill.set_input(input_name, input_data.float().numpy())
            self.prefill.set_input(valid_length_name, valid_length_data)
            self.prefill.set_input(current_length_name, current_length_data)
            self.prefill.run()
            self.prefill.sync()

        input_data = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        prefill_time = time.time() - prefill_start_time
        next_id = input_data.argmax(-1)[0]
        prefill_response = self.tokenizer.decode(next_id.tolist())
        next_id = torch.from_numpy(next_id)
        self.chat_history_ids[b] = torch.cat([all_input_id.reshape(-1), next_id], dim=-1)
        return prefill_time, input_echo_len, prefill_response, next_id

    def run_decode(self, input_datas):
        input_name = self.decode.get_input_name(0)
        input_datas = np.array(input_datas)
        decode_start_time = time.time()
        self.decode.set_input(input_name, input_datas)
        for b in range(self.batch):
            valid_length_index = 1 if b == 0 else 2 * self.nblocks * b + 3 + 2 * b - 2
            valid_length_data = np.array(self.context_lengths[b]).astype('int32')
            self.decode.set_input(self.decode.get_input_name(valid_length_index), valid_length_data)
        self.decode.run()
        self.decode.sync()
        input_datas = self.decode.get_output(self.decode.get_output_name(0)).to_host().numpy()
        decode_time = time.time() - decode_start_time
        return decode_time, input_datas

    def chat(self, questions):
        max_decode_count = 0
        all_questions = []
        all_responses = []
        input_echo_lens = []
        all_prefill_time = 0
        all_decode_time = 0
        for b in range(self.batch):
            question = questions.pop(0)
            self.current_questions[b] = question
            all_input_id = self.preprocess_prefill(question)
            prefill_time, input_echo_len, prefill_response, next_id = self.run_prefill(b, all_input_id)
            all_prefill_time += prefill_time
            self.next_ids[b] = next_id
            self.current_responses[b] = prefill_response
            self.current_echo_lens[b] = input_echo_len
        slide_len = 10  # sliding window length for decode
        skip_tokens = [0] * self.batch
        decode_responses = [''] * self.batch
        last_responses = [self.tokenizer.decode(
            self.chat_history_ids[b].tolist()[-slide_len:],
        ) for b in range(self.batch)]
        input_datas = [
            F.embedding(
                self.next_ids[b].unsqueeze(0),
                self.embedding_weight,
            ).reshape(1, -1).float().numpy() for b in range(self.batch)
        ]
        self.context_lengths = self.current_echo_lens
        self.show_response()

        while not all(self.decode_break[:self.batch]):
            decode_time, input_datas = self.run_decode(np.array(input_datas))
            max_decode_count += 1
            all_decode_time += decode_time
            self.next_ids = [input_datas[b].argmax(-1) for b in range(self.batch)]
            self.next_ids = [torch.from_numpy(next_id) for next_id in self.next_ids]
            for b in range(self.batch):
                if self.decode_break[b] >= self.decode_length:
                    self.decode_break[b] = True
                if self.next_ids[b] == self.tokenizer.eos_token_id:
                    self.current_responses[b] += decode_responses[b]
                    self.decode_break[b] = True
                else:
                    self.context_lengths[b] += 1
                if not self.decode_break[b]:
                    self.chat_history_ids[b] = torch.cat([self.chat_history_ids[b], self.next_ids[b]], dim=-1)
                    decode_responses[b] = self.tokenizer.decode(
                        self.chat_history_ids[b].tolist()[-(slide_len+1)-skip_tokens[b]:],
                    )[len(last_responses[b]):]
                else:
                    decode_responses[b] = ''
                if decode_responses[b] != '' and is_valid_char(ord(decode_responses[b][-1])):
                    self.current_responses[b] += decode_responses[b]
                    last_responses[b] = self.tokenizer.decode(
                        self.chat_history_ids[b].tolist()[-slide_len:],
                    )
                    skip_tokens[b] = 0
                else:
                    skip_tokens[b] += 1
            self.show_response()
            for b in range(self.batch):
                if self.decode_break[b] == True and len(questions) != 0:
                    question = questions.pop(0)
                    all_questions.append(self.current_questions[b])
                    all_responses.append(self.current_responses[b])
                    input_echo_lens.append(self.current_echo_lens[b])
                    self.current_questions[b] = question
                    all_input_id = self.preprocess_prefill(question)
                    prefill_time, input_echo_len, prefill_response, next_id = self.run_prefill(b, all_input_id)
                    all_prefill_time += prefill_time
                    self.next_ids[b] = next_id
                    self.current_responses[b] = prefill_response
                    self.current_echo_lens[b] = input_echo_len
                    self.context_lengths[b] = input_echo_len
                    self.decode_break[b] = False
                    skip_tokens[b] = 0
                    last_responses[b] = self.tokenizer.decode(self.chat_history_ids[b].tolist()[-slide_len:])

            input_datas = []
            for b in range(self.batch):
                next_id = torch.from_numpy(np.array(self.next_ids[b]))
                input_datas.append(
                    F.embedding(
                        next_id.unsqueeze(
                            0,
                        ), self.embedding_weight,
                    ).reshape(1, -1).float().numpy(),
                )
        sys.stdout.write("\033[0m")
        sys.stdout.flush()
        for b in range(self.batch):
            all_questions.append(self.current_questions[b])
            all_responses.append(self.current_responses[b])
            input_echo_lens.append(self.current_echo_lens[b])
        return all_questions, all_responses, sum(input_echo_lens), max_decode_count*self.batch, all_prefill_time, all_decode_time


if __name__ == "__main__":

    args = get_args()
    if HOUMO_TARGET == 'xh2':
        hmqwen = HmQwenXh2(args.model_dir, args.prefill_length, args.decode_length, nblocks=args.nblocks, batch=args.batch, forbid_flush=args.forbid_flush)
    questions = ["1+1=?", "你好", "请介绍一下存算一体技术的优势", "请介绍一下时间晶体", "介绍一下后摩智能"]
    start_time = time.time()
    questions, responses, input_tokens, decode_tokens, prefill_time, decode_time = hmqwen.chat(questions)
    for i in range(len(questions)):
        print(f"{"\033[35m"}{"Question{}:".format(i)}")
        print(f"{"\033[96m"}{questions[i]}")
        print(f"{"\033[35m"}{"Response{}:".format(i)}")
        print(f"{"\033[96m"}{responses[i]}")

    total_time = time.time() - start_time

    logger.success(f"total: {input_tokens} tokens, cost {total_time:.3f} s")
    logger.success(f"prefill time: {prefill_time * 1000:.3f} ms, {input_tokens / prefill_time:.2f} tokens/s")
    decode_latency = decode_time * 1000 / (decode_tokens)
    logger.success(f"decode average time: {decode_latency:.3f} ms, {1000 / decode_latency:.2f} tokens/s")
    res_latency = total_time * 1000 / (decode_tokens + len(responses))
    logger.success(f"end2end average time: {res_latency:.3f} ms, {1000 / res_latency:.2f} tokens/s")
