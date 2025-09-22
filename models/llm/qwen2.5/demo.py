#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import re
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

HOUMO_TARGET = os.getenv('HOUMO_TARGET')

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
        '--tokenizer_dir',
        dest='tokenizer_dir',
        type=str,
        default="qwen2.5-7b-instruct-hf",
        help='tokenizer dir',
    )
    parser.add_argument(
        '--embedding_path',
        dest='embedding_path',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, 'hmquant', 'quant_embedding.pt'),
        help='houmo embedding weight path',
    )
    parser.add_argument(
        '--prefill_path',
        dest='prefill_path',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, "qwen2.5_prefill.hmm"),
        help='houmo prefill model path',
    )
    parser.add_argument(
        '--decode_path',
        dest='decode_path',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, "qwen2.5_decode.hmm"),
        help='houmo decode model path',
    )
    parser.add_argument(
        '--ndevice',
        dest='ndevice',
        type=int,
        default=1,
        choices=[1, 2],
        help='device number, only xh2 support',
    )
    args = parser.parse_args()
    return args

class HmQwen:

    def __init__(self, prefill_path, decode_path, embedding_path, tokenizer_dir):
        weight_manager = tcim.runtime.WeightManager(0)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option1)
        logger.info("prefill model loaded")
        self.nblocks = self.get_nblocks()
        dummy_tensor_names = [
            f'model_layers_{i}_self_attn_kcache_input' for i in range(self.nblocks)
        ]
        dummy_tensor_names += [
            f'model_layers_{i}_self_attn_vcache_input' for i in range(self.nblocks)
        ]
        option2.set_dummy_tensors(dummy_tensor_names)
        self.decode = tcim.runtime.load(decode_path, option=option2)
        logger.info("decode model loaded")
        prefill_input_shape = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape
        self.prefill_length = prefill_input_shape[1]*prefill_input_shape[0]
        self.embedding_len = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[2]
        self.context_max_length = self.decode.get_input_info(self.decode.get_input_name(3)).shape[2]
        self.batch = self.decode.get_input_info(self.decode.get_input_name(0)).shape[0]
        # set kvcache input
        for i in range(self.nblocks):
            kcache = self.prefill.get_input(f'model_layers_{i}_self_attn_kcache_input')
            self.decode.set_input(f'model_layers_{i}_self_attn_kcache_input', kcache)
            vcache = self.prefill.get_input(f'model_layers_{i}_self_attn_vcache_input')
            self.decode.set_input(f'model_layers_{i}_self_attn_vcache_input', vcache)
        # set decode input
        current_length_input_1 = np.array([1]).astype("int16")
        self.decode.set_input("current_length", current_length_input_1)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        embedding_weight = torch.load(embedding_path, map_location="cpu")
        self.embedding_weight = embedding_weight.reshape(-1, self.embedding_len)

    def get_nblocks(self):
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        pattern = r'^model_layers_(\d+)_self_attn_kcache_input$'
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def chat(self, question):
        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))
        start_time = time.time()
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": question,}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        text = self.tokenizer.batch_decode(inputs.input_ids)[0]
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()
        if input_echo_len >= self.context_max_length:
            logger.error(f"Question long than {self.context_max_length}, please shorten it!")
            sys.exit(1)

        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        for round in range(prefill_loop_round):
            valid_length = round * self.prefill_length
            if round == prefill_loop_round - 1:
                current_length = input_echo_len - round * self.prefill_length
                input_ids = all_input_ids[:, round * self.prefill_length: input_echo_len]
            else:
                current_length = self.prefill_length
                input_ids = all_input_ids[:, round * self.prefill_length: (round + 1) * self.prefill_length]
            inputs_embeds = F.embedding(input_ids, self.embedding_weight)
            effective_length = input_ids.size(-1)
            _pad_embeds = torch.zeros(1, self.prefill_length - effective_length, inputs_embeds.size(-1),
                                      dtype=inputs_embeds.dtype, device=inputs_embeds.device)

            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(4, self.prefill_length // 4, self.embedding_len)
            valid_length_data = np.array([valid_length]).astype("int16")
            current_length_data = np.array([current_length]).astype("int16")
            self.prefill.set_input("input_1", input_data.numpy())
            self.prefill.set_input("valid_length", valid_length_data)
            self.prefill.set_input("current_length", current_length_data)
            self.prefill.run()
            self.prefill.sync()

        input_data = self.prefill.get_output("Output_lm_head_add_list_0").numpy()
        next_id = input_data.argmax(-1)
        prefill_response = self.tokenizer.decode(next_id.tolist())
        prefill_time = time.time() - start_time
        chat_history_ids = all_input_ids[0]
        next_id = torch.from_numpy(next_id)

        chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)
        input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(1, 1, -1)
        all_response = prefill_response
        context_length = input_echo_len
        logger.success("response:")
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)

        decode_count = 0
        skip_tokens = 0
        slide_len = 10  # sliding window length for decode
        last_response = self.tokenizer.decode(chat_history_ids.tolist()[-slide_len:])

        start_time = time.time()
        while True:
            if context_length >= self.context_max_length:
                logger.info(f"context length greater than {self.context_max_length}, break!")
                break

            self.decode.set_input("input_1", input_data.numpy())
            valid_length_data = np.array(context_length).astype("int16")
            self.decode.set_input("valid_length", valid_length_data)
            self.decode.run()
            self.decode.sync()
            input_data = self.decode.get_output("Output_lm_head_add_list_0").numpy()
            decode_count += 1

            next_id = input_data.argmax(-1)
            next_id = torch.from_numpy(next_id)
            if next_id == self.tokenizer.eos_token_id:
                print(decode_response, end="",flush=True)
                all_response += decode_response
                break

            chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)
            decode_response = self.tokenizer.decode(chat_history_ids.tolist()[-(slide_len+1)-skip_tokens:])[len(last_response):]
            if decode_response != '' and is_valid_char(ord(decode_response[-1])):
                print(decode_response, end="", flush=True)
                all_response += decode_response
                last_response = self.tokenizer.decode(chat_history_ids.tolist()[-slide_len:])
                skip_tokens = 0
            else:
                skip_tokens += 1

            input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(1, 1, -1)
            context_length = context_length + 1

        decode_time = time.time() - start_time
        print("\033[0m")

        return all_response, input_echo_len, decode_count + 1, prefill_time, decode_time

class HmQwenXh2:

    def __init__(self, prefill_path, decode_path, embedding_path, tokenizer_dir, ndevice):
        self.ndevice = ndevice
        if self.ndevice==1:
            weight_manager = tcim.runtime.WeightManager(0)
        elif self.ndevice==2:
            dev_manager = tcim.runtime.DevManager([1,0], "Xh2HalBackend")
            weight_manager = tcim.runtime.WeightManager(dev_manager)
        else:
            raise ValueError("Unsupport device number!")
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option1)
        logger.info("prefill model loaded")
        self.nblocks = self.get_nblocks()
        dummy_tensor_names = [
            f'model_layers_{i}_self_attn_kcache_input' for i in range(self.nblocks)
        ]
        dummy_tensor_names += [
            f'model_layers_{i}_self_attn_vcache_input' for i in range(self.nblocks)
        ]
        option2.set_dummy_tensors(dummy_tensor_names)
        self.decode = tcim.runtime.load(decode_path, option=option2)
        logger.info("decode model loaded")
        self.prefill_length = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[1]
        self.embedding_len = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[2]
        self.context_max_length = self.decode.get_input_info(self.decode.get_input_name(3)).shape[2]
        self.batch = self.decode.get_input_info(self.decode.get_input_name(0)).shape[0]
        for i in range(3, 2 * self.nblocks + 3):
            cache = self.prefill.get_input(self.prefill.get_input_name(i))
            self.decode.set_input(self.decode.get_input_name(i), cache)
        # set decode input
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode.get_input_name(2)
        self.decode.set_input(decode_current_length_name, current_length_input_1)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        embedding_weight = torch.load(embedding_path, map_location="cpu", weights_only=True)['weight']
        self.embedding_weight = embedding_weight.reshape(-1, self.embedding_len)

    def get_nblocks(self):
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        pattern = r'^model_layers_(\d+)_self_attn_kcache_input$'
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def chat(self, question):
        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))
        start_time = time.time()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question,}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        text = self.tokenizer.batch_decode(inputs.input_ids)[0]
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()
        if input_echo_len >= self.context_max_length:
            logger.error(f"Question long than {self.context_max_length}, please shorten it!")
            sys.exit(1)

        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        for round in range(prefill_loop_round):
            valid_length = round * self.prefill_length
            if round == prefill_loop_round - 1:
                current_length = input_echo_len - round * self.prefill_length
                input_ids = all_input_ids[:, round * self.prefill_length: input_echo_len]
            else:
                current_length = self.prefill_length
                input_ids = all_input_ids[:, round * self.prefill_length: (round + 1) * self.prefill_length]
            inputs_embeds = F.embedding(input_ids, self.embedding_weight)
            effective_length = input_ids.size(-1)
            _pad_embeds = torch.zeros(1, self.prefill_length - effective_length, inputs_embeds.size(-1),
                                      dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            # [256, 1, 4096] ==> [4, 64, 4096]
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(1, self.prefill_length, self.embedding_len)
            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")
            input_name = self.prefill.get_input_name(0)
            valid_length_name = self.prefill.get_input_name(1)
            current_length_name = self.prefill.get_input_name(2)
            self.prefill.set_input(input_name, input_data.numpy())
            self.prefill.set_input(valid_length_name, valid_length_data)
            self.prefill.set_input(current_length_name, current_length_data)
            self.prefill.run()
            self.prefill.sync()

        input_data = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        next_id = input_data.argmax(-1)[0]
        prefill_response = self.tokenizer.decode(next_id)
        prefill_time = time.time() - start_time
        chat_history_ids = all_input_ids[0]
        next_id = torch.from_numpy(next_id)

        chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)
        input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(1, 1, -1)
        all_response = prefill_response
        context_length = input_echo_len
        logger.success("response:")
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)

        decode_count = 0
        skip_tokens = 0
        slide_len = 10  # sliding window length for decode
        last_response = self.tokenizer.decode(chat_history_ids.tolist()[-slide_len:])

        start_time = time.time()
        while True:
            if context_length >= self.context_max_length:
                logger.info(f"context length greater than {self.context_max_length}, break!")
                break

            input_name = self.decode.get_input_name(0)
            valid_length_name = self.decode.get_input_name(1)
            self.decode.set_input(input_name, input_data.numpy())
            valid_length_data = np.array(context_length).astype("int32")
            self.decode.set_input(valid_length_name, valid_length_data)
            self.decode.run()
            self.decode.sync()
            input_data = self.decode.get_output(self.decode.get_output_name(0)).numpy()
            decode_count += 1

            next_id = input_data.astype(np.float32).argmax(-1)[0]
            next_id = torch.from_numpy(next_id)
            if next_id == self.tokenizer.eos_token_id:
                print(decode_response, end="",flush=True)
                all_response += decode_response
                break

            chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)
            decode_response = self.tokenizer.decode(chat_history_ids.tolist()[-(slide_len+1)-skip_tokens:])[len(last_response):]
            if decode_response != '' and is_valid_char(ord(decode_response[-1])):
                print(decode_response, end="", flush=True)
                all_response += decode_response
                last_response = self.tokenizer.decode(chat_history_ids.tolist()[-slide_len:])
                skip_tokens = 0
            else:
                skip_tokens += 1

            input_data = F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(1, 1, -1)
            context_length = context_length + 1

        decode_time = time.time() - start_time
        print("\033[0m")

        return all_response, input_echo_len, decode_count + 1, prefill_time, decode_time


if __name__ == "__main__":

    args = get_args()
    if HOUMO_TARGET == 'xh1':
        hmqwen = HmQwen(args.prefill_path, args.decode_path, args.embedding_path, args.tokenizer_dir)
    elif HOUMO_TARGET == 'xh2':
        hmqwen = HmQwenXh2(args.prefill_path, args.decode_path, args.embedding_path, args.tokenizer_dir, args.ndevice)
    question = "请介绍一下存算一体技术的优势"

    start_time = time.time()
    response, input_tokens, output_tokens, prefill_time, decode_time = hmqwen.chat(question)
    total_time = time.time() - start_time

    logger.success(f"Total Input: {input_tokens} tokens, Output {output_tokens} tokens, Prefill Cost {prefill_time*1000:.3f} ms, Decode Cost {decode_time*1000:.3f} ms")
    logger.success(f"Prefill Speed: {input_tokens / prefill_time:.2f} tokens/s")
    logger.success(f"TTFT (Time to First Token): {prefill_time * 1000:.3f} ms")
    logger.success(f"TPOT (Time Per Output Token): {(output_tokens - 1) / decode_time:.2f} tokens/s")
    logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
    logger.success(f"TPS (Tokens Per Second): {output_tokens / total_time:.2f} tokens/s")
