#!/usr/bin/python3
# -*- coding: utf-8 -*-
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


TOKENIZER_PATH = "qwen2.5-7b-instruct-hf"
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

    def __init__(self, model_dir, prefill_length, decode_length, batch=1, nblocks=28):
        self.batch = batch
        self.prefill_length = prefill_length
        self.decode_length = decode_length
        self.nblocks = nblocks
        weight_manager = tcim.runtime.WeightManager(0)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        dummy_tensor_names = [f'model_layers_{i}_self_attn_kcache_input' for i in range(nblocks)]
        dummy_tensor_names += [f'model_layers_{i}_self_attn_vcache_input' for i in range(nblocks)]
        dummy_tensor_names += [f'model_layers_{i}_self_attn_kcache_history_sum' for i in range(nblocks)]
        option2.set_dummy_tensors(dummy_tensor_names)
        self.prefill_model = tcim.runtime.load(os.path.join(model_dir, "deepseek_prefill.hmm"), option = option1)
        self.decode_model = tcim.runtime.load(os.path.join(model_dir, "deepseek_decode.hmm"), option = option2)
        # set kvcache input
        for i in range(nblocks):
            kcache = self.prefill_model.get_input(f'model_layers_{i}_self_attn_kcache_input')
            self.decode_model.set_input(f'model_layers_{i}_self_attn_kcache_input', kcache)
            vcache = self.prefill_model.get_input(f'model_layers_{i}_self_attn_vcache_input')
            self.decode_model.set_input(f'model_layers_{i}_self_attn_vcache_input', vcache)
            kcache_history_sum = self.prefill_model.get_input(f'model_layers_{i}_self_attn_kcache_history_sum')
            self.decode_model.set_input(f'model_layers_{i}_self_attn_kcache_history_sum', kcache_history_sum)
        # set decode input
        current_length_input_1 = np.array([1]).astype("int16")
        self.decode_model.set_input("current_length", current_length_input_1)

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        embedding_weight = torch.load(EMBEDDING_PATH, map_location="cpu")
        self.embedding_weight = embedding_weight.reshape(-1, 3584)

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
        if input_echo_len >= self.decode_length:
            logger.error(f"Question long than {self.decode_length}, please shorten it!")
            return f"Question long than {self.decode_length}, please shorten it!"

        # clear kcache_history_sum before prefill
        for i in range(self.nblocks):
            kcache_history_sum = self.prefill_model.get_input(f'model_layers_{i}_self_attn_kcache_history_sum')
            kcache_history_sum_init = np.zeros(kcache_history_sum.info.shape, dtype=kcache_history_sum.info.dtype)
            self.prefill_model.set_input(f'model_layers_{i}_self_attn_kcache_history_sum', kcache_history_sum_init)

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
            # [256, 1, 3584] ==> [4, 64, 3584]
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(4, self.prefill_length // 4, 3584)
            valid_length_data = np.array([valid_length]).astype("int16")
            current_length_data = np.array([current_length]).astype("int16")
            self.prefill_model.set_input("input_1", input_data.numpy())
            self.prefill_model.set_input("valid_length", valid_length_data)
            self.prefill_model.set_input("current_length", current_length_data)
            self.prefill_model.run()
            self.prefill_model.sync()

        input_data = self.prefill_model.get_output("Output_lm_head_add_list_0").numpy()
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
            if context_length >= self.decode_length:
                logger.info(f"context length greater than {self.decode_length}, break!")
                break

            self.decode_model.set_input("input_1", input_data.numpy())
            valid_length_data = np.array(context_length).astype("int16")
            self.decode_model.set_input("valid_length", valid_length_data)
            self.decode_model.run()
            self.decode_model.sync()
            input_data = self.decode_model.get_output("Output_lm_head_add_list_0").numpy()
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

        return all_response, decode_count + 1, prefill_time, decode_time

class HmQwenXh2:

    def __init__(self, model_dir, prefill_length, decode_length, batch=1, nblocks=28):
        self.batch = batch
        self.prefill_length = prefill_length
        self.decode_length = decode_length
        self.nblocks = nblocks
        weight_manager = tcim.runtime.WeightManager(0)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        dummy_tensor_names = [f'model_layers_{i}_self_attn_kcache_input' for i in range(nblocks)]
        dummy_tensor_names += [f'model_layers_{i}_self_attn_vcache_input' for i in range(nblocks)]
        option2.set_dummy_tensors(dummy_tensor_names)
        self.prefill = tcim.runtime.load(os.path.join(model_dir, "qwen2.5_prefill.hmm"), option = option1)
        self.decode = tcim.runtime.load(os.path.join(model_dir, "qwen2.5_decode.hmm"), option = option2)
        # set kvcache input
        # for i in range(nblocks):
        #     kcache = self.prefill.get_input(f'model_layers_{i}_self_attn_kcache_input')
        #     self.decode.set_input(f'model_layers_{i}_self_attn_kcache_input', kcache)
        #     vcache = self.prefill.get_input(f'model_layers_{i}_self_attn_vcache_input')
        #     self.decode.set_input(f'model_layers_{i}_self_attn_vcache_input', vcache)
        for i in range(3, 2 * nblocks + 3):
            cache = self.prefill.get_input(self.prefill.get_input_name(i))
            self.decode.set_input(self.decode.get_input_name(i), cache)
        # set decode input
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode.get_input_name(2)
        self.decode.set_input(decode_current_length_name, current_length_input_1)

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        embedding_weight = torch.load(EMBEDDING_PATH, map_location="cpu", weights_only=True)['weight']
        self.embedding_weight = embedding_weight.reshape(-1, 3584)

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
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        text = self.tokenizer.batch_decode(inputs.input_ids)[0]
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()
        if input_echo_len >= self.decode_length:
            logger.error(f"Question long than {self.decode_length}, please shorten it!")
            return f"Question long than {self.decode_length}, please shorten it!"

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
            # [256, 1, 3584] ==> [4, 64, 3584]
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(1, self.prefill_length, 3584)
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
            if context_length >= self.decode_length:
                logger.info(f"context length greater than {self.decode_length}, break!")
                break

            input_name = self.decode.get_input_name(0)
            valid_length_name = self.decode.get_input_name(1)
            self.decode.set_input(input_name, input_data.numpy())
            valid_length_data = np.array(context_length - 1).astype("int32")
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

        return all_response, decode_count + 1, prefill_time, decode_time


if __name__ == "__main__":

    args = get_args()
    if HOUMO_TARGET == 'xh1':
        hmqwen = HmQwen(args.model_dir, args.prefill_length, args.decode_length, nblocks=args.nblocks)
    elif HOUMO_TARGET == 'xh2':
        hmqwen = HmQwenXh2(args.model_dir, args.prefill_length, args.decode_length, nblocks=args.nblocks)
    question = "请介绍一下存算一体技术的优势"

    start_time = time.time()
    response, tokens, prefill_time, decode_time = hmqwen.chat(question)
    total_time = time.time() - start_time

    logger.success(f"total: {tokens} tokens, cost {total_time:.3f} s")
    logger.success(f"prefill time: {prefill_time * 1000:.3f} ms, {1 / prefill_time:.2f} tokens/s")
    decode_latency = decode_time * 1000 / (tokens - 1)
    logger.success(f"decode average time: {decode_latency:.3f} ms, {1000 / decode_latency:.2f} tokens/s")
    res_latency = total_time * 1000 / tokens
    logger.success(f"end2end average time: {res_latency:.3f} ms, {1000 / res_latency:.2f} tokens/s")
