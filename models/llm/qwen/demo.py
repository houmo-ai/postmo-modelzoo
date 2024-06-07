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
EMBEDDING_WEIGHT_PATH = "output/H30/result"
question = "请介绍一下存算一体技术的优势"

def run_model_decode(model, input_data, valid_length):
    input_name0 = "input"
    input_name1 = "valid_length"
    input_name2 = "current_length"
    input_data0 = input_data
    input_data1 = np.array([valid_length - 1]).astype("int16")
    input_data2 = np.array([1]).astype("int16")
    model.set_input(input_name0, input_data0)
    model.set_input(input_name1, input_data1)
    model.set_input(input_name2, input_data2)
    model.run()
    model.sync()
    # output = model.get_output("layers31_resadd2", True)
    # output_eval = output.reshape([1, 1, 4096])
    # out_shape = [1, 1, 1, 4096]
    # output_eval = output_eval.reshape(out_shape)
    # return torch.empty((0), dtype=torch.int)


def run_model_prefill(model, input_data, current_length, valid_length=0):
    input_name0 = "input"
    input_name1 = "valid_length"
    input_name2 = "current_length"
    input_data0 = input_data
    input_data1 = np.array([valid_length]).astype("int16")
    input_data2 = np.array([current_length]).astype("int16")

    model.set_input(input_name0, input_data0)
    model.set_input(input_name1, input_data1)
    model.set_input(input_name2, input_data2)
    model.run()
    model.sync()
    # output = model.get_output("layers31_resadd2", True)
    # output_eval = output.reshape([4, 64, 4096])
    # out_shape = [1, 4, 64, 4096]
    # output_eval = output_eval.reshape(out_shape)
    # return output_eval


def run_model_decode_head(model):
    model.run()
    model.sync()
    output = model.get_output("lm_head", True)
    return output


def run_model_prefill_head(model, gather_index):
    input1_name = "seq_length"
    input1_data = np.array([gather_index]).astype("int16")
    model.set_input(input1_name, input1_data)
    model.run()
    model.sync()
    output = model.get_output("lm_head", True)
    return output


class HmQwen:

    def __init__(self):
        weight_manager = tcim.runtime.create_weight_manager()
        self.decode_model = tcim.runtime.load("qwen_decode.hmm", weight_manager=weight_manager)
        self.prefill_model = tcim.runtime.load("qwen_prefill.hmm", weight_manager=weight_manager)
        self.decode_head_model = tcim.runtime.load("qwen_decode_head.hmm", weight_manager=weight_manager)
        self.prefill_head_model = tcim.runtime.load("qwen_prefill_head.hmm", weight_manager=weight_manager)
        self.qwen1_5tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        self.embedding_weight = torch.load(os.path.join(EMBEDDING_WEIGHT_PATH, "qwen15_quant_embedding.pth"), map_location="cpu")

    def chat(self, question):
        start_time = time.time()
        max_length = pad_length = 256
        self.embedding_weight = self.embedding_weight.reshape(-1, 4096)
        messages = [{"role": "user", "content": question,}]
        text = self.qwen1_5tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.qwen1_5tokenizer([text], return_tensors="pt")
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.shape[1]
        if input_echo_len > 2048:
            logger.error(f"Input length larger than 2048 !")
            exit(-1)
        eos_token_id = [
            self.qwen1_5tokenizer.eos_token_id,
        ]
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
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous() # [n, 1, 4096]
            effective_length = input_ids.size(-1)
            if pad_length - effective_length < 0:
                logger.error(f"Input length larger than 256")
                exit(-1)
            _pad_embeds = torch.zeros([pad_length - effective_length, 1, inputs_embeds.size(-1)], dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            padded_embeds = torch.cat([inputs_embeds, _pad_embeds],dim=0) # [256, 1, 4096]
            network_input = padded_embeds.view(4, 64, 4096)
            input_data = network_input.detach().numpy()
            run_model_prefill(self.prefill_model, input_data, current_length, valid_length)

        prefill_output_addr = self.prefill_model.get_dev_output("layers31_resadd2")
        self.prefill_head_model.set_input("layers31_resadd2", prefill_output_addr)
        input_data = run_model_prefill_head(self.prefill_head_model, gather_index)
        output_ids_all = torch.empty((0), dtype=torch.int)
        logits = torch.from_numpy(input_data)
        output_ids = logits.argmax(dim=-1) # [1]
        output_ids_all = torch.cat((output_ids_all, output_ids), 0)
        # response = self.qwen1_5tokenizer.decode(output_ids_all.tolist())
        generated_ids = output_ids_all.reshape((1, *output_ids_all.shape))
        generated_ids = torch.cat((input_ids, generated_ids), 1)
        generated_ids = [
            oids[len(iids):] for iids, oids in zip(input_ids, generated_ids)
        ]
        response = self.qwen1_5tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        prefill_time = time.time() - start_time
        start_time = time.time()
        input_data = F.embedding(output_ids, self.embedding_weight)
        input_data = input_data.view(1, 1, 4096)
        input_data = input_data.detach().numpy()
        qa_len = input_echo_len
        qa_len = qa_len + 1
        round_count = 0
        try:
            while True:
                round_count = round_count + 1
                if round_count > 2048:
                    break
                run_model_decode(self.decode_model, input_data, qa_len)
                decode_output_addr = self.decode_model.get_dev_output("layers31_resadd2")
                self.decode_head_model.set_input("layers31_resadd2", decode_output_addr)
                input_data = run_model_decode_head(self.decode_head_model)
                logits = torch.from_numpy(input_data)
                output_ids = logits.argmax(dim=-1) # [1]
                if output_ids.item() in eos_token_id or effective_length >= max_length:
                    break

                logits = torch.from_numpy(input_data)
                ids = logits.argmax(dim=-1) # [1]
                input_data = F.embedding(ids, self.embedding_weight)
                input_data = input_data.view(1, 1, 4096)
                input_data = input_data.detach().numpy()
                # decode_response = self.qwen1_5tokenizer.decode(output_ids.tolist())
                output_ids_all = torch.cat((output_ids_all, output_ids), 0)
                generated_ids = output_ids_all.reshape((1, *output_ids_all.shape))
                generated_ids = torch.cat((input_ids, generated_ids), 1)
                generated_ids = [
                    oids[len(iids):] for iids, oids in zip(input_ids, generated_ids)
                ]
                response = self.qwen1_5tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                qa_len = qa_len + 1
        except Exception as err:
            logger.error(f"Unexpected {err}, {type(err)}")
            logger.error(traceback.format_exc())
        decode_time = time.time() - start_time
        return response, round_count, prefill_time, decode_time


if __name__ == "__main__":
    hmqwen = HmQwen()
    logger.success("question:\n{}".format(question))
    start_time = time.time()
    response, tokens, prefill_time, decode_time = hmqwen.chat(question)
    total_time = time.time() - start_time
    logger.success("response:\n{}".format(response))
    logger.success("total: {} tokens, cost {:.3f} s".format(tokens, total_time))
    logger.success("prefill time: {:.3f} ms, {:.2f} tokens/s".format(prefill_time * 1000, 1 / prefill_time))
    decode_latency = decode_time * 1000 / (tokens - 1)
    logger.success("decode average time: {:.3f} ms, {:.2f} tokens/s".format(decode_latency, 1000 / decode_latency))
    res_latency = total_time * 1000 / tokens
    logger.success("end2end average time: {:.3f} ms, {:.2f} tokens/s".format(res_latency, 1000 / res_latency))
