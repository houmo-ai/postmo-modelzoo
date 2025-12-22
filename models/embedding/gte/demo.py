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

HOUMO_TARGET = os.getenv("HOUMO_TARGET")

def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default="gte_Qwen2-1.5B-instruct",
        help="tokenizer dir",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"),
        help="houmo embedding weight path",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "gte_prefill.hmm"),
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=1,
        choices=[1, 2],
        help="device number, only xh2 support",
    )
    args = parser.parse_args()
    return args


class HmGte:

    def __init__(self, prefill_path, embedding_path, tokenizer_dir, ndevice):
        self.ndevice = ndevice
        if self.ndevice == 1:
            weight_manager = tcim.runtime.WeightManager(0)
        elif self.ndevice == 2:
            dev_manager = tcim.runtime.DevManager([1, 0], "Xh2HalBackend")
            weight_manager = tcim.runtime.WeightManager(dev_manager)
        else:
            raise ValueError("Unsupport device number!")
        option1 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option1)
        logger.info("prefill model loaded")
        self.nblocks = self.get_nblocks()
        self.prefill_length = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        self.embedding_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]
        self.context_max_length = self.prefill.get_input_info(
            self.prefill.get_input_name(4)
        ).shape[2]

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )
        embedding_weight = torch.load(embedding_path, map_location="cpu")
        self.embedding_weight = embedding_weight.reshape(-1, self.embedding_len)

    def get_nblocks(self):
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def chat(self, question):
        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))
        start_time = time.time()
        inputs = self.tokenizer(question, return_tensors="pt", add_special_tokens=False)
        all_input_ids = inputs["input_ids"]
        all_input_ids = torch.cat((all_input_ids, torch.tensor([[151643]])), dim=1)
        input_echo_len = all_input_ids.numel()
        if input_echo_len >= self.context_max_length:
            logger.error(
                f"Question long than {self.context_max_length}, please shorten it!"
            )
            sys.exit(1)

        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        for round in range(prefill_loop_round):
            valid_length = round * self.prefill_length
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
            inputs_embeds = F.embedding(input_ids, self.embedding_weight)
            effective_length = input_ids.size(-1)
            _pad_embeds = torch.zeros(
                1,
                self.prefill_length - effective_length,
                inputs_embeds.size(-1),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(
                1, self.prefill_length, self.embedding_len
            )
            past_seq_length = round * 256
            position_id = torch.arange(
                past_seq_length, past_seq_length + self.prefill_length, dtype=torch.long
            )
            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")
            position_id_data = np.array([position_id]).astype("int32")
            input_name = self.prefill.get_input_name(0)
            valid_length_name = self.prefill.get_input_name(1)
            current_length_name = self.prefill.get_input_name(2)
            position_id_name = self.prefill.get_input_name(3)
            self.prefill.set_input(input_name, input_data.numpy())
            self.prefill.set_input(valid_length_name, valid_length_data)
            self.prefill.set_input(current_length_name, current_length_data)
            self.prefill.set_input(position_id_name, position_id_data)
            self.prefill.run()
            self.prefill.sync()

        output_data = torch.from_numpy(
            self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        )
        prefill_time = time.time() - start_time

        return output_data, input_echo_len, prefill_time


if __name__ == "__main__":

    args = get_args()
    if HOUMO_TARGET == "xh2":
        hmgte = HmGte(
            args.prefill_path,
            args.embedding_path,
            args.tokenizer_dir,
            args.ndevice,
        )
    queries = [
        "how much protein should a female eat",
        "summit define",
    ]
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]
    query_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    queries_process = [query_prefix + query for query in queries]
    query_emb = []
    doc_emb = []
    input_tokens = 0
    prefill_times = 0

    start_time = time.time()
    for query in queries_process:
        output_data, input_token, prefill_time = hmgte.chat(query)
        query_emb.append(output_data)
        input_tokens += input_token
        prefill_times += prefill_time
    for doc in documents:
        output_data, input_token, prefill_time = hmgte.chat(doc)
        doc_emb.append(output_data)
        input_tokens += input_token
        prefill_times += prefill_time
    total_time = time.time() - start_time

    query_emb_c = torch.concat(query_emb, axis=0)
    doc_emb_c = torch.concat(doc_emb, axis=0)
    scores = (query_emb_c @ doc_emb_c.T) * 100
    logger.success("scores:")
    print("\033[1;95m{}".format(scores.tolist()))
    logger.success(
        f"Total Input: {input_tokens} tokens, Prefill Cost {prefill_times*1000:.3f} ms"
    )
    logger.success(f"Prefill Speed: {input_tokens / prefill_time:.2f} tokens/s")
    logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
