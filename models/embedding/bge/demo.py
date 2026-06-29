#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   This document provides sample Python inference scripts for bge-m3 and bge-reranker-m3-v2
# based on the M50 device, along with time consumption statistics.
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
import json
import time
import argparse
import onnxruntime as ort
import onnx
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Any, Optional, Tuple, Union, List

from hmatc.utils import logger
from hmatc.utils.utils import first_not_none, get_model_configs

import tcim_lite

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

SUPPORTED_MODEL_TYPES = ["onnx", "houmo"]
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_tokenizer_dirs(model_config: dict) -> tuple[str, str]:
    repo_ids = model_config.get("modelscope_repo", [])
    reranker_dir = "bge-reranker-v2-m3"
    embedder_dir = "bge-m3"
    if len(repo_ids) >= 1:
        reranker_dir = repo_ids[0].rsplit("/", maxsplit=1)[-1]
    if len(repo_ids) >= 2:
        embedder_dir = repo_ids[1].rsplit("/", maxsplit=1)[-1]
    return embedder_dir, reranker_dir


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        help="model name",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size",
    )
    parser.add_argument(
        "--embedding_tokenizer_dir",
        dest="embedding_tokenizer_dir",
        type=str,
        default=None,
        help="embedder tokenizer dir",
    )
    parser.add_argument(
        "--reranker_tokenizer_dir",
        dest="reranker_tokenizer_dir",
        type=str,
        default=None,
        help="reranker tokenizer dir",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=None,
        help="houmo embedding model path",
    )
    parser.add_argument(
        "--reranker_path",
        dest="reranker_path",
        type=str,
        default=None,
        help="houmo reranker model path",
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="all",
        help="select all, embedder or reranker",
    )
    parser.add_argument(
        "--device_idx",
        dest="device_idx",
        type=int,
        default=0,
        help="Houmo device index",
    )
    parser.add_argument(
        "--model_type",
        dest="model_type",
        type=str,
        default="houmo",
        help="onnx or houmo",
    )
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    embedder_name = model_config.get("embedder_name", "bge-m3")
    reranker_name = model_config.get("reranker_name", "bge-reranker-v2-m3")
    default_embedder_dir, default_reranker_dir = get_default_tokenizer_dirs(
        model_config
    )
    args.embedding_tokenizer_dir = first_not_none(
        args.embedding_tokenizer_dir, default_embedder_dir
    )
    args.reranker_tokenizer_dir = first_not_none(
        args.reranker_tokenizer_dir, default_reranker_dir
    )
    args.embedding_path = first_not_none(
        args.embedding_path,
        os.path.join("output", HOUMO_TARGET, f"{embedder_name}-{args.model_size}.hmm"),
    )
    args.reranker_path = first_not_none(
        args.reranker_path,
        os.path.join("output", HOUMO_TARGET, f"{reranker_name}-{args.model_size}.hmm"),
    )
    return args


class ONNXTensorInfo(object):
    shape = None
    dtype = None


def get_detailed_instruct(instruction_format: str, instruction: str, sentence: str):
    if "\\n" in instruction_format:
        instruction_format = instruction_format.replace("\\n", "\n")
    return instruction_format.format(instruction, sentence)


class HmBGEReRanker(nn.Module):
    def __init__(self, model_path, tokenizer_dir, device_id=0, model_type="houmo"):
        super().__init__()
        self.model_path = model_path
        self.model_type = model_type
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            logger.error(f"Not supported model type: {self.model_type}")
            assert 0
        if self.model_type == "houmo":
            wt_manager = tcim_lite.runtime.WeightManager(device_id)
            option = tcim_lite.runtime.Option(wt_manager)
            self.engine = tcim_lite.runtime.load(self.model_path, option)
            self.input_infos = {}
            for idx in range(self.engine.get_num_inputs()):
                input_name = self.engine.get_input_name(idx)
                self.input_infos[input_name] = self.engine.get_input_info(input_name)
            self.output_name = self.engine.get_output_name(0)
        elif self.model_type == "onnx":
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
                "CPUExecutionProvider",
            ]
            self.engine = ort.InferenceSession(self.model_path, providers=providers)
            self.input_infos = {}
            for idx, tensor in enumerate(self.engine.get_inputs()):
                self.input_infos[tensor.name] = ONNXTensorInfo()
                self.input_infos[tensor.name].shape = tensor.shape
                elem_type_str = str(tensor.type).split("(")[-1].split(")")[0].upper()
                onnx_dtype = onnx.TensorProto.DataType.Value(elem_type_str)
                self.input_infos[tensor.name].dtype = (
                    onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(onnx_dtype).name
                )
            self.output_name = self.engine.get_outputs()[0].name

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=False, cache_dir=None
        )
        self.batch_size = self.input_infos["input_ids"].shape[0]
        self.max_length = self.input_infos["input_ids"].shape[1]
        self.query_max_length = self.max_length * 3 // 4
        self.pooling_method = "cls"
        self.normalize_embeddings = True
        self.passage_instruction_format = "{}{}"
        self.passage_instruction_for_rerank = None
        self.query_instruction_format = "{}{}"
        self.query_instruction_for_rerank = None
        self.get_token_types_ids(f"{tokenizer_dir}/config.json")

    def get_token_types_ids(self, config_file):
        if not os.path.exists(config_file):
            logger.error(
                f"{config_file} is not exists, don not get token_type_ids default!"
            )
            assert 0
        with open(config_file, encoding="utf-8") as reader:
            text = reader.read()
        config = json.loads(text)
        max_position_embeddings = config["max_position_embeddings"]
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def get_detailed_inputs(self, sentence_pairs: Union[str, List[str]]):
        if isinstance(sentence_pairs, str):
            sentence_pairs = [sentence_pairs]

        if self.query_instruction_for_rerank is not None:
            if self.passage_instruction_for_rerank is None:
                return [
                    [
                        get_detailed_instruct(
                            self.query_instruction_format,
                            self.query_instruction_for_rerank,
                            sentence_pair[0],
                        ),
                        sentence_pair[1],
                    ]
                    for sentence_pair in sentence_pairs
                ]
            else:
                return [
                    [
                        get_detailed_instruct(
                            self.query_instruction_format,
                            self.query_instruction_for_rerank,
                            sentence_pair[0],
                        ),
                        get_detailed_instruct(
                            self.passage_instruction_format,
                            self.passage_instruction_for_rerank,
                            sentence_pair[1],
                        ),
                    ]
                    for sentence_pair in sentence_pairs
                ]
        else:
            if self.passage_instruction_for_rerank is None:
                return [
                    [sentence_pair[0], sentence_pair[1]]
                    for sentence_pair in sentence_pairs
                ]
            else:
                return [
                    [
                        sentence_pair[0],
                        get_detailed_instruct(
                            self.passage_instruction_format,
                            self.passage_instruction_for_rerank,
                            sentence_pair[1],
                        ),
                    ]
                    for sentence_pair in sentence_pairs
                ]

    def reranker(
        self,
        sentences_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        **kwargs: Any,
    ) -> np.ndarray:
        self.passage_instruction_for_rerank = kwargs.get("passage_instruction", None)
        self.passage_instruction_format = kwargs.get(
            "passage_instruction_format", "{}{}"
        )
        self.query_instruction_for_rerank = kwargs.get("query_instruction", None)
        self.query_instruction_format = kwargs.get("query_instruction_format", "{}{}")

        if isinstance(sentences_pairs[0], str):
            sentences_pairs = [sentences_pairs]
        sentences_pairs = self.get_detailed_inputs(sentences_pairs)

        all_inputs = []
        for start_index in range(0, len(sentences_pairs), self.batch_size):
            sentences_batch = sentences_pairs[
                start_index : start_index + self.batch_size
            ]
            queries = [s[0] for s in sentences_batch]
            passages = [s[1] for s in sentences_batch]
            queries_inputs_batch = self.tokenizer(
                queries,
                return_tensors=None,
                add_special_tokens=False,
                max_length=self.query_max_length,
                truncation=True,
            )["input_ids"]
            passages_inputs_batch = self.tokenizer(
                passages,
                return_tensors=None,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
            )["input_ids"]
            for q_inp, d_inp in zip(queries_inputs_batch, passages_inputs_batch):
                item = self.tokenizer.prepare_for_model(
                    q_inp,
                    d_inp,
                    truncation="only_second",
                    max_length=self.max_length,
                    padding=False,
                )
                all_inputs.append(item)

        length_sorted_idx = np.argsort([-len(x["input_ids"]) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

        all_scores = []
        for start_index in range(0, len(all_inputs_sorted), self.batch_size):
            sentences_batch = all_inputs_sorted[
                start_index : start_index + self.batch_size
            ]
            inputs = self.tokenizer.pad(
                sentences_batch, padding=True, return_tensors="pt"
            )
            cur_batch = inputs["input_ids"].size(0)
            cur_seq_len = inputs["input_ids"].size(1)
            if "token_type_ids" not in inputs:
                if hasattr(self, "token_type_ids"):
                    buffered_token_type_ids = self.token_type_ids[:, :cur_seq_len]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                        cur_batch, cur_seq_len
                    )
                    inputs["token_type_ids"] = buffered_token_type_ids_expanded
                else:
                    inputs["token_type_ids"] = torch.zeros(
                        inputs["input_ids"].shape, dtype=torch.long, device="cpu"
                    )
            input_dict = {}
            for input_name in self.input_infos.keys():
                if cur_batch < self.batch_size or cur_seq_len < self.max_length:
                    input_data = torch.zeros(
                        self.input_infos[input_name].shape, dtype=torch.long
                    )
                    input_data[:cur_batch, :cur_seq_len] = inputs[input_name]
                else:
                    input_data = inputs[input_name]
                input_data = input_data.numpy().astype(
                    self.input_infos[input_name].dtype
                )
                input_dict[input_name] = input_data

            if self.model_type == "houmo":
                for input_name in input_dict.keys():
                    self.engine.set_input(input_name, input_dict[input_name])
                self.engine.run()
                self.engine.sync()
                cur_scores = self.engine.get_output(self.output_name).numpy()[
                    :cur_batch, ...
                ]
            elif self.model_type == "onnx":
                outputs = self.engine.run(None, input_dict)[0]
                cur_scores = outputs[:cur_batch, ...]
            else:
                logger.error(f"Not supported model type: {self.model_type}")
                assert 0
            all_scores.extend(cur_scores.flatten().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]
        return all_scores


class HmBGEM3(nn.Module):
    def __init__(self, model_path, tokenizer_dir, device_id=0, model_type="houmo"):
        super().__init__()
        self.model_path = model_path
        self.model_type = model_type
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            logger.error(f"Not supported model type: {self.model_type}")
            assert 0
        if self.model_type == "houmo":
            wt_manager = tcim_lite.runtime.WeightManager(device_id)
            option = tcim_lite.runtime.Option(wt_manager)
            self.engine = tcim_lite.runtime.load(self.model_path, option)
            self.input_infos = {}
            for idx in range(self.engine.get_num_inputs()):
                input_name = self.engine.get_input_name(idx)
                self.input_infos[input_name] = self.engine.get_input_info(input_name)
            self.output_name = self.engine.get_output_name(0)
        elif self.model_type == "onnx":
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
                "CPUExecutionProvider",
            ]
            self.engine = ort.InferenceSession(self.model_path, providers=providers)
            self.input_infos = {}
            for idx, tensor in enumerate(self.engine.get_inputs()):
                self.input_infos[tensor.name] = ONNXTensorInfo()
                self.input_infos[tensor.name].shape = tensor.shape
                elem_type_str = str(tensor.type).split("(")[-1].split(")")[0].upper()
                onnx_dtype = onnx.TensorProto.DataType.Value(elem_type_str)
                self.input_infos[tensor.name].dtype = (
                    onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(onnx_dtype).name
                )
            self.output_name = self.engine.get_outputs()[0].name

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=False, cache_dir=None
        )
        self.batch_size = self.input_infos["input_ids"].shape[0]
        self.max_length = self.input_infos["input_ids"].shape[1]
        self.pooling_method = "cls"
        self.normalize_embeddings = True
        self.get_token_types_ids(f"{tokenizer_dir}/config.json")

    def get_token_types_ids(self, config_file):
        if not os.path.exists(config_file):
            logger.error(
                f"{config_file} is not exists, don not get token_type_ids default!"
            )
            assert 0
        with open(config_file, encoding="utf-8") as reader:
            text = reader.read()
        config = json.loads(text)
        max_position_embeddings = config["max_position_embeddings"]
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def pooling(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if self.pooling_method == "cls":
            return last_hidden_state[:, 0]
        elif self.pooling_method == "mean":
            s = torch.sum(
                last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1
            )
            d = attention_mask.sum(dim=1, keepdims=True).float()
            return s / d
        else:
            raise NotImplementedError(
                f"pooling method {self.pooling_method} not implemented"
            )

    def embedder(self, sentences, **kwargs: Any) -> np.ndarray:
        instruction = kwargs.get("instruction", None)
        instruction_format = kwargs.get("instruction_format", "{}{}")
        if instruction is not None:
            if isinstance(sentences, str):
                sentences = get_detailed_instruct(
                    instruction_format, instruction, sentences
                )
            else:
                sentences = [
                    get_detailed_instruct(instruction_format, instruction, sentence)
                    for sentence in sentences
                ]

        if isinstance(sentences, str):
            sentences = [sentences]

        all_inputs = []
        for start_index in range(0, len(sentences), self.batch_size):
            sentences_batch = sentences[start_index : start_index + self.batch_size]
            inputs_batch = self.tokenizer(
                sentences_batch, truncation=True, max_length=self.max_length
            )
            inputs_batch = [
                {k: inputs_batch[k][i] for k in inputs_batch.keys()}
                for i in range(len(sentences_batch))
            ]
            all_inputs.extend(inputs_batch)

        length_sorted_idx = np.argsort([-len(x["input_ids"]) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

        all_embeddings = []
        for start_index in range(0, len(sentences), self.batch_size):
            inputs_batch = all_inputs_sorted[
                start_index : start_index + self.batch_size
            ]
            inputs_batch = self.tokenizer.pad(
                inputs_batch, padding=True, return_tensors="pt"
            )

            cur_batch = inputs_batch["input_ids"].size(0)
            cur_seq_len = inputs_batch["input_ids"].size(1)
            if "token_type_ids" not in inputs_batch:
                if hasattr(self, "token_type_ids"):
                    buffered_token_type_ids = self.token_type_ids[:, :cur_seq_len]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                        cur_batch, cur_seq_len
                    )
                    inputs_batch["token_type_ids"] = buffered_token_type_ids_expanded
                else:
                    inputs_batch["token_type_ids"] = torch.zeros(
                        inputs_batch["input_ids"].shape, dtype=torch.long, device="cpu"
                    )
            input_dict = {}
            for input_name in self.input_infos.keys():
                if cur_batch < self.batch_size or cur_seq_len < self.max_length:
                    input_data = torch.zeros(
                        self.input_infos[input_name].shape, dtype=torch.long
                    )
                    input_data[:cur_batch, :cur_seq_len] = inputs_batch[input_name]
                else:
                    input_data = inputs_batch[input_name]
                input_data = input_data.numpy().astype(
                    self.input_infos[input_name].dtype
                )
                input_dict[input_name] = input_data

            if self.model_type == "houmo":
                for input_name in input_dict.keys():
                    self.engine.set_input(input_name, input_dict[input_name])
                self.engine.run()
                self.engine.sync()
                last_hidden_state = self.engine.get_output(self.output_name).numpy()
            elif self.model_type == "onnx":
                last_hidden_state = self.engine.run(None, input_dict)[0]
            else:
                logger.error(f"Not supported model type: {self.model_type}")
                assert 0
            last_hidden_state = torch.from_numpy(last_hidden_state)
            embeddings = self.pooling(last_hidden_state, inputs_batch["attention_mask"])
            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, dim=-1)
            embeddings = embeddings.cpu().numpy()
            embeddings = embeddings[:cur_batch, ...]
            all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]

        return all_embeddings


def semantic_search_np(query_emb, corpus_embs, top_k=3):
    if len(query_emb.shape) > 1:
        query_emb = query_emb.squeeze()
    q = query_emb / np.linalg.norm(query_emb)
    c = corpus_embs / np.linalg.norm(corpus_embs, axis=1, keepdims=True)

    scores = np.dot(c, q)

    top_k_idx = np.argsort(scores)[::-1][:top_k]

    return [(int(i), float(scores[i])) for i in top_k_idx]


def xh2_demo(args):
    corpus = [
        "北京有很多著名的牛肉面馆。",
        "上海是中国的金融中心。",
        "故宫是北京著名的旅游景点。",
        "兰州牛肉面是非常有名的中国美食。",
        "深圳以科技产业闻名。",
        "北京海淀区有一家西北牛肉面馆非常好吃。",
    ]

    query = "北京哪里有好吃的牛肉面"

    pairs = None
    if args.mode == "embedder" or args.mode == "all":
        hmbgem3 = HmBGEM3(
            args.embedding_path,
            args.embedding_tokenizer_dir,
            args.device_idx,
            args.model_type,
        )
        # Encode the corpus into vectors
        start_time = time.time()
        corpus_embeddings = hmbgem3.embedder(corpus).astype(np.float32)
        embedder_time = time.time() - start_time
        logger.info("Time to encode corpus: %.6fms" % (embedder_time * 1000))

        # User Query
        start_time = time.time()
        query_embedding = hmbgem3.embedder(query).astype(np.float32)
        embedder_time = time.time() - start_time
        logger.info("User query encoding time: %.6fms" % (embedder_time * 1000))

        hits_np = semantic_search_np(query_embedding, corpus_embeddings, top_k=3)

        logger.info("==== 粗召回结果 (bge-m3) ====")

        for hit in hits_np:
            logger.info(f"{corpus[hit[0]]} score:{hit[1]:.6f}")

        if args.mode == "all":
            pairs = [[query, corpus[hit[0]]] for hit in hits_np]

    if args.mode == "reranker" or args.mode == "all":
        hmbgereranker = HmBGEReRanker(
            args.reranker_path,
            args.reranker_tokenizer_dir,
            args.device_idx,
            args.model_type,
        )
        if pairs is None:
            pairs = [[query, cor] for cor in corpus]

        # Send the query and candidate documents to the reranker for scoring
        start_time = time.time()
        scores = hmbgereranker.reranker(pairs)
        reranker_time = time.time() - start_time
        logger.info(
            "Time taken to reorder the corpus and query statements: %.6fms"
            % (reranker_time * 1000)
        )

        # Results of refined ovulation
        reranked = sorted(zip(pairs, scores), key=lambda x: x[1], reverse=True)

        logger.info("==== 精排结果 (bge-reranker-v2-m3) ====")
        for pair, score in reranked:
            logger.info(f"doc: {pair[1]}  →  score: {score:.4f}")


if __name__ == "__main__":

    args = get_args()
    xh2_demo(args)
