#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   Laguna-S-2.1 text generation demo on HOUMO AI XH2 devices.
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

import argparse
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoTokenizer

import tcim_lite as tcim
from hmatc.python.get_hm_devices import get_hm_devices
from hmatc.utils.perf_infomations import InferencePerformanceTracker, PERFTYPE
from hmatc.utils.utils import first_not_none, get_model_configs, parse_context_length

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_SYSTEM_PROMPT = (
	"You are a helpful, conversationally-fluent assistant made by Poolside. "
	"You are here to be helpful to users through natural language conversations."
)
SLIDING_WINDOW = 512
SW_MASK_PREFILL_WIDTH = 768
SW_MASK_DECODE_WIDTH = 512


def get_default_tokenizer_dir(model_config: dict) -> str:
	repo_ids = model_config.get("modelscope_repo", [])
	if repo_ids:
		return repo_ids[0].rsplit("/", maxsplit=1)[-1]
	return "Laguna-S-2.1"


def is_valid_char(cp: int) -> bool:
	if (
		(0x4E00 <= cp <= 0x9FFF)
		or (0x3400 <= cp <= 0x4DBF)
		or (0x20000 <= cp <= 0x2A6DF)
		or (0x2A700 <= cp <= 0x2B73F)
		or (0x2B740 <= cp <= 0x2B81F)
		or (0x2B820 <= cp <= 0x2CEAF)
		or (0xF900 <= cp <= 0xFAFF)
		or (0x2F800 <= cp <= 0x2FA1F)
		or (0x0041 <= cp <= 0x005A)
		or (0x0061 <= cp <= 0x007A)
	):
		return True
	return False


def parse_args() -> argparse.Namespace:
	"""Parse commandline."""
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
	parser.add_argument("--model_name", dest="model_name", type=str, default=None, help="model name")
	parser.add_argument("--model_size", dest="model_size", type=str, default=None, help="model size")
	parser.add_argument("--tokenizer_dir", dest="tokenizer_dir", type=str, default=None, help="tokenizer dir")
	parser.add_argument("--embedding_path", dest="embedding_path", type=str, default=None, help="houmo embedding weight path")
	parser.add_argument("--prefill_path", dest="prefill_path", type=str, default=None, help="houmo prefill model path")
	parser.add_argument("--decode_path", dest="decode_path", type=str, default=None, help="houmo decode model path")
	parser.add_argument("--batch", dest="batch", type=int, default=None, help="batch size")
	parser.add_argument("--ndevice", dest="ndevice", type=int, default=None, help="device number, only xh2 support")
	parser.add_argument("--context_length", dest="context_length", type=int, default=None, help="context length")
	parser.add_argument("--prefill_length", dest="prefill_length", type=int, default=None, help="prefill length")
	parser.add_argument("--system_prompt", dest="system_prompt", type=str, default=None, help="system prompt")
	parser.add_argument("--question", dest="question", type=str, default="请介绍一下存算一体技术的优势", help="question to ask")
	parser.add_argument("--repetition_penalty", dest="repetition_penalty", type=float, default=1.0, help="sampling repetition_penalty")
	parser.add_argument("--topk", dest="topk", type=int, default=None, help="sampling top-k")
	parser.add_argument("--temperature", dest="temperature", type=float, default=1.0, help="sampling temperature")
	parser.add_argument("--history", dest="history", action="store_true", help="keep chat history")
	parser.add_argument("--max_new_tokens", dest="max_new_tokens", type=int, default=32768, help="maximum number of new tokens")
	args = parser.parse_args()
	default_size, default_name, model_configs = get_model_configs(args.config_path)
	args.model_name = first_not_none(args.model_name, default_name)
	args.model_size = first_not_none(args.model_size, default_size)
	model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
	args.batch = first_not_none(args.batch, model_config.get("batch", 1))
	args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
	args.prefill_length = first_not_none(args.prefill_length, model_config.get("prefill_length", 256))
	args.context_length = first_not_none(args.context_length, parse_context_length(model_config.get("context_length", "256k")))
	args.tokenizer_dir = first_not_none(args.tokenizer_dir, get_default_tokenizer_dir(model_config))
	args.embedding_path = first_not_none(args.embedding_path, os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"))
	args.prefill_path = first_not_none(args.prefill_path, os.path.join("output", HOUMO_TARGET, f"{args.model_name}-{args.model_size}_prefill.hmm"))
	args.decode_path = first_not_none(args.decode_path, os.path.join("output", HOUMO_TARGET, f"{args.model_name}-{args.model_size}_decode.hmm"))
	if args.ndevice > 1:
		args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
		args.decode_path = args.decode_path.replace(".hmm", ".hmms")
	return args


class SamplingManager:
	def __init__(self, temperature: float = 1.0, top_k: Optional[int] = None, repetition_penalty: float = 1.0):
		self.temperature, self.top_k = temperature, top_k
		self.repetition_penalty = repetition_penalty

	def sample(self, logits: np.ndarray, previous_tokens: List[int]) -> int:
		values = logits[0, -1].astype(np.float32, copy=True)
		for token_id in set(previous_tokens):
			if 0 <= token_id < len(values) and self.repetition_penalty != 1.0:
				values[token_id] = values[token_id] * self.repetition_penalty if values[token_id] < 0 else values[token_id] / self.repetition_penalty
		if self.temperature <= 0:
			raise ValueError("Temperature must larger than 0")
		values /= self.temperature
		if self.top_k and self.top_k > 0:
			indices = np.argpartition(values, -min(self.top_k, len(values)))[-min(self.top_k, len(values)):]
			mask = np.ones(len(values), dtype=bool)
			mask[indices] = False
			values[mask] = -np.inf
		return int(np.argmax(values))


class HmLaguna:
	"""Laguna runtime wrapper; prefill and decode share the same KV tensors."""

	def __init__(self, prefill_path, decode_path, embedding_path, tokenizer_dir, ndevice=1):
		self.perf_tracker = InferencePerformanceTracker()
		self.ndevice = ndevice
		dev_manager = tcim.runtime.DevManager(get_hm_devices(self.ndevice), "Xh2HalBackend")
		weight_manager = tcim.runtime.WeightManager(dev_manager)
		self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
		self.prefill = tcim.runtime.load(prefill_path, option=tcim.runtime.Option(weight_manager))
		self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)
		cache_names = [self.prefill.get_input_name(i) for i in range(self.prefill.get_num_inputs()) if "cache" in self.prefill.get_input_name(i).lower()]
		option = tcim.runtime.Option(weight_manager)
		option.set_dummy_tensors(cache_names)
		self.perf_tracker.perf_start(PERFTYPE.DECODE_LOAD_TIME)
		self.decode = tcim.runtime.load(decode_path, option=option)
		self.perf_tracker.perf_end(PERFTYPE.DECODE_LOAD_TIME)
		self._bind_caches(cache_names)
		self.prefill_length = self.prefill.get_input_info("input_1").shape[1]
		self.context_max_length = args.context_length
		self.embedding_weight = self._load_embedding(embedding_path)
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
		self.samplingmanager = SamplingManager(args.temperature, args.topk, args.repetition_penalty)
		self.context_length = 0
		self.perf_tracker.reset_perf_time()

	def _bind_caches(self, cache_names):
		for name in cache_names:
			info = self.prefill.get_input_info(name)
			zeros = np.zeros(info.shape, dtype=np.dtype(info.dtype))
			self.prefill.set_input(name, zeros)
			self.decode.set_input(name, self.prefill.get_dev_input(name))

	@staticmethod
	def _load_embedding(path):
		saved = torch.load(path, map_location="cpu", weights_only=True)
		if not isinstance(saved, dict) or "weight" not in saved:
			raise ValueError("quant_embedding.pt must contain a weight tensor")
		return saved["weight"].float()

	@staticmethod
	def _sliding_mask(q_len, past_len, width):
		mask = np.full((1, 1, q_len, width), np.finfo(np.float16).min, dtype=np.float16)
		for q in range(q_len):
			absolute_position = past_len + q
			end = min(width, absolute_position + 1)
			start = max(0, end - SLIDING_WINDOW)
			mask[0, 0, q, start:end] = 0
		return mask

	def _prepare_run_inputs(self, input_ids, prefill):
		token_ids = torch.as_tensor(input_ids, dtype=torch.long)
		if token_ids.ndim == 1:
			token_ids = token_ids.unsqueeze(0)
		embeds = F.embedding(token_ids, self.embedding_weight).numpy().astype(np.float16)
		if prefill:
			embeds = np.pad(
				embeds,
				((0, 0), (0, self.prefill_length - embeds.shape[1]), (0, 0)),
			)
			return embeds, self.prefill_length, SW_MASK_PREFILL_WIDTH
		return embeds, 1, SW_MASK_DECODE_WIDTH

	def _set_run_inputs(self, model, embeds, q_len, width, past_len, current_len):
		model.set_input("input_1", embeds)
		model.set_input("valid_length", np.array([past_len], dtype=np.int32))
		model.set_input("current_length", np.array([current_len], dtype=np.int32))
		model.set_input("sliding_attention_mask", self._sliding_mask(q_len, past_len, width))

	def _run(self, model, input_ids, past_len, current_len, prefill):
		total_type = PERFTYPE.PREFILL_TOTAL_TIME if prefill else PERFTYPE.DECODE_TOTAL_TIME
		embed_type = PERFTYPE.PREFILL_EMBED_TIME if prefill else PERFTYPE.DECODE_EMBED_TIME
		input_type = PERFTYPE.PREFILL_INPUT_TIME if prefill else PERFTYPE.DECODE_INPUT_TIME
		infer_type = PERFTYPE.PREFILL_INFER_TIME if prefill else PERFTYPE.DECODE_INFER_TIME
		output_type = PERFTYPE.PREFILL_OUTPUT_TIME if prefill else PERFTYPE.DECODE_OUTPUT_TIME
		self.perf_tracker.perf_start(total_type)
		self.perf_tracker.perf_start(embed_type)
		embeds, q_len, width = self._prepare_run_inputs(input_ids, prefill)
		self.perf_tracker.perf_end(embed_type)
		self.perf_tracker.perf_start(input_type)
		self._set_run_inputs(model, embeds, q_len, width, past_len, current_len)
		self.perf_tracker.perf_end(input_type)
		self.perf_tracker.perf_start(infer_type)
		model.run()
		model.sync()
		self.perf_tracker.perf_end(infer_type)
		self.perf_tracker.perf_start(output_type)
		output = model.get_output("logits").numpy()
		self.perf_tracker.perf_end(output_type)
		self.perf_tracker.perf_end(total_type)
		return output

	def chat(self, question):
		if not args.history:
			self.context_length = 0
			self._bind_caches([self.prefill.get_input_name(i) for i in range(self.prefill.get_num_inputs()) if "cache" in self.prefill.get_input_name(i).lower()])
		system_prompt = DEFAULT_SYSTEM_PROMPT if args.system_prompt is None else args.system_prompt
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": question},
		]
		text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
		ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
		if len(ids) >= self.context_max_length:
			raise ValueError(f"Question longer than {self.context_max_length}, please shorten it")
		logger.success("question:")
		print("\033[1;95m{}\033[0m".format(question))
		logger.success("response:")
		print("\033[1;95m", end="", flush=True)
		generated = []
		for start in range(0, len(ids), self.prefill_length):
			chunk = ids[start:start + self.prefill_length]
			current = len(chunk)
			padded = chunk + [self.tokenizer.pad_token_id] * (self.prefill_length - current)
			logits = self._run(self.prefill, padded, self.context_length, current, True)
			self.context_length += current
		next_id = int(np.argmax(logits[0, 0]))
		decode_count = 0
		printed_response = ""
		pending_response = ""
		if next_id not in (2, 24):
			generated.append(next_id)
			pending_response = self.tokenizer.decode(generated, skip_special_tokens=True)
			if pending_response:
				print(pending_response, end="", flush=True)
				printed_response = pending_response
				pending_response = ""

		for _ in range(min(args.max_new_tokens - 1, args.context_length - self.context_length - 1)):
			logits = self._run(self.decode, [next_id], self.context_length, 1, False)
			next_id = self.samplingmanager.sample(logits, ids + generated)
			self.context_length += 1
			decode_count += 1
			if next_id in (2, 24):
				break
			generated.append(next_id)
			decoded_response = self.tokenizer.decode(generated, skip_special_tokens=True)
			if decoded_response.startswith(printed_response):
				pending_response = decoded_response[len(printed_response) :]
				if pending_response and is_valid_char(ord(pending_response[-1])):
					print(pending_response, end="", flush=True)
					printed_response = decoded_response
					pending_response = ""
		if pending_response:
			print(pending_response, end="", flush=True)
		print("\033[0m")
		self.perf_tracker.set_basic_info(1, len(ids), decode_count)
		self.perf_tracker.show_summary()


if __name__ == "__main__":
	args = parse_args()
	if args.batch != 1:
		raise ValueError("Laguna-S-2.1 only supports batch=1")
	model = HmLaguna(args.prefill_path, args.decode_path, args.embedding_path, args.tokenizer_dir, args.ndevice)
	model.chat(args.question)
