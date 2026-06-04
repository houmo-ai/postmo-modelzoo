#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import re
import sys
import math
import time
import argparse
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from loguru import logger

import tcim_lite as tcim

from hmatc.utils.perf_infomations import (
    InferencePerformanceTracker,
    InferenceMetrics,
    PERFTYPE,
)

from hmatc.python.get_hm_devices import get_hm_devices
from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", "Only support HOUMO_TARGET: xh2."
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_tokenizer_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "gpt-oss")
    model_size = model_config.get("model_size", "20b")
    return f"{model_name}-{model_size}"


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
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default=None,
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
        default=None,
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default=None,
        help="houmo decode model path",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        choices=[1, 2],
        help="device number, only xh2 support",
    )
    parser.add_argument(
        "--question",
        dest="question",
        type=str,
        default="请介绍一下存算一体技术的优势",
        help="question to ask",
    )
    parser.add_argument(
        "--repetition_penalty",
        dest="repetition_penalty",
        type=float,
        default=1.0,
        help="sampling repetition_penalty",
    )
    parser.add_argument(
        "--topk",
        dest="topk",
        type=int,
        default=None,
        help="sampling top-k",
    )
    parser.add_argument(
        "--topp",
        dest="topp",
        type=float,
        default=1.0,
        help="sampling top-p",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=1.0,
        help="sampling temperature",
    )
    parser.add_argument(
        "--it",
        dest="it",
        action="store_true",
        help="interactive mode",
    )
    parser.add_argument(
        "--history",
        dest="history",
        action="store_true",
        help="keep chat history",
    )
    args = parser.parse_args()
    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    if args.tokenizer_dir is None:
        args.tokenizer_dir = get_default_tokenizer_dir(model_config)
    if args.prefill_path is None:
        args.prefill_path = os.path.join(
            "output", HOUMO_TARGET, f"{args.model_name}-{args.model_size}_prefill.hmm"
        )
    if args.decode_path is None:
        args.decode_path = os.path.join(
            "output", HOUMO_TARGET, f"{args.model_name}-{args.model_size}_decode.hmm"
        )
    if args.ndevice > 1:
        args.prefill_path = (
            args.prefill_path.replace(".hmm", ".hmms")
            if args.prefill_path.endswith(".hmm")
            else args.prefill_path
        )
        args.decode_path = (
            args.decode_path.replace(".hmm", ".hmms")
            if args.decode_path.endswith(".hmm")
            else args.decode_path
        )
    return args


class SamplingManager:
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        min_tokens_to_keep: int = 1,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.min_tokens_to_keep = min_tokens_to_keep

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            raise ValueError("Temperature must larger than 0")

        return logits / self.temperature

    def apply_repetition_penalty(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        if self.repetition_penalty == 1.0 or not previous_tokens:
            return logits

        adjusted_logits = logits.copy()
        for token_id in set(previous_tokens):
            if 0 <= token_id < len(logits):
                if logits[token_id] < 0:
                    adjusted_logits[token_id] = (
                        logits[token_id] * self.repetition_penalty
                    )
                else:
                    adjusted_logits[token_id] = (
                        logits[token_id] / self.repetition_penalty
                    )

        return adjusted_logits

    def apply_top_k(self, probs: np.ndarray) -> np.ndarray:
        if self.top_k is None or self.top_k <= 0:
            return probs

        top_k = min(self.top_k, len(probs))

        if top_k <= 0:
            return probs

        top_k_indices = np.argpartition(probs, -top_k)[-top_k:]

        mask = np.ones_like(probs, dtype=bool)
        mask[top_k_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0

        if np.sum(filtered_probs) > 0:
            normalized_probs = filtered_probs / np.sum(filtered_probs)
        else:
            normalized_probs = np.ones_like(probs) / len(probs)

        return normalized_probs

    def apply_top_p(self, probs: np.ndarray) -> np.ndarray:
        if self.top_p >= 1.0:
            return probs

        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        cumulative_probs = np.cumsum(sorted_probs)

        cutoff_indices = np.where(cumulative_probs >= self.top_p)[0]

        if len(cutoff_indices) > 0:
            cutoff_index = cutoff_indices[0]
            if cutoff_index < self.min_tokens_to_keep - 1:
                cutoff_index = self.min_tokens_to_keep - 1

            selected_indices = sorted_indices[: cutoff_index + 1]
        else:
            selected_indices = sorted_indices

        mask = np.ones_like(probs, dtype=bool)
        mask[selected_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0

        if np.sum(filtered_probs) > 0:
            normalized_probs = filtered_probs / np.sum(filtered_probs)
        else:
            normalized_probs = np.ones_like(probs) / len(probs)

        return normalized_probs

    def process_logits(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        processed_logits = logits.copy()
        # 1. apply repetition penalty
        processed_logits = self.apply_repetition_penalty(
            processed_logits, previous_tokens
        )

        # 2. apply softmax
        # not using softmax in case of long time cost
        probs = processed_logits
        # probs = self.softmax(processed_logits)

        # 3. apply top-k
        probs = self.apply_top_k(probs)

        # 4. apply top-p
        probs = self.apply_top_p(probs)

        # 5. apply temperature
        probs = self.apply_temperature(probs)
        return probs

    def sample(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> int:
        logits = logits[0][0]
        probs = self.process_logits(logits, previous_tokens)
        if np.all(probs == 0):
            probs = np.ones_like(probs) / len(probs)

        # sampled_index = np.random.choice(len(probs), p=probs)
        sampled_index = probs.argmax(-1)

        return np.array([sampled_index])

    def get_processed_probs(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        return self.process_logits(logits, previous_tokens)


class HmGpt:

    def __init__(
        self, prefill_path, decode_path, embedding_path, tokenizer_dir, ndevice
    ):
        self.ndevice = ndevice
        dev_manager = tcim.runtime.DevManager(
            get_hm_devices(self.ndevice), "Xh2HalBackend"
        )
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option1)
        logger.info("prefill model loaded")
        self.nblocks = self.get_nblocks()
        dummy_tensor_names = [
            f"model_layers_{i}_self_attn_kcache_input" for i in range(self.nblocks)
        ]
        dummy_tensor_names += [
            f"model_layers_{i}_self_attn_vcache_input" for i in range(self.nblocks)
        ]
        option2.set_dummy_tensors(dummy_tensor_names)
        self.decode = tcim.runtime.load(decode_path, option=option2)
        logger.info("decode model loaded")
        self.samplingmanager = SamplingManager(
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            repetition_penalty=args.repetition_penalty,
        )
        self.prefill_length = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        self.embedding_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]
        self.context_max_length = self.decode.get_input_info(
            self.decode.get_input_name(5)
        ).shape[2]

        for i in range(5, 2 * self.nblocks + 5):
            cache = self.prefill.get_input(self.prefill.get_input_name(i))
            self.decode.set_input(self.decode.get_input_name(i), cache)
        # set decode input
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode.get_input_name(2)
        self.decode.set_input(decode_current_length_name, current_length_input_1)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )
        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=True
        )["weight"]
        self.embedding_weight = embedding_weight.reshape(-1, self.embedding_len)
        self.context_length = 0
        self.window_size = 128
        self.reduce_value = 0
        self.perf_tracker = InferencePerformanceTracker()

    def create_window_mask(
        self,
        fill_length: int,
        old_cache_length: int,
        new_cache_length: int,
        window_size: int,
        mask_type: str = "window",
    ) -> np.ndarray:
        if mask_type == "window":
            assert window_size > 0, "window_size must be > 0 for window mask"

        input_act_length = ((fill_length + window_size - 1 + 15) // 16) * 16
        mask = np.full(
            (1, 1, fill_length, input_act_length), -65504.0, dtype=np.float16
        )

        for i in range(new_cache_length):
            global_pos = old_cache_length + i
            global_start = max(0, global_pos - window_size + 1)

            start_idx = global_start
            end_idx = global_pos + 1

            if end_idx - start_idx > window_size:
                start_idx = end_idx - window_size

            if i == 0:
                self.reduce_value = start_idx

            start_idx -= self.reduce_value
            end_idx -= self.reduce_value

            mask[0, 0, i, start_idx:end_idx] = 0.0

        return mask

    def create_global_mask(
        self,
        fill_length: int,
        old_cache_length: int,
        new_cache_length: int,
        context_length: int,
    ) -> np.ndarray:
        assert context_length > 0, "context_length must be > 0 for global mask"
        assert fill_length > 0, "fill_length must be > 0 for global mask"
        assert old_cache_length >= 0, "old_cache_length must be > 0 for global mask"
        assert new_cache_length > 0, "new_cache_length must be > 0 for global mask"
        assert (
            new_cache_length <= fill_length
        ), f"new_cache_length({new_cache_length}) must be <= fill_length({fill_length}) for global mask"
        assert (
            context_length >= fill_length
        ), f"context_length({context_length}) must be >= fill_length({fill_length}) for global mask"

        mask = np.full((1, 1, fill_length, context_length), -65504.0, dtype=np.float16)
        for i in range(new_cache_length):
            mask[0, 0, i, : old_cache_length + i + 1] = 0.0
        return mask

    def get_nblocks(self):
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def chat(self, question):
        self.generated_ids = []
        if not args.history:
            self.context_length = 0

        logger.success("question:")
        print("\033[1;95m{}\033[0m".format(question))

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOKEN_TIME)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": question,
            },
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOKEN_TIME)

        if input_echo_len >= self.context_max_length:
            logger.error(
                f"Question long than {self.context_max_length}, please shorten it!"
            )
            sys.exit(1)

        input_name = self.prefill.get_input_name(0)
        valid_length_name = self.prefill.get_input_name(1)
        current_length_name = self.prefill.get_input_name(2)
        local_attention_mask_name = self.prefill.get_input_name(3)
        global_attention_mask_name = self.prefill.get_input_name(4)
        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        for round in range(prefill_loop_round):
            valid_length = round * self.prefill_length + self.context_length
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

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
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
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)

            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")
            local_attention_mask_data = self.create_window_mask(
                self.prefill_length, valid_length, effective_length, self.window_size
            )
            global_attention_mask_data = self.create_global_mask(
                self.prefill_length,
                valid_length,
                effective_length,
                self.context_max_length,
            )

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
            self.prefill.set_input(input_name, input_data.numpy())
            self.prefill.set_input(valid_length_name, valid_length_data)
            self.prefill.set_input(current_length_name, current_length_data)
            self.prefill.set_input(local_attention_mask_name, local_attention_mask_data)
            self.prefill.set_input(
                global_attention_mask_name, global_attention_mask_data
            )
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
            self.prefill.run()
            self.prefill.sync()
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
        input_data = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)

        next_id = input_data.argmax(-1)[0]
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)

        prefill_response = self.tokenizer.decode(next_id)
        logger.success("response:")
        print("\033[1;95m{}".format(prefill_response), end="", flush=True)
        chat_history_ids = all_input_ids[0]
        next_id = torch.from_numpy(next_id)
        self.generated_ids.append(next_id)

        chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)
        all_response = prefill_response
        self.context_length += input_echo_len

        decode_count = 0
        skip_tokens = 0
        slide_len = 10  # sliding window length for decode
        last_response = self.tokenizer.decode(chat_history_ids.tolist()[-slide_len:])
        input_name = self.decode.get_input_name(0)
        valid_length_name = self.decode.get_input_name(1)
        local_attention_mask_name = self.decode.get_input_name(3)
        global_attention_mask_name = self.decode.get_input_name(4)

        # Decode loop for generating subsequent tokens
        while True:
            # Stop generation if context length exceeds maximum limit
            if self.context_length >= self.context_max_length:
                logger.info(
                    f"context length greater than {self.context_max_length}, break!"
                )
                break

            self.perf_tracker.perf_start(PERFTYPE.DECODE_TOTAL_TIME)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
            input_data = F.embedding(
                next_id.unsqueeze(0), self.embedding_weight
            ).reshape(1, 1, -1)
            self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
            self.decode.set_input(input_name, input_data.numpy())
            valid_length_data = np.array(self.context_length).astype("int32")
            self.decode.set_input(valid_length_name, valid_length_data)
            local_attention_mask_data = self.create_window_mask(
                1, self.context_length, 1, self.window_size
            )
            global_attention_mask_data = self.create_global_mask(
                1, self.context_length, 1, self.context_max_length
            )
            self.decode.set_input(local_attention_mask_name, local_attention_mask_data)
            self.decode.set_input(
                global_attention_mask_name, global_attention_mask_data
            )
            self.perf_tracker.perf_end(PERFTYPE.DECODE_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_INFER_TIME)
            self.decode.run()
            self.decode.sync()
            self.perf_tracker.perf_end(PERFTYPE.DECODE_INFER_TIME)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_OUTPUT_TIME)
            input_data = self.decode.get_output(self.decode.get_output_name(0)).numpy()
            self.perf_tracker.perf_end(PERFTYPE.DECODE_OUTPUT_TIME)

            decode_count += 1

            next_id = self.samplingmanager.sample(input_data, self.generated_ids)
            next_id = torch.from_numpy(next_id)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_TOKEN_TIME)

            # Check for end-of-sequence token
            if next_id == self.tokenizer.eos_token_id:
                if "decode_response" in locals():
                    print(decode_response, end="", flush=True)
                    all_response += decode_response
                self.perf_tracker.perf_end(PERFTYPE.DECODE_TOKEN_TIME)
                self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)
                break

            # Update chat history with new token
            chat_history_ids = torch.cat([chat_history_ids, next_id], dim=-1)
            self.generated_ids.append(next_id)

            # Convert token id to text (within DECODE_TOKEN_TIME scope)
            decode_response = self.tokenizer.decode(
                chat_history_ids.tolist()[-(slide_len + 1) - skip_tokens :]
            )[len(last_response) :]
            self.perf_tracker.perf_end(PERFTYPE.DECODE_TOKEN_TIME)

            self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)

            # Validate and print decoded text (outside timing scope)
            if decode_response != "" and is_valid_char(ord(decode_response[-1])):
                print(decode_response, end="", flush=True)
                all_response += decode_response
                last_response = self.tokenizer.decode(
                    chat_history_ids.tolist()[-slide_len:]
                )
                skip_tokens = 0
            else:
                skip_tokens += 1

            self.context_length = self.context_length + 1

        print("\033[0m")

        # Set basic performance metrics for reporting
        self.perf_tracker.set_basic_info(
            batch_size=1,
            input_seq_length=input_echo_len,
            output_seq_length=decode_count,
        )


if __name__ == "__main__":
    args = get_args()

    hmqwen = HmGpt(
        args.prefill_path,
        args.decode_path,
        args.embedding_path,
        args.tokenizer_dir,
        args.ndevice,
    )
    if args.it:
        from prompt_toolkit import prompt
    try:
        while True:
            if args.it:
                try:
                    question = prompt("Input your instruction here: ").strip()
                    if question.lower() in ("stop", "exit", "quit", ""):
                        break
                    if not question:
                        print("输入不能为空，请重新输入。")
                        continue
                except (EOFError, KeyboardInterrupt):
                    print("\n程序结束")
                    break
            else:
                question = args.question

            try:
                hmqwen.chat(question)
                hmqwen.perf_tracker.show_summary()
            except Exception as e:
                print(f"聊天过程中出错: {e}")
                if not args.it:
                    break
                continue
            if not args.it:
                break

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
