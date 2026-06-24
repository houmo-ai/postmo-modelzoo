#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Qwen3-VL-Embedding Demo - Python script for running Qwen3-VL-Embedding
# inference on HOUMO AI device.
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
import re
import sys
import time
import math
import argparse
from pathlib import Path
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tcim_lite as tcim
from loguru import logger
from PIL import Image, ImageOps

from hmatc.python.get_hm_devices import get_hm_devices
from hmatc.utils.perf_infomations import (
    InferencePerformanceTracker,
    PERFTYPE,
)
from hmatc.utils.utils import first_not_none, get_model_configs
from processing_qwen3_vl import Qwen3VLProcessor

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
HOUMO_PIC_PATH = os.getenv(
    "HOUMO_PIC_PATH", str(Path(__file__).resolve().parents[3] / "data" / "pic")
)
DEFAULT_INSTRUCTION = "Represent the user's input."


def get_default_tokenizer_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "qwen3-vl-embedding")
    model_size = model_config.get("model_size", "8b")
    return f"{model_name}-{model_size}"


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
        "--vit_path",
        dest="vit_path",
        type=str,
        default=None,
        help="houmo visual model path",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=None,
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--image",
        dest="image",
        type=str,
        default=f"{HOUMO_PIC_PATH}/beach.jpeg",
        help="demo image path",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number",
    )
    parser.add_argument(
        "--max_size_w",
        dest="max_size_w",
        type=int,
        default=None,
        help="max image width for vision filename suffix",
    )
    parser.add_argument(
        "--max_size_h",
        dest="max_size_h",
        type=int,
        default=None,
        help="max image height for vision filename suffix",
    )
    parser.add_argument(
        "--max_size_t",
        dest="max_size_t",
        type=int,
        default=None,
        help="max temporal size for vision filename suffix",
    )
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    args.max_size_w = first_not_none(
        args.max_size_w, model_config.get("max_size_w", 896)
    )
    args.max_size_h = first_not_none(
        args.max_size_h, model_config.get("max_size_h", 896)
    )
    args.max_size_t = first_not_none(args.max_size_t, model_config.get("max_size_t", 2))
    if args.tokenizer_dir is None:
        args.tokenizer_dir = get_default_tokenizer_dir(model_config)
    if args.vit_path is None:
        args.vit_path = os.path.join(
            "output",
            HOUMO_TARGET,
            f"{args.model_name}-{args.model_size}_visual_{args.max_size_w}x{args.max_size_h}x{args.max_size_t}.hmm",
        )
    if args.prefill_path is None:
        args.prefill_path = os.path.join(
            "output", HOUMO_TARGET, f"{args.model_name}-{args.model_size}_prefill.hmm"
        )
    if args.ndevice > 1 and args.prefill_path.endswith(".hmm"):
        args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
    return args


def show_statistics(input_tokens, prefill_time, total_time):
    logger.success(
        f"Total Input: {input_tokens} tokens, Prefill Cost {prefill_time*1000:.3f} ms"
    )
    logger.success(f"Prefill Speed: {input_tokens / prefill_time:.2f} tokens/s")
    logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")


class HmQwen3VLEmbedder(object):
    IMAGE_TOKEN_ID = 151655
    VISION_START_TOKEN_ID = 151652

    def __init__(self, args):
        self.args = args
        self.ndevice = args.ndevice
        self.perf_tracker = InferencePerformanceTracker()
        dev_manager = tcim.runtime.DevManager(
            get_hm_devices(self.ndevice), "Xh2HalBackend"
        )
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        option = tcim.runtime.Option(weight_manager)
        self.perf_tracker.perf_start(PERFTYPE.VISION_LOAD_TIME)
        self.vit_model = tcim.runtime.load(args.vit_path, option=option)
        self.perf_tracker.perf_end(PERFTYPE.VISION_LOAD_TIME)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self.prefill = tcim.runtime.load(args.prefill_path, option=option)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)
        logger.info("visual and prefill model loaded successfully")

        self.processor = Qwen3VLProcessor.from_pretrained(args.tokenizer_dir)
        self.patch_size = int(getattr(self.processor.image_processor, "patch_size", 16))
        self.merge_size = int(getattr(self.processor.image_processor, "merge_size", 2))
        self.temporal_patch_size = int(
            getattr(self.processor.image_processor, "temporal_patch_size", 2)
        )

        embedding = torch.load(args.embedding_path, map_location="cpu", weights_only=False)
        if isinstance(embedding, torch.nn.Embedding):
            embedding = embedding.weight
        elif isinstance(embedding, dict) and "weight" in embedding:
            embedding = embedding["weight"]
        self.embedding = embedding.float()

        self.prefill_input_names = [
            self.prefill.get_input_name(i) for i in range(self.prefill.get_num_inputs())
        ]
        self.vit_output_names = [
            self.vit_model.get_output_name(i) for i in range(self.vit_model.get_num_outputs())
        ]
        self.nblocks = self.get_nblocks()
        self.prefill_length = self.prefill.get_input_info("input_1").shape[1]
        self.hidden_dims = self.prefill.get_input_info("input_1").shape[2]
        self.context_max_length = self.prefill.get_input_info(
            "model_layers_0_self_attn_kcache_input"
        ).shape[2]
        self.image_size_h = self.vit_model.get_input_info("pixel_values").shape[-2]
        self.image_size_w = self.vit_model.get_input_info("pixel_values").shape[-1]
        self.image_token_count = self.vit_model.get_output_info(
            self.vit_model.get_output_name(0)
        ).shape[1]
        self.num_deepstack = sum(
            1 for name in self.vit_output_names if name.startswith("deepstack_feature")
        )
        self.num_deepstack_in = sum(
            1 for name in self.prefill_input_names if name.startswith("deepstack_image_embed")
        )
        self.context_length = 0
        self.prefill_time = 0
        self.num_images = 0

        logger.info(
            f"Loaded Qwen3-VL-Embedding [{args.model_name}-{args.model_size}] | "
            f"blocks={self.nblocks} prefill_length={self.prefill_length} "
            f"ctx_max={self.context_max_length} hidden={self.hidden_dims} "
            f"image_size={self.image_size_w}x{self.image_size_h} "
            f"image_tokens={self.image_token_count} deepstack={self.num_deepstack}"
        )
        self.perf_tracker.reset_perf_time()

    def get_nblocks(self):
        """Calculate number of transformer blocks from input tensor names."""
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        return sum(1 for item in self.prefill_input_names if re.match(pattern, item))

    def reset(self):
        self.context_length = 0
        self.prefill_time = 0
        for input_name in self.prefill_input_names:
            if "model_layers" in input_name:
                cache = self.prefill.get_dev_input(input_name)
                self.prefill.set_input(input_name, cache)

    def format_model_input(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        instruction: Optional[str] = DEFAULT_INSTRUCTION,
    ) -> List[Dict[str, Any]]:
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
                instruction = instruction + "."
        content: List[Dict[str, Any]] = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]
        if image is None and not text:
            content.append({"type": "text", "text": "NULL"})
            return conversation
        if image is not None:
            image_ref = image if image.startswith(("http://", "https://")) else "file://" + image
            content.append(
                {
                    "type": "image",
                    "image": image_ref,
                    "min_pixels": self.processor.image_processor.min_pixels,
                    "max_pixels": self.processor.image_processor.max_pixels,
                }
            )
        if text:
            content.append({"type": "text", "text": text})
        return conversation

    def load_and_process_image(self, image_path):
        target_w, target_h = self.args.max_size_w, self.args.max_size_h
        if target_w != self.image_size_w or target_h != self.image_size_h:
            raise ValueError(
                f"config image size {target_w}x{target_h} does not match model input "
                f"{self.image_size_w}x{self.image_size_h}"
            )
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        if (orig_w, orig_h) != (target_w, target_h):
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
            image = ImageOps.expand(
                image,
                border=(0, 0, target_w - new_w, target_h - new_h),
                fill=(114, 114, 114),
            )
        return image

    def preprocess(self, text: Optional[str], image: Optional[str]):
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOKEN_TIME)
        conversation = self.format_model_input(text=text, image=image)
        prompt = self.processor.apply_chat_template(
            [conversation], add_generation_prompt=True, tokenize=False
        )
        if image is not None:
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOKEN_TIME)
            self.perf_tracker.perf_start(PERFTYPE.VISION_TOTAL_TIME)
            self.perf_tracker.perf_start(PERFTYPE.VISION_PREPROCESS_TIME)
            image_inputs = [self.load_and_process_image(image)]
            inputs = self.processor(
                text=prompt,
                images=image_inputs,
                videos=None,
                padding=True,
                do_resize=False,
                return_tensors="pt",
            )
            self.perf_tracker.perf_end(PERFTYPE.VISION_PREPROCESS_TIME)
            self.num_images += 1
        else:
            inputs = self.processor(
                text=prompt,
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOKEN_TIME)
        return inputs

    def get_rope_index(
        self, input_ids: torch.Tensor, image_grid_thw: Optional[torch.Tensor]
    ) -> torch.Tensor:
        seq = input_ids[0].tolist()
        if image_grid_thw is None:
            ramp = torch.arange(len(seq), dtype=torch.long)
            return ramp.view(1, -1).expand(3, -1).contiguous()

        image_index = 0
        llm_pos_ids_list: List[torch.Tensor] = []
        st = 0
        vision_start_indices = [
            i for i, token_id in enumerate(seq) if token_id == self.VISION_START_TOKEN_ID
        ]
        image_nums = sum(
            1
            for i in vision_start_indices
            if i + 1 < len(seq) and seq[i + 1] == self.IMAGE_TOKEN_ID
        )
        for _ in range(image_nums):
            ed = seq.index(self.IMAGE_TOKEN_ID, st)
            t = int(image_grid_thw[image_index][0])
            h = int(image_grid_thw[image_index][1])
            w = int(image_grid_thw[image_index][2])
            image_index += 1
            llm_grid_t = t
            llm_grid_h = h // self.merge_size
            llm_grid_w = w // self.merge_size
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )
            t_index = (
                torch.arange(llm_grid_t)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(seq):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(seq) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )
        return torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

    def run_visual(self, vit_input):
        self.perf_tracker.perf_start(PERFTYPE.VISION_INPUT_TIME)
        self.vit_model.set_input("pixel_values", vit_input.numpy())
        self.perf_tracker.perf_end(PERFTYPE.VISION_INPUT_TIME)
        self.perf_tracker.perf_start(PERFTYPE.VISION_INFER_TIME)
        self.vit_model.run()
        self.vit_model.sync()
        self.perf_tracker.perf_end(PERFTYPE.VISION_INFER_TIME)
        self.perf_tracker.perf_start(PERFTYPE.VISION_OUTPUT_TIME)
        outputs = []
        for output_name in self.vit_output_names:
            outputs.append(torch.Tensor(self.vit_model.get_output(output_name).numpy()))
        self.perf_tracker.perf_end(PERFTYPE.VISION_OUTPUT_TIME)
        self.perf_tracker.perf_end(PERFTYPE.VISION_TOTAL_TIME)
        return outputs[0], outputs[1 : 1 + self.num_deepstack]

    def build_inputs_embeds(
        self, inputs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], int, Optional[torch.Tensor]]:
        input_ids = inputs["input_ids"]
        prompt_len = int(input_ids.shape[1])
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
        inputs_embeds = F.embedding(input_ids, self.embedding).to(torch.float16)
        deepstack_tensors = tuple(
            torch.zeros_like(inputs_embeds) for _ in range(self.num_deepstack_in)
        )
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)

        image_grid_thw = inputs.get("image_grid_thw")
        n_image_tokens = int((input_ids == self.IMAGE_TOKEN_ID).sum().item())
        if n_image_tokens > 0:
            hm_pixel_values = inputs["hm_pixel_values"][0].half()
            expected_shape = tuple(self.vit_model.get_input_info("pixel_values").shape)
            if tuple(hm_pixel_values.shape) != expected_shape:
                raise ValueError(
                    f"visual input shape {tuple(hm_pixel_values.shape)} != {expected_shape}"
                )
            image_embeds, deepstack = self.run_visual(hm_pixel_values)
            image_embeds = image_embeds.reshape(-1, self.hidden_dims).to(inputs_embeds)
            if image_embeds.shape[0] != n_image_tokens:
                raise ValueError(
                    f"image tokens={n_image_tokens} but visual emitted "
                    f"{image_embeds.shape[0]} features"
                )
            self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
            image_mask = (
                (input_ids == self.IMAGE_TOKEN_ID).unsqueeze(-1).expand_as(inputs_embeds)
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            deepstack_outputs = []
            for layer_index in range(self.num_deepstack_in):
                layer_embed = torch.zeros_like(inputs_embeds)
                if layer_index < len(deepstack):
                    feat = deepstack[layer_index].reshape(-1, self.hidden_dims).to(
                        layer_embed
                    )
                    layer_embed = layer_embed.masked_scatter(image_mask, feat)
                deepstack_outputs.append(layer_embed)
            deepstack_tensors = tuple(deepstack_outputs)
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)
        return inputs_embeds, deepstack_tensors, prompt_len, image_grid_thw

    def run_prefill(
        self,
        inputs_embeds: torch.Tensor,
        deepstack_tensors: Tuple[torch.Tensor, ...],
        position_ids: torch.Tensor,
        prompt_len: int,
    ):
        if prompt_len >= self.context_max_length:
            logger.error(
                f"Question long than {self.context_max_length}, please shorten it!"
            )
            sys.exit(1)

        steps = math.ceil(prompt_len / self.prefill_length)
        input_sequence_length = steps * self.prefill_length
        if input_sequence_length > inputs_embeds.shape[1]:
            pad_len = input_sequence_length - inputs_embeds.shape[1]
            pad_embed = torch.zeros(
                1, pad_len, self.hidden_dims, dtype=inputs_embeds.dtype
            )
            inputs_embeds = torch.cat([inputs_embeds, pad_embed], dim=1)
            deepstack_tensors = tuple(
                torch.cat([item, torch.zeros_like(pad_embed)], dim=1)
                for item in deepstack_tensors
            )
            pad_pos = position_ids[:, -1:].expand(3, pad_len)
            position_ids = torch.cat([position_ids, pad_pos], dim=1)

        time_position_ids = position_ids[0].to(torch.int32)
        height_position_ids = position_ids[1].to(torch.int32)
        width_position_ids = position_ids[2].to(torch.int32)
        last_hidden = None
        past_seq_length = torch.tensor([0], dtype=torch.int32)

        for i in range(steps):
            start = i * self.prefill_length
            end = (i + 1) * self.prefill_length
            current_input_length = min(end, prompt_len) - start
            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
            self.prefill.set_input(
                "input_1", inputs_embeds[:, start:end, :].detach().numpy()
            )
            self.prefill.set_input(
                "time_position_ids", time_position_ids[start:end].detach().numpy()
            )
            self.prefill.set_input(
                "height_position_ids", height_position_ids[start:end].detach().numpy()
            )
            self.prefill.set_input(
                "width_position_ids", width_position_ids[start:end].detach().numpy()
            )
            self.prefill.set_input("valid_length", past_seq_length.detach().numpy())
            self.prefill.set_input(
                "current_length",
                torch.tensor([current_input_length], dtype=torch.int32).detach().numpy(),
            )
            for deepstack_idx in range(self.num_deepstack_in):
                self.prefill.set_input(
                    f"deepstack_image_embed_{deepstack_idx}",
                    deepstack_tensors[deepstack_idx][:, start:end, :].detach().numpy(),
                )
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
            prefill_start = time.time()
            self.prefill.run()
            self.prefill.sync()
            self.prefill_time += time.time() - prefill_start
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)
            self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
            last_hidden = torch.Tensor(self.prefill.get_output("hidden_states").numpy())
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)
            past_seq_length += current_input_length
        return last_hidden, prompt_len - (steps - 1) * self.prefill_length

    @torch.no_grad()
    def encode(self, item: Dict[str, Any], prompt_name="document"):
        self.reset()
        logger.success(prompt_name + ":")
        print("\033[1;95m{}\033[0m".format(item))
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        inputs = self.preprocess(text=item.get("text"), image=item.get("image"))
        input_ids = inputs["input_ids"]
        input_echo_len = input_ids.numel()
        inputs_embeds, deepstack_tensors, prompt_len, image_grid_thw = (
            self.build_inputs_embeds(inputs)
        )
        position_ids = self.get_rope_index(input_ids, image_grid_thw)
        last_hidden, last_chunk_valid_len = self.run_prefill(
            inputs_embeds, deepstack_tensors, position_ids, prompt_len
        )
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)
        embeddings = last_hidden[:, last_chunk_valid_len - 1, :].float()
        return embeddings, input_echo_len, self.prefill_time

    def process(self, items: List[Dict[str, Any]]):
        embeddings = []
        input_tokens = 0
        prefill_times = 0
        for index, item in enumerate(items):
            prompt_name = "query" if index < 4 else "document"
            embedding, input_token, prefill_time = self.encode(item, prompt_name)
            embeddings.append(embedding)
            input_tokens += input_token
            prefill_times += prefill_time
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        self.perf_tracker.set_basic_info(
            batch_size=1,
            input_seq_length=input_tokens,
            output_seq_length=0,
            num_images=self.num_images,
        )
        return embeddings, input_tokens, prefill_times

    def show_performance_summary(self, input_tokens: int, total_time: float):
        metrics = self.perf_tracker.current_metrics
        prefill_infos = metrics.prefill_perf_infos
        vision_infos = metrics.vision_perf_infos
        prefill_total_ms = (
            prefill_infos.tokenizer_time
            + prefill_infos.embedding_time
            + prefill_infos.setinput_time
            + prefill_infos.infer_time
            + prefill_infos.getoutput_time
        )
        e2e_ms = total_time * 1000
        prefill_speed = (
            input_tokens / (prefill_infos.infer_time / 1000)
            if prefill_infos.infer_time > 0
            else 0.0
        )
        e2e_speed = input_tokens / total_time if total_time > 0 else 0.0
        vision_speed = (
            self.num_images / (vision_infos.infer_time / 1000)
            if vision_infos.infer_time > 0 and self.num_images > 0
            else 0.0
        )

        logger.success("=" * 100)
        logger.success("                    Qwen3-VL-Embedding Performance Summary")
        logger.success("=" * 100)
        logger.success("Configuration Details:")
        logger.success(f"  Batch Size: {1:>6}")
        logger.success(f"  Total Input Length: {input_tokens:>6} tokens")
        logger.success(f"  Number of Images: {self.num_images:>6} images")
        if metrics.vision_load_time > 0:
            logger.success(f"  Vision Model Load Time: {metrics.vision_load_time:>7.2f} ms")
        if metrics.prefill_load_time > 0:
            logger.success(f"  Prefill Model Load Time: {metrics.prefill_load_time:>7.2f} ms")

        if self.num_images > 0:
            logger.success("Vision Stage Performance:")
            logger.success(f"  Total Time: {vision_infos.vision_total_time:>7.2f} ms")
            logger.success(
                f"  Preprocessing Time: {vision_infos.vision_preprocess_time:>7.2f} ms"
            )
            logger.success(f"  API SetInput Time: {vision_infos.setinput_time:>7.2f} ms")
            logger.success(
                f"  API Inference Time: {vision_infos.infer_time:>7.2f} ms | "
                f"Speed: {vision_speed:>7.2f} images/s"
            )
            logger.success(f"  API GetOutput Time: {vision_infos.getoutput_time:>7.2f} ms")

        logger.success("Prefill Stage Performance:")
        logger.success(f"  Total Time: {prefill_total_ms:>7.2f} ms")
        logger.success(f"  Tokenization Time: {prefill_infos.tokenizer_time:>7.2f} ms")
        logger.success(f"  Embedding Time: {prefill_infos.embedding_time:>7.2f} ms")
        logger.success(f"  API SetInput Time: {prefill_infos.setinput_time:>7.2f} ms")
        logger.success(
            f"  API Inference Time: {prefill_infos.infer_time:>7.2f} ms | "
            f"Prefill Speed: {prefill_speed:>7.2f} tokens/s"
        )
        logger.success(f"  API GetOutput Time: {prefill_infos.getoutput_time:>7.2f} ms")

        logger.success("Overall Performance Metrics:")
        logger.success(f"  E2E Latency (End-to-End): {e2e_ms:>7.2f} ms")
        logger.success(f"  E2E TPS (Input Throughput): {e2e_speed:>7.2f} tokens/s")
        logger.success("=" * 100)


def main():
    args = get_args()
    logger.info(f"args: {args}")
    model = HmQwen3VLEmbedder(args)

    queries = [
        {"text": "A woman playing with her dog on a beach at sunset."},
        {"text": "Pet owner training dog outdoors near water."},
        {"text": "Woman surfing on waves during a sunny day."},
        {"text": "City skyline view from a high-rise building at night."},
    ]
    documents = [
        {
            "text": "A woman shares a joyful moment with her golden retriever on a "
            "sun-drenched beach at sunset, as the dog offers its paw in a "
            "heartwarming display of companionship and trust."
        },
        {"image": args.image},
        {
            "text": "A woman shares a joyful moment with her golden retriever on a "
            "sun-drenched beach at sunset, as the dog offers its paw in a "
            "heartwarming display of companionship and trust.",
            "image": args.image,
        },
    ]

    start_time = time.time()
    embeddings, input_tokens, prefill_times = model.process(queries + documents)
    similarity_scores = embeddings[:4] @ embeddings[4:].T
    total_time = time.time() - start_time

    doc_labels = ["Doc 0 [text]", "Doc 1 [image]", "Doc 2 [text+image]"]
    print("\n" + "=" * 100)
    header = f"{'Query':<50} " + " ".join(f"{label:<18}" for label in doc_labels)
    print(header)
    print("=" * 100)
    for i, query in enumerate(queries):
        row = f"{query['text'][:48]:<50} "
        for j in range(3):
            row += f"{similarity_scores[i][j].item():<18.4f}"
        print(row)
    print("=" * 100)

    print("\nBest match per query:")
    for i, query in enumerate(queries):
        best_idx = int(similarity_scores[i].argmax().item())
        best_score = similarity_scores[i][best_idx].item()
        print(
            f"  Q{i}: \"{query['text'][:40]}\"  ->  {doc_labels[best_idx]}  "
            f"(score: {best_score:.4f})"
        )

    show_statistics(input_tokens, prefill_times, total_time)
    model.show_performance_summary(input_tokens, total_time)


if __name__ == "__main__":
    main()
