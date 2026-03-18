#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Qwen2.5-VL Inference Demo - Python script for running Qwen2.5-VL
# automatic speech recognition on HOUMO AI device.
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
import logging
import argparse
from typing import List, Tuple, Optional
from loguru import logger

logging.basicConfig(level=logging.ERROR)
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
import torch
import torch.nn.functional as F
import numpy as np
import tcim_lite as tcim
from PIL import Image
from processing_qwen2_5_vl import Qwen2_5_VLProcessor
from utils import get_rope_index, QRawToYuv

from hmatc.utils.perf_infomations import InferencePerformanceTracker, InferenceMetrics, PERFTYPE

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

TARGET_TYPE = torch.float16


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default="qwen2.5-vl",
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
        default=os.path.join("output", HOUMO_TARGET, "qwen2.5-vl_visual.hmm"),
        help="houmo visual model path",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen2.5-vl_prefill.hmm"),
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen2.5-vl_decode.hmm"),
        help="houmo decode model path",
    )
    parser.add_argument(
        "--repetition_penalty",
        dest="repetition_penalty",
        type=float,
        default=1.1,
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
    args = parser.parse_args()
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
        logits = logits[0]
        if HOUMO_TARGET == "xh2":
            logits = logits[0]
        probs = self.process_logits(logits, previous_tokens)
        if np.all(probs == 0):
            probs = np.ones_like(probs) / len(probs)

        # sampled_index = np.random.choice(len(probs), p=probs)
        sampled_index = probs.argmax(-1)

        return np.array([[sampled_index]])

    def get_processed_probs(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        return self.process_logits(logits, previous_tokens)


class Qwen25VL:
    def __init__(
        self,
        vit_path,
        prefill_path,
        decode_path,
        tokenizer_dir,
        embedding_path,
        window_size=112,
        spatial_merge_size=2,
        patch_size=14,
    ):
        self.perf_tracker = InferencePerformanceTracker()
        weight_manager = tcim.runtime.WeightManager(0)
        option0 = tcim.runtime.Option(weight_manager)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        self.perf_tracker.perf_start(PERFTYPE.VISION_LOAD_TIME)
        self.vit_model = tcim.runtime.load(os.path.join(vit_path), option=option0)
        self.perf_tracker.perf_end(PERFTYPE.VISION_LOAD_TIME)
        logger.info("vit model loaded")
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self.prefill = tcim.runtime.load(os.path.join(prefill_path), option=option1)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)
        logger.info("prefill model loaded")
        self.nblocks = self.get_nblocks()
        dummy_tensor_names = [
            f"model_layers_{i}_self_attn_kcache_input" for i in range(self.nblocks)
        ]
        dummy_tensor_names += [
            f"model_layers_{i}_self_attn_vcache_input" for i in range(self.nblocks)
        ]
        option2.set_dummy_tensors(dummy_tensor_names)
        self.perf_tracker.perf_start(PERFTYPE.DECODE_LOAD_TIME)
        self.decode = tcim.runtime.load(os.path.join(decode_path), option=option2)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_LOAD_TIME)
        logger.info("decode model loaded")
        self.samplingmanager = SamplingManager(
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            repetition_penalty=args.repetition_penalty,
        )
        self.processor = Qwen2_5_VLProcessor.from_pretrained(tokenizer_dir)
        self.device = torch.device("cpu")
        prefill_shape = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[:2]
        self.prefill_shape = torch.Size(prefill_shape)
        self.prefill_len = self.prefill_shape.numel()
        self.window_size = window_size
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.batch_size = 1
        self.max_size_t = 2
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.eos_token_id = [151645, 151643]
        # set mode
        self.rgb2yuv = QRawToYuv(input_color_type="RGB", toYUV_format="YUV444")

        self.embedding_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]

        self.context_max_length = self.decode.get_input_info(
            self.decode.get_input_name(6)
        ).shape[2]
        self.image_shape = self.vit_model.get_input_info(
            self.vit_model.get_input_name(0)
        ).shape[3:]

        for i in range(self.nblocks):
            kcache = self.prefill.get_input(f"model_layers_{i}_self_attn_kcache_input")
            vcache = self.prefill.get_input(f"model_layers_{i}_self_attn_vcache_input")
            self.decode.set_input(f"model_layers_{i}_self_attn_kcache_input", kcache)
            self.decode.set_input(f"model_layers_{i}_self_attn_vcache_input", vcache)
        self.decode.set_input("current_length", np.array([1]).astype("int16"))
        self.embedding = torch.load(embedding_path, weights_only=False)
        if HOUMO_TARGET == "xh2":
            self.embedding = self.embedding.weight
        self.hidden_dims = self.embedding.shape[-1]

        self.perf_tracker.reset_perf_time()

    def get_nblocks(self):
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def create_template(self, prompt, image_dir):
        content_list = []
        if image_dir:
            for img_path in image_dir:
                content_list.append({"type": "image", "image": img_path})
        content_list.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content_list}]
        return messages

    def preprocess(self, prompt, image_dir):
        messages = self.create_template(prompt, image_dir)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        resized_image_inputs = None
        video_inputs = None
        if image_dir:
            resized_image_inputs = []
            try:
                from qwen_vl_utils import process_vision_info

                image_inputs, video_inputs = process_vision_info(messages)
                for image_input in image_inputs:
                    resized_image_input = image_input.resize(
                        (self.image_shape[1], self.image_shape[0])
                    )
                    resized_image_inputs.append(resized_image_input)
            except:
                for content in messages[0]["content"]:
                    if content["type"] == "image":
                        image_input = Image.open(content["image"])
                        resized_image_input = image_input.resize(
                            ((self.image_shape[1], self.image_shape[0]))
                        )
                        resized_image_inputs.append(resized_image_input)

        inputs = self.processor(
            text=[text],
            images=resized_image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def preprocess_visual(self, inputs):
        visual_inputs = dict()
        hidden_states = []
        window_indexes = []
        window_masks = []
        for batch in range(inputs["hm_pixel_values"].shape[0]):
            hidden_state = (
                inputs["hm_pixel_values"][batch]
                .unsqueeze(0)
                .repeat(self.batch_size, 1, 1, 1)
            )
            hidden_state = (
                hidden_state.unsqueeze(2).repeat(1, 1, self.max_size_t, 1, 1).squeeze(0)
            )
            window_index, cu_window_seqlens = self.get_window_index(
                inputs["image_grid_thw"][batch].unsqueeze(0)
            )
            cu_window_seqlens = torch.tensor(
                cu_window_seqlens, device=self.device, dtype=torch.int32
            )
            cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
            seq_len = cu_window_seqlens[-1]
            attention_mask = torch.full(
                [1, seq_len, seq_len],
                torch.iinfo(torch.int16).min,
                device=inputs["hm_pixel_values"].device,
                dtype=torch.int16,
            )
            for i in range(1, len(cu_window_seqlens)):
                attention_mask[
                    ...,
                    cu_window_seqlens[i - 1] : cu_window_seqlens[i],
                    cu_window_seqlens[i - 1] : cu_window_seqlens[i],
                ] = 0

            hidden_states.append(hidden_state.to(self.device))
            window_indexes.append(window_index.to(self.device))
            window_masks.append(attention_mask.to(self.device))
        visual_inputs["window_index"] = torch.stack(window_indexes)
        visual_inputs["window_mask"] = torch.stack(window_masks)
        visual_inputs["hidden_states"] = torch.stack(hidden_states)

        return visual_inputs

    def run_visual(self, inputs):
        vit_model_outputs = list()
        for batch in range(inputs["hidden_states"].shape[0]):
            self.perf_tracker.perf_start(PERFTYPE.VISION_INPUT_TIME)
            self.vit_model.set_input(
                self.vit_model.get_input_name(0),
                inputs["hidden_states"][batch].unsqueeze(0).numpy().astype(np.float16),
            )
            self.vit_model.set_input(
                self.vit_model.get_input_name(1),
                inputs["window_index"][batch].numpy().astype(np.int32),
            )
            if self.vit_model.get_num_inputs() == 3:
                self.vit_model.set_input(
                    self.vit_model.get_input_name(2),
                    inputs["window_mask"][batch].numpy().astype(np.float16),
                )
            self.perf_tracker.perf_end(PERFTYPE.VISION_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.VISION_INFER_TIME)
            self.vit_model.run()
            self.vit_model.sync()
            self.perf_tracker.perf_end(PERFTYPE.VISION_INFER_TIME)

            self.perf_tracker.perf_start(PERFTYPE.VISION_OUTPUT_TIME)
            vit_model_output = (
                self.vit_model.get_output(self.vit_model.get_output_name(0))
                .numpy()
                .astype(np.float16)
            )
            self.perf_tracker.perf_end(PERFTYPE.VISION_OUTPUT_TIME)
            vit_model_outputs.append(torch.tensor(vit_model_output))

        return torch.cat(vit_model_outputs, dim=0)

    def preprocess_prefill(self, inputs, image_features):
        image_grid_thw = None
        input_ids = inputs["input_ids"].cpu()
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
        inputs_embeds = F.embedding(input_ids, self.embedding).cpu()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)
        mask = input_ids == 151655  # <image> token id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded
        if image_features is not None:
            image_features = image_features.type(TARGET_TYPE)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
            image_grid_thw = inputs["image_grid_thw"]

        position_ids, rope_deltas = get_rope_index(
            input_ids,
            image_grid_thw,
            None,  # video_grid_thw is None
            None,
            inputs["attention_mask"].cpu(),
        )

        time_position_ids = position_ids[0][0]
        hight_position_ids = position_ids[1][0]
        width_position_ids = position_ids[2][0]
        return (
            inputs_embeds,
            time_position_ids,
            hight_position_ids,
            width_position_ids,
            rope_deltas,
        )

    def create_prefill_inputs(
        self,
        inputs_embeds,
        time_position_ids,
        hight_position_ids,
        width_position_ids,
        pre_gen_idx,
    ):
        x = inputs_embeds[
            :,
            pre_gen_idx * self.prefill_len : (pre_gen_idx + 1) * self.prefill_len,
        ]
        x_time = time_position_ids[
            pre_gen_idx * self.prefill_len : (pre_gen_idx + 1) * self.prefill_len
        ]
        x_hight = hight_position_ids[
            pre_gen_idx * self.prefill_len : (pre_gen_idx + 1) * self.prefill_len
        ]
        x_width = width_position_ids[
            pre_gen_idx * self.prefill_len : (pre_gen_idx + 1) * self.prefill_len
        ]
        p_current_length = torch.tensor([self.prefill_len])
        p_valid_length = (p_current_length * pre_gen_idx).to(TARGET_TYPE)
        prefill_inputs = dict(
            input_1=x,
            valid_length=p_valid_length,
            current_length=p_current_length,
            time_position_ids=x_time,
            hight_position_ids=x_hight,
            width_position_ids=x_width,
        )
        return prefill_inputs

    def run_prefill(
        self, inputs_embeds, time_position_ids, hight_position_ids, width_position_ids
    ):
        current_length = inputs_embeds.shape[1]
        if current_length >= self.context_max_length:
            logger.error(
                f"Question long than {self.context_max_length}, please shorten it!"
            )
            sys.exit(1)
        if current_length > self.prefill_len:
            pre_gen_nums = current_length // self.prefill_len
            for pre_gen_idx in range(pre_gen_nums):
                prefill_inputs = self.create_prefill_inputs(
                    inputs_embeds,
                    time_position_ids,
                    hight_position_ids,
                    width_position_ids,
                    pre_gen_idx,
                )
                self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
                self.prefill.set_input(
                    self.prefill.get_input_name(0),
                    prefill_inputs["input_1"].detach().numpy(),
                )
                self.prefill.set_input(
                    self.prefill.get_input_name(1),
                    prefill_inputs["time_position_ids"].detach().numpy(),
                )
                self.prefill.set_input(
                    self.prefill.get_input_name(2),
                    prefill_inputs["hight_position_ids"].detach().numpy(),
                )
                self.prefill.set_input(
                    self.prefill.get_input_name(3),
                    prefill_inputs["width_position_ids"].detach().numpy(),
                )
                self.prefill.set_input(
                    self.prefill.get_input_name(4),
                    prefill_inputs["valid_length"].detach().numpy(),
                )
                self.prefill.set_input(
                    self.prefill.get_input_name(5),
                    prefill_inputs["current_length"].detach().numpy(),
                )
                self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

                self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
                self.prefill.run()
                self.prefill.sync()
                self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)

                self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
                prefill_output = self.prefill.get_output(
                    self.prefill.get_output_name(0)
                )
                self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)
        else:
            pre_gen_nums = 0

        current_length = current_length % self.prefill_len
        prefill_shape = list(self.prefill_shape)
        prefill_shape.append(self.hidden_dims)
        x = torch.zeros(prefill_shape, dtype=TARGET_TYPE)
        x[:, :current_length] = inputs_embeds[:, -current_length:]

        x_time = torch.zeros(self.prefill_len, dtype=TARGET_TYPE)
        x_hight = torch.zeros(self.prefill_len, dtype=TARGET_TYPE)
        x_width = torch.zeros(self.prefill_len, dtype=TARGET_TYPE)
        x_time[:current_length] = time_position_ids[-current_length:]
        x_hight[:current_length] = hight_position_ids[-current_length:]
        x_width[:current_length] = width_position_ids[-current_length:]
        current_length = torch.tensor([current_length])
        valid_length = (torch.tensor([self.prefill_len]) * pre_gen_nums).to(TARGET_TYPE)
        prefill_inputs = dict(
            input_1=x,
            valid_length=valid_length,
            current_length=current_length,
            time_position_ids=x_time,
            hight_position_ids=x_hight,
            width_position_ids=x_width,
        )

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
        self.prefill.set_input(
            self.prefill.get_input_name(0), prefill_inputs["input_1"].detach().numpy()
        )
        self.prefill.set_input(
            self.prefill.get_input_name(1),
            prefill_inputs["time_position_ids"].detach().numpy(),
        )
        self.prefill.set_input(
            self.prefill.get_input_name(2),
            prefill_inputs["hight_position_ids"].detach().numpy(),
        )
        self.prefill.set_input(
            self.prefill.get_input_name(3),
            prefill_inputs["width_position_ids"].detach().numpy(),
        )
        self.prefill.set_input(
            self.prefill.get_input_name(4),
            prefill_inputs["valid_length"].detach().numpy(),
        )
        self.prefill.set_input(
            self.prefill.get_input_name(5),
            prefill_inputs["current_length"].detach().numpy(),
        )
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
        self.prefill.run()
        self.prefill.sync()
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
        prefill_output = self.prefill.get_output(self.prefill.get_output_name(0))
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)
        next_id = prefill_output.numpy().argmax(-1)
        return next_id, valid_length, current_length

    def chat_vit_prefill(self, image_dir, prompt, system_prompt=None):
        self.generated_ids = []
        self.decode_time = 0
        image_features = None
        inputs = self.preprocess(prompt, image_dir)
        self.perf_tracker.perf_start(PERFTYPE.VISION_TOTAL_TIME)
        if image_dir != None:
            self.perf_tracker.perf_start(PERFTYPE.VISION_PREPROCESS_TIME)
            visual_inputs = self.preprocess_visual(inputs)
            self.perf_tracker.perf_end(PERFTYPE.VISION_PREPROCESS_TIME)
            image_features = self.run_visual(visual_inputs)
        self.perf_tracker.perf_end(PERFTYPE.VISION_TOTAL_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)
        (
            inputs_embeds,
            time_position_ids,
            hight_position_ids,
            width_position_ids,
            self.rope_deltas,
        ) = self.preprocess_prefill(inputs, image_features)

        self.next_id, valid_length, current_length = self.run_prefill(
            inputs_embeds, time_position_ids, hight_position_ids, width_position_ids
        )
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)
        next_str = self.processor.tokenizer.decode(torch.tensor(self.next_id.item()))
        logger.success("response:")
        print("\033[1;95m{}".format(next_str), end="", flush=True)
        self.generated_ids.append(self.next_id.item())
        self.context_length = valid_length.item() + current_length.item() + 1
        return inputs_embeds.shape[1]

    def chat_decoder(self):
        if self.context_length >= self.context_max_length:
            logger.error(
                f"Context length long than {self.context_max_length}, stop run decode model!"
            )
            return None
        decoder_pids = self.context_length + self.rope_deltas.item() - 1
        self.perf_tracker.perf_start(PERFTYPE.DECODE_TOTAL_TIME)
        self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
        x = F.embedding(torch.from_numpy(self.next_id).unsqueeze(0), self.embedding)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)
        decoder_inputs = dict(
            input_1=x,
            valid_length=torch.tensor(self.context_length - 1),
            current_length=torch.tensor([1]).long(),
            time_position_ids=torch.tensor(decoder_pids),
            hight_position_ids=torch.tensor(decoder_pids),
            width_position_ids=torch.tensor(decoder_pids),
        )
        if HOUMO_TARGET == "xh2":
            decoder_inputs["input_1"] = decoder_inputs["input_1"].squeeze(0)
        self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
        self.decode.set_input(
            self.decode.get_input_name(0), decoder_inputs["input_1"].detach().numpy()
        )
        self.decode.set_input(
            self.decode.get_input_name(1),
            decoder_inputs["time_position_ids"].detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(2),
            decoder_inputs["hight_position_ids"].detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(3),
            decoder_inputs["width_position_ids"].detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(4),
            decoder_inputs["valid_length"].detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(5),
            decoder_inputs["current_length"].detach().numpy(),
        )
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_INFER_TIME)
        self.decode.run()
        self.decode.sync()
        self.perf_tracker.perf_end(PERFTYPE.DECODE_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.DECODE_OUTPUT_TIME)
        decoder_output = self.decode.get_output(self.decode.get_output_name(0))
        self.perf_tracker.perf_end(PERFTYPE.DECODE_OUTPUT_TIME)
        self.next_id = self.samplingmanager.sample(
            decoder_output.numpy(), self.generated_ids
        )
        self.generated_ids.append(self.next_id.item())
        if self.next_id.item() in self.eos_token_id:
            return None
        self.perf_tracker.perf_start(PERFTYPE.DECODE_TOKEN_TIME)
        next_str = self.processor.tokenizer.decode(self.next_id.item())
        self.perf_tracker.perf_end(PERFTYPE.DECODE_TOKEN_TIME)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)
        self.context_length += 1
        return next_str


if __name__ == "__main__":
    args = get_args()
    qwen25vl = Qwen25VL(
        args.vit_path,
        args.prefill_path,
        args.decode_path,
        args.tokenizer_dir,
        args.embedding_path,
    )
    # image_dir = None
    image_dir = ["../../../data/pic/beach.jpeg"]

    image_num = 0 if not image_dir else len(image_dir)

    prompt = "请描述图片内容。"
    logger.success("question:")
    print("\033[1;95m{}\033[0m".format(prompt))
    input_tokens = qwen25vl.chat_vit_prefill(image_dir, prompt=prompt)

    decode_count = 0
    while True:
        next_str = qwen25vl.chat_decoder()
        decode_count += 1
        if next_str is None:
            break
        print(next_str, end="", flush=True)
    print("\033[0m")

    qwen25vl.perf_tracker.set_basic_info(
        batch_size=1,
        input_seq_length=input_tokens,
        output_seq_length=decode_count,
        num_images=image_num,
    )

    qwen25vl.perf_tracker.show_summary()
