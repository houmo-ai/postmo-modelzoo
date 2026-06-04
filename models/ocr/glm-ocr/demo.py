#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   GLM-OCR Inference Demo - Python script for running GLM-OCR
# automatic optical character recognition on HOUMO AI device.
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
import math
import time
import argparse
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image, ImageOps
from loguru import logger
import itertools

import tcim_lite as tcim
from hmatc.python.get_hm_devices import get_hm_devices
from hmatc.utils.utils import first_not_none, get_model_configs

TARGET_TYPE = torch.float16
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def find_glmocr_sdk_root() -> Optional[str]:
    candidates = [
        os.path.join(os.path.dirname(__file__), "GLM-OCR"),
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "examples",
            "llm",
            "glm_ocr",
            "GLM-OCR",
        ),
        os.path.join(os.getcwd(), "examples", "llm", "glm_ocr", "GLM-OCR"),
    ]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isdir(candidate):
            return candidate
    return None


SDK_ROOT = find_glmocr_sdk_root()
if SDK_ROOT is not None and SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)


def resolve_path(path: Optional[str], base_dir: Optional[str] = None) -> Optional[str]:
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path

    candidates = [os.path.abspath(path)]
    if base_dir is not None:
        candidates.append(os.path.abspath(os.path.join(base_dir, path)))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    candidates.append(os.path.abspath(os.path.join(repo_root, path)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def get_default_tokenizer_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "glm-ocr")
    model_size = model_config.get("model_size", "0.9b")
    return f"{model_name}-{model_size}"


def parse_visual_size_from_path(path: str) -> Optional[Tuple[int, int, int]]:
    match = re.search(
        r"_visual_(\d+)x(\d+)x(\d+)\.(?:hmm|hmms)$", os.path.basename(path)
    )
    if match is None:
        return None
    return tuple(int(item) for item in match.groups())


def build_messages(image_path: str, prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_inputs(
    processor, messages: List[Dict[str, Any]], device: Optional[torch.device] = None
):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    if device is not None:
        inputs = inputs.to(device)
    return inputs


def get_rope_index(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    spatial_merge_size: int = 2,
    image_token_id: int = 151339,
    video_start_token_id: int = 151343,
    video_end_token_id: int = 151344,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ported from HF `GlmOcrModel.get_rope_index` with minor simplification.
    """
    if input_ids is None:
        raise ValueError("input_ids is required")

    mrope_position_deltas: List[torch.Tensor] = []

    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        video_group_index = 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, sample_input_ids in enumerate(total_input_ids):
            sample_input_ids = sample_input_ids[attention_mask[i] == 1]
            input_tokens = sample_input_ids.tolist()

            input_token_type = []
            video_check_flag = False
            for token in input_tokens:
                if token == video_start_token_id:
                    video_check_flag = True
                elif token == video_end_token_id:
                    video_check_flag = False

                if token == image_token_id and not video_check_flag:
                    input_token_type.append("image")
                elif token == image_token_id and video_check_flag:
                    input_token_type.append("video")
                else:
                    input_token_type.append("text")

            input_type_group = []
            for key, group in itertools.groupby(
                enumerate(input_token_type), lambda x: x[1]
            ):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            llm_pos_ids_list = []
            video_frame_num = 1

            for modality_type, start_idx, end_idx in input_type_group:
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )

                if modality_type == "image":
                    if image_grid_thw is None:
                        raise ValueError(
                            "image_grid_thw is required when image token exists"
                        )
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        int(t.item()),
                        int(h.item()) // spatial_merge_size,
                        int(w.item()) // spatial_merge_size,
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
                        torch.stack([t_index, h_index, w_index]) + st_idx
                    )

                    image_index += 1
                    video_frame_num = 1

                elif modality_type == "video":
                    if video_grid_thw is None:
                        raise ValueError(
                            "video_grid_thw is required when video token exists"
                        )
                    t, h, w = (
                        video_frame_num,
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        int(t),
                        int(h.item()) // spatial_merge_size,
                        int(w.item()) // spatial_merge_size,
                    )

                    for t_idx in range(llm_grid_t):
                        t_index = (
                            torch.tensor(t_idx)
                            .view(-1, 1)
                            .expand(-1, llm_grid_h * llm_grid_w)
                            .flatten()
                        )
                        h_index = (
                            torch.arange(llm_grid_h)
                            .view(1, -1, 1)
                            .expand(1, -1, llm_grid_w)
                            .flatten()
                        )
                        w_index = (
                            torch.arange(llm_grid_w)
                            .view(1, 1, -1)
                            .expand(1, llm_grid_h, -1)
                            .flatten()
                        )
                        llm_pos_ids_list.append(
                            torch.stack([t_index, h_index, w_index]) + st_idx
                        )

                    video_group_index += 1
                    if video_group_index >= video_grid_thw[video_index][0]:
                        video_index += 1
                        video_group_index = 0
                    video_frame_num += 1

                else:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    video_frame_num = 1

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )

        deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(
            1
        )
        return position_ids, deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[
            0
        ]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

    return position_ids, mrope_position_deltas


def scatter_image_embeds(
    input_ids: torch.Tensor,
    token_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    n_image_tokens = int((input_ids == image_token_id).sum().item())
    if n_image_tokens == 0:
        return token_embeds

    if image_embeds.dim() != 2:
        raise ValueError(
            f"image_embeds must be rank-2, got shape={tuple(image_embeds.shape)}"
        )
    if image_embeds.shape[0] != n_image_tokens:
        raise ValueError(
            f"Image tokens/features mismatch, tokens={n_image_tokens}, features={image_embeds.shape[0]}"
        )

    image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(token_embeds)
    image_embeds = image_embeds.to(token_embeds.device, token_embeds.dtype)
    return token_embeds.masked_scatter(image_mask, image_embeds)


def show_statictic_info(hmglm_ocr, input_tokens, output_tokens):
    logger.success(
        f"Output {input_tokens} tokens, Prefill Cost {hmglm_ocr.prefill_time*1000:.3f} ms"
    )
    logger.success(
        f"Prefill Speed: {(input_tokens) / hmglm_ocr.prefill_time:.2f} tokens/s"
    )
    logger.success(
        f"Output {output_tokens} tokens, Decode Cost {hmglm_ocr.decode_time*1000:.3f} ms"
    )
    logger.success(
        f"Decode Speed: {(output_tokens) / hmglm_ocr.decode_time:.2f} tokens/s"
    )
    logger.success(f"TTFT (Time to First Token): {hmglm_ocr.ttft_time * 1000:.3f} ms")
    logger.success(
        f"TPOT (Time Per Output Token): {hmglm_ocr.decode_time * 1000 / (output_tokens):.3f} ms/token"
    )
    logger.success(
        f"E2E Latency (End-to-End Latency): {(hmglm_ocr.ttft_time + hmglm_ocr.decode_time):.3f} seconds"
    )
    logger.success(
        f"E2E TPS (End-to-End Tokens Per Second): {output_tokens / (hmglm_ocr.ttft_time + hmglm_ocr.decode_time):.2f} tokens/s"
    )


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
        "--vit_path",
        dest="vit_path",
        type=str,
        nargs="+",
        default=None,
        help="houmo visual model path(s)",
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
        "--layout_path",
        dest="layout_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "output", HOUMO_TARGET, "ppdoclayoutv3.hmm"
        ),
        help="houmo PP-DocLayoutV3 model path",
    )
    parser.add_argument(
        "--layout_batch_size",
        dest="layout_batch_size",
        type=int,
        default=1,
        help="PP-DocLayoutV3 layout batch size",
    )
    parser.add_argument(
        "--layout_threshold",
        dest="layout_threshold",
        type=float,
        default=0.3,
        help="PP-DocLayoutV3 post-process threshold",
    )
    parser.add_argument(
        "--layout_use_polygon",
        dest="layout_use_polygon",
        action="store_true",
        help="use PP-DocLayoutV3 polygons for crop/visualization",
    )
    parser.add_argument(
        "--image",
        dest="image",
        type=str,
        default="../../../data/pic/ocr.jpeg",
        help="image input path",
    )
    parser.add_argument(
        "--pdf",
        dest="pdf",
        type=str,
        default=None,
        help="PDF input path; full pipeline renders pages before layout OCR",
    )
    parser.add_argument(
        "--pdf_dpi",
        dest="pdf_dpi",
        type=int,
        default=200,
        help="DPI for full-pipeline PDF rendering",
    )
    parser.add_argument(
        "--pdf_max_pages",
        dest="pdf_max_pages",
        type=int,
        default=None,
        help="maximum PDF pages for full pipeline",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt",
        type=str,
        default="Text Recognition:",
        help="OCR prompt used for each layout region",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "work_dirs",
            "glm_ocr_npu_full_pipeline_demo",
        ),
        help="full-pipeline output directory",
    )
    parser.add_argument(
        "--max_new_tokens",
        dest="max_new_tokens",
        type=int,
        default=256,
        help="maximum decode tokens for each OCR region",
    )
    parser.add_argument(
        "--no_full_pipeline",
        dest="full_pipeline",
        action="store_false",
        help="disable layout full pipeline and run the legacy single-image flow",
    )
    parser.add_argument(
        "--image_size",
        dest="image_size",
        type=int,
        nargs=2,
        default=None,
        help="size of the input image",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number, only xh2 support",
    )
    parser.set_defaults(full_pipeline=True)
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

    image_size_w = model_config.get("image_size_w", 672)
    image_size_h = model_config.get("image_size_h", 672)
    max_size_t = model_config.get("max_size_t", 2)
    args.max_size_t = max_size_t
    args.image_size = list(
        first_not_none(args.image_size, [image_size_w, image_size_h])
    )

    model_prefix = f"{args.model_name}-{args.model_size}"
    if args.vit_path is None:
        args.vit_path = [
            os.path.join(
                "output",
                HOUMO_TARGET,
                f"{model_prefix}_visual_{image_size_w}x{image_size_h}x{max_size_t}.hmm",
            )
        ]
    if args.prefill_path is None:
        args.prefill_path = os.path.join(
            "output", HOUMO_TARGET, f"{model_prefix}_prefill.hmm"
        )
    if args.decode_path is None:
        args.decode_path = os.path.join(
            "output", HOUMO_TARGET, f"{model_prefix}_decode.hmm"
        )
    if args.ndevice > 1:
        args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        args.decode_path = args.decode_path.replace(".hmm", ".hmms")

    return args


def should_fallback_to_legacy_ocr(args, base_dir: str) -> bool:
    if not args.full_pipeline:
        return True

    layout_path = resolve_path(args.layout_path, base_dir)
    if layout_path is not None and os.path.exists(layout_path):
        args.layout_path = layout_path
        return False

    message = (
        f"PP-DocLayoutV3 HMM not found: {layout_path}. "
        "Fallback to legacy single-image OCR."
    )
    if args.pdf is not None:
        raise FileNotFoundError(
            f"PP-DocLayoutV3 HMM not found: {layout_path}. "
            "Legacy OCR fallback only supports image input; please provide "
            "--image or a valid --layout_path for PDF input."
        )
    logger.warning(message)
    return True


class HmGLM_OCR:

    def __init__(self, args):
        self.ndevice = args.ndevice
        self.dev_manager = tcim.runtime.DevManager(
            get_hm_devices(self.ndevice), "Xh2HalBackend"
        )
        self.weight_manager = tcim.runtime.WeightManager(self.dev_manager)
        option2 = tcim.runtime.Option(self.weight_manager)
        option3 = tcim.runtime.Option(self.weight_manager)
        self.visual_specs = self._load_visual_specs(
            args.vit_path,
            args.image_size,
            args.max_size_t,
            self.weight_manager,
        )
        self.vit_model = self.visual_specs[0]["model"]
        logger.info(f"{len(self.visual_specs)} vit model(s) loaded")
        self.prefill = tcim.runtime.load(args.prefill_path, option=option2)
        logger.info("prefill model loaded")
        self.nblocks = self._get_nblocks()
        dummy_tensor_names = [
            f"model_layers_{i}_self_attn_kcache_input" for i in range(self.nblocks)
        ]
        dummy_tensor_names += [
            f"model_layers_{i}_self_attn_vcache_input" for i in range(self.nblocks)
        ]
        option2.set_dummy_tensors(dummy_tensor_names)
        self.decode = tcim.runtime.load(args.decode_path, option=option3)
        logger.info("decode model loaded")
        self.prefill_length = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        self.embedding_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]
        self.context_max_length = self.decode.get_input_info(
            self.decode.get_input_name(4)
        ).shape[2]
        self.batch = self.decode.get_input_info(self.decode.get_input_name(0)).shape[0]

        for i in range(4, 2 * self.nblocks + 4):
            cache = self.prefill.get_input(self.prefill.get_input_name(i))
            self.decode.set_input(self.decode.get_input_name(i), cache)
        # set decode input
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode.get_input_name(2)
        self.decode.set_input(decode_current_length_name, current_length_input_1)

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(args.tokenizer_dir)

        embedding_weight = torch.load(
            args.embedding_path, map_location="cpu", weights_only=False
        )
        self.token_embedding = embedding_weight.weight.reshape(
            -1, self.embedding_len
        ).float()
        self.stop_token_ids = [151329, 59246]
        self.pad_token_id = 59246
        self.image_token_id = 59280
        self.video_start_token_id = 151343
        self.video_end_token_id = 151344
        for token in ["<|user|>", "<|assistant|>", "<|observation|>", "<eop>"]:
            token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and int(token_id) >= 0:
                self.stop_token_ids.append(int(token_id))
        self.context_length = 0
        self.spatial_merge_size = 2
        self.image_size_w, self.image_size_h = args.image_size
        self.generated_ids = []
        self.vit_time = 0
        self.prefill_time = 0
        self.decode_time = 0

    def _load_visual_specs(
        self,
        vit_paths: Union[str, List[str]],
        image_size: List[int],
        max_size_t: int,
        weight_manager,
    ) -> List[Dict[str, Any]]:
        if isinstance(vit_paths, str):
            vit_paths = [vit_paths]

        visual_specs = []
        multi_visual = len(vit_paths) > 1
        for path in vit_paths:
            parsed_size = parse_visual_size_from_path(path)
            if parsed_size is None:
                if multi_visual:
                    raise ValueError(
                        f"Cannot parse visual size from '{path}'. "
                        "Multi visual mode requires filenames like '*_visual_336x336x2.hmm'."
                    )
                width, height = image_size
                temporal_size = max_size_t
            else:
                width, height, temporal_size = parsed_size
            if width <= 0 or height <= 0 or temporal_size <= 0:
                raise ValueError(
                    f"Invalid visual size for '{path}': {width}x{height}x{temporal_size}"
                )

            option = tcim.runtime.Option(weight_manager)
            model = tcim.runtime.load(path, option=option)
            visual_specs.append(
                {
                    "path": path,
                    "width": int(width),
                    "height": int(height),
                    "max_size_t": int(temporal_size),
                    "model": model,
                }
            )
            logger.info(
                f"vit model loaded: path={path}, size={width}x{height}x{temporal_size}"
            )
        return visual_specs

    def _select_visual_spec(self, orig_w: int, orig_h: int) -> Dict[str, Any]:
        candidates = [
            spec
            for spec in self.visual_specs
            if spec["width"] >= orig_w and spec["height"] >= orig_h
        ]
        if candidates:
            return min(
                candidates,
                key=lambda spec: (
                    spec["width"] * spec["height"],
                    max(spec["width"], spec["height"]),
                ),
            )
        return max(
            self.visual_specs,
            key=lambda spec: (
                spec["width"] * spec["height"],
                -max(spec["width"], spec["height"]),
            ),
        )

    def _get_nblocks(self):
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_start_token_id=self.video_start_token_id,
            video_end_token_id=self.video_end_token_id,
        )

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        input_ids = data["input_ids"]
        seq_length = input_ids.shape[1]

        assert self.token_embedding is not None, "Token embedding is not available."
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
        assert (
            seq_length <= self.context_max_length
        ), f"Input sequence too long: max={self.context_max_length}, got={seq_length}"

        attention_mask = data.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones((1, seq_length), dtype=torch.long)
        else:
            attention_mask = attention_mask

        # Pad to input_sequence_length
        if self.input_sequence_length > seq_length:
            pad_len = self.input_sequence_length - seq_length
            padding_ids = torch.zeros((1, pad_len), dtype=torch.long).fill_(
                self.pad_token_id
            )
            input_ids = torch.cat([input_ids, padding_ids], dim=-1)
            padding_mask = torch.zeros((1, pad_len), dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, padding_mask], dim=-1)

        inputs_embeds = F.embedding(input_ids, self.token_embedding).cpu()

        # Scatter image embeddings into token embeddings
        n_image_tokens = int(torch.sum(input_ids == self.image_token_id).item())
        if n_image_tokens > 0 and data.get("image_embeds", None) is not None:
            image_embeds = data["image_embeds"]
            inputs_embeds = scatter_image_embeds(
                input_ids=input_ids,
                token_embeds=inputs_embeds,
                image_embeds=image_embeds,
                image_token_id=self.image_token_id,
            )
        elif data.get("image_embeds", None) is not None:
            image_embeds = data["image_embeds"]
            expected_tokens = int(image_embeds.shape[0])
            unique_ids, counts = torch.unique(input_ids, return_counts=True)
            matched = unique_ids[counts == expected_tokens]
            if matched.numel() == 1:
                detected_image_token_id = int(matched[0].item())
                self.image_token_id = detected_image_token_id
                inputs_embeds = scatter_image_embeds(
                    input_ids=input_ids,
                    token_embeds=inputs_embeds,
                    image_embeds=image_embeds,
                    image_token_id=self.image_token_id,
                )

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."

        if past_seq_length == 0:
            # Prefill: compute full rope index
            image_grid_thw = data.get("image_grid_thw", None)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw
            position_ids, rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
            # Fix rope_deltas for right-padding: get_rope_index computes
            # delta = max_position + 1 - total_len, where total_len includes
            # padding.  But the decode loop passes the *original* (unpadded)
            # seq_length as past_seq_length, so the delta must be relative to
            # the original length. Add back the padding offset.
            pad_len = self.input_sequence_length - seq_length
            if pad_len > 0:
                rope_deltas = rope_deltas + pad_len
            self.rope_deltas = rope_deltas
        else:
            # Decode: use cached rope_deltas
            assert (
                self.rope_deltas is not None
            ), f"rope_deltas is None but past_seq_length={past_seq_length}"
            batch_size, embed_seq_len, _ = inputs_embeds.shape
            delta = past_seq_length + self.rope_deltas
            position_ids = torch.arange(embed_seq_len)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        return (
            inputs_embeds,
            position_ids.to(dtype=torch.float16),
            torch.tensor([past_seq_length], dtype=torch.int32),
            torch.tensor([seq_length], dtype=torch.int32),
        )

    def _run_visual(self, inputs: torch.Tensor, vit_model) -> torch.Tensor:
        vit_model.set_input(
            vit_model.get_input_name(0),
            inputs.numpy().astype(np.float16),
        )
        start_time = time.time()
        vit_model.run()
        vit_model.sync()
        self.vit_time += time.time() - start_time
        vit_model_output = vit_model.get_output(
            vit_model.get_output_name(0)
        ).numpy()

        return torch.tensor(vit_model_output)

    def _run_prefill(self, data):
        input_ids = data["input_ids"]
        input_seq_len = input_ids.shape[-1]
        steps = (input_seq_len + self.prefill_length - 1) // self.prefill_length
        self.input_sequence_length = self.prefill_length * steps
        (
            inputs_embeds,
            position_ids,
            _,
            _,
        ) = self.prepare_inputs(data)
        valid_length_data = 0
        inputs_embeds_list = torch.split(
            inputs_embeds, split_size_or_sections=256, dim=1
        )
        position_ids_list = torch.split(position_ids, split_size_or_sections=256, dim=2)
        current_length = input_ids.shape[1]
        if current_length >= self.context_max_length:
            logger.error(
                f"Question long than {self.context_max_length}, please shorten it!"
            )
            sys.exit(1)

        for i in range(steps):
            current_length_data = min(
                self.prefill_length, input_seq_len - i * self.prefill_length
            )
            self.prefill.set_input(
                self.prefill.get_input_name(0),
                inputs_embeds_list[i].detach().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(1),
                position_ids_list[i].detach().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(2),
                torch.tensor([valid_length_data], dtype=torch.int32).detach().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(3),
                torch.tensor([current_length_data], dtype=torch.int32).detach().numpy(),
            )
            start_time = time.time()
            self.prefill.run()
            self.prefill.sync()
            self.prefill_time += time.time() - start_time

            prefill_output = self.prefill.get_output(self.prefill.get_output_name(0))
            valid_length_data += self.prefill_length
        next_id = prefill_output.numpy().argmax(-1)

        return next_id, valid_length_data - self.prefill_length, current_length_data

    def chat_vit_prefill(
        self,
        image_path: str,
        prompt: str,
        stream: bool = True,
    ) -> str:
        self.ttft_time = 0
        self.generated_ids = []
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        visual_spec = self._select_visual_spec(orig_w, orig_h)
        target_w = int(visual_spec["width"])
        target_h = int(visual_spec["height"])
        logger.info(
            f"selected vit model: path={visual_spec['path']}, "
            f"size={target_w}x{target_h}x{visual_spec['max_size_t']}, "
            f"image={orig_w}x{orig_h}"
        )
        if (orig_w, orig_h) != (target_w, target_h):
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            image = ImageOps.expand(
                image, border=(0, 0, pad_w, pad_h), fill=(114, 114, 114)
            )

        messages = build_messages(image, prompt)
        inputs = build_inputs(self.processor, messages)

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        image_features = self._run_visual(pixel_values, visual_spec["model"])
        data_prefill = {
            "input_ids": input_ids,
            "image_embeds": image_features,
            "past_seq_length": 0,
            "image_grid_thw": image_grid_thw,
        }
        self.next_id, valid_length, current_length = self._run_prefill(data_prefill)
        next_token_id = int(np.asarray(self.next_id).reshape(-1)[0])
        next_str = (
            ""
            if next_token_id in self.stop_token_ids
            else self.processor.tokenizer.decode(torch.tensor(next_token_id))
        )
        self.ttft_time += time.time() - start_time
        if stream:
            logger.success("response:")
            print("\033[1;95m{}".format(next_str), end="", flush=True)
        self.context_length = valid_length + current_length + 1
        return input_ids.shape[1]

    def chat_decoder(self):
        if self.context_length >= self.context_max_length:
            logger.error(
                f"Context length long than {self.context_max_length}, stop run decode model!"
            )
            return None
        self.input_sequence_length = 1
        data_decode = {
            "input_ids": torch.tensor(self.next_id),
            "past_seq_length": self.context_length - 1,
        }
        (
            inputs_embeds,
            position_ids,
            past_seq_length,
            _,
        ) = self.prepare_inputs(data_decode)
        self.decode.set_input(
            self.decode.get_input_name(0), inputs_embeds.detach().numpy()
        )
        self.decode.set_input(
            self.decode.get_input_name(1),
            position_ids.detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(2),
            past_seq_length.detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(3),
            torch.tensor([self.input_sequence_length], dtype=torch.int32)
            .detach()
            .numpy(),
        )
        start_time = time.time()
        self.decode.run()
        self.decode.sync()
        self.decode_time += time.time() - start_time
        decoder_output = self.decode.get_output(self.decode.get_output_name(0))
        self.next_id = np.argmax(decoder_output.numpy(), axis=-1)
        self.generated_ids.append(self.next_id.item())
        if self.next_id.item() in self.stop_token_ids:
            return None
        next_str = self.processor.tokenizer.decode(self.next_id.item())
        self.context_length += 1
        return next_str

    def generate(
        self,
        image_path: str,
        prompt: str = "Text Recognition:",
        max_new_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        input_tokens = self.chat_vit_prefill(image_path, prompt=prompt, stream=stream)
        del input_tokens

        text_parts = []
        first_token_id = int(np.asarray(self.next_id).reshape(-1)[0])
        if first_token_id in self.stop_token_ids:
            return ""
        text_parts.append(self.processor.tokenizer.decode(first_token_id))

        decode_count = 0
        while max_new_tokens is None or decode_count < max(0, max_new_tokens - 1):
            next_str = self.chat_decoder()
            decode_count += 1
            if next_str is None:
                break
            text_parts.append(next_str)
            if stream:
                print(next_str, end="", flush=True)

        return "".join(text_parts)


if __name__ == "__main__":
    args = get_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    legacy_ocr = should_fallback_to_legacy_ocr(args, base_dir)
    hmglm_ocr = HmGLM_OCR(args)
    if not legacy_ocr:
        from glm_ocr_npu_full_pipeline import run_npu_full_pipeline

        run_npu_full_pipeline(
            args,
            hmglm_ocr,
            logger,
            base_dir=base_dir,
        )
        sys.exit(0)

    image_dir = resolve_path(args.image, base_dir)
    prompt = args.prompt
    logger.success("question:")
    print("\033[1;95m{}\033[0m".format(prompt))
    input_tokens = hmglm_ocr.chat_vit_prefill(image_dir, prompt=prompt, stream=True)

    decode_count = 0
    while decode_count < max(0, args.max_new_tokens - 1):
        next_str = hmglm_ocr.chat_decoder()
        decode_count += 1
        if next_str is None:
            break
        print(next_str, end="", flush=True)
    print("\033[0m")
    show_statictic_info(hmglm_ocr, input_tokens, decode_count)
