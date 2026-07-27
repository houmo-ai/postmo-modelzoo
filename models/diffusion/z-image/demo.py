#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Z-Image-Turbo Inference Demo - Python script for running
#   precompiled hmm models on HOUMO AI device.
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
from dataclasses import dataclass, field
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from loguru import logger
import torch
import torch.nn as nn
from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from prettytable import PrettyTable
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import tcim_lite as tcim

from hmatc.python.get_hm_devices import get_hm_devices
from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_PROMPT = "一只长得像蝴蝶一样缤纷绚丽的奇异花朵，开在丛林中，散发着柔和的光芒"

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


def load_json_file(path: Union[str, Path]) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


@dataclass
class StageMetric:
    elapsed: float = 0.0
    count: int = 0

    def add(self, elapsed: float, count: int = 1):
        self.elapsed += elapsed
        self.count += count

    def set(self, elapsed: float, count: Optional[int] = None):
        self.elapsed = elapsed
        if count is not None:
            self.count = count


@dataclass
class ZImagePerf:
    metrics: Dict[str, StageMetric] = field(default_factory=dict)
    _active: Dict[str, float] = field(default_factory=dict)
    batch_size: int = 0
    total_images: int = 0
    num_inference_steps: int = 0

    DEFAULT_STAGES = (
        "text_encode",
        "prepare_latents",
        "denoise",
        "dit",
        "dit_preprocess",
        "dit_runtime",
        "dit_postprocess",
        "scheduler",
        "vae_decode",
        "postprocess",
        "total",
    )

    def __post_init__(self):
        for name in self.DEFAULT_STAGES:
            self.metrics.setdefault(name, StageMetric())

    def metric(self, name: str) -> StageMetric:
        return self.metrics.setdefault(name, StageMetric())

    def add(self, name: str, elapsed: float, count: int = 1):
        self.metric(name).add(elapsed, count)

    def set(self, name: str, elapsed: float, count: Optional[int] = None):
        self.metric(name).set(elapsed, count)

    def start(self, name: str):
        if name in self._active:
            raise RuntimeError(f"Generation perf event already started: {name}")
        self._active[name] = time.perf_counter()

    def stop(self, name: str, count: int = 1) -> float:
        start_time = self._active.pop(name, None)
        if start_time is None:
            raise RuntimeError(f"Generation perf event was not started: {name}")
        elapsed = time.perf_counter() - start_time
        self.add(name, elapsed, count=count)
        return elapsed

    def snapshot(self):
        if self._active:
            raise RuntimeError(f"Unclosed generation perf events: {list(self._active)}")
        return (
            {name: metric.elapsed for name, metric in self.metrics.items()},
            {name: metric.count for name, metric in self.metrics.items()},
        )

    def elapsed(self, name: str) -> float:
        return self.metric(name).elapsed

    def count(self, name: str) -> int:
        return self.metric(name).count

    def set_generation_shape(
        self,
        *,
        batch_size: int,
        total_images: int,
        num_inference_steps: int,
    ):
        self.batch_size = batch_size
        self.total_images = total_images
        self.num_inference_steps = num_inference_steps

    def get_generation_rows(self):
        total_time = self.elapsed("total")
        total_images = self.total_images
        num_inference_steps = self.num_inference_steps
        dit_count = self.count("dit")
        return [
            {
                "name": "generation_end_to_end",
                "time": total_time,
                "count": total_images,
                "total": total_time,
                "notes": "generate_image total",
            },
            {
                "name": "  text_encode",
                "time": self.elapsed("text_encode"),
                "count": self.batch_size,
                "total": total_time,
                "notes": "prompt encoder",
            },
            {
                "name": "  prepare_latents",
                "time": self.elapsed("prepare_latents"),
                "count": total_images,
                "total": total_time,
                "notes": "latent init",
            },
            {
                "name": "  denoise_total",
                "time": self.elapsed("denoise"),
                "count": num_inference_steps,
                "total": total_time,
                "notes": "all denoising steps",
            },
            {
                "name": "    dit_total",
                "time": self.elapsed("dit"),
                "count": dit_count,
                "total": self.elapsed("denoise"),
                "notes": "share of denoising",
            },
            {
                "name": "      dit_preprocess",
                "time": self.elapsed("dit_preprocess"),
                "count": self.count("dit_preprocess"),
                "total": self.elapsed("dit"),
                "notes": "t_embedder/patchify/masks",
            },
            {
                "name": "      dit_runtime",
                "time": self.elapsed("dit_runtime"),
                "count": self.count("dit_runtime"),
                "total": self.elapsed("dit"),
                "notes": "HMM runtime",
            },
            {
                "name": "      dit_postprocess",
                "time": self.elapsed("dit_postprocess"),
                "count": self.count("dit_postprocess"),
                "total": self.elapsed("dit"),
                "notes": "crop/unpatchify",
            },
            {
                "name": "    scheduler_total",
                "time": self.elapsed("scheduler"),
                "count": self.count("scheduler"),
                "total": self.elapsed("denoise"),
                "notes": "share of denoising",
            },
            {
                "name": "  vae_decode",
                "time": self.elapsed("vae_decode"),
                "count": self.count("vae_decode"),
                "total": total_time,
                "notes": "latent to image tensor",
            },
            {
                "name": "  postprocess",
                "time": self.elapsed("postprocess"),
                "count": 1,
                "total": total_time,
                "notes": "image processor",
            },
        ]

    def log_generation_notes(self, width: int, height: int):
        total_images = self.total_images
        num_inference_steps = self.num_inference_steps
        total_time = self.elapsed("total")
        logger.info(
            f"Generation config: resolution={width}x{height}, "
            f"prompts={self.batch_size}, "
            f"images={total_images}, steps={num_inference_steps}"
        )
        if num_inference_steps > 0:
            logger.info(
                f"Generation average: denoise_per_step="
                f"{self.elapsed('denoise') / num_inference_steps:.3f} s, "
                f"dit_per_step={self.elapsed('dit') / num_inference_steps:.3f} s"
            )
        if total_images > 0 and total_time > 0:
            logger.info(
                f"Generation throughput: latency_per_image="
                f"{total_time / total_images:.3f} s, "
                f"throughput={total_images / total_time:.3f} images/s"
            )

    @staticmethod
    def get_init_rows(profile: Dict[str, Union[float, "ZImagePerf"]]):
        init_total = profile["init_total"]
        runtime_total = profile["load_runtime_models"]
        return [
            {
                "name": "init_total",
                "time": init_total,
                "total": init_total,
                "notes": "model setup end-to-end",
            },
            {
                "name": "  load_demo_dependencies",
                "time": profile["load_demo_dependencies"],
                "total": init_total,
                "notes": "tokenizer/scheduler/config",
            },
            {
                "name": "  load_runtime_models",
                "time": runtime_total,
                "total": init_total,
                "notes": "all HMM runtimes",
            },
            {
                "name": "    load_encoder_runtime",
                "time": profile["load_encoder_runtime"],
                "total": runtime_total,
                "notes": "share of runtime loading",
            },
            {
                "name": "    load_dit_runtime",
                "time": profile["load_dit_runtime"],
                "total": runtime_total,
                "notes": "share of runtime loading",
            },
            {
                "name": "    load_vae_runtime",
                "time": profile["load_vae_runtime"],
                "total": runtime_total,
                "notes": "share of runtime loading",
            },
        ]

    @staticmethod
    def log_table(title: str, rows: List[Dict[str, Union[str, float]]]):
        # 将初始化和生成阶段的 profile 数据统一格式化，便于定位瓶颈是在
        # host 侧预处理、HMM runtime，还是 scheduler/VAE 后处理。
        table = PrettyTable()
        table.field_names = [
            "Stage",
            "Count",
            "Time(s)",
            "Avg(s)",
            "Percent",
            "Notes",
        ]
        table.align["Stage"] = "l"
        table.align["Count"] = "r"
        table.align["Time(s)"] = "r"
        table.align["Avg(s)"] = "r"
        table.align["Percent"] = "r"
        table.align["Notes"] = "l"

        for row in rows:
            name = str(row["name"])
            elapsed = float(row["time"])
            count = int(row.get("count", 1))
            average = elapsed / count if count > 0 else 0.0
            total = float(row.get("total", elapsed))
            percent = elapsed / total * 100 if total > 0 else 0.0
            notes = str(row.get("notes", ""))
            table.add_row(
                [
                    name,
                    count,
                    f"{elapsed:.3f}",
                    f"{average:.3f}",
                    f"{percent:.1f}%",
                    notes,
                ]
            )

        logger.info(f"{title}\n{table}")

    @classmethod
    def log_summary(
        cls,
        profile: Dict[str, Union[float, "ZImagePerf"]],
        width: int,
        height: int,
    ):
        logger.info("=" * 80)
        rows = cls.get_init_rows(profile)
        generation_perf = profile.get("generation")
        if isinstance(generation_perf, cls):
            rows.extend(generation_perf.get_generation_rows())
        cls.log_table("Performance summary:", rows)
        if isinstance(generation_perf, cls):
            generation_perf.log_generation_notes(width, height)


class TcimRuntimeModel:
    # tcim_lite runtime 的轻量封装
    def __init__(self, model_path: Union[str, Path], option):
        self.model_path = str(model_path)
        self.model = tcim.runtime.load(self.model_path, option=option)
        self.input_names = [
            self.model.get_input_name(i) for i in range(self.model.get_num_inputs())
        ]
        self.output_names = [
            self.model.get_output_name(i) for i in range(self.model.get_num_outputs())
        ]

    def get_input_shape(self, index: int):
        return self.model.get_input_info(self.input_names[index]).shape

    def get_input_shape_by_name(self, input_name: str):
        for name in self.input_names:
            if name == input_name or name.split(".")[0] == input_name:
                return self.model.get_input_info(name).shape
        raise KeyError(f"Input {input_name} not found in {self.input_names}")

    def get_input_name(self, index: int):
        return self.input_names[index]

    def reset_cache_inputs(self, start_index: int = 3):
        # Text encoder HMM 将 KV cache 暴露为普通输入。编码新 prompt 前清零
        # cache，避免不同 prompt 之间串状态。
        for index in range(start_index, len(self.input_names)):
            self.model.set_input(
                self.input_names[index],
                np.zeros(self.get_input_shape(index), dtype=np.float16),
            )

    def __call__(self, *inputs):
        for name, value in zip(self.input_names, inputs):
            self.model.set_input(name, to_numpy(value))

        self.model.run()
        self.model.sync()

        outputs = []
        for name in self.output_names:
            tensor = torch.from_numpy(self.model.get_dev_output(name).to_host().numpy())
            outputs.append(tensor)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class ZImageTextEncoderRunner:
    """
    ZImage text encoder inference wrapper backed by tcim_lite runtime.
    """

    def __init__(
        self,
        prefill_model: TcimRuntimeModel,
        token_embedding_file: str,
        input_sequence_length: int = 256,
        tokenizer=None,
    ):
        self.prefill_model = prefill_model
        self.tokenizer = tokenizer
        # token embedding 保留在 host 侧用 PyTorch 执行；编译后的 text encoder
        # HMM 接收的是 embeddings，而不是 token ids。
        token_embedding_state_dict = torch.load(
            token_embedding_file,
            map_location="cpu",
            weights_only=True,
        )
        self.token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        ).to(torch.float16)
        self.token_embedding.load_state_dict(token_embedding_state_dict)

        self.input_sequence_length = input_sequence_length
        self.pad_token_id = self.tokenizer.eos_token_id

    def reset_cache(self):
        self.prefill_model.reset_cache_inputs(start_index=3)

    def get_input_sequence_length(self):
        return self.input_sequence_length

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
    ) -> torch.FloatTensor:
        return self.prefill_model(inputs_embeds, past_seq_length, current_input_length)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TimestepEmbedder(nn.Module):
    # 在 host 侧重建 timestep embedding 子模块。编译后的 DiT HMM 直接接收
    # AdaLN timestep embedding，因此每次调用 DiT 前先用 PyTorch 计算这一小段。
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period, dtype=torch.float32, device=t.device))
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        return self.mlp(t_freq)


class ZImageTransformerHelper:
    """
    DiT HMM 的 host 侧辅助类。

    编译后的 DiT 只负责核心 transformer 计算，demo 需要在 host 侧补齐三类工作：
    1. 重建 timestep embedder，生成 DiT 需要的 AdaLN 条件输入；
    2. 将 latent 图像切成静态长度的 patch token 序列，并生成对应位置/mask 信息；
    3. 将 DiT 输出的 patch token 还原成 VAE/scheduler 使用的 latent 布局。
    """

    def __init__(
        self,
        t_embedder_state_dict: dict,
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        patch_size: int = 2,
        f_patch_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ):
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.patch_size = patch_size
        self.f_patch_size = f_patch_size
        self.dtype = dtype

        # 从量化导出的权重中恢复 timestep embedder 的结构尺寸，保证 host 侧
        # 生成的 AdaLN embedding 与原始 transformer 以及编译后 DiT HMM 对齐。
        out_size = t_embedder_state_dict["mlp.2.weight"].shape[0]
        mid_size = t_embedder_state_dict["mlp.0.weight"].shape[0]
        frequency_embedding_size = t_embedder_state_dict["mlp.0.weight"].shape[1]
        self.t_embedder = TimestepEmbedder(
            min(out_size, ADALN_EMBED_DIM),
            mid_size=mid_size,
            frequency_embedding_size=frequency_embedding_size,
        ).to(dtype)
        self.t_embedder.load_state_dict(t_embedder_state_dict)

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        # 生成 3D token 坐标网格，坐标格式为 (frame, height, width)。text token
        # 和 image token 会使用不同起点，避免拼接后的位置 id 冲突。
        if start is None:
            start = (0 for _ in size)
        axes = [
            torch.arange(x0, x0 + span, dtype=torch.int32, device=device)
            for x0, span in zip(start, size)
        ]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def _patchify_image(self, image: torch.Tensor):
        # 将 latent 的 C/F/H/W 布局切成 patch token 序列，匹配编译后 DiT 的输入。
        p_h = p_w = self.patch_size
        p_f = self.f_patch_size
        channels, frames, height, width = image.size()
        f_tokens = frames // p_f
        h_tokens = height // p_h
        w_tokens = width // p_w
        image = image.view(channels, f_tokens, p_f, h_tokens, p_h, w_tokens, p_w)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
            f_tokens * h_tokens * w_tokens,
            p_f * p_h * p_w * channels,
        )
        return image, (frames, height, width), (f_tokens, h_tokens, w_tokens)

    def _pad_with_ids(
        self,
        feat: torch.Tensor,
        pos_grid_size: tuple,
        pos_start: tuple,
        device: torch.device,
    ):
        # DiT 的序列长度在编译时固定为 SEQ_MULTI_OF 的倍数。features、position ids
        # 和 mask 需要一起 padding，保持 HMM 运行时要求的静态 shape。
        ori_len = len(feat)
        pad_len = (-ori_len) % SEQ_MULTI_OF
        ori_pos_ids = self.create_coordinate_grid(
            size=pos_grid_size,
            start=pos_start,
            device=device,
        ).flatten(0, 2)
        if pad_len > 0:
            pad_pos_ids = (
                self.create_coordinate_grid((1, 1, 1), (0, 0, 0), device)
                .flatten(0, 2)
                .repeat(pad_len, 1)
            )
            pos_ids = torch.cat([ori_pos_ids, pad_pos_ids], dim=0)
            padded_feat = torch.cat([feat, feat[-1:].repeat(pad_len, 1)], dim=0)
            pad_mask = torch.cat(
                [
                    torch.zeros(ori_len, dtype=torch.bool, device=device),
                    torch.ones(pad_len, dtype=torch.bool, device=device),
                ]
            )
        else:
            pos_ids = ori_pos_ids
            padded_feat = feat
            pad_mask = torch.zeros(ori_len, dtype=torch.bool, device=device)

        return padded_feat, pos_ids, pad_mask, ori_len + pad_len

    def patchify_and_embed(self, all_image, all_cap_feats):
        # 将一个 batch 的 prompt hidden states 和 latent 图像都整理成 DiT 输入序列。
        # prompt token 先占用序列前段，image token 的位置起点接在 prompt 后面。
        device = all_image[0].device
        all_img_out, all_img_size, all_img_pos_ids, all_img_pad_mask = [], [], [], []
        all_cap_out, all_cap_pos_ids, all_cap_pad_mask = [], [], []

        for image, cap_feat in zip(all_image, all_cap_feats):
            # prompt hidden states 先 padding 到编译要求的序列粒度，并生成对应 mask。
            cap_out, cap_pos_ids, cap_pad_mask, cap_len = self._pad_with_ids(
                cap_feat,
                (len(cap_feat) + (-len(cap_feat)) % SEQ_MULTI_OF, 1, 1),
                (1, 0, 0),
                device,
            )
            all_cap_out.append(cap_out)
            all_cap_pos_ids.append(cap_pos_ids)
            all_cap_pad_mask.append(cap_pad_mask)

            # latent 先切成 patch token，再接在 prompt token 之后编码位置。
            img_patches, size, (f_tokens, h_tokens, w_tokens) = self._patchify_image(
                image
            )
            img_out, img_pos_ids, img_pad_mask, _ = self._pad_with_ids(
                img_patches,
                (f_tokens, h_tokens, w_tokens),
                (cap_len + 1, 0, 0),
                device,
            )
            all_img_out.append(img_out)
            all_img_size.append(size)
            all_img_pos_ids.append(img_pos_ids)
            all_img_pad_mask.append(img_pad_mask)

        return (
            all_img_out,
            all_cap_out,
            all_img_size,
            all_img_pos_ids,
            all_cap_pos_ids,
            all_img_pad_mask,
            all_cap_pad_mask,
        )

    def unpatchify(self, x, size):
        # 将 DiT 输出的 patch token 还原成 latent C/F/H/W；reshape 前丢弃 padding token。
        p_h = p_w = self.patch_size
        p_f = self.f_patch_size
        for index, item in enumerate(x):
            frames, height, width = size[index]
            ori_len = (frames // p_f) * (height // p_h) * (width // p_w)
            x[index] = (
                item[:ori_len]
                .view(
                    frames // p_f,
                    height // p_h,
                    width // p_w,
                    p_f,
                    p_h,
                    p_w,
                    self.out_channels,
                )
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, frames, height, width)
            )
        return x


class HmZImage:
    """
    Z-Image-Turbo demo 的主控类。

    该类把一次文生图流程拆成几个明确阶段：
    1. 加载 tokenizer/scheduler/transformer 配置等 host 侧依赖；
    2. 加载 text encoder、DiT、VAE 三个已编译 HMM 子模型；
    3. 编码 prompt，初始化 latent，循环调用 DiT 去噪；
    4. 通过 VAE 解码 latent，并记录各阶段耗时。
    """

    def __init__(self, args: argparse.Namespace):
        init_start = time.perf_counter()
        self.args = args
        self.embedding_path = args.embedding_path
        # profile 用于贯穿初始化和生成阶段的耗时统计，最后由
        # log_performance_summary 统一打印。
        self.profile = {}
        self.perf: Optional[ZImagePerf] = None
        self._past_seq_length = 0
        self.deps_dir = args.deps_dir

        # 先加载不在 HMM 里的 demo 依赖：tokenizer、scheduler 配置、
        # timestep embedder 权重和 transformer 静态配置。
        dependency_start = time.perf_counter()
        self._load_demo_dependencies()
        self.profile["load_demo_dependencies"] = time.perf_counter() - dependency_start

        # 再加载三个设备侧 HMM runtime。后续推理只通过封装后的 runtime 调用。
        runtime_start = time.perf_counter()
        text_encoder_runtime, self.dit, self.vae = self._load_runtime_models()
        self.profile["load_runtime_models"] = time.perf_counter() - runtime_start
        # 运行时 shape 是图像分辨率和 token 长度的单一依据。这样 demo 参数会和
        # 实际编译出的 HMM 文件保持一致，不需要用户手动传 height/width。
        self.height, self.width = self._infer_image_size_from_vae()
        self.image_token_len, self.text_token_len = (
            self._infer_transformer_input_lengths()
        )
        vae_scale = self.vae_scale_factor * 2
        if self.height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {self.height})."
            )
        if self.width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {self.width})."
            )
        # text_encoder runner 将 host 侧 embedding lookup 和 HMM prefill runtime
        # 组合成一个可调用对象，供 prompt 编码阶段使用。
        self.text_encoder = ZImageTextEncoderRunner(
            prefill_model=text_encoder_runtime,
            token_embedding_file=str(self.embedding_path),
            input_sequence_length=self.text_token_len,
            tokenizer=self.tokenizer,
        )

        self.profile["init_total"] = time.perf_counter() - init_start
        logger.info(f"Demo initialized with model {args.model_name}-{args.model_size}.")

    def _load_demo_dependencies(self):
        # 这些依赖来自 hmquant/hf_config，不是设备 HMM 本体，但决定了输入预处理、
        # scheduler 时间步和 latent/image shape 的解释方式。
        logger.info(f"Model: {self.args.model_name}-{self.args.model_size}")
        logger.info(f"Loading demo dependencies from {self.deps_dir}...")
        logger.info(f"Inference steps: {self.args.num_inference_steps}")
        logger.info(f"Number of images per prompt: {self.args.num_images_per_prompt}")

        deps_dir = Path(self.deps_dir)
        tokenizer_dir = deps_dir / "tokenizer"
        scheduler_config_path = deps_dir / "scheduler" / "scheduler_config.json"
        t_embedder_path = deps_dir / "t_embedder.pt"
        transformer_config_path = deps_dir / "transformer_config.json"

        logger.info(f"Tokenizer directory: {tokenizer_dir}")
        logger.info(f"Scheduler config path: {scheduler_config_path}")
        logger.info(f"T embedder path: {t_embedder_path}")
        logger.info(f"Transformer config path: {transformer_config_path}")

        # 这些文件由量化流程导出，提供驱动三个 HMM 子模型所需的 host 侧小组件。
        transformer_config = load_json_file(transformer_config_path)
        scheduler_config = load_json_file(scheduler_config_path)
        t_embedder_state_dict = torch.load(
            t_embedder_path,
            map_location="cpu",
            weights_only=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        self.vae_scale_factor = transformer_config.get("vae_scale_factor", 8)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.transformer = ZImageTransformerHelper(
            t_embedder_state_dict=t_embedder_state_dict,
            in_channels=transformer_config.get("in_channels", 16),
            out_channels=transformer_config.get(
                "out_channels",
                transformer_config.get("in_channels", 16),
            ),
            patch_size=transformer_config.get("patch_size", 2),
            f_patch_size=transformer_config.get("f_patch_size", 1),
            dtype=torch.float16,
        )
        self.transformer.t_embedder = self.transformer.t_embedder.to(
            dtype=torch.float16
        )

    def _load_runtime_models(self):
        # 加载三个已编译子模型到后摩设备 runtime：
        # text_encoder 负责 prompt embedding，DiT 负责去噪，VAE 负责 latent 解码成图。
        dev_manager = tcim.runtime.DevManager(
            get_hm_devices(self.args.ndevice), "Xh2HalBackend"
        )
        wm_encoder = tcim.runtime.WeightManager(dev_manager)
        wm_dit = tcim.runtime.WeightManager(dev_manager)
        wm_vae = tcim.runtime.WeightManager(dev_manager)

        text_encoder_path = self.args.encoder_path
        dit_path = self.args.dit_path
        vae_path = self.args.vae_path

        logger.info(f"Loading encoder runtime from {text_encoder_path}...")
        start = time.perf_counter()
        text_encoder_model = TcimRuntimeModel(
            text_encoder_path, tcim.runtime.Option(wm_encoder)
        )
        self.profile["load_encoder_runtime"] = time.perf_counter() - start
        logger.info(f"Loading dit runtime from {dit_path}...")
        start = time.perf_counter()
        dit_model = TcimRuntimeModel(dit_path, tcim.runtime.Option(wm_dit))
        self.profile["load_dit_runtime"] = time.perf_counter() - start
        logger.info(f"Loading vae runtime from {vae_path}...")
        start = time.perf_counter()
        vae_model = TcimRuntimeModel(vae_path, tcim.runtime.Option(wm_vae))
        self.profile["load_vae_runtime"] = time.perf_counter() - start

        return text_encoder_model, dit_model, vae_model

    def log_performance_summary(self):
        # 生成阶段是可选的：如果只初始化模型但未调用 generate_image，
        # 这里仍然可以打印初始化耗时。
        ZImagePerf.log_summary(self.profile, self.width, self.height)

    def _perf_start(self, key: str):
        if self.perf is not None:
            self.perf.start(key)

    def _perf_stop(self, key: str, count: int = 1):
        if self.perf is not None:
            self.perf.stop(key, count=count)

    def _finish_generation_perf(self, total_start: float, total_images: int):
        if self.perf is None:
            return
        self.perf.set("total", time.perf_counter() - total_start, count=total_images)
        self.profile["generation"] = self.perf

    def _infer_image_size_from_vae(self):
        # VAE HMM 输入是 latent N/C/H/W。最终图像尺寸由 latent H/W 乘以
        # transformer config 中的 VAE scale factor 得到。
        vae_input_shape = self.vae.get_input_shape(0)
        if len(vae_input_shape) != 4:
            raise ValueError(f"Unexpected VAE input shape: {vae_input_shape}")

        _, latent_channels, latent_height, latent_width = vae_input_shape
        if latent_channels != self.transformer.in_channels:
            raise ValueError(
                f"VAE latent channels {latent_channels} do not match "
                f"transformer input channels {self.transformer.in_channels}."
            )
        if latent_height <= 0 or latent_width <= 0:
            raise ValueError(f"VAE input shape must be static: {vae_input_shape}")

        height = latent_height * self.vae_scale_factor
        width = latent_width * self.vae_scale_factor
        logger.info(
            f"Inferred image size from VAE input shape {vae_input_shape}: "
            f"{height}x{width}"
        )
        return height, width

    def _infer_transformer_input_lengths(self):
        # DiT HMM 使用静态序列长度编译。这里从输入 shape 读取长度，确保 prompt
        # 截断/padding 和 latent patching 都与 HMM 对齐。
        latent_mask_shape = self.dit.get_input_shape_by_name("latent_mask")
        cap_feats_shape = self.dit.get_input_shape_by_name("cap_feats")
        if len(latent_mask_shape) != 2:
            raise ValueError(f"Unexpected DIT latent_mask shape: {latent_mask_shape}")
        if len(cap_feats_shape) != 2:
            raise ValueError(f"Unexpected DIT cap_feats shape: {cap_feats_shape}")

        image_token_len = int(latent_mask_shape[1])
        text_token_len = int(cap_feats_shape[0])
        if image_token_len <= 0 or text_token_len <= 0:
            raise ValueError(
                "DIT input shapes must be static, got "
                f"latent_mask={latent_mask_shape}, cap_feats={cap_feats_shape}"
            )

        logger.info(
            "Inferred DIT input lengths from HMM: "
            f"image_token_len={image_token_len}, text_token_len={text_token_len}"
        )
        return image_token_len, text_token_len

    def _prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # 去噪从高斯 latent 噪声开始。latent H/W 由推导出的输出图像尺寸和
        # VAE scale factor 共同决定。
        height = 2 * (int(self.height) // (self.vae_scale_factor * 2))
        width = 2 * (int(self.width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = torch.randn(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)
        return latents

    def _text_encoder_infer_one(self, input_ids, output_sequence_length=None):
        # 编译后的 text encoder 只处理固定 prefill 长度。长 prompt 会被切成多段，
        # 同一个 prompt 内部通过 HMM KV cache 串起上下文。
        self._past_seq_length = 0
        self.text_encoder.reset_cache()
        input_sequence_length = self.text_encoder.get_input_sequence_length()
        seq_length = input_ids.shape[-1]
        output_sequence_length = (
            seq_length if output_sequence_length is None else output_sequence_length
        )

        output_hidden = None
        prefill_loop_round = math.ceil(seq_length / input_sequence_length)

        for idx in range(prefill_loop_round):
            valid_length = idx * input_sequence_length + self._past_seq_length
            start = idx * input_sequence_length
            end = min((idx + 1) * input_sequence_length, seq_length)
            current_length = end - start
            current_input_ids = input_ids[:, start:end]

            if current_length < input_sequence_length:
                padding_input_ids = torch.full(
                    (1, input_sequence_length - current_length),
                    self.text_encoder.pad_token_id,
                    dtype=current_input_ids.dtype,
                    device=current_input_ids.device,
                )
                current_input_ids = torch.cat(
                    [current_input_ids, padding_input_ids], dim=-1
                )
            inputs_embeds = self.text_encoder.token_embedding(current_input_ids)

            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")

            # 每次向 text encoder HMM 输入一段 embeddings 和序列长度信息；
            # 最后一轮输出包含该 prompt 的 hidden states。
            output_hidden = self.text_encoder(
                inputs_embeds,
                valid_length_data,
                current_length_data,
            )

        if output_hidden is None:
            raise RuntimeError("Text encoder produced no output.")
        return output_hidden[0, :output_sequence_length, :]

    def _text_encoder_infer(self, all_input_ids, attention_mask=None):
        # Text encoder HMM 按单条 prompt 运行。这里遍历 batch，并用 attention_mask
        # 截掉 tokenizer padding 后的无效 hidden states。
        prompt_embeds = []
        for index in range(all_input_ids.shape[0]):
            output_sequence_length = None
            if attention_mask is not None:
                output_sequence_length = int(attention_mask[index].sum().item())
            prompt_embeds.append(
                self._text_encoder_infer_one(
                    all_input_ids[index : index + 1],
                    output_sequence_length=output_sequence_length,
                )
            )
        return prompt_embeds

    def _prepare_dit_condition_inputs(self, cap_feats, cap_pad_mask, valid_len):
        # 将 prompt features 和 masks 扩展到 DiT 编译时固定的 text token 长度。
        # padding 区域使用较大的负值 mask，使 attention 忽略这些位置。
        image_token_len = self.image_token_len
        text_token_len = self.text_token_len

        attn_mask = torch.zeros((1, image_token_len), dtype=torch.bool)
        pad_len = text_token_len - valid_len
        cap_mask = torch.zeros((1, valid_len))
        cap_feats = torch.concat(
            [
                cap_feats[0],
                torch.zeros(pad_len, cap_feats[0].shape[1]),
            ],
            dim=0,
        )
        cap_mask = torch.concat(
            [cap_mask, torch.ones(pad_len).unsqueeze(0) * -65504],
            dim=1,
        )

        cap_pad_mask = (
            torch.concat([cap_pad_mask[0], torch.ones(pad_len)]).half().unsqueeze(-1)
        )
        n_cap_pad_mask = 1 - cap_pad_mask

        return (
            attn_mask.half(),
            cap_feats.half(),
            cap_mask.half(),
            cap_pad_mask,
            n_cap_pad_mask,
        )

    def _transformer_infer(
        self,
        latent_model_input_list,
        timestep_model_input,
        prompt_embeds_model_input,
    ):
        # DiT HMM 的实际调用入口。输入是当前 timestep 的 latent batch 和 prompt
        # hidden states，输出是 scheduler.step 需要的噪声预测 latent。
        if len(latent_model_input_list) != len(prompt_embeds_model_input):
            raise ValueError(
                "latent batch size must match prompt embeds batch size, "
                f"got {len(latent_model_input_list)} latents and "
                f"{len(prompt_embeds_model_input)} prompt embeds."
            )
        timestep_model_input = timestep_model_input * 1000.0
        batch_outputs = []

        with torch.no_grad():
            for batch_index, (latent_input, prompt_embeds) in enumerate(
                zip(latent_model_input_list, prompt_embeds_model_input)
            ):
                self._perf_start("dit_preprocess")
                current_timestep = timestep_model_input[batch_index : batch_index + 1]
                # 准备 DiT HMM 之外的输入：timestep AdaLN embedding、latent patch
                # tokens 和 prompt 条件 mask。
                adaln_input = self.transformer.t_embedder(current_timestep).to(
                    torch.float16
                )

                (
                    x,
                    cap_feats,
                    x_size,
                    x_pos_ids,
                    cap_pos_ids,
                    x_pad_mask,
                    cap_pad_mask,
                ) = self.transformer.patchify_and_embed([latent_input], [prompt_embeds])
                del x_pos_ids
                del x_pad_mask
                del cap_pos_ids

                valid_len = int(cap_feats[0].shape[0])
                image_token_len = self.image_token_len
                text_token_len = self.text_token_len
                if x[0].shape[0] != image_token_len:
                    raise ValueError(
                        f"DIT model expects {image_token_len} image tokens, "
                        f"but current latent produced {x[0].shape[0]} tokens."
                    )
                if valid_len > text_token_len:
                    raise ValueError(
                        f"Prompt embedding length {valid_len} exceeds DIT text "
                        f"token length {text_token_len}."
                    )

                (
                    attn_mask,
                    cap_feats,
                    cap_mask,
                    cap_pad_mask,
                    n_cap_pad_mask,
                ) = self._prepare_dit_condition_inputs(
                    cap_feats,
                    cap_pad_mask,
                    valid_len,
                )
                self._perf_stop("dit_preprocess")

                self._perf_start("dit_runtime")
                # 一次 DiT HMM 调用会预测当前 scheduler timestep 下单张图的噪声残差。
                output = self.dit(
                    x[0],
                    attn_mask,
                    adaln_input,
                    cap_feats,
                    cap_mask,
                    cap_pad_mask,
                    n_cap_pad_mask,
                )
                self._perf_stop("dit_runtime")

                self._perf_start("dit_postprocess")
                # 裁掉多余 token，并将预测出的 patch token 还原成 scheduler 需要的 latent 布局。
                output = output[:, : image_token_len + valid_len]

                unpatchify_res = self.transformer.unpatchify(
                    list(output.unbind(dim=0)), x_size
                )

                batch_outputs.extend(unpatchify_res)
                self._perf_stop("dit_postprocess")

        return batch_outputs

    def _vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        # 编译后的 VAE HMM batch shape 固定，因此逐张 latent 解码后再拼接。
        decoded_images = []
        for batch_index in range(latents.shape[0]):
            decoded_images.append(self.vae(latents[batch_index : batch_index + 1]))
        return torch.cat(decoded_images, dim=0)

    def _encode_single_prompt(self, prompt: Union[str, List[str]], system_prompt: Optional[str] = None):
        # 将用户输入 prompt 规范化为 tokenizer 可处理的 chat 文本，再调用
        # text encoder HMM 得到每条 prompt 的 hidden states。
        if isinstance(prompt, str):
            prompt = [prompt]

        normalized_prompts = []
        for prompt_item in prompt:
            # tokenization 前套用原始 Z-Image-Turbo text encoder 使用的 Qwen 风格
            # chat template。
            effective_system_prompt = system_prompt
            messages = []
            if effective_system_prompt:
                messages.append({"role": "system", "content": effective_system_prompt})
            messages.append({"role": "user", "content": prompt_item})
            normalized_prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )

        text_inputs = self.tokenizer(
            normalized_prompts,
            padding="max_length",
            max_length=self.text_token_len,
            truncation=True,
            return_tensors="pt",
        )
        prompt_hidden_states = self._text_encoder_infer(
            all_input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        )
        return prompt_hidden_states

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        system_prompt: Optional[str] = None,
    ):
        # 允许调用方直接传入 prompt_embeds，便于复用文本编码结果或做外部调试。
        if prompt_embeds is None:
            prompt_embeds = self._encode_single_prompt(prompt, system_prompt=system_prompt)
        return prompt_embeds

    def generate_image(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int,
        seed: int = 42,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: int = 1,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: str = "pil",
        system_prompt: Optional[str] = None,
    ):
        # 对外的完整生成接口。默认返回 PIL 图像；output_type="latent" 时提前
        # 返回最终 latent，便于调试 DiT/scheduler 而跳过 VAE 解码。
        logger.info(f"Generating image with prompt: {prompt}")
        total_start = time.perf_counter()
        self.perf = ZImagePerf()

        generator = torch.Generator("cpu").manual_seed(seed)

        if prompt is not None:
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
        elif prompt_embeds is not None:
            batch_size = len(prompt_embeds)
        else:
            raise ValueError("Either prompt or prompt_embeds must be provided.")

        if prompt_embeds is None:
            # 1. Prompt -> token ids -> host embedding lookup -> text encoder HMM
            # hidden states。该结果会在所有去噪 step 中复用。
            self._perf_start("text_encode")
            prompt_embeds = self._encode_prompt(
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                system_prompt=system_prompt,
            )
            self._perf_stop("text_encode", count=batch_size)

        # 2. 创建初始 latent 噪声。每个 prompt 生成多张图时，先扩展 latent batch，
        # 再在下方复制 prompt embeddings。
        self._perf_start("prepare_latents")
        latents = self._prepare_latents(
            batch_size * num_images_per_prompt,
            self.transformer.in_channels,
            torch.float32,
            "cpu",
            generator,
            latents,
        )
        self._perf_stop(
            "prepare_latents",
            count=batch_size * num_images_per_prompt,
        )

        if num_images_per_prompt > 1:
            prompt_embeds = [
                pe for pe in prompt_embeds for _ in range(num_images_per_prompt)
            ]

        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", self.image_token_len),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0

        # 3. 构建 scheduler timesteps。Z-Image 的 shift 与分辨率相关，因此
        # calculate_shift 依赖 latent token 数量。
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            "cpu",
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order,
            0,
        )

        self._perf_start("denoise")
        with tqdm(total=num_inference_steps) as progress_bar:
            for index, timestep_value in enumerate(timesteps):
                # 4. 去噪循环：DiT 在 latent 空间预测噪声，scheduler 再更新 latents
                # 进入下一个 timestep。
                timestep = timestep_value.expand(latents.shape[0])
                timestep = (1000 - timestep) / 1000
                # t_norm = timestep[0].item()

                latent_model_input = latents.to(self.transformer.dtype)
                prompt_embeds_model_input = prompt_embeds
                timestep_model_input = timestep

                latent_model_input = latent_model_input.unsqueeze(2)

                self._perf_start("dit")
                model_out_list = self._transformer_infer(
                    list(latent_model_input.unbind(dim=0)),
                    timestep_model_input,
                    prompt_embeds_model_input,
                )
                self._perf_stop("dit", count=len(prompt_embeds_model_input))

                noise_pred = torch.stack(
                    [tensor.float() for tensor in model_out_list],
                    dim=0,
                )
                noise_pred = -noise_pred.squeeze(2)
                self._perf_start("scheduler")
                latents = self.scheduler.step(
                    noise_pred.to(torch.float32),
                    timestep_value,
                    latents,
                    return_dict=False,
                )[0]
                self._perf_stop("scheduler")

                if index == len(timesteps) - 1 or (
                    (index + 1) > num_warmup_steps
                    and (index + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._perf_stop("denoise", count=num_inference_steps)
        if self.perf is not None:
            self.perf.set_generation_shape(
                batch_size=batch_size,
                total_images=latents.shape[0],
                num_inference_steps=num_inference_steps,
            )

        if output_type == "latent":
            self._finish_generation_perf(total_start, latents.shape[0])
            return latents

        # 5. 将最终 latents 从模型缩放还原到 VAE 输入分布，然后解码并后处理成 PIL 图像。
        latents = (latents.to(torch.float16) / 0.3611) + 0.1159
        self._perf_start("vae_decode")
        image = self._vae_decode(latents)
        self._perf_stop("vae_decode", count=latents.shape[0])

        self._perf_start("postprocess")
        image = self.image_processor.postprocess(image, output_type=output_type)
        self._perf_stop("postprocess")
        self._finish_generation_perf(total_start, latents.shape[0])

        return image


def get_output_path(output: Union[str, Path], index: int, image_count: int) -> Path:
    output_path = Path(output)
    suffix = output_path.suffix or ".png"
    if not output_path.suffix:
        output_path = output_path.with_suffix(suffix)

    if image_count > 1:
        output_path = output_path.with_name(
            f"{output_path.stem}_{index}{output_path.suffix}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def get_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model_name", type=str, default=None, help="model name")
    parser.add_argument("--model_size", type=str, default=None, help="model size")
    parser.add_argument("--deps_dir", type=str, default=os.path.join("output", HOUMO_TARGET, "hmquant", "hf_config"), help="directory containing extracted demo dependencies")
    parser.add_argument("--embedding_path", type=str, default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"), help="token embedding weight path")
    parser.add_argument("--encoder_path", type=str, default=None, help="compiled text encoder hmm path")
    parser.add_argument("--dit_path", type=str, default=None, help="compiled dit hmm path")
    parser.add_argument("--vae_path", type=str, default=None, help="compiled vae hmm path")
    parser.add_argument("--ndevice", type=int, default=1, help="device number")
    parser.add_argument("--output", type=str, default="output_turbo_xh2a.png", help="output image path")
    parser.add_argument("--prompt", type=str, nargs="+", action="append", default=None, help="positive prompt. Pass one prompt, multiple quoted prompts, or repeat --prompt.")
    parser.add_argument("--system_prompt", type=str, default=None, help="system prompt to control prompt encoding")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="number of images to generate per prompt")
    parser.add_argument("--num_inference_steps", type=int, default=8, help="number of inference steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # fmt: on

    if args.prompt is None:
        args.prompt = DEFAULT_PROMPT
    else:
        prompts = [prompt for prompt_group in args.prompt for prompt in prompt_group]
        args.prompt = prompts[0] if len(prompts) == 1 else prompts

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    # CLI 参数优先级高于 config.yaml；未显式传入的模型名、规格和设备数使用默认配置。
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))

    # build.py 默认用该前缀输出三个 HMM 文件；测试自定义编译模型时仍可显式传路径。
    model_path_prefix = f"./output/{HOUMO_TARGET}/{args.model_name}-{args.model_size}"
    args.encoder_path = first_not_none(
        args.encoder_path, f"{model_path_prefix}_encoder.hmm"
    )
    args.dit_path = first_not_none(args.dit_path, f"{model_path_prefix}_dit.hmm")
    args.vae_path = first_not_none(args.vae_path, f"{model_path_prefix}_vae.hmm")

    if args.ndevice > 1:
        if args.encoder_path.endswith(".hmm"):
            args.encoder_path = args.encoder_path.replace(".hmm", ".hmms")
        if args.dit_path.endswith(".hmm"):
            args.dit_path = args.dit_path.replace(".hmm", ".hmms")
        if args.vae_path.endswith(".hmm"):
            args.vae_path = args.vae_path.replace(".hmm", ".hmms")

    return args


def main():
    args = get_args()
    # 主流程：
    #   加载依赖和 runtime -> 编码 prompt -> DiT 去噪 latent
    #   -> VAE 解码 -> 保存图片并打印性能统计。
    model = HmZImage(args)
    images = model.generate_image(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        system_prompt=args.system_prompt,
    )

    for i, image in enumerate(images):
        output_path = get_output_path(args.output, i, len(images))
        logger.info(
            f"Image {i} generated with shape: {image.size}, saving image to {output_path}..."
        )
        image.save(output_path)

    model.log_performance_summary()


if __name__ == "__main__":
    main()
