# Copyright (c) 2026 HOUMO AI
#
# File: minicpm_v45_process.py
# Description:
#   CPU preprocessing, stage input construction, and text postprocessing.
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

from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TORCHAO_FORCE_SKIP_LOADING_SO_FILES", "1")
logging.getLogger("torchao").setLevel(logging.ERROR)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedTokenizerFast

from houmo_engine import ModelProcess
from houmo_engine.core.types import StageInputs
from houmo_engine.perf import PerfTracker

from minicpm_v45_types import (
    MiniCPMV45Request,
    PrefillRequest,
    PreparedMiniCPMV45Request,
)

PATCH_SIZE = 14
TOKEN_CAPACITY = 1600
POSITIONS_PER_SIDE = 70
VIDEO_MAX_NUM_FRAMES = 180
VIDEO_MAX_PACKING = 6
VIDEO_TIME_SCALE = 0.1


def _uniform_sample(length: int, count: int) -> list[int]:
    gap = length / count
    return [int(index * gap + gap / 2) for index in range(count)]


def _encode_video(path: str, choose_fps: float, group_capacity: int) -> tuple[list[Image.Image], list[list[int]]]:
    import cv2
    import numpy as np
    from scipy.spatial import cKDTree

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"failed to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or length <= 0:
        raise ValueError(f"video has no readable frames: {path}")
    duration = length / fps
    max_packing = min(int(group_capacity), VIDEO_MAX_PACKING)
    if max_packing <= 0:
        raise ValueError(f"visual model batch must be positive, got {group_capacity}")
    if choose_fps * int(duration) <= VIDEO_MAX_NUM_FRAMES:
        packing = 1
        frame_count = round(min(choose_fps, round(fps)) * min(VIDEO_MAX_NUM_FRAMES, duration))
    else:
        packing = math.ceil(duration * choose_fps / VIDEO_MAX_NUM_FRAMES)
        if packing <= max_packing:
            frame_count = round(duration * choose_fps)
        else:
            frame_count = round(VIDEO_MAX_NUM_FRAMES * max_packing)
            packing = max_packing
    frame_count = max(1, frame_count)

    frame_indices = _uniform_sample(length, frame_count)
    capture = cv2.VideoCapture(path)
    frames = []
    for index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"failed to read frame {index} from video: {path}")
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB"))
    capture.release()

    frame_times = np.asarray(frame_indices, dtype=np.float32) / fps
    time_scale = np.arange(0, duration, VIDEO_TIME_SCALE)
    temporal_ids = cKDTree(time_scale[:, None]).query(frame_times[:, None])[1].astype(np.int32)
    groups = [temporal_ids[i : i + packing].tolist() for i in range(0, len(temporal_ids), packing)]
    return frames, groups


def _temporal_pos_embed_cache(embed_dim: int, size: int) -> torch.Tensor:
    positions = np.arange(size, dtype=np.float32)
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / 10000.0**omega
    values = np.einsum("m,d->md", positions, omega)
    return torch.from_numpy(np.concatenate([np.sin(values), np.cos(values)], axis=-1)).float()


def _temporal_group_inputs(
    pixel_slices: list[tuple[np.ndarray, np.ndarray]],
    temporal_ids: list[int],
    patch_capacity: int,
    group_capacity: int,
    embedding_dim: int,
) -> tuple[tuple[np.ndarray, ...], int]:
    if not 1 <= len(pixel_slices) <= group_capacity:
        raise ValueError(f"video group must contain 1..{group_capacity} frames")
    pixels = np.zeros((group_capacity, 3, PATCH_SIZE, patch_capacity * PATCH_SIZE), dtype=np.float16)
    position_ids = np.zeros((group_capacity, patch_capacity), dtype=np.int32)
    attention_bias = np.full(
        (group_capacity, 1, 1, patch_capacity), np.finfo(np.float16).min, dtype=np.float16
    )
    pos_embed = np.zeros((group_capacity, patch_capacity, embedding_dim), dtype=np.float16)
    temporal_embed = np.zeros_like(pos_embed)
    key_bias = np.full(
        (1, 1, 1, group_capacity * patch_capacity), np.finfo(np.float16).min, dtype=np.float16
    )
    temporal_cache = _temporal_pos_embed_cache(embedding_dim, max(max(temporal_ids, default=0) + 1, 1))
    for frame_index, ((packed, size), temporal_id) in enumerate(zip(pixel_slices, temporal_ids, strict=True)):
        height, width = map(int, size)
        count = height * width
        if count > patch_capacity:
            raise ValueError(f"video frame patch count {count} exceeds capacity {patch_capacity}")
        pixels[frame_index, ..., : packed.shape[-1]] = packed[0].astype(np.float16)
        rows = np.floor(np.arange(height) * POSITIONS_PER_SIDE / height).astype(np.int32)
        cols = np.floor(np.arange(width) * POSITIONS_PER_SIDE / width).astype(np.int32)
        position_ids[frame_index, :count] = (rows[:, None] * POSITIONS_PER_SIDE + cols[None, :]).reshape(-1)
        attention_bias[frame_index, 0, 0, :count] = 0
        pos_embed[frame_index, :count] = _sincos_position_embedding(
            height, width, embedding_dim
        ).reshape(count, embedding_dim)
        if temporal_id >= 0:
            temporal_embed[frame_index] = temporal_cache[int(temporal_id)].numpy()
        start = frame_index * patch_capacity
        key_bias[0, 0, 0, start : start + count] = 0
    return (pixels, position_ids, attention_bias, pos_embed, temporal_embed, key_bias), 64


def _load_tokenizer(path: Path) -> PreTrainedTokenizerFast:
    with (path / "tokenizer_config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    for key in ("auto_map", "tokenizer_class", "added_tokens_decoder"):
        config.pop(key, None)
    return PreTrainedTokenizerFast(tokenizer_file=str(path / "tokenizer.json"), **config)


def _load_embedding(path: Path, embedding_dim: int) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value["weight"]
    elif hasattr(value, "weight"):
        value = value.weight
    return value.reshape(-1, embedding_dim).float()


def _sincos_position_embedding(height: int, width: int, dim: int) -> np.ndarray:
    if dim % 4:
        raise ValueError("embedding_dim must be divisible by four")
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    half = dim // 4
    omega = 1.0 / (10000 ** (np.arange(half, dtype=np.float32) / half))
    yy = np.concatenate((np.sin(y[..., None] * omega), np.cos(y[..., None] * omega)), -1)
    xx = np.concatenate((np.sin(x[..., None] * omega), np.cos(x[..., None] * omega)), -1)
    return np.concatenate((np.broadcast_to(xx, (height, width, half * 2)),
                           np.broadcast_to(yy, (height, width, half * 2))), -1)


class MiniCPMV45ImagePreprocessor:
    def __init__(self):
        self.token_capacity = TOKEN_CAPACITY

    def _position_ids(self, height: int, width: int) -> np.ndarray:
        rows = np.floor(np.arange(height) * POSITIONS_PER_SIDE / height).astype(np.int32)
        cols = np.floor(np.arange(width) * POSITIONS_PER_SIDE / width).astype(np.int32)
        ids = rows[:, None] * POSITIONS_PER_SIDE + cols[None, :]
        result = np.zeros((1, self.token_capacity), dtype=np.int32)
        result[0, : height * width] = ids.reshape(-1)
        return result

    def build_unit_inputs(self, pixels: np.ndarray, target_size: Sequence[int]) -> tuple[dict[str, np.ndarray], int]:
        height, width = map(int, target_size)
        count = height * width
        if count > self.token_capacity:
            raise ValueError(f"visual unit [{height}, {width}] exceeds HMM capacity")
        expected = (1, 3, PATCH_SIZE, count * PATCH_SIZE)
        if tuple(pixels.shape) != expected:
            raise ValueError(f"unexpected packed pixel shape {pixels.shape}, expected {expected}")
        padded = np.zeros((1, 3, PATCH_SIZE, TOKEN_CAPACITY * PATCH_SIZE), dtype=np.float16)
        padded[..., : pixels.shape[-1]] = pixels.astype(np.float16)
        bias = np.full((1, 1, 1, TOKEN_CAPACITY), np.finfo(np.float16).min, dtype=np.float16)
        bias[..., :count] = 0
        pos = np.zeros((1, TOKEN_CAPACITY, 4096), dtype=np.float16)
        pos[0, :count] = _sincos_position_embedding(height, width, 4096).reshape(count, 4096)
        return {"pixel_values": padded, "position_ids": self._position_ids(height, width),
                "attention_bias": bias, "resampler_pos_embed": pos,
                "resampler_key_bias": bias.copy()}, 64

    def split_units(self, pixels: np.ndarray, sizes: np.ndarray):
        result, offset = [], 0
        for size in sizes:
            height, width = map(int, size)
            length = height * width * PATCH_SIZE
            result.append((pixels[..., offset:offset + length], size))
            offset += length
        if offset != pixels.shape[-1]:
            raise ValueError("packed pixels and target sizes do not match")
        return result


class MiniCPMV45Process(ModelProcess):
    def __init__(self, tokenizer_dir: Path, embedding_path: Path, *, prefill_length: int,
                 embedding_dim: int, max_slice_nums: int, video_group_capacity: int,
                 perf: PerfTracker):
        self.perf = perf
        self.prefill_length = prefill_length
        self.embedding_dim = embedding_dim
        self.max_slice_nums = max_slice_nums
        self.video_group_capacity = video_group_capacity
        processor_root = Path(os.getenv("HOUMO_EXAMPLES_PATH") or "/hmdd/imodelzoo") / "models" / "omni" / "minicpmo"
        sys.path.insert(0, str(processor_root))
        from image_processing_minicpmv import MiniCPMVImageProcessor
        with self.perf.scope("llm.init.processor_load"):
            self.tokenizer = _load_tokenizer(Path(tokenizer_dir))
        self.image_processor = MiniCPMVImageProcessor(max_slice_nums=max_slice_nums, scale_resolution=448, patch_size=PATCH_SIZE)
        with self.perf.scope("llm.init.embedding_load"):
            self.embedding_weight = _load_embedding(Path(embedding_path), embedding_dim)
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids("<unk>"))
        eos = self.tokenizer.eos_token_id
        self.eos_token_ids = {int(x) for x in (eos if isinstance(eos, list) else [eos])}

    def preprocess(self, request: MiniCPMV45Request) -> PreparedMiniCPMV45Request:
        image_content = ""
        vision_units = []
        image_count = 0
        with self.perf.scope("llm.vision.preprocess" if request.images or request.videos else "llm.preprocess"):
            static_images = [Image.open(path).convert("RGB") for path in request.images]
            if static_images:
                image_inputs = self.image_processor(static_images, max_slice_nums=self.max_slice_nums, return_tensors="pt")
                static_pixels = torch.cat(image_inputs["pixel_values"][0], dim=-1).unsqueeze(0)
                static_sizes = image_inputs["tgt_sizes"][0]
                static_units = MiniCPMV45ImagePreprocessor().split_units(
                    static_pixels.detach().cpu().numpy(), static_sizes.detach().cpu().numpy()
                )
                for index, (unit_pixels, size) in enumerate(static_units):
                    values, effective = MiniCPMV45ImagePreprocessor().build_unit_inputs(unit_pixels, size)
                    vision_units.append(StageInputs(
                        tuple(values[name] for name in ("pixel_values", "position_ids", "attention_bias", "resampler_pos_embed", "resampler_key_bias")),
                        {"effective_tokens": effective, "profile": "vision_1x"},
                    ))
                image_content += "\n".join(
                    self.image_processor.get_slice_image_placeholder(image.size, index, self.max_slice_nums)
                    for index, image in enumerate(static_images)
                ) + "\n"
                image_count += len(static_units)
            if request.videos:
                video_frames = []
                video_groups = []
                for path in request.videos:
                    frames, groups = _encode_video(path, request.video_fps, self.video_group_capacity)
                    video_frames.extend(frames)
                    video_groups.extend(groups)
                video_inputs = self.image_processor(video_frames, max_slice_nums=1, return_tensors="pt")
                video_pixels = torch.cat(video_inputs["pixel_values"][0], dim=-1).unsqueeze(0)
                video_sizes = video_inputs["tgt_sizes"][0]
                video_units = MiniCPMV45ImagePreprocessor().split_units(video_pixels.detach().cpu().numpy(), video_sizes.detach().cpu().numpy())
                offset = 0
                for group in video_groups:
                    group_units = video_units[offset : offset + len(group)]
                    offset += len(group)
                    values, effective = _temporal_group_inputs(group_units, group, 1600, self.video_group_capacity, self.embedding_dim)
                    vision_units.append(StageInputs(values, {"effective_tokens": effective, "profile": "vision_6x"}))
                    image_content += self.image_processor.get_slice_image_placeholder(video_frames[offset - len(group)].size, 0, 1, use_image_id=False) + "\n"
                image_count += len(video_groups)
            messages = ([{"role": "system", "content": request.system_prompt}] if request.system_prompt else [])
            messages.append({"role": "user", "content": f"{image_content}\n{request.prompt}" if image_content else request.prompt})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"]
            token_embeds = F.embedding(input_ids, self.embedding_weight)
        return PreparedMiniCPMV45Request(
            input_ids, token_embeds, None, None,
            int(input_ids.shape[1]),
            image_count,
            None,
            vision_units,
        )

    def merge_vision(self, prepared: PreparedMiniCPMV45Request, features: torch.Tensor) -> torch.Tensor:
        count = int((prepared.input_ids == self.image_token_id).sum().item())
        if features.shape[0] != count:
            raise ValueError(f"image token/features mismatch: {count} vs {features.shape[0]}")
        mask = (prepared.input_ids == self.image_token_id).unsqueeze(-1).expand_as(prepared.token_embeds)
        return prepared.token_embeds.masked_scatter(mask, features.to(prepared.token_embeds.dtype))

    def prepare_vision(self, prepared: PreparedMiniCPMV45Request):
        if prepared.vision_units is not None:
            yield from prepared.vision_units
            return
        pixels = prepared.pixel_values.detach().cpu().numpy()
        sizes = prepared.target_sizes.detach().cpu().numpy()
        units = MiniCPMV45ImagePreprocessor().split_units(pixels, sizes)
        if prepared.temporal_groups is None:
            for unit_pixels, size in units:
                values, effective = MiniCPMV45ImagePreprocessor().build_unit_inputs(unit_pixels, size)
                yield StageInputs(
                    tuple(values[name] for name in (
                        "pixel_values", "position_ids", "attention_bias",
                        "resampler_pos_embed", "resampler_key_bias",
                    )),
                    {"effective_tokens": effective},
                )
            return

        offset = 0
        for group in prepared.temporal_groups:
            group_units = units[offset : offset + len(group)]
            offset += len(group)
            values, effective = _temporal_group_inputs(
                group_units, group, 1600, self.video_group_capacity, self.embedding_dim
            )
            yield StageInputs(
                values,
                {"effective_tokens": effective},
            )
        if offset != len(units):
            raise ValueError(f"video groups consumed {offset} frames, got {len(units)}")

    def prepare_prefill(self, request: PrefillRequest, start: int) -> StageInputs:
        current = min(self.prefill_length, request.token_embeds.shape[1] - start)
        chunk = request.token_embeds[:, start:start + current]
        padded = torch.zeros((1, self.prefill_length, self.embedding_dim), dtype=chunk.dtype)
        padded[:, :current] = chunk
        return StageInputs((padded, np.array([start], np.int32), np.array([current], np.int32)), {"current_length": current})

    def prepare_decode(self, token: int, position: int) -> StageInputs:
        embed = F.embedding(torch.tensor([[token]], dtype=torch.long), self.embedding_weight)
        return StageInputs((embed, np.array([position], np.int32), np.array([1], np.int32)), {})

    def postprocess(self, state, *, final: bool = False) -> str:
        with self.perf.scope("llm.text.postprocess"):
            text = self.tokenizer.decode(state.generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if not final and text.endswith("\\"):
                text = text[:-1]
            text = text.replace("\\n", "\n")
        delta = text[len(state.emitted_text):] if text.startswith(state.emitted_text) else ""
        state.emitted_text = text
        return delta
