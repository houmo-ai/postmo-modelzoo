# Copyright (c) 2026 HOUMO AI
#
# File: process.py
# Description:
#   MiniCPM-V 4.6 input and output Process implementation.
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

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("TORCHAO_FORCE_SKIP_LOADING_SO_FILES", "1")
logging.getLogger("torchao").setLevel(logging.ERROR)

import transformers.utils as transformers_utils
import transformers.utils.import_utils as transformers_import_utils


def _torchao_unavailable(*args, **kwargs) -> bool:
    del args, kwargs
    return False


transformers_import_utils.is_torchao_available = _torchao_unavailable
transformers_utils.is_torchao_available = _torchao_unavailable

from transformers import AutoProcessor

from ...core import ModelProcess
from ...core.types import GenerationState, StageInputs, StageOutputs
from ...perf import PerfTracker


IMAGE_TOKEN_ID = 248056
PATCH_SIZE = 14
TOKEN_CAPACITY = 1536
POSITIONS_PER_SIDE = 70


@dataclass
class PreparedRequest:
    input_ids: torch.Tensor
    token_embeds: torch.Tensor
    position_ids: torch.Tensor
    vision_inputs: tuple[StageInputs, ...] = ()

    @property
    def uses_vision(self) -> bool:
        return bool(self.vision_inputs)


def _load_embedding(path, embedding_size: int) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value["weight"]
    else:
        value = value.weight
    return value.reshape(-1, embedding_size).float()


def _window_index_4x(height: int, width: int) -> np.ndarray:
    if height % 2 or width % 2:
        raise ValueError(
            f"4x vision requires an even patch grid, got [{height}, {width}]"
        )
    index = np.arange(height * width, dtype=np.int32).reshape(height, width)
    return (
        index.reshape(height // 2, 2, width // 2, 2)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )


def _window_index_16x(height: int, width: int) -> np.ndarray:
    if height % 4 or width % 4:
        raise ValueError(
            f"16x vision requires a patch grid divisible by 4, got [{height}, {width}]"
        )
    index = np.arange(height * width, dtype=np.int32).reshape(height, width)
    index = (
        index.reshape(height // 2, 2, width // 2, 2)
        .transpose(0, 2, 1, 3)
        .reshape(height // 2, width // 2, 4)
    )
    return (
        index.reshape(height // 4, 2, width // 4, 2, 4)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1)
    )


class MiniCPMV46ImagePreprocessor:
    """Convert native packed patches to fixed-shape visual HMM inputs."""

    def __init__(
        self,
        downsample_mode: str,
        token_capacity: int = TOKEN_CAPACITY,
        patch_size: int = PATCH_SIZE,
        positions_per_side: int = POSITIONS_PER_SIDE,
    ):
        if downsample_mode not in {"4x", "16x"}:
            raise ValueError("downsample_mode must be '4x' or '16x'")
        self.downsample_mode = downsample_mode
        self.token_capacity = token_capacity
        self.patch_size = patch_size
        self.positions_per_side = positions_per_side

    def _window_index(self, height: int, width: int) -> np.ndarray:
        if self.downsample_mode == "16x":
            return _window_index_16x(height, width)
        return _window_index_4x(height, width)

    def _position_ids(
        self, height: int, width: int, window_index: np.ndarray
    ) -> np.ndarray:
        rows = np.floor(
            np.arange(height) * self.positions_per_side / height
        ).astype(np.int32)
        cols = np.floor(
            np.arange(width) * self.positions_per_side / width
        ).astype(np.int32)
        base = rows[:, None] * self.positions_per_side + cols[None, :]
        output = np.zeros((1, self.token_capacity), dtype=np.int32)
        output[0, : height * width] = base.reshape(-1)[window_index]
        return output

    def build_unit_inputs(
        self, pixel_values: np.ndarray, target_size: Sequence[int]
    ) -> StageInputs:
        height, width = map(int, target_size)
        patch_count = height * width
        if patch_count > self.token_capacity:
            raise ValueError(
                f"visual unit [{height}, {width}] has {patch_count} patches, "
                f"exceeding HMM capacity {self.token_capacity}"
            )
        expected_shape = (1, 3, self.patch_size, patch_count * self.patch_size)
        if tuple(pixel_values.shape) != expected_shape:
            raise ValueError(
                f"unexpected packed pixel shape {pixel_values.shape}, "
                f"expected {expected_shape}"
            )

        window_index = self._window_index(height, width)
        patches = pixel_values.reshape(
            1, 3, self.patch_size, patch_count, self.patch_size
        )
        reordered = patches[:, :, :, window_index, :].reshape(
            1, 3, self.patch_size, -1
        )
        padded_pixels = np.zeros(
            (1, 3, self.patch_size, self.token_capacity * self.patch_size),
            dtype=np.float16,
        )
        padded_pixels[..., : reordered.shape[-1]] = reordered.astype(np.float16)

        neg_inf = np.finfo(np.float16).min
        pre_bias = np.full(
            (1, 1, 1, self.token_capacity), neg_inf, dtype=np.float16
        )
        pre_bias[..., :patch_count] = 0
        tensors = [
            padded_pixels,
            self._position_ids(height, width, window_index),
            pre_bias,
        ]
        divisor = 4
        if self.downsample_mode == "16x":
            post_bias = np.full(
                (1, 1, 1, self.token_capacity // 4),
                neg_inf,
                dtype=np.float16,
            )
            post_bias[..., : patch_count // 4] = 0
            tensors.append(post_bias)
            divisor = 16
        return StageInputs(
            tensors=tuple(tensors),
            metadata={"effective_tokens": patch_count // divisor},
        )

    def split_units(
        self, packed_pixel_values: np.ndarray, target_sizes: np.ndarray
    ) -> tuple[StageInputs, ...]:
        units = []
        offset = 0
        for target_size in target_sizes:
            height, width = map(int, target_size)
            packed_width = height * width * self.patch_size
            pixels = packed_pixel_values[..., offset : offset + packed_width]
            units.append(self.build_unit_inputs(pixels, target_size))
            offset += packed_width
        if offset != packed_pixel_values.shape[-1]:
            raise ValueError(
                f"packed pixels consumed {offset}, actual width is "
                f"{packed_pixel_values.shape[-1]}"
            )
        return tuple(units)


class MiniCPMV46Process(ModelProcess):
    """MiniCPM-V 4.6 preprocessing and streamed token postprocessing."""

    def __init__(
        self,
        tokenizer_path,
        embedding_path,
        embedding_size: int,
        *,
        downsample_mode: str = "16x",
        max_slice_nums: int = 36,
        perf: PerfTracker,
    ):
        self.downsample_mode = downsample_mode
        self.max_slice_nums = max_slice_nums
        self.perf = perf
        self.image_preprocessor = MiniCPMV46ImagePreprocessor(downsample_mode)
        with self.perf.scope("llm.init.processor_load"):
            self.processor = AutoProcessor.from_pretrained(str(tokenizer_path))
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = int(
            getattr(self.processor, "image_token_id", IMAGE_TOKEN_ID)
        )
        with self.perf.scope("llm.init.embedding_load"):
            self.embedding_weight = _load_embedding(embedding_path, embedding_size)

    @staticmethod
    def normalize_images(images) -> list[str]:
        if images is None:
            return []
        if isinstance(images, (str, Path)):
            images = [images]
        return [str(image) for image in images]

    @staticmethod
    def attention_mask(length: int, valid_length: int) -> np.ndarray:
        mask = np.zeros((1, length), dtype=np.float16)
        mask[0, :valid_length] = 1
        return mask

    def preprocess(
        self,
        prompt: str,
        images: Sequence[str],
        system_prompt: str | None,
    ) -> PreparedRequest:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if images:
            content = [{"type": "image", "url": path} for path in images]
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        if images:
            self.perf.start("llm.vision.preprocess")
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "downsample_mode": self.downsample_mode,
                    "max_slice_nums": self.max_slice_nums,
                },
            )
        finally:
            if images:
                self.perf.end("llm.vision.preprocess")

        input_ids = inputs["input_ids"]
        token_embeds = F.embedding(input_ids, self.embedding_weight)
        vision_inputs = ()
        if images:
            packed_pixels = inputs["pixel_values"].detach().cpu().numpy()
            target_sizes = inputs["target_sizes"].detach().cpu().numpy()
            vision_inputs = self.image_preprocessor.split_units(
                packed_pixels, target_sizes
            )
        position_ids = torch.arange(
            int(input_ids.shape[1]), dtype=torch.int32
        ).reshape(1, -1)
        return PreparedRequest(
            input_ids=input_ids,
            token_embeds=token_embeds,
            position_ids=position_ids,
            vision_inputs=vision_inputs,
        )

    def merge_vision(
        self, request: PreparedRequest, outputs: Sequence[StageOutputs]
    ) -> None:
        image_embeds = torch.cat(
            [
                torch.from_numpy(output.tensors[0])[
                    :, : output.metadata["effective_tokens"], :
                ]
                for output in outputs
            ],
            dim=1,
        ).squeeze(0)
        image_mask = (
            (request.input_ids == self.image_token_id)
            .unsqueeze(-1)
            .expand_as(request.token_embeds)
        )
        image_token_count = int(image_mask[..., 0].sum().item())
        if image_embeds.shape[0] != image_token_count:
            raise ValueError(
                "image token/features mismatch: "
                f"{image_token_count} vs {image_embeds.shape[0]}"
            )
        request.token_embeds = request.token_embeds.masked_scatter(
            image_mask, image_embeds.to(request.token_embeds.dtype)
        )

    def prepare_prefill_chunk(
        self,
        request: PreparedRequest,
        start: int,
        prefill_length: int,
        embedding_size: int,
    ) -> StageInputs:
        input_length = int(request.input_ids.shape[1])
        end = min(start + prefill_length, input_length)
        current_length = end - start
        chunk = request.token_embeds[:, start:end]
        padded = torch.zeros(
            1, prefill_length, embedding_size, dtype=chunk.dtype
        )
        padded[:, :current_length] = chunk
        positions = request.position_ids[:, start:end]
        if current_length < prefill_length:
            positions = torch.cat(
                [
                    positions,
                    positions[:, -1:].expand(-1, prefill_length - current_length),
                ],
                dim=-1,
            )
        return StageInputs(
            tensors=(
                padded,
                positions,
                positions,
                positions,
                np.array([start], dtype=np.int32),
                np.array([current_length], dtype=np.int32),
                self.attention_mask(prefill_length, current_length),
            )
        )

    def prepare_decode(self, token: int, position: int) -> StageInputs:
        embedding = F.embedding(
            torch.tensor([[token]], dtype=torch.long), self.embedding_weight
        )
        positions = np.array([[position]], dtype=np.int32)
        return StageInputs(
            tensors=(
                embedding,
                positions,
                positions,
                positions,
                np.array([position], dtype=np.int32),
            )
        )

    @staticmethod
    def _normalize_generated_text(text: str, *, final: bool = False) -> str:
        if not final and text.endswith("\\"):
            text = text[:-1]
        return text.replace("\\n", "\n")

    def postprocess(self, state: GenerationState, *, final: bool = False) -> str:
        text = self._normalize_generated_text(
            self.tokenizer.decode(
                state.generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            final=final,
        )
        if not final and text.endswith("\ufffd"):
            return ""
        if not text.startswith(state.emitted_text):
            return ""
        delta = text[len(state.emitted_text) :]
        if delta:
            state.emitted_text = text
        return delta
