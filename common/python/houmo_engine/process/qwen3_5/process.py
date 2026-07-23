# Copyright (c) 2026 HOUMO AI
#
# File: process.py
# Description:
#   Qwen3.5 input and output Process implementation.
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

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.image_processing_utils import BatchFeature
from transformers.video_processing_utils import BaseVideoProcessor

from ...core import ModelProcess
from ...core.types import GenerationState, StageInputs, StageOutputs
from ...perf import PerfTracker
from .image_processing import Qwen3_5ImageProcessor
from .processing import Qwen3_5Processor
from .vision_processing import process_vision_info


SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2


class _DummyVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __call__(self, videos=None, **kwargs):
        del videos, kwargs
        return BatchFeature(data={})


@dataclass
class PreparedRequest:
    input_ids: torch.Tensor
    token_embeds: torch.Tensor
    positions: torch.Tensor
    vision_values: Sequence[torch.Tensor] | None = None
    image_grid_thw: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    @property
    def uses_vision(self) -> bool:
        return self.vision_values is not None


def _build_processor(
    tokenizer_path,
    max_size_h: int,
    max_size_w: int,
    patch_size: int,
) -> Qwen3_5Processor:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    template_path = Path(tokenizer_path) / "chat_template.jinja"
    if chat_template is None and template_path.exists():
        chat_template = template_path.read_text(encoding="utf-8")
    image_processor = Qwen3_5ImageProcessor(
        patch_size=patch_size,
        merge_size=SPATIAL_MERGE_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        min_pixels=max_size_h * max_size_w,
        max_pixels=max_size_h * max_size_w,
    )
    return Qwen3_5Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=_DummyVideoProcessor(),
        chat_template=chat_template,
    )


def _load_embedding(path, embedding_size: int) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value["weight"]
    elif hasattr(value, "weight"):
        value = value.weight
    return value.reshape(-1, embedding_size).float()


def _is_stable_char(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x0041 <= codepoint <= 0x005A
        or 0x0061 <= codepoint <= 0x007A
    )


class Qwen35Process(ModelProcess):
    """Qwen3.5 request preprocessing and token postprocessing."""

    def __init__(
        self,
        tokenizer_path,
        embedding_path,
        embedding_size: int,
        *,
        max_size_h: int = 896,
        max_size_w: int = 896,
        patch_size: int = 16,
        perf: PerfTracker,
    ):
        self.max_size_h = max_size_h
        self.max_size_w = max_size_w
        self.patch_size = patch_size
        self.perf = perf
        self.processor = _build_processor(
            tokenizer_path, max_size_h, max_size_w, patch_size
        )
        self.tokenizer = self.processor.tokenizer
        self.embedding_weight = _load_embedding(embedding_path, embedding_size)

    @staticmethod
    def normalize_images(images):
        if images is None:
            return None
        if isinstance(images, (str, Path)):
            images = [images]
        return [str(image) if isinstance(image, Path) else image for image in images]

    @staticmethod
    def attention_mask(length: int, valid_length: int) -> np.ndarray:
        mask = np.zeros((1, length), dtype=np.float16)
        mask[0, :valid_length] = 1
        return mask

    @staticmethod
    def text_positions(start: int, length: int) -> torch.Tensor:
        positions = torch.arange(start, start + length, dtype=torch.long)
        return positions.view(1, -1).expand(3, -1)

    def preprocess(
        self,
        prompt: str,
        images,
        system_prompt: str | None,
    ) -> PreparedRequest:
        if images:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": "image",
                                "image": image,
                                "resized_height": self.max_size_h,
                                "resized_width": self.max_size_w,
                            }
                            for image in images
                        ],
                        {"type": "text", "text": prompt},
                    ],
                }
            )
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            with self.perf.scope("llm.vision.preprocess"):
                image_inputs, video_inputs = process_vision_info(messages)
                model_inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    min_pixels=self.max_size_h * self.max_size_w,
                    max_pixels=self.max_size_h * self.max_size_w,
                    patch_size=self.patch_size,
                    merge_size=SPATIAL_MERGE_SIZE,
                    padding=True,
                    return_tensors="pt",
                )
            input_ids = model_inputs["input_ids"]
            return PreparedRequest(
                input_ids=input_ids,
                token_embeds=F.embedding(input_ids, self.embedding_weight),
                positions=torch.empty(0),
                vision_values=model_inputs["hm_pixel_values"],
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=model_inputs["attention_mask"],
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        return PreparedRequest(
            input_ids=input_ids,
            token_embeds=F.embedding(input_ids, self.embedding_weight),
            positions=torch.empty(0),
        )


    def merge_vision(
        self,
        request: PreparedRequest,
        vision_outputs: StageOutputs,
        state: GenerationState,
    ) -> None:
        image_embeds = vision_outputs.tensors[0]
        image_mask = (request.input_ids == self.processor.image_token_id).unsqueeze(-1)
        image_mask = image_mask.expand_as(request.token_embeds)
        if int(image_mask[..., 0].sum()) != image_embeds.shape[0]:
            raise ValueError("image token and vision feature counts differ")
        request.token_embeds = request.token_embeds.masked_scatter(
            image_mask, image_embeds.to(request.token_embeds.dtype)
        )
        positions, state.rope_deltas = self._vision_positions(
            request.input_ids,
            request.image_grid_thw,
            request.attention_mask,
        )
        request.positions = positions[:, 0, :] + state.context_length

    def prepare_prefill_chunk(
        self,
        request: PreparedRequest,
        state: GenerationState,
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
        valid_length = state.context_length + start
        if request.uses_vision:
            chunk_positions = request.positions[:, start:end]
            if current_length < prefill_length:
                chunk_positions = torch.cat(
                    [
                        chunk_positions,
                        chunk_positions[:, -1:].expand(
                            -1, prefill_length - current_length
                        ),
                    ],
                    dim=1,
                )
        else:
            chunk_positions = self.text_positions(valid_length, prefill_length)
        return StageInputs(
            tensors=(
                padded,
                chunk_positions[0:1],
                chunk_positions[1:2],
                chunk_positions[2:3],
                np.array([valid_length], dtype=np.int32),
                np.array([current_length], dtype=np.int32),
                self.attention_mask(prefill_length, current_length),
            ),
            metadata={"current_length": current_length},
        )

    def prepare_decode(self, token: int, state: GenerationState) -> StageInputs:
        embedding = F.embedding(
            torch.tensor([[token]], dtype=torch.long), self.embedding_weight
        )
        position = state.context_length
        if state.rope_deltas is not None:
            position += int(state.rope_deltas.item())
        positions = np.array([[position]], dtype=np.int32)
        return StageInputs(
            tensors=(
                embedding,
                positions,
                positions,
                positions,
                np.array([state.context_length], dtype=np.int32),
            )
        )

    def postprocess(self, state: GenerationState, *, final: bool = False) -> str:
        text = self.tokenizer.decode(state.generated_ids)
        delta = text[len(state.emitted_text) :]
        if final:
            state.emitted_text = text
            return delta
        if delta and _is_stable_char(delta[-1]):
            state.emitted_text = text
            return delta
        return ""

    def _vision_positions(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.zeros(
            3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype
        )
        image_index = 0
        deltas = []
        for batch_index, ids in enumerate(input_ids):
            ids = ids[attention_mask[batch_index] == 1]
            token_ids = ids.tolist()
            parts = []
            start = 0
            while self.processor.image_token_id in token_ids[start:]:
                end = token_ids.index(self.processor.image_token_id, start)
                grid_t, grid_h, grid_w = image_grid_thw[image_index].tolist()
                image_index += 1
                grid_h //= SPATIAL_MERGE_SIZE
                grid_w //= SPATIAL_MERGE_SIZE
                offset = int(parts[-1].max().item() + 1) if parts else 0
                text_length = end - start
                parts.append(
                    torch.arange(text_length).view(1, -1).expand(3, -1) + offset
                )
                time_ids = (
                    torch.arange(grid_t)
                    .view(-1, 1)
                    .expand(-1, grid_h * grid_w)
                    .flatten()
                )
                height_ids = (
                    torch.arange(grid_h)
                    .view(1, -1, 1)
                    .expand(grid_t, -1, grid_w)
                    .flatten()
                )
                width_ids = (
                    torch.arange(grid_w)
                    .view(1, 1, -1)
                    .expand(grid_t, grid_h, -1)
                    .flatten()
                )
                parts.append(
                    torch.stack([time_ids, height_ids, width_ids])
                    + text_length
                    + offset
                )
                start = end + grid_t * grid_h * grid_w
            if start < len(token_ids):
                offset = int(parts[-1].max().item() + 1) if parts else 0
                parts.append(
                    torch.arange(len(token_ids) - start).view(1, -1).expand(3, -1)
                    + offset
                )
            value = torch.cat(parts, dim=1)
            positions[:, batch_index, attention_mask[batch_index] == 1] = value
            deltas.append(value.max() + 1 - len(input_ids[batch_index]))
        return positions, torch.tensor(deltas).unsqueeze(1)
