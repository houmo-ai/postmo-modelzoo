# Copyright (c) 2026 HOUMO AI
#
# File: minicpm_v45_types.py
# Description:
#   Layer contracts and CPU state for MiniCPM-V 4.5.
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from houmo_engine.core.types import GenerationState


@dataclass
class MiniCPMV45Paths:
    tokenizer_dir: Path
    embedding_path: Path
    prefill_path: Path
    decode_path: Path
    vision_path: Path | None
    video_vision_path: Path | None = None


@dataclass
class MiniCPMV45Request:
    prompt: str
    images: list[str]
    videos: list[str]
    system_prompt: str | None
    video_fps: float = 3.0


@dataclass
class PreparedMiniCPMV45Request:
    input_ids: Any
    token_embeds: Any
    pixel_values: Any | None
    target_sizes: Any | None
    input_length: int
    image_count: int
    temporal_groups: list[list[int]] | None = None
    vision_units: Any | None = None


@dataclass
class PrefillRequest:
    token_embeds: Any


@dataclass
class MiniCPMV45State(GenerationState):
    input_length: int = 0
    decode_tokens: int = 0
    image_count: int = 0


__all__ = [
    "MiniCPMV45Paths",
    "MiniCPMV45Request",
    "PreparedMiniCPMV45Request",
    "PrefillRequest",
    "MiniCPMV45State",
]
