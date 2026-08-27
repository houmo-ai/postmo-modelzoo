# Copyright (c) 2026 HOUMO AI
#
# File: qwen_types.py
# Description:
#   Qwen model-local layer data contracts.
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

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from houmo_engine.core.types import GenerationState, StageInputs


@dataclass
class PreparedRequest:
    input_ids: torch.Tensor
    token_embeds: torch.Tensor
    positions: torch.Tensor
    vision_values: Any = None
    image_grid_thw: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    @property
    def uses_vision(self) -> bool:
        return self.vision_values is not None


@dataclass
class Qwen36MtpPreparedRequest:
    """Tokenized single-turn Qwen3.6 MTP request."""

    input_ids: np.ndarray


@dataclass
class Qwen36MtpGenerationState(GenerationState):
    """Engine-owned CPU state specific to Qwen3.6 MTP generation."""

    mtp_context_length: int = 0
    pending_token: int | None = None
    draft_anchor_hidden: np.ndarray | torch.Tensor | None = None
    finish_reason: str | None = None


@dataclass
class VerifyResult:
    draft_tokens: list[int]
    accepted_count: int
    next_token: int
    next_hidden: np.ndarray


__all__ = [
    "PreparedRequest",
    "Qwen36MtpPreparedRequest",
    "Qwen36MtpGenerationState",
    "VerifyResult",
]
