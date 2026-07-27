# Copyright (c) 2026 HOUMO AI
#
# File: types.py
# Description:
#   Shared data types for Houmo Python Engine layers.
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
from enum import Enum
from typing import Any


class Stage(str, Enum):
    VISION = "vision"
    ENCODE = "encode"
    PREFILL = "prefill"
    DECODE = "decode"
    MTP_PREFILL = "mtp_prefill"
    DRAFT = "draft"
    VERIFY = "verify"


@dataclass
class StageInputs:
    tensors: tuple[Any, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageOutputs:
    tensors: tuple[Any, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationState:
    """Engine-owned CPU state for one generation session.

    This object contains only request/session bookkeeping used to prepare the
    next model stage and produce streaming output. Device-side state such as KV
    cache, convolution cache, recurrent state, and runtime handles belongs to
    the corresponding Module and must not be stored here.
    """

    # Number of tokens already occupying the model context. It includes the
    # prompt tokens submitted by prefill and generated tokens that have been
    # fed through decode. With history enabled, the value is retained between
    # requests; otherwise Engine.clear_session() resets it to zero.
    context_length: int = 0

    # Model-specific RoPE position correction calculated during multimodal
    # preprocessing. Qwen3.5 uses it to translate the next text decode position
    # after image tokens have used three-dimensional temporal/height/width
    # positions. Text-only requests leave this as None.
    rope_deltas: Any = None

    # Token IDs generated for the current request, beginning with the token
    # sampled from the final prefill logits. The Engine uses this list for
    # sampling penalties, stopping decisions, and detokenization. It is reset
    # for every request even when the device context is retained for history.
    generated_ids: list[int] = field(default_factory=list)

    # Complete decoded text already returned to the caller. Process compares
    # the latest full detokenized text with this prefix to emit only the new,
    # stable suffix and prevent duplicated streaming output.
    emitted_text: str = ""
