# Copyright (c) 2026 HOUMO AI
#
# File: types.py
# Description:
#   Backend-independent request, stage, and result data types.
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

"""Backend-independent request, model-stage, and result types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StopReason(str, Enum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"
    CONTEXT_CAPACITY = "context_capacity"
    CANCELLED = "cancelled"
    ERROR = "error"


class SessionStatus(str, Enum):
    VALID = "valid"
    EXECUTING = "executing"
    INVALID = "invalid"


@dataclass(frozen=True)
class EngineRequest:
    request_id: str
    prompt: str
    max_new_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        if isinstance(self.max_new_tokens, bool) or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")


@dataclass(frozen=True)
class PrefillInputs:
    """Complete logical prompt inputs; Module owns runtime chunking."""

    embeddings: Any
    input_length: int
    positions: Any | None = None
    attention_mask: Any | None = None

    def __post_init__(self) -> None:
        if isinstance(self.input_length, bool) or self.input_length <= 0:
            raise ValueError("input_length must be greater than zero")


@dataclass(frozen=True)
class DecodeInputs:
    embedding: Any
    position: Any | None = None


@dataclass(frozen=True)
class PrefillOutputs:
    logits: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodeOutputs:
    logits: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleResult:
    token_id: int

    def __post_init__(self) -> None:
        if isinstance(self.token_id, bool) or self.token_id < 0:
            raise ValueError("token_id must be a non-negative integer")


@dataclass(frozen=True)
class OutputChunk:
    request_id: str
    sequence_no: int
    text_delta: str
    token_ids: tuple[int, ...]
    is_final: bool
    stop_reason: StopReason | None = None


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    stop_reason: StopReason
    input_tokens: int
    sampled_tokens: int
    visible_tokens: int
    submitted_decode_tokens: int
    output_chunks: int
