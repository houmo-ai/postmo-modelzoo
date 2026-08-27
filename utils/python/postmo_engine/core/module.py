# Copyright (c) 2026 HOUMO AI
#
# File: module.py
# Description:
#   Minimal model-semantic Module contract and session state.
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

"""Minimal model-semantic Module contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .errors import InvalidSessionError
from .types import DecodeInputs, DecodeOutputs, PrefillInputs, PrefillOutputs, SessionStatus


@dataclass
class ModuleSessionState:
    status: SessionStatus = SessionStatus.VALID
    context_length: int = 0


class PostMoModule(ABC):
    """Own model state, Prefill chunking, Decode execution, and context length."""

    session_state: ModuleSessionState

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Return the number of successfully committed logical tokens."""

    @property
    @abstractmethod
    def context_max_length(self) -> int:
        """Return the model context capacity."""

    @property
    def remaining_context(self) -> int:
        return self.context_max_length - self.context_length

    @abstractmethod
    def prefill(self, inputs: PrefillInputs) -> PrefillOutputs:
        """Execute one logical Prefill, including internal Chunking."""

    @abstractmethod
    def decode(self, inputs: DecodeInputs) -> DecodeOutputs:
        """Execute one logical Decode step."""

    @abstractmethod
    def clear_session(self) -> None:
        """Reset device state so a new request behaves like a fresh session."""

    def _require_valid_session(self) -> None:
        if self.session_state.status is SessionStatus.INVALID:
            raise InvalidSessionError("Module session is invalid; call clear_session()")

    def _begin_execution(self) -> None:
        self._require_valid_session()
        self.session_state.status = SessionStatus.EXECUTING

    def _mark_valid(self) -> None:
        self.session_state.status = SessionStatus.VALID

    def _mark_invalid(self) -> None:
        self.session_state.status = SessionStatus.INVALID
