# Copyright (c) 2026 HOUMO AI
#
# File: processor.py
# Description:
#   Minimal processor contract for model input and output conversion.
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

"""Minimal processor contract."""

from abc import ABC, abstractmethod
from typing import Any

from .types import DecodeInputs, DecodeOutputs, PrefillInputs, PrefillOutputs


class PostMoProcessor(ABC):
    """Convert user-facing text and model outputs without owning cache state."""

    @abstractmethod
    def preprocess(self, prompt: str) -> Any:
        """Prepare one request without creating Runtime-specific inputs."""

    @abstractmethod
    def build_prefill_inputs(self, prepared: Any) -> PrefillInputs:
        """Build the complete logical prompt for Module.prefill()."""

    @abstractmethod
    def build_decode_inputs(self, token_id: int) -> DecodeInputs:
        """Build one logical decode input."""

    @abstractmethod
    def process_prefill_outputs(self, outputs: PrefillOutputs) -> Any:
        """Convert Prefill output into the next sampling input."""

    @abstractmethod
    def process_decode_outputs(self, outputs: DecodeOutputs) -> Any:
        """Convert Decode output into the next sampling input."""

    @abstractmethod
    def decode_text(
        self,
        token_ids: tuple[int, ...],
        emitted_text: str,
        *,
        final: bool = False,
    ) -> str:
        """Return a stable text delta without owning generation state."""
