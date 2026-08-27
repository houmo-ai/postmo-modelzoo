# Copyright (c) 2026 HOUMO AI
#
# File: engine.py
# Description:
#   Minimal public Engine contract for request orchestration.
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

"""Minimal Engine contract."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from .capabilities import EngineCapabilities
from .types import EngineRequest, OutputChunk


class PostMoEngine(ABC):
    """Own request orchestration, stopping, sampling, and streaming."""

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Return capabilities that are callable by this Engine."""

    @abstractmethod
    def generate(self, request: EngineRequest) -> Iterator[OutputChunk]:
        """Synchronously stream output chunks and a final chunk."""

    @abstractmethod
    def clear_session(self) -> None:
        """Reset the underlying Module session."""
