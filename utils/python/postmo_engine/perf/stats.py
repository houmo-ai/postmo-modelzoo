# Copyright (c) 2026 HOUMO AI
#
# File: stats.py
# Description:
#   Performance scope statistics and report data structures.
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
from math import inf
from typing import Any


@dataclass
class ScopeStats:
    path: str
    count: int | None = 0
    total_ms: float = 0.0
    min_ms: float | None = inf
    max_ms: float | None = 0.0

    @property
    def avg_ms(self) -> float | None:
        if self.count is None:
            return None
        return self.total_ms / self.count if self.count else 0.0

    def add(self, elapsed_ms: float) -> None:
        if self.count is None or self.min_ms is None or self.max_ms is None:
            raise RuntimeError("cannot add samples to an aggregate scope")
        self.count += 1
        self.total_ms += elapsed_ms
        self.min_ms = min(self.min_ms, elapsed_ms)
        self.max_ms = max(self.max_ms, elapsed_ms)

    def copy(self) -> "ScopeStats":
        return ScopeStats(
            path=self.path,
            count=self.count,
            total_ms=self.total_ms,
            min_ms=self.min_ms,
            max_ms=self.max_ms,
        )

@dataclass
class PerfReport:
    scopes: dict[str, ScopeStats] = field(default_factory=dict)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    derived: dict[str, dict[str, float]] = field(default_factory=dict)
    speeds: dict[str, tuple[float, str]] = field(default_factory=dict)
