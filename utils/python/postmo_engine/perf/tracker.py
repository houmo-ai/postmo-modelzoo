# Copyright (c) 2026 HOUMO AI
#
# File: tracker.py
# Description:
#   Performance scope and metric tracking implementation.
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

import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from loguru import logger

from .formatter import format_report
from .metrics import derive_metrics, derive_speeds
from .stats import PerfReport, ScopeStats


class PerfTracker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._scopes: dict[str, ScopeStats] = {}
        self._metrics: dict[str, dict[str, Any]] = {}
        self._active: dict[str, int] = {}

    @classmethod
    def create(cls, enabled: bool = False) -> "PerfTracker":
        return cls(enabled=enabled)

    @staticmethod
    def _validate_path(path: str) -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("perf path must be a non-empty string")
        if path.startswith(".") or path.endswith(".") or ".." in path:
            raise ValueError(f"invalid perf path: {path!r}")

    def _record(self, path: str, elapsed_ms: float) -> None:
        stats = self._scopes.get(path)
        if stats is None:
            stats = ScopeStats(path=path)
            self._scopes[path] = stats
        stats.add(elapsed_ms)

    def _check_unfinished(self) -> None:
        if not self._active:
            return
        paths = ", ".join(sorted(repr(path) for path in self._active))
        raise RuntimeError(f"perf paths were started but not ended: {paths}")

    @contextmanager
    def scope(self, path: str):
        if not self.enabled:
            yield
            return
        self._validate_path(path)
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            self._record(path, elapsed_ms)

    def start(self, path: str) -> None:
        if not self.enabled:
            return
        self._validate_path(path)
        if path in self._active:
            raise RuntimeError(f"perf path is already started: {path!r}")
        self._active[path] = time.perf_counter_ns()

    def end(self, path: str) -> None:
        if not self.enabled:
            return
        self._validate_path(path)
        started_ns = self._active.pop(path, None)
        if started_ns is None:
            raise RuntimeError(f"perf path was not started: {path!r}")
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        self._record(path, elapsed_ms)

    def set_audio_length(self, seconds: float) -> None:
        if not self.enabled:
            return
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise TypeError("audio length must be a number")
        if seconds <= 0:
            raise ValueError("audio length must be greater than zero")
        self.set_metrics("asr", audio_length_s=float(seconds))

    def set_metrics(self, path: str, **metrics: Any) -> None:
        if not self.enabled:
            return
        self._validate_path(path)
        self._metrics.setdefault(path, {}).update(metrics)

    def summary(self) -> PerfReport:
        self._check_unfinished()
        report = PerfReport(
            scopes={path: stats.copy() for path, stats in self._scopes.items()},
            metrics={path: dict(values) for path, values in self._metrics.items()},
        )
        report.derived = derive_metrics(report)
        report.speeds = derive_speeds(report)
        return report

    def print_summary(self) -> None:
        if not self.enabled:
            return
        sys.stdout.flush()
        logger.opt(raw=True, colors=True).success(
            "<green>{}</green>\n",
            format_report(self.summary()),
        )

    def reset(self, preserve_prefixes: tuple[str, ...] = ()) -> None:
        self._check_unfinished()
        for prefix in preserve_prefixes:
            self._validate_path(prefix)
        self._scopes = {
            path: stats
            for path, stats in self._scopes.items()
            if any(path == prefix or path.startswith(f"{prefix}.") for prefix in preserve_prefixes)
        }
        self._metrics = {
            path: metrics
            for path, metrics in self._metrics.items()
            if any(path == prefix or path.startswith(f"{prefix}.") for prefix in preserve_prefixes)
        }
