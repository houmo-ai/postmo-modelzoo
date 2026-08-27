# Copyright (c) 2026 HOUMO AI
#
# File: progress.py
# Description:
#   Dependency-free terminal progress reporting for performance runs.
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

"""Small dependency-free terminal progress reporter."""

from __future__ import annotations

import sys
import threading
from typing import TextIO


class ProgressReporter:
    """Render progress to stderr without affecting the performance report."""

    def __init__(self, *, enabled: bool | None = None, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = self.stream.isatty() if enabled is None else enabled
        self._label = ""
        self._total = 0
        self._current = 0
        self._last_length = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def begin(self, label: str, total: int) -> None:
        self._label = label
        self._total = max(1, int(total))
        self._current = 0
        self._render()
        if self.enabled:
            self._stop.clear()
            self._thread = threading.Thread(target=self._refresh, daemon=True)
            self._thread.start()

    def phase(self, label: str) -> None:
        self._label = label

    def reset_total(self, total: int, *, current: int = 0) -> None:
        self._total = max(1, int(total))
        self._current = min(max(0, int(current)), self._total)

    def update(self, current: int) -> None:
        self._current = min(max(0, int(current)), self._total)

    def finish(self, *, newline: bool = True) -> None:
        if not self.enabled:
            return
        self._current = self._total
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._render()
        if newline:
            self.close()

    def close(self) -> None:
        """Finish the current terminal line after all iterations."""
        if not self.enabled:
            return
        self.stream.write("\n")
        self.stream.flush()

    def _render(self) -> None:
        if not self.enabled:
            return
        width = 30
        completed = int(width * self._current / self._total)
        bar = "=" * completed + ">" + " " * max(0, width - completed - 1)
        text = f"\r{self._label}: [{bar}] {self._current}/{self._total}"
        if len(text) < self._last_length:
            text += " " * (self._last_length - len(text))
        self.stream.write(text)
        self.stream.flush()
        self._last_length = len(text)

    def _refresh(self) -> None:
        while not self._stop.wait(0.1):
            self._render()
