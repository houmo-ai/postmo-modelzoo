# Copyright (c) 2025 HOUMO AI
#
# File: resource_lock.py
# Description:
#  Cross-Suite File-Based Resource Locking.
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

from __future__ import annotations

import enum
import fcntl
import logging
import os
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class ResourceLockError(RuntimeError):
    """Report a file-lock acquisition or lifecycle failure."""


class FileResourceLock:
    """Coordinate shared resources between concurrently running test processes."""

    class LockMode(enum.Enum):
        NO_LOCK = None
        READ_ONLY = fcntl.LOCK_SH
        WRITE = fcntl.LOCK_EX

    MAX_RESOURCE_ACCESS_TIME_OUT = 7200.0

    def __init__(
        self,
        lock_file: str | Path,
        lock_mode: LockMode,
        lock_purpose: str,
        *,
        timeout_seconds: float = MAX_RESOURCE_ACCESS_TIME_OUT,
        poll_interval_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.lock_file = str(lock_file)
        self.lock_mode = lock_mode
        self.lock_purpose = lock_purpose
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.clock = clock
        self.sleeper = sleeper
        self.lock_fd: int | None = None

    def __enter__(self):
        if self.lock_mode == self.LockMode.NO_LOCK:
            return None
        self.acquire_lock()
        return self.lock_fd

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.lock_mode != self.LockMode.NO_LOCK:
            self.release_lock()

    def acquire_lock(self) -> None:
        if self.lock_mode not in (self.LockMode.READ_ONLY, self.LockMode.WRITE):
            raise ResourceLockError(f"Unsupported lock mode: {self.lock_mode}")
        lock_folder = os.path.dirname(self.lock_file)
        if lock_folder and not os.path.exists(lock_folder):
            os.makedirs(lock_folder, exist_ok=True)
            os.chmod(lock_folder, 0o777)
        original_umask = os.umask(0)
        try:
            fd = os.open(self.lock_file, os.O_RDWR | os.O_CREAT, 0o666)
        finally:
            os.umask(original_umask)

        deadline = self.clock() + self.timeout_seconds
        logged_wait = False
        while self.clock() < deadline:
            try:
                fcntl.flock(fd, self.lock_mode.value | fcntl.LOCK_NB)
            except (IOError, OSError):
                if not logged_wait:
                    logger.info(
                        "%d waiting for %s lock: %s for %s",
                        os.getpid(),
                        self.lock_mode,
                        self.lock_file,
                        self.lock_purpose,
                    )
                    logged_wait = True
                self.sleeper(self.poll_interval_seconds)
                continue
            self.lock_fd = fd
            return

        os.close(fd)
        raise ResourceLockError(
            f"Timed out acquiring {self.lock_mode.name} lock {self.lock_file} " f"for {self.lock_purpose}"
        )

    def release_lock(self) -> None:
        if self.lock_fd is None:
            raise ResourceLockError(f"Lock is not held: {self.lock_file}")
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(self.lock_fd)
            self.lock_fd = None


ModelResourceLock = FileResourceLock

__all__ = ["FileResourceLock", "ModelResourceLock", "ResourceLockError"]
