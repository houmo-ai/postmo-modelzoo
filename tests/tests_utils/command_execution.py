# Copyright (c) 2025 HOUMO AI
#
# File: command_execution.py
# Description:
#  Cross-Suite Subprocess Execution, Streaming Output, and Diagnostics.
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

import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO, cast

logger = logging.getLogger(__name__)


class OutputCaptureMode(str, Enum):
    """Describe whether stderr is captured separately or merged into stdout."""

    SEPARATE = "separate"
    COMBINED = "combined"


@dataclass(frozen=True)
class CommandSpec:
    """Describe one subprocess execution without suite-specific behavior."""

    name: str
    argv: tuple[str, ...]
    cwd: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    capture_mode: OutputCaptureMode = OutputCaptureMode.SEPARATE
    allow_nonzero_exit: bool = False
    log_file: Path | None = None
    mirror_to_console: bool = False
    timestamp_log_lines: bool = False
    suppressed_display_substrings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandResult:
    """Capture a completed command and both output streams."""

    command: CommandSpec
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def succeeded(self) -> bool:
        """Return whether the process exited with code zero."""
        return self.return_code == 0

    @property
    def combined_output(self) -> str:
        """Combine non-empty stdout and stderr for explicit caller validation."""
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


class CommandExecutionError(RuntimeError):
    """Report a generic command infrastructure failure."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic_fields: Mapping[str, object] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.diagnostic_fields = dict(diagnostic_fields or {})
        self.details = dict(details or {})

    def format_diagnostic(self) -> str:
        """Render stable diagnostics without assuming a specific test suite."""
        lines = [self.__class__.__name__, self.message]
        if self.diagnostic_fields:
            lines.append(
                " ".join(f"{key}={value}" for key, value in self.diagnostic_fields.items() if value is not None)
            )
        lines.extend(f"{key}: {value}" for key, value in self.details.items())
        return "\n".join(lines)


def output_reports_failure(output: str) -> bool:
    """Apply the legacy explicit failure-marker predicate to supplied text."""
    return ("Fail" in output and "- Failed" not in output) or "[error]" in output


class StreamingLogSink(AbstractContextManager["StreamingLogSink"]):
    """Write command output to an optional log file and console in real time."""

    def __init__(self, command: CommandSpec) -> None:
        self.command = command
        self._stream: TextIO | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> "StreamingLogSink":
        if self.command.log_file is not None:
            self.command.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.command.log_file.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.flush()
                self._stream.close()
                self._stream = None

    def write(self, text: str, *, channel: str = "stdout") -> None:
        """Write one captured chunk without affecting the captured result."""
        if not text:
            return
        suppressed = any(marker in text for marker in self.command.suppressed_display_substrings)
        with self._lock:
            if self._stream is not None and not suppressed:
                rendered = text
                if self.command.timestamp_log_lines:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    rendered = f"[{timestamp}] {text}"
                self._stream.write(rendered)
                self._stream.flush()
            if self.command.mirror_to_console and not suppressed:
                console = sys.stderr if channel == "stderr" else sys.stdout
                console.write(text)
                console.flush()

    def write_line(self, text: str, *, channel: str = "stdout") -> None:
        self.write(text if text.endswith("\n") else f"{text}\n", channel=channel)


class CommandRunner:
    """Execute commands with bounded waits and process-group cleanup."""

    _TERMINATE_GRACE_SECONDS = 5.0
    _READER_JOIN_SECONDS = 5.0

    def run(
        self,
        command: CommandSpec,
        *,
        diagnostic_fields: Mapping[str, object] | None = None,
    ) -> CommandResult:
        started = time.monotonic()
        environment = os.environ.copy()
        environment.update({key: str(value) for key, value in command.environment.items()})
        environment.setdefault("PYTHONUNBUFFERED", "1")
        stderr_target = subprocess.STDOUT if command.capture_mode == OutputCaptureMode.COMBINED else subprocess.PIPE
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        reader_errors: list[str] = []
        process: subprocess.Popen[str] | None = None
        readers: list[threading.Thread] = []

        with StreamingLogSink(command) as sink:
            sink.write_line(f"$ {shlex.join(command.argv)}")
            try:
                process = subprocess.Popen(
                    list(command.argv),
                    cwd=command.cwd,
                    env=environment,
                    text=True,
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=stderr_target,
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as error:
                sink.write_line(f"[command-start-error] {error}", channel="stderr")
                raise CommandExecutionError(
                    f"Failed to start command: {command.name}",
                    diagnostic_fields=diagnostic_fields,
                    details={"argv": command.argv, "cwd": command.cwd, "error": error},
                ) from error

            assert process.stdout is not None
            readers.append(
                self._start_reader(
                    cast(TextIO, process.stdout),
                    stdout_chunks,
                    sink,
                    reader_errors,
                    channel="stdout",
                )
            )
            if stderr_target != subprocess.STDOUT:
                assert process.stderr is not None
                readers.append(
                    self._start_reader(
                        cast(TextIO, process.stderr),
                        stderr_chunks,
                        sink,
                        reader_errors,
                        channel="stderr",
                    )
                )
            try:
                return_code = process.wait(timeout=command.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                sink.write_line(
                    f"[command-timeout] exceeded {command.timeout_seconds} seconds",
                    channel="stderr",
                )
                self._terminate_process_group(process, sink)
                self._join_readers(process, readers)
                raise CommandExecutionError(
                    f"Command timed out: {command.name}",
                    diagnostic_fields=diagnostic_fields,
                    details={
                        "argv": command.argv,
                        "cwd": command.cwd,
                        "timeout_seconds": command.timeout_seconds,
                        "stdout_tail": "".join(stdout_chunks)[-2000:],
                        "stderr_tail": "".join(stderr_chunks)[-2000:],
                    },
                ) from error
            except BaseException:
                self._terminate_process_group(process, sink)
                self._join_readers(process, readers)
                raise

            self._join_readers(process, readers)
            if reader_errors:
                sink.write_line(
                    "[command-stream-error] " + "; ".join(reader_errors),
                    channel="stderr",
                )
                raise CommandExecutionError(
                    f"Failed to capture command output: {command.name}",
                    diagnostic_fields=diagnostic_fields,
                    details={
                        "argv": command.argv,
                        "cwd": command.cwd,
                        "reader_errors": tuple(reader_errors),
                    },
                )

        result = CommandResult(
            command=command,
            return_code=return_code,
            stdout="".join(stdout_chunks),
            stderr="" if stderr_target == subprocess.STDOUT else "".join(stderr_chunks),
            duration_seconds=time.monotonic() - started,
        )
        if not result.succeeded and not command.allow_nonzero_exit:
            raise CommandExecutionError(
                f"Command returned non-zero exit code: {command.name}",
                diagnostic_fields=diagnostic_fields,
                details={
                    "argv": command.argv,
                    "cwd": command.cwd,
                    "return_code": result.return_code,
                    "stdout_tail": result.stdout[-2000:],
                    "stderr_tail": result.stderr[-2000:],
                },
            )
        return result

    @staticmethod
    def _start_reader(
        stream: TextIO,
        chunks: list[str],
        sink: StreamingLogSink,
        errors: list[str],
        *,
        channel: str,
    ) -> threading.Thread:
        def read_stream() -> None:
            try:
                while True:
                    chunk = stream.readline()
                    if not chunk:
                        break
                    chunks.append(chunk)
                    sink.write(chunk, channel=channel)
            except (OSError, ValueError) as error:
                if not stream.closed:
                    errors.append(f"{channel}: {error}")
            finally:
                try:
                    stream.close()
                except OSError:
                    pass

        reader = threading.Thread(
            target=read_stream,
            name=f"command-{channel}-reader",
            daemon=True,
        )
        reader.start()
        return reader

    def _join_readers(self, process: subprocess.Popen[str], readers: Sequence[threading.Thread]) -> None:
        for reader in readers:
            reader.join(timeout=self._READER_JOIN_SECONDS)
        if any(reader.is_alive() for reader in readers):
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
            for reader in readers:
                reader.join(timeout=1.0)

    def _terminate_process_group(self, process: subprocess.Popen[str], sink: StreamingLogSink) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=self._TERMINATE_GRACE_SECONDS)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            sink.write_line(
                "[command-timeout] SIGTERM grace expired; sending SIGKILL",
                channel="stderr",
            )
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait()


__all__ = [
    "CommandExecutionError",
    "CommandResult",
    "CommandRunner",
    "CommandSpec",
    "OutputCaptureMode",
    "StreamingLogSink",
    "output_reports_failure",
]
