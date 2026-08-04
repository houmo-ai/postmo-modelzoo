# Copyright (c) 2026 HOUMO AI
#
# File: test_api_runner.py
# Description:
#  Unit tests for API command execution, failure reporting, and C++ demo
#    configuration and build orchestration.
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

"""Contract tests for API command success classification."""

from pathlib import Path

import pytest

from tests.apis_tests.test_apis_utils import _command_succeeded, _compile_cpp_exec
from tests.tests_utils.command_execution import CommandResult, CommandSpec

pytestmark = pytest.mark.unit


def _result(
    *, stdout: str = "", stderr: str = "", return_code: int = 0
) -> CommandResult:
    """Build a command result for success-classification tests."""
    return CommandResult(
        command=CommandSpec("api-command-probe", ("true",)),
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=0.0,
    )


def test_command_succeeded_rejects_failure_marker_in_stderr() -> None:
    """Treat a legacy stderr failure marker as a failed API command."""
    assert not _command_succeeded(
        _result(stdout="normal output\n", stderr="[error] inference failed\n")
    )


def test_command_succeeded_accepts_clean_stderr() -> None:
    """Do not treat unrelated stderr diagnostics as command failures."""
    assert _command_succeeded(
        _result(
            stdout="normal output\n", stderr="warning: optional component unavailable\n"
        )
    )


def test_command_succeeded_can_disable_output_marker_check() -> None:
    """Preserve the explicit opt-out used by callers that allow marker text."""
    assert _command_succeeded(
        _result(stderr="[error] expected diagnostic\n"),
        check_output=False,
    )


class _RecordingRunner:
    """Record command specifications while reporting successful execution."""

    def __init__(self) -> None:
        self.commands: list[CommandSpec] = []

    def run(self, command: CommandSpec) -> CommandResult:
        """Record one command and return a successful result."""
        self.commands.append(command)
        return CommandResult(
            command=command,
            return_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.0,
        )


def test_compile_cpp_exec_configures_cmake_with_one_source_path(
    tmp_path: Path,
) -> None:
    """Do not pass the build directory as a second CMake position argument."""
    runner = _RecordingRunner()

    _compile_cpp_exec(
        tmp_path,
        runner,
        tmp_path / "api.log",
        ["-DENABLE_PROBE=ON"],
    )

    assert [command.argv for command in runner.commands] == [
        (
            "cmake",
            f"-DCMAKE_INSTALL_PREFIX={tmp_path}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_PROBE=ON",
            "..",
        ),
        ("make", "-j"),
        ("make", "install"),
    ]
    assert all(command.cwd == tmp_path / "build" for command in runner.commands)
