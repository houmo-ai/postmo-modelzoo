# Copyright (c) 2026 HOUMO AI
#
# File: test_hmatc_runner.py
# Description:
#  Unit tests for HMATC command construction, result handling, configuration
#    discovery, and flow orchestration.
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

"""Unit tests for the standalone HMATC functional-test runner."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.hmatc_tests import test_hmatc_utils as hmatc_runner
from tests.hmatc_tests.test_hmatc_utils import (
    _command_succeeded,
    _hmatc_command,
    _perf_models,
    _run_hmatc,
    _run_hmatc_functional_cases,
    _run_hmatc_matrix,
    _run_hmatc_perf_cases,
)
from tests.tests_utils.command_execution import CommandResult, CommandSpec

pytestmark = pytest.mark.unit


class _RecordingRunner:
    """Record commands and return queued command results."""

    def __init__(self, outcomes=()) -> None:
        self.commands: list[CommandSpec] = []
        self._outcomes = list(outcomes)

    def run(self, command: CommandSpec) -> CommandResult:
        """Record one command and return its configured outcome."""
        self.commands.append(command)
        return_code, stdout, stderr = (
            self._outcomes.pop(0) if self._outcomes else (0, "ok\n", "")
        )
        return CommandResult(command, return_code, stdout, stderr, 0.01)


def _result(
    *, return_code: int = 0, stdout: str = "", stderr: str = ""
) -> CommandResult:
    """Build one standalone result for success-classification tests."""
    command = CommandSpec("probe", ("true",))
    return CommandResult(command, return_code, stdout, stderr, 0.01)


def test_hmatc_command_preserves_console_and_log_contract(tmp_path: Path) -> None:
    """Retain the legacy HMATC command presentation settings."""
    log_file = tmp_path / "hmatc.log"
    command = _hmatc_command(
        "hmatc-demo",
        ("hmatc", "demo", "--config", "demo.yml"),
        cwd=tmp_path,
        log_file=log_file,
    )

    assert command.name == "hmatc-demo"
    assert command.argv == ("hmatc", "demo", "--config", "demo.yml")
    assert command.cwd == tmp_path
    assert command.log_file == log_file
    assert command.allow_nonzero_exit
    assert command.mirror_to_console
    assert command.timestamp_log_lines
    assert command.suppressed_display_substrings == ("MB/s",)


@pytest.mark.parametrize(
    ("return_code", "stdout", "expected"),
    (
        (0, "normal output", True),
        (1, "normal output", False),
        (0, "Fail", False),
        (0, "[error] inference failed", False),
        (0, "case - Failed is a table label", True),
    ),
)
def test_command_succeeded_preserves_legacy_stdout_rules(
    return_code: int,
    stdout: str,
    expected: bool,
) -> None:
    """Combine exit status with the legacy stdout failure-marker predicate."""
    assert (
        _command_succeeded(_result(return_code=return_code, stdout=stdout)) is expected
    )


def test_command_succeeded_does_not_scan_separate_stderr() -> None:
    """Document that this legacy HMATC runner validates stdout only."""
    assert _command_succeeded(_result(stderr="[error] diagnostic on stderr"))


@pytest.mark.parametrize(
    ("hmatc_type", "expected_suffix"),
    (
        ("quant", ()),
        ("compare", ("--data_path", "input.jpg")),
        ("perf", ("-wn", "10", "-sn", "500", "-tn", "8")),
    ),
)
def test_run_hmatc_builds_subcommand_specific_arguments(
    tmp_path: Path,
    hmatc_type: str,
    expected_suffix: tuple[str, ...],
) -> None:
    """Add compare and perf options without affecting other subcommands."""
    runner = _RecordingRunner()

    assert _run_hmatc(
        runner,
        tmp_path,
        {"data_path": "input.jpg"},
        "config.yml",
        hmatc_type,
        tmp_path / "run.log",
    )

    assert runner.commands[0].argv == (
        "hmatc",
        hmatc_type,
        "--config",
        "config.yml",
        *expected_suffix,
    )


@pytest.mark.parametrize(("backend", "expected_ncore"), (("xh1", "4"), ("xh2", "2")))
def test_perf_models_runs_quant_build_and_perf_in_order(
    tmp_path: Path,
    backend: str,
    expected_ncore: str,
) -> None:
    """Use the backend-specific build core count before performance execution."""
    runner = _RecordingRunner()

    assert _perf_models(
        runner,
        tmp_path,
        "perf.yml",
        tmp_path / "perf.log",
        backend=backend,
    )

    assert [command.name for command in runner.commands] == [
        "hmatc-perf-quant",
        "hmatc-perf-build",
        "hmatc-perf",
    ]
    assert runner.commands[1].argv[-2:] == ("--ncore", expected_ncore)
    assert runner.commands[2].argv[-6:] == ("-wn", "10", "-sn", "1000", "-tn", "8")


def test_perf_models_stops_after_quant_failure(tmp_path: Path) -> None:
    """Do not run build or perf after quant fails."""
    runner = _RecordingRunner(((1, "quant failed", ""),))

    assert not _perf_models(
        runner,
        tmp_path,
        "perf.yml",
        tmp_path / "perf.log",
        backend="xh2",
    )
    assert [command.name for command in runner.commands] == ["hmatc-perf-quant"]


def test_functional_cases_are_sorted_and_aggregated(
    tmp_path: Path, monkeypatch
) -> None:
    """Run every subcommand for every config in deterministic path order."""
    config_dir = tmp_path / "func_test"
    config_dir.mkdir()
    for name in ("b.yml", "a.yml"):
        (config_dir / name).write_text("model: {}\n", encoding="utf-8")
    calls = []

    def run_case(runner, workspace, model_info, config_yml, hmatc_type, log_file):
        del runner, workspace, model_info, log_file
        calls.append((Path(config_yml).name, hmatc_type))
        return not (Path(config_yml).name == "b.yml" and hmatc_type == "build")

    monkeypatch.setattr(hmatc_runner, "_run_hmatc", run_case)
    passed = _run_hmatc_functional_cases(
        SimpleNamespace(),
        tmp_path,
        {},
        tmp_path,
        ("quant", "build"),
        tmp_path / "run.log",
    )

    assert not passed
    assert calls == [
        ("a.yml", "quant"),
        ("a.yml", "build"),
        ("b.yml", "quant"),
        ("b.yml", "build"),
    ]


def test_perf_cases_are_sorted_and_forward_backend(tmp_path: Path, monkeypatch) -> None:
    """Run performance configs deterministically with the selected backend."""
    config_dir = tmp_path / "perf_test"
    config_dir.mkdir()
    for name in ("z.yml", "a.yml"):
        (config_dir / name).write_text("model: {}\n", encoding="utf-8")
    calls = []

    def run_perf(runner, workspace, config_yml, log_file, *, backend):
        del runner, workspace, log_file
        calls.append((Path(config_yml).name, backend))
        return True

    monkeypatch.setattr(hmatc_runner, "_perf_models", run_perf)

    assert _run_hmatc_perf_cases(
        SimpleNamespace(),
        tmp_path,
        tmp_path,
        tmp_path / "perf.log",
        "xh1",
    )
    assert calls == [("a.yml", "xh1"), ("z.yml", "xh1")]


@pytest.mark.parametrize(("backend", "expected_perf_calls"), (("xh1", 1), ("xh2", 0)))
def test_hmatc_matrix_runs_extra_perf_cases_only_on_xh1(
    tmp_path: Path,
    monkeypatch,
    backend: str,
    expected_perf_calls: int,
) -> None:
    """Keep the standalone xh1-only performance matrix behavior."""
    calls = {"functional": 0, "perf": 0}

    def functional(*args):
        calls["functional"] += 1
        return True

    def perf(*args):
        calls["perf"] += 1
        return True

    monkeypatch.setattr(hmatc_runner, "_run_hmatc_functional_cases", functional)
    monkeypatch.setattr(hmatc_runner, "_run_hmatc_perf_cases", perf)

    assert _run_hmatc_matrix(
        "resnet50",
        {},
        SimpleNamespace(backend=backend),
        SimpleNamespace(),
        tmp_path,
        tmp_path / "run.log",
    )
    assert calls == {"functional": 1, "perf": expected_perf_calls}
