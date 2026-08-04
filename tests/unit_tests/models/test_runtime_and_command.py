# Copyright (c) 2026 HOUMO AI
#
# File: test_runtime_and_command.py
# Description:
#  Unit tests for runtime context, command execution, diagnostics, timeouts,
#    device markers, and output streaming.
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

"""Unit tests extracted from the former model-flow contract suite: test_runtime_and_command.py."""

import pytest
import sys
import threading
import time
from pathlib import (
    Path,
)
from tests.models_tests.model_workflow.backend_flow_policies import (
    GET_MODEL_COMMAND_TIMEOUT_SECONDS,
)
from tests.models_tests.model_workflow.flow_contracts import (
    CommandExecutionError,
    CommandResult,
    CommandSpec,
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    DiagnosticContext,
    ModelFamily,
    ModelFlow,
)
from tests.tests_utils.command_execution import (
    CommandRunner,
    CommandSpec as CrossSuiteCommandSpec,
)
from tests.tests_utils.pytest_support import (
    parse_device_markers,
)
from tests.tests_utils.runtime_context import (
    TCaseType,
    TestRuntimeContext as RuntimeContext,
)

pytestmark = pytest.mark.unit


def test_command_result_combined_output_skips_empty_streams() -> None:
    command = CommandSpec("combined", ("true",))
    assert CommandResult(command, 0, "stdout", "stderr", 0.1).combined_output == (
        "stdout\nstderr"
    )
    assert CommandResult(command, 0, "stdout", "", 0.1).combined_output == "stdout"


def test_diagnostic_context_derives_case_and_phase_without_mutation() -> None:
    base = DiagnosticContext(
        run_id="run",
        model_name="demo",
        family=ModelFamily.CV,
        backend="xh2",
        flow=ModelFlow.COMPILE,
    )
    case = base.for_case("hmm_xh2", phase="python-build")
    assert base.case_id is None and base.phase is None
    assert case.case_id == "hmm_xh2"
    assert case.phase == "python-build"
    assert "case_id=hmm_xh2" in case.as_fields()
    assert "phase=python-build" in case.as_fields()


def test_command_runner_reports_nonzero_and_timeout(tmp_path: Path) -> None:
    runner = CommandRunner()
    with pytest.raises(CommandExecutionError, match="non-zero"):
        runner.run(CommandSpec("false", ("sh", "-c", "exit 7")))
    log_file = tmp_path / "timeout.log"
    with pytest.raises(CommandExecutionError, match="timed out"):
        runner.run(
            CommandSpec(
                "timeout",
                (
                    sys.executable,
                    "-c",
                    "import time; print('before-timeout', flush=True); time.sleep(2)",
                ),
                timeout_seconds=0.1,
                log_file=log_file,
            )
        )
    timeout_log = log_file.read_text(encoding="utf-8")
    assert "before-timeout" in timeout_log
    assert "[command-timeout]" in timeout_log


def test_model_commands_default_to_eight_hour_timeout() -> None:
    assert DEFAULT_COMMAND_TIMEOUT_SECONDS == 28_800
    assert CommandSpec("default-timeout", ("true",)).timeout_seconds == 28_800
    assert GET_MODEL_COMMAND_TIMEOUT_SECONDS == 14_400


def test_cross_suite_commands_do_not_inherit_model_timeout() -> None:
    assert CrossSuiteCommandSpec("shared-default", ("true",)).timeout_seconds is None


def test_runtime_context_preserves_legacy_separate_stage_resolution(
    tmp_path: Path,
) -> None:
    no_infer = RuntimeContext.from_environment(
        {"HOUMO_TARGET": "xh2", "SKIP_INFER": "ON"},
        tests_root=tmp_path,
        asic=False,
        host_platform="x86_64",
    )
    infer = RuntimeContext.from_environment(
        {"HOUMO_TARGET": "xh2", "SKIP_INFER": "OFF"},
        tests_root=tmp_path,
        asic=True,
        host_platform="aarch64",
    )
    assert no_infer.test_type == TCaseType.SEPARATE_NO_INFER
    assert infer.test_type == TCaseType.SEPARATE_INFER


def test_device_marker_parser_retains_tokens_and_values() -> None:
    markers = parse_device_markers(("imodelzoo", "ndevice_2", "dev_mem_24g"))
    assert markers.ndevice == 2
    assert markers.device_mem == "24g"
    assert markers.result_directory_name == "ndevice_2_dev_mem_24g"


def test_command_runner_streams_output_before_process_completion(
    tmp_path: Path,
) -> None:
    log_file = tmp_path / "streaming.log"
    runner = CommandRunner()
    result_holder = []
    error_holder = []

    def run_command() -> None:
        try:
            result_holder.append(
                runner.run(
                    CommandSpec(
                        "streaming",
                        (
                            sys.executable,
                            "-c",
                            "import sys, time; "
                            "print('first-line', flush=True); "
                            "print('stderr-line', file=sys.stderr, flush=True); "
                            "time.sleep(0.5); "
                            "print('last-line', flush=True)",
                        ),
                        log_file=log_file,
                    )
                )
            )
        except Exception as error:  # pragma: no cover - asserted below
            error_holder.append(error)

    worker = threading.Thread(target=run_command)
    worker.start()
    deadline = time.monotonic() + 2.0
    observed_while_running = False
    while time.monotonic() < deadline:
        if log_file.is_file() and "first-line" in log_file.read_text(encoding="utf-8"):
            observed_while_running = worker.is_alive()
            break
        time.sleep(0.01)
    worker.join(timeout=3.0)

    assert not worker.is_alive()
    assert not error_holder
    assert observed_while_running
    assert len(result_holder) == 1
    result = result_holder[0]
    assert result.stdout == "first-line\nlast-line\n"
    assert result.stderr == "stderr-line\n"
    log_text = log_file.read_text(encoding="utf-8")
    assert log_text.startswith("$ ")
    assert "first-line" in log_text
    assert "stderr-line" in log_text
    assert "last-line" in log_text


def test_command_runner_logs_start_failure_and_unbuffers_python(
    tmp_path: Path,
) -> None:
    runner = CommandRunner()
    environment_result = runner.run(
        CommandSpec(
            "environment",
            (
                sys.executable,
                "-c",
                "import os; print(os.environ.get('PYTHONUNBUFFERED'))",
            ),
        )
    )
    assert environment_result.stdout == "1\n"

    log_file = tmp_path / "start-error.log"
    with pytest.raises(CommandExecutionError, match="Failed to start"):
        runner.run(
            CommandSpec(
                "missing",
                (str(tmp_path / "missing-command"),),
                log_file=log_file,
            )
        )
    log_text = log_file.read_text(encoding="utf-8")
    assert "missing-command" in log_text
    assert "[command-start-error]" in log_text
