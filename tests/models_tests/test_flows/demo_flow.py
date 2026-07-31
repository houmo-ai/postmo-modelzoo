# Copyright (c) 2025 HOUMO AI
#
# File: demo_flow.py
# Description:
#  Demo Flow Implementation for Shell, HMATC, and Python Model Demos.
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

"""Execute model demos across test.sh, HMATC, Python, and multibatch paths."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ...tests_utils.command_execution import output_reports_failure
from ...tests_utils.platform_device import check_device_info, is_asic_platform
from ...tests_utils.python_environment import PythonEnvironment, as_python_environment
from ...tests_utils.resource_lock import ModelResourceLock
from ...tests_utils.runtime_context import TCaseType
from ..model_workflow.cache_path_resolver import resolve_case_paths
from ..model_workflow.parameter_matrix import ParameterMatrix, render_case_options
from ..model_workflow.python_environment import prepare_python_environment
from ..model_workflow.flow_contracts import (
    ArtifactValidationError,
    CommandResult,
    CommandSpec,
    FlowDisposition,
    FlowRequest,
    FlowResult,
    ModelFamily,
    ModelFlow,
)
from ..model_workflow.backend_flow_policies import FamilyFlowPolicy, should_check_output_failure
from .artifact_preparation import ensure_inference_artifacts
from .inference_flow_support import (
    common_skip_reason,
    validated_result,
)
from .hmatc_flow_support import persist_separate_workspace, run_hmatc_cases

__all__ = ["DemoFlowHandler"]


logger = logging.getLogger(__name__)
_FAILURE_OUTPUT_TAIL_CHARS = 4000
_TEST_SCRIPT_NAME = "test.sh"


@dataclass(frozen=True)
class DemoFlowHandler:
    """Select and run the demo implementations allowed by family and platform."""

    policy: FamilyFlowPolicy

    def run(self, request: FlowRequest, services) -> FlowResult:
        """Execute the demo flow handler and return its structured result."""
        skip = common_skip_reason(request, ModelFlow.DEMO, self.policy.family)
        if skip:
            return FlowResult(FlowDisposition.SKIPPED, skip)
        core_skip = _demo_device_skip_reason(request, services)
        if core_skip:
            return FlowResult(FlowDisposition.SKIPPED, core_skip)

        commands: list[CommandResult] = []
        shell_failures: list[str] = []

        shell_results, shell_errors = self._run_test_sh_stage(request, services)
        commands.extend(shell_results)
        shell_failures.extend(shell_errors)

        if request.context.release and not request.config.demo_enabled():
            return validated_result(
                ModelFlow.DEMO,
                commands,
                shell_failures,
                "release demo.py is disabled",
            )

        with services.workspace_manager.open(request.context.source_dir, phase="demo") as workspace:
            preparation = ensure_inference_artifacts(request, services, workspace, self.policy)
            prep_failures = list(preparation.failures)
            commands.extend(preparation.commands)
            if prep_failures:
                return validated_result(
                    ModelFlow.DEMO,
                    commands,
                    [*shell_failures, *prep_failures],
                    "demo preparation failed",
                )
            if request.context.test_type == TCaseType.SEPARATE_NO_INFER:
                persist_separate_workspace(request, workspace, self.policy)
                return FlowResult(
                    FlowDisposition.PREPARED_ONLY,
                    "demo artifacts prepared for separate infer",
                    commands=tuple(commands),
                )

            results, case_failures = self._run_demo_implementation(request, services, workspace)
            commands.extend(results)
            shell_failures.extend(case_failures)

        return validated_result(ModelFlow.DEMO, commands, shell_failures)

    def _run_test_sh_stage(self, request, services):
        """Run the optional ASIC test.sh stage outside the demo workspace."""
        if not is_asic_platform() or request.context.test_type == TCaseType.SEPARATE_NO_INFER:
            return [], []
        with services.workspace_manager.open(request.context.source_dir, phase="demo_test_sh") as workspace:
            if not (workspace / _TEST_SCRIPT_NAME).is_file():
                return [], []
            python = (
                prepare_python_environment(workspace, request.context.log_file, activated=True)
                if self.policy.family == ModelFamily.CV
                else PythonEnvironment("python3", {})
            )
            return _run_test_sh(request, services, workspace, environment=python.environment)

    def _run_demo_implementation(self, request, services, workspace):
        """Run either HMATC demo cases or Python demo scripts."""
        python = prepare_python_environment(
            workspace,
            request.context.log_file,
            activated=self.policy.family == ModelFamily.CV,
        )
        if request.config.has_section("hmdemo_params") and request.context.platform != "aarch64":
            return run_hmatc_cases(
                request,
                services,
                workspace,
                section_name="hmdemo_params",
                subcommand="demo",
                environment=python.environment,
            )
        results = []
        failures = []
        for demo_name in ("demo", "demo_multibatch"):
            demo_results, demo_failures = _run_python_demo_cases(request, services, workspace, python, demo_name)
            results.extend(demo_results)
            failures.extend(demo_failures)
        return results, failures


def _demo_device_skip_reason(request: FlowRequest, services) -> str | None:
    """Return the device-specific reason for skipping demo execution."""
    if not is_asic_platform() or request.context.platform != "aarch64":
        return None
    backend = request.context.diagnostic.backend
    get_params = request.config.backend_section("get_model_params", backend) or {}
    has_hmm = "hmm" in tuple(get_params.get("type", ()))
    core_requirement = request.config.supported_core_count(backend)
    if (
        not request.config.has_section("demo_params")
        or not has_hmm
        or not check_device_info(
            core_requirement,
            backend=backend,
            runner=services.command_runner,
        )
    ):
        return "aarch64 demo requires downloadable HMMs and a supported core count"
    return None


def _run_test_sh(
    request,
    services,
    workspace: Path,
    *,
    environment: Mapping[str, str] | None = None,
):
    """Run test shell script and return its structured execution outcome."""
    cases = _test_sh_cases(
        request.config.value("test_sh_params"),
        request.context.diagnostic.backend,
        request.context.diagnostic,
    )
    results = []
    failures = []
    for index, args in enumerate(cases):
        result = services.command_runner.run(
            CommandSpec(
                f"test-sh[{index}]",
                ("bash", _TEST_SCRIPT_NAME, *args),
                cwd=workspace,
                allow_nonzero_exit=True,
                log_file=request.context.log_file,
                environment=environment or {},
            ),
            diagnostic_fields=request.context.diagnostic.for_case(index, phase="demo-test-sh").as_mapping(),
        )
        results.append(result)
        check_output = should_check_output_failure(request.config.model_name, ModelFlow.DEMO)
        if not result.succeeded or (check_output and output_reports_failure(result.combined_output)):
            output_marker_failure = check_output and output_reports_failure(result.combined_output)
            _log_failed_command(_TEST_SCRIPT_NAME, index, result, output_marker_failure)
            failures.append(f"{_TEST_SCRIPT_NAME} case {index} failed")
    return results, failures


def _log_failed_command(
    label: str,
    index: int,
    result: CommandResult,
    output_marker_failure: bool,
) -> None:
    """Log bounded command output and execution metadata for a failed case."""
    logger.error(
        "%s case failed: index=%s name=%s return_code=%s "
        "duration=%.3fs output_marker_failure=%s cwd=%s argv=%r\n"
        "stdout_tail:\n%s\nstderr_tail:\n%s",
        label,
        index,
        result.command.name,
        result.return_code,
        result.duration_seconds,
        output_marker_failure,
        result.command.cwd,
        result.command.argv,
        _output_tail(result.stdout),
        _output_tail(result.stderr),
    )


def _output_tail(output: str, *, limit: int = _FAILURE_OUTPUT_TAIL_CHARS) -> str:
    """Return a bounded command-output tail suitable for failure diagnostics."""
    if not output:
        return "<empty>"
    if len(output) <= limit:
        return output.rstrip()
    return "<truncated>\n" + output[-limit:].lstrip()


def _test_sh_cases(params, backend, diagnostic) -> list[tuple[str, ...]]:
    """Normalize supported test_sh_params layouts into shell argument cases."""
    if isinstance(params, Mapping) and backend in params:
        params = params[backend]
    if not params:
        return [()]
    if isinstance(params, Mapping):
        list_values = [value for value in params.values() if isinstance(value, list)]
        if not list_values:
            raise ArtifactValidationError(
                "test_sh_params dictionary values must contain parameter lists",
                context=diagnostic,
            )
        return _render_test_sh_columns(params, max(len(value) for value in list_values))
    return _render_test_sh_lists(params, diagnostic)


def _render_test_sh_columns(params: Mapping, case_count: int) -> list[tuple[str, ...]]:
    """Render a column-oriented test.sh parameter mapping."""
    return [
        tuple(
            token
            for name, values in params.items()
            for token in _render_test_sh_value(
                name,
                values[index] if isinstance(values, list) and index < len(values) else values,
            )
        )
        for index in range(case_count)
    ]


def _render_test_sh_value(name, value) -> tuple[str, ...]:
    """Render one test.sh parameter value."""
    option = str(name) if str(name).startswith("-") else f"--{name}"
    if isinstance(value, bool):
        return (option,) if value else ()
    return (option, str(value)) if value is not None else ()


def _render_test_sh_lists(params, diagnostic) -> list[tuple[str, ...]]:
    """Validate and normalize list-oriented test.sh parameters."""
    if not isinstance(params, list) or not all(isinstance(case, (list, tuple)) for case in params):
        raise ArtifactValidationError(
            "test_sh_params must be an argument-list or column dictionary",
            context=diagnostic,
        )
    return [tuple(str(value) for value in case) for case in params]


def _run_python_demo_cases(request, services, workspace, python, demo_name):
    """Run configured Python demo cases and collect command-level failures."""
    python = as_python_environment(python)
    backend = request.context.diagnostic.backend
    if demo_name not in request.config.support_flow.get(backend, ()):
        return [], []
    params = request.config.backend_section(f"{demo_name}_params", backend)
    if params is None:
        return [], [f"missing {demo_name}_params.{backend}"]
    matrix = ParameterMatrix.from_columns(params, location=f"{request.config.model_name}.{demo_name}_params.{backend}")
    results = []
    failures = []
    lock_file = request.context.result_cache_dir / "lock.lock"
    request.context.result_cache_dir.mkdir(parents=True, exist_ok=True)
    with ModelResourceLock(str(lock_file), ModelResourceLock.LockMode.WRITE, f"execute {demo_name}"):
        for case in matrix.cases:
            resolved = resolve_case_paths(
                case,
                request.context.model_cache_dir,
                request.context.result_cache_dir,
            )
            script = resolved.values.get("script")
            script_name = str(script) if script not in (None, "default") else f"{demo_name}.py"
            result = services.command_runner.run(
                CommandSpec(
                    f"{demo_name}[{case.index}]",
                    (
                        python.executable,
                        script_name,
                        *render_case_options(resolved, skipped_keys={"script"}),
                    ),
                    cwd=workspace,
                    allow_nonzero_exit=True,
                    log_file=request.context.log_file,
                    environment=python.environment,
                ),
                diagnostic_fields=request.context.diagnostic.for_case(case.index, phase=demo_name).as_mapping(),
            )
            results.append(result)
            check_output = should_check_output_failure(request.config.model_name, ModelFlow.DEMO)
            if not result.succeeded or (check_output and output_reports_failure(result.combined_output)):
                output_marker_failure = check_output and output_reports_failure(result.combined_output)
                _log_failed_command(
                    demo_name,
                    case.index,
                    result,
                    output_marker_failure,
                )
                failures.append(f"{demo_name} case {case.index} failed")
    return results, failures
