# Copyright (c) 2025 HOUMO AI
#
# File: perf_flow.py
# Description:
#  Model Performance Flow, Runner Selection, and Metric Validation.
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

"""Run performance cases and validate metrics from HMATC or Python runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...tests_utils.command_execution import output_reports_failure
from ...tests_utils.python_environment import as_python_environment
from ...tests_utils.resource_lock import ModelResourceLock
from ...tests_utils.runtime_context import TCaseType
from ..model_workflow.cache_path_resolver import resolve_case_paths
from ..model_workflow.parameter_matrix import (
    ParameterCase,
    ParameterMatrix,
    render_case_options,
)
from ..model_workflow.python_environment import (
    prepare_python_environment,
)
from ..model_workflow.flow_contracts import (
    ArtifactValidationError,
    CommandResult,
    CommandSpec,
    FlowDisposition,
    FlowRequest,
    FlowResult,
    ModelFlow,
    ValidationResult,
)
from ..model_workflow.backend_flow_policies import FamilyFlowPolicy, should_check_output_failure
from .artifact_preparation import ensure_inference_artifacts
from .inference_flow_support import (
    common_skip_reason,
    resolve_python_script,
    validated_result,
)
from .hmatc_flow_support import persist_separate_workspace, run_hmatc_cases
from ..model_workflow.perf_metric_validation import extract_perf_metrics, validate_perf_metrics

__all__ = ["PerfFlowHandler"]


@dataclass(frozen=True)
class PerfFlowHandler:
    """Select the configured perf runner and validate its extracted metrics."""

    policy: FamilyFlowPolicy

    def run(self, request: FlowRequest, services) -> FlowResult:
        """Execute the perf flow handler and return its structured result."""
        skip = common_skip_reason(request, ModelFlow.PERF, self.policy.family)
        if skip:
            return FlowResult(FlowDisposition.SKIPPED, skip)
        if not request.config.has_section("perf_metrics"):
            return FlowResult(FlowDisposition.SKIPPED, "missing perf_metrics")

        commands: list[CommandResult] = []
        failures: list[str] = []
        validation: ValidationResult | None = None
        with services.workspace_manager.open(request.context.source_dir, phase="perf") as workspace:
            preparation = ensure_inference_artifacts(request, services, workspace, self.policy)
            prep_failures = list(preparation.failures)
            commands.extend(preparation.commands)
            failures.extend(prep_failures)
            if failures:
                return validated_result(ModelFlow.PERF, commands, failures, "perf preparation failed")
            if request.context.test_type == TCaseType.SEPARATE_NO_INFER:
                persist_separate_workspace(request, workspace, self.policy)
                return FlowResult(
                    FlowDisposition.PREPARED_ONLY,
                    "perf artifacts prepared for separate infer",
                    commands=tuple(commands),
                )

            baseline, behavior = request.config.perf_contract(
                request.context.diagnostic.backend, request.context.platform
            )
            python = prepare_python_environment(workspace, request.context.log_file, activated=True)
            results, case_failures = _run_perf_runner(request, services, workspace, python, behavior)
            commands.extend(results)
            failures.extend(case_failures)
            if not failures:
                validation, parse_failures = _validate_perf_result(behavior, baseline, results, request)
                failures.extend(parse_failures)

        if validation is not None and failures:
            validation = ValidationResult(
                False,
                "; ".join(failures),
                metrics=validation.metrics,
                failures=tuple(failures),
            )
        return FlowResult(
            FlowDisposition.EXECUTED,
            "perf completed" if not failures else "perf failed",
            commands=tuple(commands),
            validation=validation or ValidationResult(not failures, "perf completed", failures=tuple(failures)),
        )


def _run_perf_runner(request, services, workspace, python, behavior):
    """Dispatch to the configured performance runner."""
    if behavior.runner == "hmatc":
        return run_hmatc_cases(
            request,
            services,
            workspace,
            section_name="hmperf_params",
            subcommand="perf",
            environment=python.environment,
        )
    if behavior.runner == "demo":
        return _run_python_perf_case(request, services, workspace, python)
    return _run_custom_perf_case(request, services, workspace, python, behavior.custom_script)


def _validate_perf_result(behavior, baseline, results, request):
    """Parse and validate one performance runner result."""
    try:
        source_text = _select_perf_text(behavior.source, results, request.context.log_file)
        actual = extract_perf_metrics(source_text, behavior)
        validation = validate_perf_metrics(
            actual,
            baseline,
            behavior,
            minimum_ratio=0.1 if request.context.release else 0.95,
        )
        return validation, list(validation.failures)
    except Exception as error:
        return None, [f"performance parsing failed: {error}"]


def _run_python_perf_case(request, services, workspace, python):
    """Run the configured Python performance case and validate its output."""
    python = as_python_environment(python)
    params = request.config.backend_section("demo_params", request.context.diagnostic.backend)
    if params is None:
        return [], ["demo perf requires demo_params"]
    matrix = ParameterMatrix.from_columns(
        params,
        location=f"{request.config.model_name}.demo_params.{request.context.diagnostic.backend}",
    )
    if not matrix.cases:
        return [], ["demo perf has no parameter cases"]
    case = resolve_case_paths(
        matrix.cases[0],
        request.context.model_cache_dir,
        request.context.result_cache_dir,
    )
    script = case.values.get("script")
    script_name = resolve_python_script(workspace, script, default="demo.py")
    lock_file = request.context.result_cache_dir / "lock.lock"
    request.context.result_cache_dir.mkdir(parents=True, exist_ok=True)
    with ModelResourceLock(str(lock_file), ModelResourceLock.LockMode.WRITE, "execute demo perf"):
        result = services.command_runner.run(
            CommandSpec(
                "demo-perf[0]",
                (
                    python.executable,
                    script_name,
                    *render_case_options(case, skipped_keys={"script"}),
                ),
                cwd=workspace,
                allow_nonzero_exit=True,
                log_file=request.context.log_file,
                environment=python.environment,
            ),
            diagnostic_fields=request.context.diagnostic.for_case(0, phase="demo-perf").as_mapping(),
        )
    failures = [] if result.succeeded else ["demo perf command failed"]
    if should_check_output_failure(request.config.model_name, ModelFlow.PERF) and output_reports_failure(
        result.combined_output
    ):
        failures.append("demo perf output reported failure")
    return [result], failures


def _run_custom_perf_case(request, services, workspace, python, specification):
    """Run custom performance case and return its structured execution outcome."""
    python = as_python_environment(python)
    if specification is None:
        return [], ["custom perf runner has no command specification"]
    params = request.config.backend_section(specification.params_section, request.context.diagnostic.backend)
    if params is None:
        return [], [f"custom perf requires {specification.params_section}." f"{request.context.diagnostic.backend}"]
    matrix = ParameterMatrix.from_columns(
        params,
        ignored_keys={"prerequisites"},
        location=(
            f"{request.config.model_name}.{specification.params_section}." f"{request.context.diagnostic.backend}"
        ),
    )
    if specification.case_index >= len(matrix.cases):
        return [], [f"custom perf case_index {specification.case_index} is out of range"]
    resolved = resolve_case_paths(
        matrix.cases[specification.case_index],
        request.context.model_cache_dir,
        request.context.result_cache_dir,
    )
    missing = [key for key in specification.parameter_keys if key not in resolved.values]
    if missing:
        return [], [f"custom perf parameters are missing: {missing}"]
    selected = ParameterCase(
        resolved.index,
        {key: resolved.values[key] for key in specification.parameter_keys},
    )
    script_name = resolve_python_script(
        workspace,
        specification.script,
        default=specification.script,
    )
    result = services.command_runner.run(
        CommandSpec(
            f"custom-perf[{resolved.index}]",
            (
                python.executable,
                script_name,
                *render_case_options(selected),
            ),
            cwd=workspace,
            allow_nonzero_exit=True,
            log_file=request.context.log_file,
            environment=python.environment,
        ),
        diagnostic_fields=request.context.diagnostic.for_case(resolved.index, phase="custom-perf").as_mapping(),
    )
    output = result.combined_output
    failures = []
    if not result.succeeded or output_reports_failure(output):
        failures.append("custom perf command failed")
    return [result], failures


def _select_perf_text(source, results, log_file: Path) -> str:
    """Select performance text for the current request and execution context."""
    if source == "stdout":
        return "\n".join(result.stdout for result in results)
    if source == "stderr":
        return "\n".join(result.stderr for result in results)
    if source == "combined_output":
        return "\n".join(result.combined_output for result in results if result.combined_output)
    try:
        return log_file.read_text(encoding="utf-8")
    except OSError as error:
        raise ArtifactValidationError(
            "Failed to read performance log",
            details={"log_file": log_file, "error": error},
        ) from error
