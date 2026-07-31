# Copyright (c) 2025 HOUMO AI
#
# File: compile_flow.py
# Description:
#  Compilation Flow Implementations and Compiled-Artifact Publication.
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

"""Compile model artifacts through Python build scripts or HMATC."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from ...tests_utils.command_execution import output_reports_failure
from ...tests_utils.platform_device import is_asic_platform
from ...tests_utils.resource_lock import ModelResourceLock
from ...tests_utils.runtime_context import TCaseType
from ..model_workflow.artifact_cache_store import AtomicArtifactWriter
from ..model_workflow.artifact_file_scanner import prune_compiler_intermediates
from ..model_workflow.cache_path_resolver import (
    replace_case_output_dir,
    resolve_case_paths,
)
from ..model_workflow.flow_contracts import (
    ArtifactValidationError,
    CommandResult,
    CommandSpec,
    FlowDisposition,
    FlowRequest,
    FlowResult,
    ModelFlow,
    PhaseResult,
    ValidationResult,
)
from ..model_workflow.artifact_publication import (
    compiled_artifact_reusable,
    publish_compiled_artifact,
)
from ..model_workflow.backend_flow_policies import (
    FamilyFlowPolicy,
    filter_xh2_ncore4,
    hmatc_build_header,
)
from ..model_workflow.parameter_matrix import ParameterMatrix, render_case_options
from .artifact_preparation import ArtifactNeed, ensure_artifacts
from .hmatc_flow_support import run_hmatc_quant_cases

logger = logging.getLogger(__name__)


__all__ = [
    "CompileFlowHandler",
    "publish_compiled_artifact",
    "run_hmatc_build_cases",
]


def run_hmatc_build_cases(request, services, workspace: Path) -> PhaseResult:
    """Run HMATC build cases and return its structured execution outcome."""
    backend = request.context.diagnostic.backend
    columns = request.config.hmatc_columns("hmbuild_params", backend=backend)
    matrix = ParameterMatrix.from_columns(
        columns,
        location=f"{request.config.model_name}.hmbuild_params.{backend}",
    )
    cases = (
        filter_xh2_ncore4(matrix.cases, lambda case: case.values.get("ncore"))
        if backend == "xh2"
        else tuple(matrix.cases)
    )
    filtered_count = len(matrix.cases) - len(cases)
    threshold = request.config.validation_threshold(backend, "compile")
    logger.info(
        "compile threshold=%s source=%s total_cases=%s filtered_cases=%s",
        threshold.value,
        threshold.source,
        len(matrix.cases),
        filtered_count,
    )
    results: list[CommandResult] = []
    failures: list[str] = []
    asic = is_asic_platform()
    for case in cases:
        argv = (
            *hmatc_build_header(backend, asic=asic),
            *render_case_options(case, skipped_keys={"onnx"}),
        )
        result = services.command_runner.run(
            CommandSpec(
                name=f"hmatc-build[{case.index}]",
                argv=argv,
                cwd=workspace,
                allow_nonzero_exit=True,
                log_file=request.context.log_file,
            ),
            diagnostic_fields=request.context.diagnostic.for_case(case.index, phase="hmatc-build").as_mapping(),
        )
        results.append(result)
        output = result.combined_output
        if not result.succeeded or output_reports_failure(output):
            failures.append(f"hmatc build case {case.index} failed")
        elif asic and not _compile_output_passed(output, backend, float(threshold.value)):
            failures.append(f"hmatc build case {case.index} did not meet cosine threshold " f"{threshold.value}")
    # HMATC keeps every case's outputs in the same workspace directory, so this
    # runs once after the last case instead of per case.
    prune_compiler_intermediates(workspace)
    return PhaseResult(
        commands=tuple(results),
        failures=tuple(failures),
        total_cases=len(matrix.cases),
        executed_cases=len(results),
        filtered_cases=filtered_count,
    )


@dataclass(frozen=True)
class CompileFlowHandler:
    """Orchestrate compile cases and atomically publish valid outputs."""

    policy: FamilyFlowPolicy

    def run(self, request: FlowRequest, services) -> FlowResult:
        """Execute the compile flow handler and return its structured result."""
        skip = self._skip_reason(request)
        if skip:
            return FlowResult(FlowDisposition.SKIPPED, skip)
        if not request.context.source_dir.is_dir():
            raise ArtifactValidationError(
                "Model source directory does not exist",
                context=request.context.diagnostic,
                details={"source_dir": request.context.source_dir},
            )

        request.context.result_cache_dir.mkdir(parents=True, exist_ok=True)
        commands: list[CommandResult] = []
        failures: list[str] = []
        with services.workspace_manager.open(request.context.source_dir, phase="compile") as workspace:
            if request.config.has_section("hmbuild_params"):
                preparation = ensure_artifacts(
                    request,
                    services,
                    (ArtifactNeed.raw_model(),),
                    workspace=workspace,
                    policy=self.policy,
                )
                commands.extend(preparation.commands)
                if preparation.failures:
                    return FlowResult(
                        FlowDisposition.EXECUTED,
                        "raw model preparation failed",
                        commands=tuple(commands),
                        validation=ValidationResult(
                            False,
                            "raw model preparation failed",
                            failures=preparation.failures,
                        ),
                    )
                quant_phase = run_hmatc_quant_cases(request, services, workspace)
                commands.extend(quant_phase.commands)
                failures.extend(quant_phase.failures)
                build_phase = run_hmatc_build_cases(request, services, workspace)
            else:
                preparation = ensure_artifacts(
                    request,
                    services,
                    (ArtifactNeed.quant_model(),),
                    workspace=workspace,
                    policy=self.policy,
                )
                commands.extend(preparation.commands)
                if preparation.disposition == FlowDisposition.SKIPPED:
                    return FlowResult(
                        FlowDisposition.SKIPPED,
                        preparation.message,
                        commands=tuple(commands),
                    )
                if preparation.failures:
                    failure_summary = preparation.message or "quant input preparation failed"
                    return FlowResult(
                        FlowDisposition.EXECUTED,
                        failure_summary,
                        commands=tuple(commands),
                        validation=ValidationResult(
                            False,
                            failure_summary,
                            failures=preparation.failures,
                        ),
                    )
                build_phase = self._run_python_build(request, services, workspace)
            commands.extend(build_phase.commands)
            failures.extend(build_phase.failures)
            if build_phase.all_filtered and not failures:
                return FlowResult(
                    FlowDisposition.SKIPPED,
                    "all hmatc build cases were filtered by the default xh2 "
                    "ncore=4 policy; set IMODELZOO_ALLOW_XH2_NCORE4=ON to "
                    "enable them",
                    commands=tuple(commands),
                )

        validation = ValidationResult(
            passed=not failures,
            summary=("compilation completed" if not failures else "compilation failed: " + "; ".join(failures)),
            failures=tuple(failures),
        )
        return FlowResult(
            FlowDisposition.EXECUTED,
            "compilation completed",
            commands=tuple(commands),
            validation=validation,
        )

    def _skip_reason(self, request: FlowRequest) -> str | None:
        """Return why this flow should be skipped, or validate that it may run."""
        config = request.config
        context = request.context
        backend = context.diagnostic.backend
        if config.family != self.policy.family:
            raise ArtifactValidationError(
                "Compile handler received the wrong model family",
                context=context.diagnostic,
                details={
                    "expected": self.policy.family.value,
                    "actual": config.family.value,
                },
            )
        if config.obsolete or not config.supports(backend, ModelFlow.COMPILE):
            return f"{config.model_name} does not support compile on {backend}"
        if context.test_type == TCaseType.SEPARATE_INFER:
            return f"compile for {config.model_name} already ran in separate no-infer"
        if self.policy.compile_skip_in_release and context.release:
            return f"release mode skips {self.policy.family.value} compile for " f"{config.model_name}"
        if context.platform in (None, "aarch64"):
            return f"{config.model_name} does not support compile on {context.platform}"
        return None

    def _run_python_build(self, request, services, workspace: Path):
        """Run configured Python build cases and collect validation failures."""
        backend = request.context.diagnostic.backend
        params = request.config.backend_section("compile_params", backend)
        if params is None:
            return PhaseResult(failures=(f"missing compile_params.{backend}",))
        matrix = ParameterMatrix.from_columns(
            params,
            location=f"{request.config.model_name}.compile_params.{backend}",
        )
        results: list[CommandResult] = []
        failures: list[str] = []
        executed_cases = 0
        reused_cases = 0
        for case in matrix.cases:
            result, executed, reused, error = self._run_python_build_case(
                request, services, workspace, case
            )
            if result is not None:
                results.append(result)
            executed_cases += executed
            reused_cases += reused
            if error:
                failures.append(error)
        return PhaseResult(
            commands=tuple(results),
            failures=tuple(failures),
            total_cases=len(matrix.cases),
            executed_cases=executed_cases,
            reused_cases=reused_cases,
        )

    def _run_python_build_case(self, request, services, workspace, case):
        """Build and publish one Python compile case."""
        resolved = resolve_case_paths(
            case, request.context.model_cache_dir, request.context.result_cache_dir
        )
        output_value = resolved.values.get("output_dir")
        output_dir = Path(output_value) if isinstance(output_value, str) else None
        if output_dir is None:
            return None, 0, 0, f"python build case {case.index} has no output_dir"
        owned_root = _artifact_owned_root(request, output_dir)
        with ModelResourceLock(
            str(owned_root / "lock.lock"),
            ModelResourceLock.LockMode.WRITE,
            "model compiling",
        ):
            model_dir = resolved.values.get("model_dir")
            if not isinstance(model_dir, str) or not Path(model_dir).is_dir():
                return (
                    None,
                    0,
                    0,
                    f"python build case {case.index} input model_dir is missing: {model_dir}",
                )
            if compiled_artifact_reusable(request, services, resolved, output_dir):
                logger.info("Reuse compiled artifact for case %s: %s", case.index, output_dir)
                return None, 0, 1, None
            try:
                return self._execute_python_build_case(
                    request, services, workspace, case, resolved, output_dir, owned_root, model_dir
                )
            except ArtifactValidationError as error:
                return None, 0, 0, error.message

    @staticmethod
    def _execute_python_build_case(
        request, services, workspace, case, resolved, output_dir, owned_root, model_dir
    ):
        """Execute the command and atomically publish one compile result."""
        writer = AtomicArtifactWriter(
            output_dir,
            root=owned_root,
            token=f"{request.context.diagnostic.run_id}-build-{case.index}",
        )
        with writer as staging_dir:
            staged = replace_case_output_dir(resolved, staging_dir)
            result = services.command_runner.run(
                CommandSpec(
                    name=f"python-build[{case.index}]",
                    argv=("python3", "build.py", *render_case_options(staged)),
                    cwd=workspace,
                    allow_nonzero_exit=True,
                    log_file=request.context.log_file,
                ),
                diagnostic_fields=request.context.diagnostic.for_case(
                    output_dir.name, phase="python-build"
                ).as_mapping(),
            )
            if not result.succeeded or output_reports_failure(result.combined_output):
                return result, 1, 0, f"python build case {case.index} failed"
            prune_compiler_intermediates(staging_dir)
            _copy_quant_sidecar_files(Path(model_dir), staging_dir)
            publish_compiled_artifact(
                request, services, resolved, staging_dir, case_id=output_dir.name
            )
            writer.commit()
        return result, 1, 0, None


def _artifact_owned_root(request, destination: Path) -> Path:
    """Return the owned artifact root represented by a cache path."""
    resolved = destination.resolve()
    for root in (
        request.context.result_cache_dir.resolve(),
        request.context.model_cache_dir.resolve(),
    ):
        if resolved != root and root in resolved.parents:
            return root
    raise ArtifactValidationError(
        "Compile output must be inside cached_results or cached_models",
        context=request.context.diagnostic,
        details={"destination": resolved},
    )


def _copy_quant_sidecar_files(model_dir: Path, staging_dir: Path) -> None:
    """Copy quant sidecar files while preserving the expected artifact layout."""
    destination = staging_dir / "hmquant"
    for source in sorted(model_dir.glob("quant_*.pt")):
        if not source.is_file():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / source.name)


def _compile_output_passed(output: str, backend: str, threshold: float) -> bool:
    """Return whether compiler output contains an acceptable result table."""
    if backend == "xh2":
        row_pattern = re.compile(r"\|\s*([\w\-/.]+)\s*\|\s*(\d+\.\d+)\s*\|$")
    else:
        row_pattern = re.compile(
            r"^\|\s*([^|]+?)\s*\|\s*(\d+\.\d+)\s*\|\s*(\w+)\s*\|" r"\s*(\d+\.\d+)\s*\|\s*(\w+)\s*\|$"
        )
    header = None
    rows = []
    for line in output.splitlines():
        line = line.strip()
        if "cosine_dist" in line:
            header = [column.strip() for column in line.split("|") if column.strip()]
            continue
        match = row_pattern.match(line)
        if match and header:
            values = match.groups()
            rows.append(values)
    if not header or not rows:
        return False
    if backend == "xh2":
        return all(float(row[1]) >= threshold for row in rows)
    return all(float(row[1]) >= threshold and float(row[3]) >= threshold for row in rows)
