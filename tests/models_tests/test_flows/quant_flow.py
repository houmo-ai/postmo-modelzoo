# Copyright (c) 2025 HOUMO AI
#
# File: quant_flow.py
# Description:
#  Quantization Flow Implementations and Quantized-Artifact Publication.
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

"""Quantize model inputs and publish validated quantization artifacts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ...tests_utils.platform_device import check_gpu
from ...tests_utils.resource_lock import ModelResourceLock
from ...tests_utils.runtime_context import TCaseType
from ..model_workflow.artifact_cache_store import (
    ArtifactManifest,
    ArtifactType,
    AtomicArtifactWriter,
    calculate_config_fingerprint,
    copy_cache_contents,
)
from ..model_workflow.artifact_file_scanner import (
    build_required_file_roles,
    find_nonempty_artifact_files,
)
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
from ..model_workflow.backend_flow_policies import FamilyFlowPolicy
from ..model_workflow.parameter_matrix import (
    ParameterCase,
    ParameterMatrix,
    render_case_options,
)
from ..model_workflow.python_environment import prepare_python_environment
from .artifact_preparation import ArtifactNeed, ensure_artifacts
from .hmatc_flow_support import run_hmatc_quant_cases
from .hmatc_v2_flow_support import run_hmatc_v2_quant_cases

__all__ = [
    "QuantFlowHandler",
    "copy_cache_contents",
    "run_hmatc_quant_cases",
    "run_hmatc_v2_quant_cases",
]


@dataclass(frozen=True)
class QuantFlowHandler:
    """Run Python or HMATC quantization according to the model configuration."""

    policy: FamilyFlowPolicy

    def run(self, request: FlowRequest, services) -> FlowResult:
        """Execute the quant flow handler and return its structured result."""
        config = request.config
        context = request.context
        skip = self._skip_reason(request)
        if skip:
            return FlowResult(FlowDisposition.SKIPPED, skip)
        if not context.source_dir.is_dir():
            raise ArtifactValidationError(
                "Model source directory does not exist",
                context=context.diagnostic,
                details={"source_dir": context.source_dir},
            )

        command_results: list[CommandResult] = []
        failures: list[str] = []
        context.result_cache_dir.mkdir(parents=True, exist_ok=True)

        with services.workspace_manager.open(context.source_dir, phase="quant") as workspace:
            if config.uses_hmatc_v2:
                phase = run_hmatc_v2_quant_cases(request, services, workspace)
            else:
                preparation = ensure_artifacts(
                    request,
                    services,
                    (ArtifactNeed.raw_model(),),
                    workspace=workspace,
                    policy=self.policy,
                )
                command_results.extend(preparation.commands)
                if preparation.failures:
                    return FlowResult(
                        FlowDisposition.EXECUTED,
                        "raw model preparation failed",
                        commands=tuple(command_results),
                        validation=ValidationResult(
                            False,
                            "raw model preparation failed",
                            failures=preparation.failures,
                        ),
                    )
                phase = self._run_legacy_or_python_quant(
                    request, services, workspace
                )
            command_results.extend(phase.commands)
            failures.extend(phase.failures)

        validation = ValidationResult(
            passed=not failures,
            summary=("quantization completed" if not failures else "quantization failed: " + "; ".join(failures)),
            failures=tuple(failures),
        )
        return FlowResult(
            FlowDisposition.EXECUTED,
            "quantization completed",
            commands=tuple(command_results),
            validation=validation,
        )

    @staticmethod
    def _run_legacy_or_python_quant(request, services, workspace):
        """Run the unchanged HMATC v1 or Python quant implementation."""
        if request.config.has_section("hmquant_params"):
            phase = run_hmatc_quant_cases(request, services, workspace)
        else:
            phase = _run_python_quant(request, services, workspace)
        return phase

    def _skip_reason(self, request: FlowRequest) -> str | None:
        """Return why this flow should be skipped, or validate that it may run."""
        config = request.config
        context = request.context
        backend = context.diagnostic.backend
        if config.family != self.policy.family:
            raise ArtifactValidationError(
                "Quant handler received the wrong model family",
                context=context.diagnostic,
                details={
                    "expected": self.policy.family.value,
                    "actual": config.family.value,
                },
            )
        if config.obsolete or not config.supports(backend, ModelFlow.QUANT):
            return f"{config.model_name} does not support quant on {backend}"
        if context.test_type == TCaseType.SEPARATE_INFER:
            return f"quant for {config.model_name} already ran in separate no-infer"
        if context.platform in (None, "aarch64"):
            return f"{config.model_name} does not support quant on {context.platform}"
        if self.policy.quant_requires_gpu and (context.release or not check_gpu()["has_gpu"]):
            return f"{config.model_name} quant requires GPU and development mode; " f"release={int(context.release)}"
        return None


def _run_python_quant(request, services, workspace: Path):
    """Run configured Python quantization cases and publish their outputs."""
    backend = request.context.diagnostic.backend
    params = request.config.backend_section("quant_params", backend)
    if params is None:
        return PhaseResult(failures=(f"missing quant_params.{backend}",))
    python = prepare_python_environment(
        workspace,
        request.context.log_file,
        flow_type="quant",
        other_requirements=params.get("prerequisites", {}),
    )
    matrix = ParameterMatrix.from_columns(
        params,
        ignored_keys={"prerequisites"},
        location=f"{request.config.model_name}.quant_params.{backend}",
    )
    results: list[CommandResult] = []
    failures: list[str] = []
    executed_cases = 0
    lock_file = request.context.result_cache_dir / "lock.lock"
    with ModelResourceLock(str(lock_file), ModelResourceLock.LockMode.WRITE, "model quantizing"):
        for case in matrix.cases:
            result, error = _run_python_quant_case(
                request, services, workspace, python, case
            )
            if result is not None:
                results.append(result)
                executed_cases += 1
            if error:
                failures.append(error)
    return PhaseResult(
        commands=tuple(results),
        failures=tuple(failures),
        total_cases=len(matrix.cases),
        executed_cases=executed_cases,
    )


def _run_python_quant_case(request, services, workspace, python, case):
    """Execute and publish one Python quantization case."""
    resolved = resolve_case_paths(
        case, request.context.model_cache_dir, request.context.result_cache_dir
    )
    output_dir = _case_output_dir(resolved)
    if output_dir is None:
        return None, f"python quant case {case.index} has no output directory"
    writer = AtomicArtifactWriter(
        output_dir,
        root=request.context.result_cache_dir,
        token=f"{request.context.diagnostic.run_id}-quant-{case.index}",
        create_directory=False,
    )
    try:
        with writer as staging_dir:
            staged = replace_case_output_dir(resolved, staging_dir)
            result = services.command_runner.run(
                CommandSpec(
                    name=f"python-quant[{case.index}]",
                    argv=(python.executable, "ptq.py", *render_case_options(staged)),
                    cwd=workspace,
                    allow_nonzero_exit=True,
                    log_file=request.context.log_file,
                    environment=python.environment,
                ),
                diagnostic_fields=request.context.diagnostic.for_case(
                    output_dir.name, phase="python-quant"
                ).as_mapping(),
            )
            if not result.succeeded:
                return result, f"python quant case {case.index} failed"
            _flatten_hmquant(staging_dir)
            _publish_quant_artifact(
                request, services, resolved, staging_dir, case_id=output_dir.name
            )
            writer.commit()
            return result, None
    except ArtifactValidationError as error:
        return None, error.message


def _case_output_dir(case: ParameterCase) -> Path | None:
    """Return the output directory assigned to one parameter case."""
    for key in ("out-dir", "output_dir", "out_dir", "output_path"):
        value = case.values.get(key)
        if isinstance(value, str) and value:
            return Path(value)
    return None


def _flatten_hmquant(directory: Path) -> None:
    """Flatten nested HMQuant output into the canonical cache layout."""
    nested = directory / "hmquant"
    if not nested.is_dir():
        return
    for entry in nested.iterdir():
        shutil.move(str(entry), str(directory / entry.name))
    nested.rmdir()


def _publish_quant_artifact(
    request,
    services,
    case: ParameterCase,
    directory: Path,
    *,
    case_id: str | None = None,
):
    """Publish quant artifact with ownership and manifest metadata."""
    if not directory.is_dir() or not any(directory.iterdir()):
        raise ArtifactValidationError(
            "Quant command produced an empty artifact directory",
            context=request.context.diagnostic,
            details={"directory": directory, "case": case.index},
        )
    required_files: dict[str, str] = {}
    embedding = directory / "quant_embedding.pt"
    if embedding.is_file() and embedding.stat().st_size > 0:
        required_files["quant_embedding"] = embedding.name
    onnx_files = find_nonempty_artifact_files(directory, patterns=("*.onnx",))
    required_files.update(build_required_file_roles(directory, onnx_files, prefix="onnx"))
    if not required_files:
        representative_files = find_nonempty_artifact_files(directory, limit=32)
        required_files = build_required_file_roles(directory, representative_files, prefix="file")
    if not required_files:
        raise ArtifactValidationError(
            "Quant command produced no non-empty artifact files",
            context=request.context.diagnostic,
            details={"directory": directory, "case": case.index},
        )
    case_id = case_id or directory.name or f"case-{case.index}"
    services.artifact_cache.write_manifest(
        directory,
        ArtifactManifest(
            schema_version=1,
            fingerprint_version=1,
            artifact_type=ArtifactType.QUANT_MODEL.value,
            model_name=request.config.model_name,
            model_family=request.config.family.value,
            backend=request.context.diagnostic.backend,
            case_id=case_id,
            producer_flow=ModelFlow.QUANT.value,
            source_type="local_quant",
            config_fingerprint=calculate_config_fingerprint(
                {
                    "model": request.config.model_name,
                    "family": request.config.family.value,
                    "backend": request.context.diagnostic.backend,
                    "case_index": case.index,
                    "parameters": dict(case.values),
                }
            ),
            required_files=required_files,
        ),
    )
