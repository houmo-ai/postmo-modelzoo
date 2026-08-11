# Copyright (c) 2026 HOUMO AI
#
# File: hmatc_v2_flow_support.py
# Description:
#  Independent HMATC v2 Quant/Build Execution and Artifact Publication.
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

"""Execute HMATC v2 without changing the legacy HMATC workspace protocol."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ...tests_utils.command_execution import output_reports_failure
from ...tests_utils.platform_device import is_asic_platform
from ...tests_utils.resource_lock import ModelResourceLock
from ..model_workflow.artifact_cache_store import (
    ArtifactManifest,
    ArtifactRequirement,
    ArtifactType,
    AtomicArtifactWriter,
    CacheStatus,
)
from ..model_workflow.backend_flow_policies import hmatc_build_header
from ..model_workflow.cache_path_resolver import (
    RESULT_CACHE_ROOT,
    cache_case_reference,
    resolve_cached_path,
)
from ..model_workflow.flow_contracts import (
    ArtifactValidationError,
    CommandResult,
    CommandSpec,
    ConfigError,
    ModelFlow,
    PhaseResult,
)
from ..model_workflow.hmatc_v2_config import (
    hmatc_v2_case_id,
    hmatc_v2_fingerprint,
    materialize_hmatc_v2_config,
    resolve_nested_cache_paths,
)
from .artifact_preparation import prepare_hmatc_v2_raw_models

logger = logging.getLogger(__name__)

_V2_SOURCE_TYPE_QUANT = "local_hmatc_v2_quant"
_V2_SOURCE_TYPE_BUILD = "local_hmatc_v2_build"

__all__ = [
    "copy_hmatc_v2_runtime_sidecars",
    "run_hmatc_v2_build_cases",
    "run_hmatc_v2_quant_cases",
]


def run_hmatc_v2_quant_cases(
    request,
    services,
    workspace: Path,
    *,
    logical_outputs: frozenset[str] | None = None,
) -> PhaseResult:
    """Run or reuse selected HMATC v2 quant cases."""
    cases = tuple(
        case
        for case in request.config.hmatc_v2_cases("hmquant_params")
        if logical_outputs is None or case.logical_quant_output in logical_outputs
    )
    if logical_outputs is not None and len(cases) != len(logical_outputs):
        found = {case.logical_quant_output for case in cases}
        missing = sorted(logical_outputs - found)
        return PhaseResult(
            failures=(f"HMATC v2 quant cases not found for outputs: {missing}",),
            total_cases=len(logical_outputs),
        )

    commands: list[CommandResult] = []
    failures: list[str] = []
    executed = 0
    reused = 0
    for case in cases:
        case_commands, error, was_reused = _run_hmatc_v2_quant_case(request, services, workspace, case)
        commands.extend(case_commands)
        if error:
            failures.append(error)
        elif was_reused:
            reused += 1
        else:
            executed += 1
    return PhaseResult(
        commands=tuple(commands),
        failures=tuple(failures),
        total_cases=len(cases),
        executed_cases=executed,
        reused_cases=reused,
    )


def _run_hmatc_v2_quant_case(request, services, workspace, case):
    """Run one v2 quant case through a locked atomic artifact writer."""
    backend = request.context.diagnostic.backend
    destination = _result_artifact_directory(request, case.logical_quant_output)
    case_id = hmatc_v2_case_id(case)
    fingerprint = hmatc_v2_fingerprint(
        case,
        workspace,
        backend=backend,
        flow=ModelFlow.QUANT,
    )
    request.context.result_cache_dir.mkdir(parents=True, exist_ok=True)
    lock_file = request.context.result_cache_dir / f".{destination.name}.hmatc-v2.lock"
    with ModelResourceLock(str(lock_file), ModelResourceLock.LockMode.WRITE, "HMATC v2 quantizing"):
        if _hmatc_v2_artifact_reusable(
            request,
            services,
            destination,
            ArtifactType.QUANT_MODEL,
            case_id,
            fingerprint,
        ):
            logger.info(
                "Reusing HMATC v2 quant artifact: case=%s directory=%s",
                case_id,
                destination,
            )
            return (), None, True

        raw_report = prepare_hmatc_v2_raw_models(request, services, (case,))
        raw_commands = raw_report.commands
        if raw_report.failures:
            return raw_commands, "; ".join(raw_report.failures), False

        writer = AtomicArtifactWriter(
            destination,
            root=request.context.result_cache_dir,
            token=f"{request.context.diagnostic.run_id}-hmatc-v2-quant-{case_id}",
        )
        result = None
        try:
            with writer as staging:
                execution_override = resolve_nested_cache_paths(
                    case.override,
                    model_cache_dir=request.context.model_cache_dir,
                    result_cache_dir=request.context.result_cache_dir,
                )
                execution_override["save_dir"] = str(staging)
                effective_config = materialize_hmatc_v2_config(
                    case,
                    workspace,
                    staging,
                    flow=ModelFlow.QUANT,
                    fingerprint=fingerprint,
                    execution_override=execution_override,
                )
                command = CommandSpec(
                    name=f"hmatc-v2-quant[{case_id}]",
                    argv=(
                        "hmatc",
                        "quant",
                        "--target",
                        backend,
                        "--config",
                        str(effective_config),
                    ),
                    cwd=workspace,
                    allow_nonzero_exit=True,
                    log_file=request.context.log_file,
                )
                result = services.command_runner.run(
                    command,
                    diagnostic_fields=request.context.diagnostic.for_case(case_id, phase="hmatc-v2-quant").as_mapping(),
                )
                commands = (*raw_commands, result)
                if not result.succeeded or output_reports_failure(result.combined_output):
                    return commands, f"HMATC v2 quant case {case_id} failed", False
                _write_hmatc_v2_manifest(
                    request,
                    services,
                    staging,
                    artifact_type=ArtifactType.QUANT_MODEL,
                    case_id=case_id,
                    fingerprint=fingerprint,
                    source_type=_V2_SOURCE_TYPE_QUANT,
                    effective_config=effective_config,
                )
                writer.commit()
                return commands, None, False
        except (ArtifactValidationError, ConfigError, OSError) as error:
            commands = (*raw_commands, *((result,) if result is not None else ()))
            return commands, f"HMATC v2 quant case {case_id} failed: {error}", False


def run_hmatc_v2_build_cases(request, services, workspace: Path) -> PhaseResult:
    """Run or reuse HMATC v2 build cases with their matching quant artifacts."""
    build_cases = request.config.hmatc_v2_cases("hmbuild_params")
    quant_cases = request.config.hmatc_v2_cases("hmquant_params")
    quant_by_output = {case.logical_quant_output: case for case in quant_cases}
    commands: list[CommandResult] = []
    failures: list[str] = []
    executed = 0
    reused = 0
    for case in build_cases:
        quant_case = quant_by_output.get(case.logical_quant_output)
        if quant_case is None:
            failures.append(f"HMATC v2 build case {case.index} has no quant case for " f"{case.logical_quant_output}")
            continue
        if quant_case.config != case.config:
            failures.append(
                f"HMATC v2 build case {case.index} config {case.config} does not "
                f"match quant config {quant_case.config} for {case.logical_quant_output}"
            )
            continue
        case_commands, error, was_reused = _run_hmatc_v2_build_case(request, services, workspace, case, quant_case)
        commands.extend(case_commands)
        if error:
            failures.append(error)
        elif was_reused:
            reused += 1
        else:
            executed += 1
    return PhaseResult(
        commands=tuple(commands),
        failures=tuple(failures),
        total_cases=len(build_cases),
        executed_cases=executed,
        reused_cases=reused,
    )


def _run_hmatc_v2_build_case(request, services, workspace, case, quant_case):
    """Ensure one quant input, then atomically run and publish one v2 build."""
    quant_phase = run_hmatc_v2_quant_cases(
        request,
        services,
        workspace,
        logical_outputs=frozenset({case.logical_quant_output}),
    )
    if quant_phase.failures:
        return (
            quant_phase.commands,
            f"HMATC v2 build case {case.index} quant preparation failed: " + "; ".join(quant_phase.failures),
            False,
        )

    backend = request.context.diagnostic.backend
    quant_directory = _result_artifact_directory(request, case.logical_quant_output)
    quant_fingerprint = hmatc_v2_fingerprint(
        quant_case,
        workspace,
        backend=backend,
        flow=ModelFlow.QUANT,
    )
    build_output = case.logical_build_output
    if build_output is None:
        return quant_phase.commands, f"HMATC v2 build case {case.index} has no output", False
    destination = _result_artifact_directory(request, build_output)
    case_id = hmatc_v2_case_id(case)
    fingerprint = hmatc_v2_fingerprint(
        case,
        workspace,
        backend=backend,
        flow=ModelFlow.COMPILE,
        upstream_quant_fingerprint=quant_fingerprint,
    )
    lock_file = request.context.result_cache_dir / f".{destination.name}.hmatc-v2.lock"
    with ModelResourceLock(str(lock_file), ModelResourceLock.LockMode.WRITE, "HMATC v2 compiling"):
        if _hmatc_v2_artifact_reusable(
            request,
            services,
            destination,
            ArtifactType.COMPILED_MODEL,
            case_id,
            fingerprint,
        ):
            logger.info(
                "Reusing HMATC v2 build artifact: case=%s directory=%s",
                case_id,
                destination,
            )
            return quant_phase.commands, None, True

        writer = AtomicArtifactWriter(
            destination,
            root=request.context.result_cache_dir,
            token=f"{request.context.diagnostic.run_id}-hmatc-v2-build-{case_id}",
        )
        result = None
        try:
            with writer as staging:
                execution_override = resolve_nested_cache_paths(
                    case.override,
                    model_cache_dir=request.context.model_cache_dir,
                    result_cache_dir=request.context.result_cache_dir,
                )
                execution_override["save_dir"] = str(quant_directory)
                effective_config = materialize_hmatc_v2_config(
                    case,
                    workspace,
                    staging,
                    flow=ModelFlow.COMPILE,
                    fingerprint=fingerprint,
                    execution_override=execution_override,
                )
                environment = resolve_nested_cache_paths(
                    case.environment,
                    model_cache_dir=request.context.model_cache_dir,
                    result_cache_dir=request.context.result_cache_dir,
                )
                environment["HMATC_BUILD_OUTPUT_DIR"] = str(staging)
                command = CommandSpec(
                    name=f"hmatc-v2-build[{case_id}]",
                    argv=(
                        *hmatc_build_header(backend, asic=is_asic_platform()),
                        "--config",
                        str(effective_config),
                    ),
                    cwd=workspace,
                    environment=environment,
                    allow_nonzero_exit=True,
                    log_file=request.context.log_file,
                )
                result = services.command_runner.run(
                    command,
                    diagnostic_fields=request.context.diagnostic.for_case(case_id, phase="hmatc-v2-build").as_mapping(),
                )
                commands = (*quant_phase.commands, result)
                if not result.succeeded or output_reports_failure(result.combined_output):
                    return commands, f"HMATC v2 build case {case_id} failed", False
                copy_hmatc_v2_runtime_sidecars(
                    quant_directory / backend / "hmquant",
                    staging / backend / "hmquant",
                )
                _write_hmatc_v2_manifest(
                    request,
                    services,
                    staging,
                    artifact_type=ArtifactType.COMPILED_MODEL,
                    case_id=case_id,
                    fingerprint=fingerprint,
                    source_type=_V2_SOURCE_TYPE_BUILD,
                    effective_config=effective_config,
                )
                writer.commit()
                return commands, None, False
        except (ArtifactValidationError, ConfigError, OSError) as error:
            commands = (
                *quant_phase.commands,
                *((result,) if result is not None else ()),
            )
            return (
                commands,
                f"HMATC v2 build case {case_id} failed: {error}",
                False,
            )


def copy_hmatc_v2_runtime_sidecars(source: Path, destination: Path) -> None:
    """Copy only runtime `.pt` files and `hf_config` from quant to build."""
    if not source.is_dir():
        raise ArtifactValidationError(
            "HMATC v2 quant hmquant directory is missing",
            details={"source": source},
        )
    copied = 0
    for source_file in source.rglob("*.pt"):
        if not source_file.is_file():
            continue
        relative = source_file.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
        copied += 1
    source_hf_config = source / "hf_config"
    copied_hf_config = source_hf_config.is_dir()
    if copied_hf_config:
        shutil.copytree(
            source_hf_config,
            destination / "hf_config",
            dirs_exist_ok=True,
        )
    logger.info(
        "HMATC v2 runtime sidecars copied: source=%s destination=%s " "pt_files=%s hf_config=%s",
        source,
        destination,
        copied,
        copied_hf_config,
    )


def _result_artifact_directory(request, logical_path: str) -> Path:
    """Resolve and validate an HMATC v2 cached_results artifact root."""
    reference = cache_case_reference(logical_path)
    if reference is None or reference[0] != RESULT_CACHE_ROOT:
        raise ConfigError(f"HMATC v2 artifact path must be below cached_results: {logical_path}")
    resolved = Path(
        resolve_cached_path(
            logical_path,
            model_cache_dir=request.context.model_cache_dir,
            result_cache_dir=request.context.result_cache_dir,
        )
    ).resolve()
    root = request.context.result_cache_dir.resolve()
    if resolved == root or root not in resolved.parents:
        raise ConfigError(f"HMATC v2 artifact escapes result cache: {logical_path} -> {resolved}")
    return resolved


def _hmatc_v2_artifact_reusable(
    request,
    services,
    directory,
    artifact_type,
    case_id,
    fingerprint,
) -> bool:
    """Return whether a typed v2 manifest matches this logical case."""
    inspection = services.artifact_cache.inspect(
        directory,
        ArtifactRequirement(
            artifact_type,
            request.config.model_name,
            request.context.diagnostic.backend,
            case_id,
            required_roles=("effective_config",),
        ),
        expected_fingerprint=fingerprint,
    )
    if inspection.status != CacheStatus.VALID or inspection.manifest is None:
        return False
    expected_source = _V2_SOURCE_TYPE_QUANT if artifact_type == ArtifactType.QUANT_MODEL else _V2_SOURCE_TYPE_BUILD
    return inspection.manifest.source_type == expected_source


def _write_hmatc_v2_manifest(
    request,
    services,
    directory,
    *,
    artifact_type,
    case_id,
    fingerprint,
    source_type,
    effective_config,
) -> None:
    """Write the v2 identity manifest without model-specific required files."""
    if not effective_config.is_file() or effective_config.stat().st_size == 0:
        raise ArtifactValidationError(
            "HMATC v2 effective config is missing after command execution",
            details={"effective_config": effective_config},
        )
    services.artifact_cache.write_manifest(
        directory,
        ArtifactManifest(
            schema_version=1,
            fingerprint_version=1,
            artifact_type=artifact_type.value,
            model_name=request.config.model_name,
            model_family=request.config.family.value,
            backend=request.context.diagnostic.backend,
            case_id=case_id,
            producer_flow=(
                ModelFlow.QUANT.value if artifact_type == ArtifactType.QUANT_MODEL else ModelFlow.COMPILE.value
            ),
            source_type=source_type,
            config_fingerprint=fingerprint,
            required_files={"effective_config": str(effective_config.relative_to(directory))},
        ),
    )
