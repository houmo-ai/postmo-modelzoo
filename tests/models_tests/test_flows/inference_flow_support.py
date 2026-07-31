# Copyright (c) 2025 HOUMO AI
#
# File: inference_flow_support.py
# Description:
#  Shared Workspace, Artifact, Virtualenv, and Command Support for Inference Flows.
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

"""Common artifact, skip, mapping, and result helpers for inference flows."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Mapping

from ...tests_utils.resource_lock import ModelResourceLock
from ..model_workflow.artifact_cache_store import (
    ArtifactManifest,
    ArtifactRequirement,
    ArtifactType,
    AtomicArtifactWriter,
    CacheStatus,
    calculate_config_fingerprint,
)
from ..model_workflow.artifact_file_scanner import (
    build_required_file_roles,
    find_nonempty_hmm_files,
)
from ..model_workflow.artifact_publication import publish_compiled_artifact
from ..model_workflow.cache_path_resolver import (
    MODEL_CACHE_ROOT,
    cache_case_reference,
    get_model_case_artifact_id,
    resolve_case_paths,
)
from ..model_workflow.parameter_matrix import ParameterCase, ParameterMatrix
from ..model_workflow.flow_contracts import (
    ArtifactValidationError,
    CommandResult,
    FlowDisposition,
    FlowRequest,
    FlowResult,
    ModelFamily,
    ModelFlow,
    ValidationResult,
)
from ..model_workflow.backend_flow_policies import FamilyFlowPolicy

logger = logging.getLogger(__name__)


__all__ = [
    "backfill_referenced_demo_artifacts",
    "best_matching_download_case",
    "common_skip_reason",
    "mirror_downloaded_hmms",
    "mirror_local_compile_outputs",
    "prepare_inference_workspace",
    "release_hmm_case_mappings",
    "unproduced_demo_artifact_refs",
    "validate_python_compiled_artifacts",
    "validated_result",
]


def common_skip_reason(request: FlowRequest, flow: ModelFlow, family: ModelFamily) -> str | None:
    """Return the common reason for skipping an unsupported flow request."""
    config = request.config
    context = request.context
    backend = context.diagnostic.backend
    if config.family != family:
        raise ArtifactValidationError(
            f"{flow.value} handler received the wrong model family",
            context=context.diagnostic,
            details={"expected": family.value, "actual": config.family.value},
        )
    if config.obsolete or not config.supports(backend, flow):
        return f"{config.model_name} does not support {flow.value} on {backend}"
    if context.platform is None:
        return f"{config.model_name} does not support the current platform"
    if flow != ModelFlow.DEMO and context.platform == "aarch64":
        return f"{flow.value} is not supported on aarch64"
    if not context.source_dir.is_dir():
        raise ArtifactValidationError(
            "Model source directory does not exist",
            context=context.diagnostic,
            details={"source_dir": context.source_dir},
        )
    return None


def prepare_inference_workspace(
    request: FlowRequest,
    services,
    workspace: Path,
    policy: FamilyFlowPolicy,
) -> tuple[list[CommandResult], list[str]]:
    """Compatibility wrapper around declarative inference preparation."""
    from .artifact_preparation import ensure_inference_artifacts

    report = ensure_inference_artifacts(request, services, workspace, policy)
    return list(report.commands), list(report.failures)


def validate_python_compiled_artifacts(request: FlowRequest, services) -> list[str]:
    """Validate compiled artifacts in their final inference cache.

    ``cached_models`` may be used as a producer-side location for downloaded or
    locally compiled HMMs, but Python demos consume the mirrored case directory
    under ``cached_results``.  Once that final directory is valid, inference
    must not require the producer-side directory to remain available (for
    example when no-infer and infer run on different hosts).
    """
    backend = request.context.diagnostic.backend
    params = request.config.backend_section("compile_params", backend)
    if params is None:
        return []
    matrix = ParameterMatrix.from_columns(
        params,
        location=f"{request.config.model_name}.compile_params.{backend}",
    )
    failures = []
    for case in matrix.cases:
        resolved = resolve_case_paths(
            case,
            request.context.model_cache_dir,
            request.context.result_cache_dir,
        )
        configured_output = case.values.get("output_dir")
        resolved_output = resolved.values.get("output_dir")
        if not isinstance(configured_output, str) or not configured_output:
            failures.append(f"compile case {case.index} has no output_dir")
            continue
        if not isinstance(resolved_output, str) or not resolved_output:
            failures.append(f"compile case {case.index} has no resolved output_dir")
            continue
        directory, case_id = _inference_compiled_artifact_directory(
            request,
            configured_output,
            resolved_output,
            case.index,
        )
        inspection = services.artifact_cache.inspect(
            directory,
            ArtifactRequirement(
                ArtifactType.COMPILED_MODEL,
                request.config.model_name,
                backend,
                case_id,
            ),
        )
        if inspection.status == CacheStatus.VALID:
            continue
        if inspection.status == CacheStatus.LEGACY:
            try:
                publish_compiled_artifact(request, services, resolved, directory)
                continue
            except ArtifactValidationError:
                pass
        failures.append(
            f"compiled artifact case {case.index} is {inspection.status.value}: " f"{inspection.reason} ({directory})"
        )
    return failures


def _inference_compiled_artifact_directory(
    request: FlowRequest,
    configured_output: str,
    resolved_output: str,
    case_index: int,
) -> tuple[Path, str]:
    """Resolve the persistent directory consumed by Python inference.

    Compile outputs already configured below ``cached_results`` retain their
    exact resolved path. Outputs configured below ``cached_models`` are producer
    sources and are validated through their mirrored result-cache case.
    """
    source_directory = Path(resolved_output)
    reference = cache_case_reference(configured_output)
    case_id = source_directory.name or f"case-{case_index}"
    if reference is not None and reference[0] == MODEL_CACHE_ROOT:
        return request.context.result_cache_dir / case_id, case_id
    return source_directory, case_id


def backfill_referenced_demo_artifacts(request: FlowRequest, services) -> list[Path]:
    """Compatibility wrapper for preparer-owned demo artifact backfilling."""
    from .artifact_preparation import ArtifactPreparer

    preparer = getattr(services, "artifact_preparer", ArtifactPreparer())
    return preparer.backfill_referenced_demo_artifacts(request, services)


def unproduced_demo_artifact_refs(request: FlowRequest, backend: str) -> set[tuple[str, str]]:
    """Return demo artifact references that no compile output directory covers."""
    references = request.config.referenced_artifact_case_refs(backend)
    if not references:
        return set()
    compile_params = request.config.backend_section("compile_params", backend)
    if compile_params is None:
        return references
    produced: set[str] = set()
    for case in ParameterMatrix.from_columns(
        compile_params,
        location=f"{request.config.model_name}.compile_params.{backend}",
    ).cases:
        for key in ("output_dir", "model_dir"):
            value = case.values.get(key)
            if not isinstance(value, str):
                continue
            reference = cache_case_reference(value)
            if reference is not None:
                produced.add(reference[1])
    return {(root, case_id) for root, case_id in references if case_id not in produced}


def mirror_downloaded_hmms(request: FlowRequest, services, mappings: Mapping[str, str]) -> None:
    """Map downloaded case directories to cached_results references used by demos."""
    _mirror_model_cache_hmms(
        request,
        services,
        mappings,
        producer_flow=ModelFlow.GET_MODEL,
        source_type="download_mirror",
    )


def mirror_local_compile_outputs(request: FlowRequest, services) -> None:
    """Mirror local compile outputs while preserving the expected artifact layout."""
    backend = request.context.diagnostic.backend
    params = request.config.backend_section("compile_params", backend)
    if params is None:
        return
    mappings = {}
    for case in ParameterMatrix.from_columns(
        params,
        location=f"{request.config.model_name}.compile_params.{backend}",
    ).cases:
        output = case.values.get("output_dir")
        if not isinstance(output, str) or "cached_models" not in output:
            continue
        case_id = Path(output).name
        mappings[case_id] = case_id
    _mirror_model_cache_hmms(
        request,
        services,
        mappings,
        producer_flow=ModelFlow.COMPILE,
        source_type="local_compile_mirror",
    )


def _mirror_model_cache_hmms(
    request,
    services,
    mappings: Mapping[str, str],
    *,
    producer_flow: ModelFlow,
    source_type: str,
) -> None:
    """Mirror model cache HMM files while preserving the expected artifact layout."""
    request.context.result_cache_dir.mkdir(parents=True, exist_ok=True)
    lock_file = request.context.result_cache_dir / "lock.lock"
    with ModelResourceLock(str(lock_file), ModelResourceLock.LockMode.WRITE, "mirror downloaded HMMs"):
        for destination_id, source_id in mappings.items():
            source = request.context.model_cache_dir / source_id
            destination = request.context.result_cache_dir / destination_id
            if not source.is_dir() or source.resolve() == destination.resolve():
                continue
            token = getattr(request.context.diagnostic, "run_id", "download-mirror")
            writer = AtomicArtifactWriter(
                destination,
                root=request.context.result_cache_dir,
                token=f"{token}-mirror-{destination_id}",
            )
            with writer as staging:
                shutil.copytree(source, staging, dirs_exist_ok=True)
                _publish_hmm_mirror_manifest(
                    request,
                    services,
                    staging,
                    destination_id,
                    source_id,
                    producer_flow=producer_flow,
                    source_type=source_type,
                )
                writer.commit()


def _publish_hmm_mirror_manifest(
    request,
    services,
    destination: Path,
    destination_id: str,
    source_id: str,
    *,
    producer_flow: ModelFlow,
    source_type: str,
) -> None:
    """Publish HMM mirror manifest with ownership and manifest metadata."""
    hmm_files = find_nonempty_hmm_files(destination)
    if not hmm_files:
        raise ArtifactValidationError(
            "Downloaded HMM mirror contains no non-empty HMM files",
            context=request.context.diagnostic,
            details={"source_case_id": source_id, "destination": destination},
        )
    services.artifact_cache.write_manifest(
        destination,
        ArtifactManifest(
            schema_version=1,
            fingerprint_version=1,
            artifact_type=ArtifactType.COMPILED_MODEL.value,
            model_name=request.config.model_name,
            model_family=request.config.family.value,
            backend=request.context.diagnostic.backend,
            case_id=destination_id,
            producer_flow=producer_flow.value,
            source_type=source_type,
            config_fingerprint=calculate_config_fingerprint(
                {
                    "model": request.config.model_name,
                    "backend": request.context.diagnostic.backend,
                    "source_case_id": source_id,
                    "destination_case_id": destination_id,
                }
            ),
            required_files=build_required_file_roles(destination, hmm_files, prefix="hmm"),
        ),
    )


def release_hmm_case_mappings(request: FlowRequest) -> dict[str, str]:
    """Match downloaded HMM cases to compile output cases by stable parameters.

    Most configurations use the same directory name on both sides. A few older
    LLM configs (for example ``hmm_xh2`` vs ``hmm_xh2_2k``) do not, so matching
    only path names would make standalone release inference miss valid downloads.
    """
    backend = request.context.diagnostic.backend
    get_params = request.config.backend_section("get_model_params", backend)
    if get_params is None:
        return {}
    release = bool(getattr(request.context, "release", False))
    get_cases = [
        case
        for case in ParameterMatrix.from_columns(
            get_params,
            location=f"{request.config.model_name}.get_model_params.{backend}",
        ).cases
        if str(case.values.get("type")) == "hmm"
        and (not release or str(case.values.get("source_type") or "").lower() != "modelscope")
    ]
    source_cases = {get_model_case_artifact_id(case): case for case in get_cases if get_model_case_artifact_id(case)}
    if not source_cases:
        return {}

    compile_params = request.config.backend_section("compile_params", backend)
    if compile_params is None:
        referenced = request.config.referenced_result_case_ids(backend)
        return {case_id: case_id for case_id in referenced if case_id in source_cases}
    compile_cases = ParameterMatrix.from_columns(
        compile_params,
        location=f"{request.config.model_name}.compile_params.{backend}",
    ).cases
    mappings: dict[str, str] = {}
    for compile_case in compile_cases:
        output = compile_case.values.get("output_dir")
        if not isinstance(output, str) or not output:
            continue
        destination_id = Path(output).name
        if destination_id in source_cases:
            mappings[destination_id] = destination_id
            continue
        selected = best_matching_download_case(compile_case, source_cases)
        if selected is not None:
            mappings[destination_id] = selected
    return mappings


def best_matching_download_case(
    consumer_case: ParameterCase,
    source_cases: Mapping[str, ParameterCase],
    *,
    allow_ambiguous: bool = False,
) -> str | None:
    """Select a download case matching one artifact consumer.

    Release artifact mapping requires an unambiguous producer. Preparing a
    cached_models side-effect directory is less strict: tied HMM cases for the
    same model variant all download the same tokenizer/model resources, so the
    caller may request a stable first match.
    """
    aliases = (
        ("model_name", "model_name"),
        ("model-name", "model_name"),
        ("model_size", "model_size"),
        ("model-size", "model_size"),
        ("context_length", "context_length"),
        ("context-length", "context_length"),
        ("prefill_length", "prefill_length"),
        ("batch", "batch"),
        ("ndevice", "ndevice"),
        ("ncore", "ncore"),
    )
    candidates = []
    for source_id, source_case in source_cases.items():
        score = _matching_case_score(consumer_case, source_case, aliases)
        if score is not None:
            candidates.append((score, source_id))
    if not candidates:
        return next(iter(source_cases)) if len(source_cases) == 1 else None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score = candidates[0][0]
    best_ids = [source_id for score, source_id in candidates if score == best_score]
    if len(best_ids) > 1 and not allow_ambiguous:
        raise ArtifactValidationError(
            "Ambiguous release HMM mapping",
            details={
                "consumer_case": dict(consumer_case.values),
                "best_score": best_score,
                "candidate_case_ids": best_ids,
            },
        )
    return best_ids[0]


def _matching_case_score(consumer_case, source_case, aliases) -> int | None:
    """Return matching parameter count, or None when a value conflicts."""
    score = 0
    for consumer_key, source_key in aliases:
        compile_value = consumer_case.values.get(consumer_key)
        source_value = source_case.values.get(source_key)
        if compile_value in (None, "default") or source_value in (None, "default"):
            continue
        if str(compile_value) != str(source_value):
            return None
        score += 1
    return score


def _flow_failures(result: FlowResult) -> list[str]:
    """Collect command-level failures for a flow result."""
    if result.validation is not None and not result.validation.passed:
        return list(result.validation.failures or (result.validation.summary,))
    return []


def validated_result(
    flow: ModelFlow,
    commands,
    failures,
    message: str | None = None,
    *,
    metrics=None,
) -> FlowResult:
    """Build a completed flow result with structured validation details."""
    failures = list(failures)
    return FlowResult(
        FlowDisposition.EXECUTED,
        message or (f"{flow.value} completed" if not failures else f"{flow.value} failed"),
        commands=tuple(commands),
        validation=ValidationResult(
            not failures,
            f"{flow.value} validation passed" if not failures else "; ".join(failures),
            metrics=metrics or {},
            failures=tuple(failures),
        ),
    )
