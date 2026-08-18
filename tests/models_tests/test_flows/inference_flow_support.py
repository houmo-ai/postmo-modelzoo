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
from pathlib import Path
from typing import Mapping

from ..model_workflow.artifact_cache_store import (
    ArtifactRequirement,
    ArtifactType,
    CacheStatus,
)
from ..model_workflow.artifact_publication import publish_compiled_artifact
from ..model_workflow.cache_path_resolver import (
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
    "prepare_inference_workspace",
    "release_hmm_case_ids",
    "resolve_python_script",
    "unproduced_demo_artifact_refs",
    "validate_python_compiled_artifacts",
    "validated_result",
]


def resolve_python_script(workspace: Path, script, *, default: str) -> str:
    """Prefer a matching script below ``python/`` over the model-root script."""
    script_name = str(script) if script not in (None, "default") else default
    script_path = Path(script_name)
    if script_path.is_absolute() or ".." in script_path.parts:
        return script_name
    if script_path.parts and script_path.parts[0] == "python":
        return script_name
    python_script = Path("python") / script_path
    return python_script.as_posix() if (workspace / python_script).is_file() else script_name


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
    """Validate compiled artifacts in the JSON-declared output directories."""
    backend = request.context.diagnostic.backend
    params = request.config.backend_section("compile_params", backend)
    if params is None:
        return []
    matrix = ParameterMatrix.from_columns(
        params,
        location=f"{request.config.model_name}.compile_params.{backend}",
    )
    failures = (_python_compiled_artifact_failure(request, services, backend, case) for case in matrix.cases)
    return [failure for failure in failures if failure is not None]


def _python_compiled_artifact_failure(
    request: FlowRequest,
    services,
    backend: str,
    case: ParameterCase,
) -> str | None:
    """Validate one Python compile case and return its failure, if any."""
    configured_output = case.values.get("output_dir")
    if not isinstance(configured_output, str) or not configured_output:
        return f"compile case {case.index} has no output_dir"

    resolved = resolve_case_paths(
        case,
        request.context.model_cache_dir,
        request.context.result_cache_dir,
    )
    resolved_output = resolved.values.get("output_dir")
    if not isinstance(resolved_output, str) or not resolved_output:
        return f"compile case {case.index} has no resolved output_dir"

    directory = Path(resolved_output)
    inspection = services.artifact_cache.inspect(
        directory,
        ArtifactRequirement(
            ArtifactType.COMPILED_MODEL,
            request.config.model_name,
            backend,
            directory.name or f"case-{case.index}",
        ),
    )
    if inspection.status == CacheStatus.VALID:
        return None
    if inspection.status == CacheStatus.LEGACY and _publish_legacy_compiled_artifact(
        request,
        services,
        resolved,
        directory,
    ):
        return None
    return f"compiled artifact case {case.index} is {inspection.status.value}: " f"{inspection.reason} ({directory})"


def _publish_legacy_compiled_artifact(request, services, resolved, directory: Path) -> bool:
    """Publish one legacy artifact and report whether migration succeeded."""
    try:
        publish_compiled_artifact(request, services, resolved, directory)
    except ArtifactValidationError:
        return False
    return True


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


def release_hmm_case_ids(request: FlowRequest) -> frozenset[str]:
    """Return same-name release HMM cases referenced by demo configuration."""
    backend = request.context.diagnostic.backend
    get_params = request.config.backend_section("get_model_params", backend)
    if get_params is None:
        return frozenset()
    release = bool(getattr(request.context, "release", False))
    available = {
        case_id
        for case in ParameterMatrix.from_columns(
            get_params,
            location=f"{request.config.model_name}.get_model_params.{backend}",
        ).cases
        if str(case.values.get("type")) == "hmm"
        and (not release or str(case.values.get("source_type") or "").lower() != "modelscope")
        if (case_id := get_model_case_artifact_id(case))
    }
    referenced = {case_id for _, case_id in request.config.referenced_artifact_case_refs(backend)}
    return frozenset(available & referenced)


def best_matching_download_case(
    consumer_case: ParameterCase,
    source_cases: Mapping[str, ParameterCase],
    *,
    allow_ambiguous: bool = False,
) -> str | None:
    """Select a download case matching one artifact consumer.

    Preparing a cached_models side-effect directory may accept tied HMM cases
    for the same model variant because they download equivalent support files.
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
            "Ambiguous HMM download mapping",
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
