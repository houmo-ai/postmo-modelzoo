# Copyright (c) 2026 HOUMO AI
#
# File: artifact_publication.py
# Description:
#  Shared Model Artifact Publication and Reuse Contracts.
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

"""Build and publish manifests for artifacts produced by model flows."""

from __future__ import annotations

from pathlib import Path

from .artifact_cache_store import (
    ArtifactManifest,
    ArtifactRequirement,
    ArtifactType,
    CacheStatus,
    calculate_config_fingerprint,
)
from .artifact_file_scanner import (
    build_required_file_roles,
    find_nonempty_hmm_files,
)
from .flow_contracts import ArtifactValidationError, ModelFlow
from .parameter_matrix import ParameterCase


def compiled_artifact_fingerprint(request, case: ParameterCase) -> str:
    """Calculate the fingerprint used for compiled-model reuse."""
    return calculate_config_fingerprint(
        {
            "model": request.config.model_name,
            "family": request.config.family.value,
            "backend": request.context.diagnostic.backend,
            "case_index": case.index,
            "parameters": dict(case.values),
        }
    )


def publish_compiled_artifact(
    request,
    services,
    case: ParameterCase,
    directory: Path,
    *,
    case_id: str | None = None,
) -> None:
    """Publish a compiled artifact with ownership and manifest metadata."""
    hmm_files = find_nonempty_hmm_files(directory)
    if not hmm_files:
        raise ArtifactValidationError(
            "Compile command produced no non-empty HMM files",
            context=request.context.diagnostic,
            details={"directory": directory, "case": case.index},
        )
    case_id = case_id or directory.name or f"case-{case.index}"
    services.artifact_cache.write_manifest(
        directory,
        ArtifactManifest(
            schema_version=1,
            fingerprint_version=1,
            artifact_type=ArtifactType.COMPILED_MODEL.value,
            model_name=request.config.model_name,
            model_family=request.config.family.value,
            backend=request.context.diagnostic.backend,
            case_id=case_id,
            producer_flow=ModelFlow.COMPILE.value,
            source_type="local_compile",
            config_fingerprint=compiled_artifact_fingerprint(request, case),
            required_files=build_required_file_roles(
                directory, hmm_files, prefix="hmm"
            ),
        ),
    )


def compiled_artifact_reusable(
    request, services, case: ParameterCase, directory: Path
) -> bool:
    """Return whether a compiled artifact is valid or safely adoptable."""
    if not directory.is_dir():
        return False
    case_id = directory.name or f"case-{case.index}"
    inspection = services.artifact_cache.inspect(
        directory,
        ArtifactRequirement(
            ArtifactType.COMPILED_MODEL,
            request.config.model_name,
            request.context.diagnostic.backend,
            case_id,
        ),
        expected_fingerprint=compiled_artifact_fingerprint(request, case),
    )
    if inspection.status == CacheStatus.VALID:
        return True
    if inspection.status != CacheStatus.LEGACY:
        return False
    if not find_nonempty_hmm_files(directory):
        return False
    publish_compiled_artifact(request, services, case, directory)
    return True


__all__ = [
    "compiled_artifact_fingerprint",
    "compiled_artifact_reusable",
    "publish_compiled_artifact",
]
