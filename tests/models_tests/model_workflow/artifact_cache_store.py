# Copyright (c) 2025 HOUMO AI
#
# File: artifact_cache_store.py
# Description:
#  Atomic Artifact Publication, Manifests, and Cache Inspection.
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

"""Store, inspect, fingerprint, and atomically publish model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .flow_contracts import ArtifactValidationError, ConfigError


__all__ = [
    "ArtifactCache",
    "ArtifactManifest",
    "ArtifactRequirement",
    "ArtifactType",
    "AtomicArtifactWriter",
    "CacheInspection",
    "CacheStatus",
    "calculate_config_fingerprint",
    "copy_cache_contents",
]


def copy_cache_contents(source: Path, destination: Path) -> None:
    """Copy cache contents while excluding cache coordination metadata."""
    if not source.is_dir():
        return
    for entry in source.iterdir():
        if entry.name == "lock.lock" or entry.name.startswith("artifact_manifest"):
            continue
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        elif entry.is_file():
            shutil.copy2(entry, target)


class AtomicArtifactWriter(AbstractContextManager[Path]):
    """Build an artifact beside its destination and replace it only after validation.

    The previous destination remains untouched until :meth:`commit` succeeds. A
    failed command or validator therefore removes only the owned staging directory.
    """

    def __init__(
        self,
        destination: Path,
        *,
        root: Path,
        token: str,
        create_directory: bool = True,
    ) -> None:
        """Initialize the Atomic Artifact Writer."""
        self.destination = destination.resolve()
        self.root = root.resolve()
        self.create_directory = create_directory
        safe_token = re.sub(r"[^A-Za-z0-9_.-]+", "-", token) or "run"
        self.staging = self.destination.parent / (
            f".{self.destination.name}.partial-{safe_token}"
        )
        # The backup name is stable across process runs. If a process stops after
        # moving the old destination but before publishing staging, the next run
        # can therefore restore it even though its run token is different.
        self.backup = self.destination.parent / f".{self.destination.name}.backup"
        self.staging_marker = self.staging.parent / f"{self.staging.name}.owner"
        self.backup_marker = self.backup.parent / f"{self.backup.name}.owner"
        self._committed = False

    def __enter__(self) -> Path:
        """Enter the managed atomic artifact writer context."""
        self._validate_destination()
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._recover_interrupted_backup()
        self._remove_owned_temporary(self.staging)
        if self.create_directory:
            self.staging.mkdir()
        self.staging_marker.write_text("owned\n", encoding="utf-8")
        return self.staging

    def commit(self) -> Path:
        """Atomically publish the staged artifact as the destination."""
        if not self.staging.is_dir() or not self.staging_marker.is_file():
            raise ArtifactValidationError(
                "Artifact staging directory is missing or unowned",
                details={"staging": self.staging},
            )
        moved_existing = False
        try:
            if self.destination.exists():
                self.backup_marker.write_text("owned\n", encoding="utf-8")
                self.destination.replace(self.backup)
                moved_existing = True
            self.staging.replace(self.destination)
        except Exception:
            if (
                moved_existing
                and self.backup.exists()
                and not self.destination.exists()
            ):
                self.backup.replace(self.destination)
                self.backup_marker.unlink(missing_ok=True)
            raise
        self._committed = True
        self.staging_marker.unlink(missing_ok=True)
        self._remove_owned_temporary(self.backup)
        return self.destination

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Exit the writer context and clean up owned temporary resources."""
        if not self._committed:
            self._remove_owned_temporary(self.staging)

    def _validate_destination(self) -> None:
        """Validate destination and raise a structured error when invalid."""
        if self.destination == self.root or self.root not in self.destination.parents:
            raise ArtifactValidationError(
                "Artifact destination must be inside its owned cache root",
                details={"destination": self.destination, "root": self.root},
            )

    def _remove_owned_temporary(
        self, path: Path, *, require_sentinel: bool = True
    ) -> None:
        """Remove a temporary artifact only after validating cache ownership."""
        if not path.exists():
            return
        if path == self.root or self.root not in path.resolve().parents:
            raise ArtifactValidationError(
                "Refusing to remove temporary artifact outside its cache root",
                details={"temporary": path, "root": self.root},
            )
        marker = path.parent / f"{path.name}.owner"
        if require_sentinel and not marker.is_file():
            raise ArtifactValidationError(
                "Refusing to remove an unowned temporary artifact directory",
                details={"temporary": path},
            )
        if path.is_dir():
            shutil.rmtree(path)
            marker.unlink(missing_ok=True)
        else:
            raise ArtifactValidationError(
                "Temporary artifact path is not a directory",
                details={"temporary": path},
            )

    def _recover_interrupted_backup(self) -> None:
        """Recover a destination backup left by an interrupted publication."""
        if not self.backup.exists():
            self.backup_marker.unlink(missing_ok=True)
            return
        if not self.backup_marker.is_file():
            raise ArtifactValidationError(
                "Refusing to recover an unowned artifact backup",
                details={"backup": self.backup},
            )
        if self.destination.exists():
            # Publishing completed and only backup cleanup was interrupted.
            self._remove_owned_temporary(self.backup)
            return
        self.backup.replace(self.destination)
        self.backup_marker.unlink(missing_ok=True)


class ArtifactType(str, Enum):
    """Logical artifact categories tracked by manifests."""
    RAW_MODEL = "raw_model"
    QUANT_MODEL = "quant_model"
    COMPILED_MODEL = "compiled_model"


class CacheStatus(str, Enum):
    """Possible validity states returned by cache inspection."""
    VALID = "valid"
    MISSING = "missing"
    LEGACY = "legacy"
    STALE = "stale"
    CORRUPTED = "corrupted"
    IDENTITY_MISMATCH = "identity_mismatch"


@dataclass(frozen=True)
class ArtifactManifest:
    """Identity and required-file metadata persisted beside an artifact."""
    schema_version: int
    fingerprint_version: int
    artifact_type: str
    model_name: str
    model_family: str
    backend: str | None
    case_id: str
    producer_flow: str
    source_type: str
    config_fingerprint: str
    required_files: Mapping[str, str]


@dataclass(frozen=True)
class ArtifactRequirement:
    """Description of an artifact required by a consuming flow."""
    artifact_type: ArtifactType
    model_name: str
    backend: str | None
    case_id: str | None = None
    required_roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class CacheInspection:
    """Result of checking a directory against an artifact requirement."""
    status: CacheStatus
    reason: str
    manifest: ArtifactManifest | None = None


def calculate_config_fingerprint(
    payload: Mapping[str, Any], *, version: int = 1
) -> str:
    """Calculate a stable fingerprint for configuration and source inputs."""
    if version != 1:
        raise ConfigError(f"Unsupported fingerprint version: {version}")
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ArtifactCache:
    """Read and write typed manifests while preserving legacy cache support."""
    MANIFEST_NAME = "artifact_manifest.json"

    @staticmethod
    def typed_manifest_filename(artifact_type: str, case_id: str) -> str:
        """Return the typed manifest filename for an artifact type and case."""
        safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "-", case_id) or "default"
        return f"artifact_manifest.{artifact_type}.{safe_case}.json"

    def inspect(
        self,
        directory: Path,
        requirement: ArtifactRequirement,
        *,
        expected_fingerprint: str | None = None,
    ) -> CacheInspection:
        """Inspect cached artifacts and return their validation status."""
        if not directory.is_dir():
            return CacheInspection(CacheStatus.MISSING, "artifact directory is missing")
        candidates = []
        if requirement.case_id is not None:
            candidates.append(
                directory
                / self.typed_manifest_filename(
                    requirement.artifact_type.value, requirement.case_id
                )
            )
        candidates.append(directory / self.MANIFEST_NAME)
        path = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if path is None:
            return CacheInspection(CacheStatus.LEGACY, "artifact manifest is missing")
        try:
            manifest = ArtifactManifest(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError) as error:
            return CacheInspection(
                CacheStatus.CORRUPTED, f"invalid artifact manifest: {error}"
            )
        if manifest.schema_version != 1:
            return CacheInspection(
                CacheStatus.CORRUPTED,
                f"unsupported manifest schema: {manifest.schema_version}",
                manifest,
            )
        if manifest.fingerprint_version != 1:
            return CacheInspection(
                CacheStatus.STALE,
                f"unsupported fingerprint version: {manifest.fingerprint_version}",
                manifest,
            )
        mismatch = self._identity_mismatch(manifest, requirement)
        if mismatch:
            return CacheInspection(CacheStatus.IDENTITY_MISMATCH, mismatch, manifest)
        if expected_fingerprint and manifest.config_fingerprint != expected_fingerprint:
            return CacheInspection(
                CacheStatus.STALE, "config fingerprint mismatch", manifest
            )
        roles_to_validate = set(manifest.required_files) | set(
            requirement.required_roles
        )
        for role in roles_to_validate:
            relative = manifest.required_files.get(role)
            if relative is None:
                return CacheInspection(
                    CacheStatus.CORRUPTED, f"missing required role: {role}", manifest
                )
            artifact = directory / relative
            if not artifact.is_file() or artifact.stat().st_size == 0:
                return CacheInspection(
                    CacheStatus.CORRUPTED,
                    f"missing or empty artifact: {relative}",
                    manifest,
                )
        return CacheInspection(CacheStatus.VALID, "artifact is valid", manifest)

    def write_manifest(self, directory: Path, manifest: ArtifactManifest) -> Path:
        """Write artifact metadata atomically after successful publication."""
        directory.mkdir(parents=True, exist_ok=True)
        content = json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
        typed_target = directory / self.typed_manifest_filename(
            manifest.artifact_type, manifest.case_id
        )
        self._atomic_write(typed_target, content)
        # Keep the historical filename during migration. Requirement-aware readers
        # prefer the typed manifest, so multiple artifact types can share a directory.
        self._atomic_write(directory / self.MANIFEST_NAME, content)
        return typed_target

    @staticmethod
    def _atomic_write(target: Path, content: str) -> None:
        """Write text through a temporary file to avoid partial manifest state."""
        temporary = target.parent / f".{target.name}.partial-{os.getpid()}"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)

    @staticmethod
    def _identity_mismatch(
        manifest: ArtifactManifest, requirement: ArtifactRequirement
    ) -> str | None:
        """Return the first manifest identity field that fails its requirement."""
        checks = {
            "artifact_type": (manifest.artifact_type, requirement.artifact_type.value),
            "model_name": (manifest.model_name, requirement.model_name),
        }
        if requirement.backend is not None:
            checks["backend"] = (manifest.backend, requirement.backend)
        if requirement.case_id is not None:
            checks["case_id"] = (manifest.case_id, requirement.case_id)
        for name, (actual, expected) in checks.items():
            if actual != expected:
                return f"{name} mismatch: expected={expected}, actual={actual}"
        return None
