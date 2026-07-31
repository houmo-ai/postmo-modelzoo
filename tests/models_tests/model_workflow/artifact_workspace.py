# Copyright (c) 2025 HOUMO AI
#
# File: artifact_workspace.py
# Description:
#  Model Artifact Side-Effect Snapshot, Persistence, and Restoration.
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

"""Snapshot and restore files produced in separate-infer workspaces."""

from __future__ import annotations

import shutil
from pathlib import Path

from ...tests_utils.workspace import WorkspaceHandle


def snapshot_workspace_files(workspace: Path) -> dict[Path, tuple[int, int]]:
    """Capture file metadata used to identify model command side effects."""
    snapshot: dict[Path, tuple[int, int]] = {}
    for path in workspace.rglob("*"):
        if not path.is_file() or path.name == WorkspaceHandle._SENTINEL:
            continue
        stat = path.stat()
        snapshot[path.relative_to(workspace)] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def persist_workspace_outputs(
    workspace: Path,
    model_cache_dir: Path,
    before: dict[Path, tuple[int, int]],
) -> tuple[Path, ...]:
    """Persist model files created or changed by a command."""
    workspace = workspace.resolve()
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    persisted: list[Path] = []
    for path in workspace.rglob("*"):
        if not path.is_file() or path.name == WorkspaceHandle._SENTINEL:
            continue
        relative = path.relative_to(workspace)
        stat = path.stat()
        if before.get(relative) == (stat.st_size, stat.st_mtime_ns):
            continue
        target = model_cache_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        persisted.append(relative)
    return tuple(persisted)


def restore_workspace_outputs(
    model_cache_dir: Path,
    workspace: Path,
    outputs: tuple[Path, ...],
) -> None:
    """Restore explicitly recorded model command side-effect files."""
    model_cache_dir = model_cache_dir.resolve()
    workspace = workspace.resolve()
    for relative in outputs:
        source = (model_cache_dir / relative).resolve()
        if model_cache_dir not in source.parents or not source.is_file():
            continue
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


__all__ = [
    "persist_workspace_outputs",
    "restore_workspace_outputs",
    "snapshot_workspace_files",
]
