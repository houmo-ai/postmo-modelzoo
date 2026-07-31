# Copyright (c) 2025 HOUMO AI
#
# File: workspace.py
# Description:
#  Cross-Suite Owned Workspace Lifecycle Management.
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

from __future__ import annotations

import re
import shutil
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


class WorkspaceOwnershipError(RuntimeError):
    """Refuse unsafe cleanup of a directory not owned by this manager."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})

    def format_diagnostic(self) -> str:
        lines = [self.__class__.__name__, self.message]
        lines.extend(f"{key}: {value}" for key, value in self.details.items())
        return "\n".join(lines)


class WorkspaceHandle(AbstractContextManager[Path]):
    """Create and safely clean one source-adjacent copied workspace."""

    _SENTINEL = ".imodelzoo-workspace"

    def __init__(self, source_dir: Path, *, phase: str, root: Path | None = None) -> None:
        self.source_dir = source_dir.resolve()
        self.phase = re.sub(r"[^A-Za-z0-9_.-]+", "-", phase).strip("-_")
        if not self.phase:
            raise ValueError("workspace phase must contain a visible label")
        self.workspace_label = self.phase.lower().replace("-", "_")
        self.root = root.resolve() if root else None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"Workspace source does not exist: {self.source_dir}")
        workspace_root = self.root or self.source_dir.parent
        workspace_root.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.source_dir.name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"{safe_name}_{self.workspace_label}_{timestamp}"
        path = workspace_root / base_name
        collision_index = 0
        while True:
            try:
                path.mkdir()
                break
            except FileExistsError:
                collision_index += 1
                path = workspace_root / f"{base_name}_{collision_index:02d}"

        sentinel = path / self._SENTINEL
        sentinel.write_text("owned\n", encoding="utf-8")
        try:
            shutil.copytree(self.source_dir, path, dirs_exist_ok=True)
            sentinel.write_text("owned\n", encoding="utf-8")
        except Exception:
            if sentinel.is_file():
                shutil.rmtree(path)
            raise
        self.path = path
        return path

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.path is None:
            return
        path = self.path.resolve()
        sentinel = path / self._SENTINEL
        allowed_root = (self.root or self.source_dir.parent).resolve()
        if allowed_root not in path.parents or not sentinel.is_file():
            ownership_error = WorkspaceOwnershipError(
                "Refusing to delete an unowned workspace",
                details={"workspace": path, "allowed_root": allowed_root},
            )
            if exc is not None:
                if hasattr(exc, "add_note"):
                    exc.add_note(ownership_error.format_diagnostic())
                return None
            raise ownership_error
        shutil.rmtree(path)


class WorkspaceManager:
    """Open source-adjacent workspaces without changing process cwd."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root

    def open(self, source_dir: Path, *, phase: str) -> WorkspaceHandle:
        return WorkspaceHandle(source_dir, phase=phase, root=self.root)


__all__ = ["WorkspaceHandle", "WorkspaceManager", "WorkspaceOwnershipError"]
