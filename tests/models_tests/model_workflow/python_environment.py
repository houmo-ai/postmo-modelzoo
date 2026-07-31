# Copyright (c) 2025 HOUMO AI
#
# File: python_environment.py
# Description:
#  Model Requirement Resolution over the Cross-Suite Python Environment API.
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

"""Prepare Python environments and requirements for model command execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from ...tests_utils.python_environment import (
    PythonEnvironment,
    prepare_python_environment as prepare_cross_suite_python_environment,
)


def _model_requirement_paths(
    workspace: Path,
    *,
    flow_type: str,
    other_requirements: Mapping[str, Any] | None,
) -> tuple[Path, ...]:
    """Resolve model-specific dependency policy into an ordered path list."""
    paths: list[Path] = []
    if flow_type == "quant":
        paths.append(workspace / "requirements_ptq.txt")
    paths.append(workspace / "requirements.txt")

    options = dict(other_requirements or {})
    if options.get("hm_gptq", False):
        examples_path = os.environ.get("HOUMO_EXAMPLES_PATH")
        if examples_path:
            paths.append(Path(examples_path) / "hmodel" / "gptqmodel" / "requirements.txt")
    datasets_value = os.environ.get("HOUMO_DATASETS_PATH")
    datasets_path = Path(datasets_value) if datasets_value else workspace
    if not datasets_path.is_absolute():
        datasets_path = workspace / datasets_path
    for requirement in options.get("py_reqs", []):
        candidate = Path(requirement)
        if candidate.is_absolute():
            paths.append(candidate)
            continue
        workspace_candidate = workspace / candidate
        paths.append(workspace_candidate if workspace_candidate.is_file() else datasets_path / candidate)
    return tuple(path for path in paths if path.is_file())


def prepare_python_environment(
    workspace: Path,
    log_file: Path,
    *,
    flow_type: str = "default",
    other_requirements: Mapping[str, Any] | None = None,
    activated: bool = False,
) -> PythonEnvironment:
    """Resolve model dependency policy and prepare a generic venv."""
    return prepare_cross_suite_python_environment(
        workspace,
        _model_requirement_paths(
            workspace,
            flow_type=flow_type,
            other_requirements=other_requirements,
        ),
        base_environment=os.environ,
        activated=activated,
        log_file=log_file,
    )


__all__ = [
    "prepare_python_environment",
]
