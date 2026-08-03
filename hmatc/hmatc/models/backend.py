# Copyright 2025 HOUMO AI
#
# File: backend.py
# Description:
#   Backend resolution logic for HMATC models.
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
from pathlib import Path
from typing import Set

_BACKENDS = {"auto", "raw", "hmonnx", "hmm"}


def resolve_backend(model_path: str, requested_backend: str) -> str:
    backend = requested_backend.strip().lower()
    if backend not in _BACKENDS:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"Unsupported backend {requested_backend!r}. Supported backends: {supported}"
        )
    if backend != "auto":
        return backend

    path = Path(model_path)
    if not path.exists():
        raise ValueError(
            f"Cannot detect backend because model path does not exist: {path}"
        )

    detected = _detect_backends(path)
    if not detected:
        raise ValueError(f"Cannot detect a known backend from model path: {path}")
    if len(detected) > 1:
        formats = ", ".join(sorted(detected))
        raise ValueError(
            f"Backend detection is ambiguous for {path}: detected {formats}. "
            "Specify --backend explicitly."
        )
    return detected.pop()


def _detect_backends(path: Path) -> Set[str]:
    if path.is_file():
        return _detect_file_backend(path)

    detected = set()
    for child in path.iterdir():
        detected.update(_detect_file_backend(child))
    if (path / "config.json").is_file():
        detected.add("raw")
    return detected


def _detect_file_backend(path: Path) -> Set[str]:
    suffix = path.suffix.lower()
    if suffix in {".hmm", ".hmms"}:
        return {"hmm"}
    if suffix == ".onnx":
        return {"hmonnx"}
    return set()
