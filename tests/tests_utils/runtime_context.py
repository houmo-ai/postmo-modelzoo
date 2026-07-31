# Copyright (c) 2025 HOUMO AI
#
# File: runtime_context.py
# Description:
#  Explicit Cross-Suite Runtime Settings and Test-Stage Resolution.
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

import os
import platform
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import ClassVar, Mapping

from .platform_device import is_asic_platform


DEFAULT_BACKEND = "xh2"


@unique
class TCaseType(Enum):
    DEFAULT = 0
    SEPARATE_NO_INFER = 1
    SEPARATE_INFER = 2


def resolve_test_type(separate_test: str | None, *, is_asic: bool) -> TCaseType:
    """Preserve legacy SKIP_INFER plus ASIC stage selection."""
    if separate_test in {"ON", "OFF"}:
        return TCaseType.SEPARATE_INFER if is_asic else TCaseType.SEPARATE_NO_INFER
    return TCaseType.DEFAULT


@dataclass(frozen=True)
class TestRuntimeContext:
    """Hold one immutable snapshot of environment and platform facts."""

    __test__: ClassVar[bool] = False

    backend: str
    test_type: TCaseType
    release: bool
    models_path: Path
    results_path: Path
    examples_path: Path
    datasets_path: Path
    is_asic: bool
    host_platform: str
    environment: Mapping[str, str]

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        tests_root: Path | None = None,
        asic: bool | None = None,
        host_platform: str | None = None,
    ) -> "TestRuntimeContext":
        uses_process_environment = environment is None
        env = dict(os.environ if uses_process_environment else environment)
        backend = env.setdefault("HOUMO_TARGET", DEFAULT_BACKEND)
        if uses_process_environment:
            # CommandRunner starts subprocesses from os.environ. Keep the
            # process environment aligned with the resolved runtime backend so
            # model scripts see the same default selected by the test suite.
            os.environ.setdefault("HOUMO_TARGET", backend)
        resolved_tests_root = tests_root.resolve() if tests_root is not None else Path(__file__).resolve().parents[1]
        resolved_asic = is_asic_platform() if asic is None else asic
        models_path = Path(
            env.get(
                "IMODELZOO_MODELS_PATH",
                str(resolved_tests_root / f"models_{backend}"),
            )
        ).resolve()
        results_path = (resolved_tests_root / f"model_results_{backend}").resolve()
        examples_path = Path(env.get("HOUMO_EXAMPLES_PATH", str(resolved_tests_root.parent))).resolve()
        datasets_path = Path(
            env.get(
                "HOUMO_DATASETS_PATH",
                str(resolved_tests_root.parent / "data" / "datasets"),
            )
        ).resolve()
        release_value = env.get("USE_RELEASED_MODELS", "ON")
        return cls(
            backend=backend,
            test_type=resolve_test_type(env.get("SKIP_INFER"), is_asic=resolved_asic),
            release=release_value in {"on", "ON"},
            models_path=models_path,
            results_path=results_path,
            examples_path=examples_path,
            datasets_path=datasets_path,
            is_asic=resolved_asic,
            host_platform=host_platform or platform.machine(),
            environment=env,
        )


__all__ = [
    "DEFAULT_BACKEND",
    "TCaseType",
    "TestRuntimeContext",
    "resolve_test_type",
]
