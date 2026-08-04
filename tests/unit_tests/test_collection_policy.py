# Copyright (c) 2026 HOUMO AI
#
# File: test_collection_policy.py
# Description:
#  Unit tests for the root pytest collection policy that keeps framework unit
#    tests out of implicit recursion while allowing explicit marker or path use.
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

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.conftest import _unit_tests_requested

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _InvocationParams:
    dir: Path


class _Config:
    def __init__(self, root: Path, *, markexpr: str = "", args: tuple[str, ...] = ()):
        self._markexpr = markexpr
        self.args = args
        self.invocation_params = _InvocationParams(root)

    def getoption(self, name: str, *, default: str = "") -> str:
        assert name == "markexpr"
        return self._markexpr or default


@pytest.mark.parametrize(
    "expression",
    (
        "unit",
        "unit and not slow",
        "unit or compare",
        "compare and (unit or slow)",
    ),
)
def test_unit_tests_requested_accepts_marker_expressions(
    tmp_path: Path,
    expression: str,
) -> None:
    assert _unit_tests_requested(_Config(tmp_path, markexpr=expression))


@pytest.mark.parametrize("expression", ("", "compare", "unit_extra", "preunit"))
def test_unit_tests_requested_rejects_unrelated_marker_expressions(
    tmp_path: Path,
    expression: str,
) -> None:
    assert not _unit_tests_requested(_Config(tmp_path, markexpr=expression))


def test_unit_tests_requested_accepts_explicit_unit_test_path() -> None:
    unit_tests_path = Path(__file__).resolve().parent

    assert _unit_tests_requested(_Config(unit_tests_path.parent, args=(str(unit_tests_path),)))
