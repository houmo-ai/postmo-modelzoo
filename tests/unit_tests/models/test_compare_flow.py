# Copyright (c) 2026 HOUMO AI
#
# File: test_compare_flow.py
# Description:
#  Unit tests for HMATC Cosine Distance table parsing and compare-threshold
#    validation.
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

"""Contract tests for structural HMATC compare-output validation."""

import pytest

from tests.models_tests.test_flows.compare_flow import _compare_output_passed

pytestmark = pytest.mark.unit


def _table(headers: str, *rows: str) -> str:
    """Build a representative HMATC Cosine Distance table."""
    body = "\n".join(f"| {row} |" for row in rows)
    return (
        "+--------------------------------------------------------+\n"
        "|                    Cosine Distance                     |\n"
        "+--------------------------------------------------------+\n"
        f"| {headers} |\n"
        "+--------------------------------------------------------+\n"
        f"{body}\n"
        "+--------------------------------------------------------+\n"
    )


def test_compare_accepts_legacy_hmquant_table() -> None:
    """Preserve validation of the legacy HMATC comparison labels."""
    output = _table(
        "name | onnx vs hmquant | onnx vs xh2 | hmquant vs xh2",
        "output0 | 0.991181 | 0.991558 | 0.999877",
    )

    assert _compare_output_passed(output, "xh2", 0.99)


def test_compare_accepts_hmonnx_hmm_table() -> None:
    """Accept renamed intermediate and final model labels without aliases."""
    output = _table(
        "name | onnx vs hmonnx | onnx vs hmm | hmonnx vs hmm",
        "output0 | 0.999932 | 0.999932 | 1.000000",
    )

    assert _compare_output_passed(output, "xh2", 0.99)


def test_compare_validates_metrics_after_column_reordering() -> None:
    """Validate comparison metrics independently of column order."""
    output = _table(
        "name | hmonnx vs hmm | onnx vs hmm | onnx vs hmonnx",
        "output0 | 0.999950 | 0.999800 | 0.999700",
    )

    assert _compare_output_passed(output, "xh2", 0.9997)


@pytest.mark.parametrize(
    "headers",
    (
        "name | onnx vs hmonnx | onnx vs hmm",
        "name | onnx vs hmonnx | hmonnx vs hmm",
        "name | onnx vs hmm | hmonnx vs hmm",
    ),
)
def test_compare_accepts_any_two_comparison_columns(headers: str) -> None:
    """Accept any two of the three model-comparison columns."""
    output = _table(headers, "output0 | 0.999932 | 1.000000")

    assert _compare_output_passed(output, "xh2", 0.99)


def test_compare_accepts_unrestricted_nonempty_output_names() -> None:
    """Treat numeric, alphabetic, underscored, and mixed names as row labels."""
    output = _table(
        "name | source vs middle | middle vs final",
        "123 | 0.99 | 1.0",
        "output | 0.99 | 1.0",
        "output_name | 0.99 | 1.0",
        "output_123 | 0.99 | 1.0",
    )

    assert _compare_output_passed(output, "xh2", 0.99)


def test_compare_requires_every_output_to_pass() -> None:
    """Fail when any comparison metric of any output is below the threshold."""
    output = _table(
        "name | onnx vs hmonnx | onnx vs hmm | hmonnx vs hmm",
        "output0 | 0.999932 | 0.999932 | 1.000000",
        "output1 | 0.999900 | 0.850000 | 0.880000",
    )

    assert not _compare_output_passed(output, "xh2", 0.90)


def test_compare_accepts_status_columns_and_scientific_notation() -> None:
    """Ignore non-metric columns and parse general finite float formats."""
    output = _table(
        "name | source vs quantized | source vs compiled | quantized vs compiled | result",
        "output0 | 9.9e-1 | 0.999 | 1 | PASS",
    )

    assert _compare_output_passed(output, "xh1", 0.99)


def test_compare_rejects_table_without_comparison_columns() -> None:
    """Fail closed when table columns do not identify comparison metrics."""
    output = _table(
        "name | first metric | second metric | third metric",
        "output0 | 1.0 | 1.0 | 1.0",
    )

    assert not _compare_output_passed(output, "xh2", 0.9)


def test_compare_parses_timestamped_ansi_table_lines() -> None:
    """Ignore command-log prefixes and ANSI suffixes around table rows."""
    output = "\n".join(
        (
            "[2026-08-04 10:00:00] \x1b[92m| Cosine Distance |\x1b[0m",
            "[2026-08-04 10:00:00] | name | source vs middle | source vs final | middle vs final |",
            "[2026-08-04 10:00:00] | output0 | 0.99 | 0.99 | 1.0 |\x1b[0m",
        )
    )

    assert _compare_output_passed(output, "xh2", 0.99)
