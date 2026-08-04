# Copyright (c) 2025 HOUMO AI
#
# File: compare_flow.py
# Description:
#  Model Comparison Flow and Backend Cosine Validation.
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

"""Run HMATC comparison cases and validate cosine similarity results."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass

from ...tests_utils.runtime_context import TCaseType
from ..model_workflow.python_environment import prepare_python_environment
from ..model_workflow.flow_contracts import (
    CommandResult,
    FlowDisposition,
    FlowRequest,
    FlowResult,
    ModelFamily,
    ModelFlow,
)
from ..model_workflow.backend_flow_policies import CV_FLOW_POLICY, FamilyFlowPolicy
from .artifact_preparation import ensure_inference_artifacts
from .inference_flow_support import (
    common_skip_reason,
    validated_result,
)
from .hmatc_flow_support import persist_separate_workspace, run_hmatc_cases

__all__ = ["CompareFlowHandler"]


logger = logging.getLogger(__name__)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class _CompareTable:
    """Hold one parsed HMATC Cosine Distance table."""

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class CompareFlowHandler:
    """Execute the compare command after inference artifacts are prepared."""

    policy: FamilyFlowPolicy = CV_FLOW_POLICY

    def run(self, request: FlowRequest, services) -> FlowResult:
        """Execute the compare flow handler and return its structured result."""
        skip = common_skip_reason(request, ModelFlow.COMPARE, ModelFamily.CV)
        if skip:
            return FlowResult(FlowDisposition.SKIPPED, skip)
        if request.context.diagnostic.backend == "xh1" and (request.context.test_type != TCaseType.DEFAULT):
            return FlowResult(FlowDisposition.SKIPPED, "xh1 compare does not support separate mode")
        if not request.config.has_section("hmcompare_params"):
            return FlowResult(FlowDisposition.SKIPPED, "missing hmcompare_params")

        commands: list[CommandResult] = []
        failures: list[str] = []
        with services.workspace_manager.open(request.context.source_dir, phase="compare") as workspace:
            preparation = ensure_inference_artifacts(request, services, workspace, self.policy)
            prep_failures = list(preparation.failures)
            commands.extend(preparation.commands)
            failures.extend(prep_failures)
            if failures:
                return validated_result(ModelFlow.COMPARE, commands, failures, "compare preparation failed")
            if request.context.test_type == TCaseType.SEPARATE_NO_INFER:
                persist_separate_workspace(request, workspace, self.policy)
                return FlowResult(
                    FlowDisposition.PREPARED_ONLY,
                    "compare artifacts prepared for separate infer",
                    commands=tuple(commands),
                )
            threshold = float(request.config.validation_threshold(request.context.diagnostic.backend, "compare").value)
            python = prepare_python_environment(workspace, request.context.log_file, activated=True)
            results, command_failures = run_hmatc_cases(
                request,
                services,
                workspace,
                section_name="hmcompare_params",
                subcommand="compare",
                output_validator=lambda output: _compare_output_passed(
                    output, request.context.diagnostic.backend, threshold
                ),
                validation_description=f"compare cosine threshold {threshold}",
                environment=python.environment,
            )
            commands.extend(results)
            failures.extend(command_failures)
        return validated_result(ModelFlow.COMPARE, commands, failures)


def _compare_output_passed(output: str, backend: str, threshold: float) -> bool:
    """Return whether every comparison metric in the table passes."""
    del backend  # Table structure, rather than backend-specific labels, selects the metric.
    table = _parse_cosine_distance_table(output)
    if table is None:
        logger.error("HMATC compare output has no valid Cosine Distance table")
        return False
    metric_indexes = _comparison_column_indexes(table.headers)
    if len(metric_indexes) < 2:
        logger.error(
            "HMATC compare table must contain at least two 'X vs Y' columns: headers=%s",
            table.headers,
        )
        return False
    for row in table.rows:
        for metric_index in metric_indexes:
            value = _parse_finite_float(row[metric_index])
            if value is None:
                logger.error(
                    "HMATC compare metric is not numeric: output=%s header=%s value=%s",
                    row[0],
                    table.headers[metric_index],
                    row[metric_index],
                )
                return False
            if value < threshold:
                logger.error(
                    "HMATC compare metric is below threshold: output=%s header=%s actual=%s threshold=%s",
                    row[0],
                    table.headers[metric_index],
                    value,
                    threshold,
                )
                return False
    return True


def _parse_cosine_distance_table(output: str) -> _CompareTable | None:
    """Parse the first populated ASCII table under the Cosine Distance title."""
    table_lines = iter(_table_cells(line) for line in output.splitlines())
    if not _advance_to_cosine_distance_section(table_lines):
        return None
    headers = _next_compare_header(table_lines)
    if headers is None:
        return None
    rows = _collect_compare_rows(table_lines, len(headers))
    return _CompareTable(headers, rows) if rows else None


def _advance_to_cosine_distance_section(table_lines: Iterator[tuple[str, ...]]) -> bool:
    """Consume table lines through the first Cosine Distance title."""
    return any(len(cells) == 1 and cells[0].casefold() == "cosine distance" for cells in table_lines)


def _next_compare_header(table_lines: Iterator[tuple[str, ...]]) -> tuple[str, ...] | None:
    """Return the first valid compare header after the section title."""
    return next(
        (cells for cells in table_lines if len(cells) >= 2 and cells[0].casefold() == "name"),
        None,
    )


def _collect_compare_rows(
    table_lines: Iterator[tuple[str, ...]],
    column_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Collect contiguous data rows matching the compare header width."""
    rows: list[tuple[str, ...]] = []
    for cells in table_lines:
        if len(cells) == column_count and _is_compare_data_row(cells):
            rows.append(cells)
            continue
        if rows and cells:
            break
    return tuple(rows)


def _table_cells(line: str) -> tuple[str, ...]:
    """Extract pipe-delimited cells while ignoring log prefixes and ANSI codes."""
    clean = _ANSI_ESCAPE_RE.sub("", line)
    start = clean.find("|")
    end = clean.rfind("|")
    if start < 0 or end <= start:
        return ()
    return tuple(cell.strip() for cell in clean[start + 1 : end].split("|"))


def _is_compare_data_row(cells: tuple[str, ...]) -> bool:
    """Return whether a table row has a name and at least one numeric metric."""
    return bool(cells[0]) and any(_parse_finite_float(cell) is not None for cell in cells[1:])


def _comparison_column_indexes(headers: tuple[str, ...]) -> tuple[int, ...]:
    """Return all structurally identified ``X vs Y`` metric columns."""
    return tuple(index for index, header in enumerate(headers[1:], start=1) if _comparison_pair(header) is not None)


def _comparison_pair(header: str) -> tuple[str, str] | None:
    """Normalize one comparison header into its two model-role labels."""
    parts = re.split(r"\s+vs\s+", header.strip(), maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2 or not all(parts):
        return None
    return tuple(part.casefold() for part in parts)


def _parse_finite_float(value: str) -> float | None:
    """Parse one finite floating-point table cell."""
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None
