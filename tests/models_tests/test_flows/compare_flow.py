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

import re
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
    """Return whether comparison output satisfies the backend threshold."""
    if backend == "xh2":
        pattern = re.compile(r"\|\s*([^|]+?)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|$")
    else:
        pattern = re.compile(r"\|\s*([^|]+?)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\w+)\s*\|$")
    header_seen = False
    rows = []
    for line in output.splitlines():
        stripped = line.strip()
        if "onnx vs hmquant" in stripped:
            header_seen = True
            continue
        match = pattern.match(stripped)
        if match and header_seen:
            rows.append(match.groups())
    return bool(rows) and all(float(row[3]) >= threshold for row in rows)
