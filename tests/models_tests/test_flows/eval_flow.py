# Copyright (c) 2025 HOUMO AI
#
# File: eval_flow.py
# Description:
#  Model Evaluation Flow and ONNX-to-HM Metric Validation.
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

"""Run HMATC evaluation and compare HM metrics with ONNX baselines."""

from __future__ import annotations

import os
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
from ..model_workflow.backend_flow_policies import (
    CV_FLOW_POLICY,
    PARTIAL_DATASET_THRESHOLD_FACTOR,
    FamilyFlowPolicy,
)
from .artifact_preparation import ensure_inference_artifacts
from .inference_flow_support import (
    common_skip_reason,
    validated_result,
)
from .hmatc_flow_support import persist_separate_workspace, run_hmatc_cases

__all__ = ["EvalFlowHandler"]


@dataclass(frozen=True)
class EvalFlowHandler:
    """Execute eval cases and apply configured relative metric thresholds."""

    policy: FamilyFlowPolicy = CV_FLOW_POLICY

    def run(self, request: FlowRequest, services) -> FlowResult:
        """Execute the eval flow handler and return its structured result."""
        skip = common_skip_reason(request, ModelFlow.EVAL, ModelFamily.CV)
        if skip:
            return FlowResult(FlowDisposition.SKIPPED, skip)
        if not request.config.has_section("hmeval_params"):
            return FlowResult(FlowDisposition.SKIPPED, "missing hmeval_params")
        thresholds = request.config.eval_thresholds()
        if not thresholds:
            return FlowResult(FlowDisposition.SKIPPED, "missing eval_threshold")

        commands: list[CommandResult] = []
        failures: list[str] = []
        metrics: dict[str, float] = {}
        with services.workspace_manager.open(request.context.source_dir, phase="eval") as workspace:
            preparation = ensure_inference_artifacts(request, services, workspace, self.policy)
            prep_failures = list(preparation.failures)
            commands.extend(preparation.commands)
            failures.extend(prep_failures)
            if failures:
                return validated_result(ModelFlow.EVAL, commands, failures, "eval preparation failed")
            if request.context.test_type == TCaseType.SEPARATE_NO_INFER:
                persist_separate_workspace(request, workspace, self.policy)
                return FlowResult(
                    FlowDisposition.PREPARED_ONLY,
                    "eval artifacts prepared for separate infer",
                    commands=tuple(commands),
                )

            python = prepare_python_environment(workspace, request.context.log_file, activated=True)
            onnx_results, onnx_failures = run_hmatc_cases(
                request,
                services,
                workspace,
                section_name="hmeval_params",
                subcommand="eval",
                extra_argv=("--onnx",),
                environment=python.environment,
            )
            hm_results, hm_failures = run_hmatc_cases(
                request,
                services,
                workspace,
                section_name="hmeval_params",
                subcommand="eval",
                environment=python.environment,
            )
            commands.extend(onnx_results)
            commands.extend(hm_results)
            failures.extend(onnx_failures)
            failures.extend(hm_failures)
            parsed, metric_failures = _validate_eval_outputs(onnx_results, hm_results, thresholds)
            metrics.update(parsed)
            failures.extend(metric_failures)

        return validated_result(ModelFlow.EVAL, commands, failures, metrics=metrics)


def _parse_eval_metrics(output: str, names) -> dict[str, float]:
    """Parse eval metrics from normalized command output."""
    result = {}
    for name in names:
        pattern = re.compile(rf"{re.escape(str(name))}':\s*'?([^'\]\s,}}]+)'?")
        values = [match.group(1) for match in pattern.finditer(output)]
        if values:
            try:
                result[str(name)] = float(values[-1])
            except ValueError:
                pass
    return result


def _validate_eval_outputs(onnx_results, hm_results, thresholds):
    """Validate eval outputs and raise a structured error when invalid."""
    failures = []
    metrics = {}
    names = tuple(str(name) for name in thresholds)
    if len(onnx_results) != len(hm_results):
        failures.append(f"eval case count mismatch: onnx={len(onnx_results)} hm={len(hm_results)}")
    onnx_parsed = [_parse_eval_metrics(result.stdout, names) for result in onnx_results]
    hm_parsed = [_parse_eval_metrics(result.stdout, names) for result in hm_results]
    # Iterate over the longer side so unpaired cases on either side are reported
    # individually, rather than being silently dropped by zip() truncation.
    max_cases = max(len(onnx_parsed), len(hm_parsed))
    for index in range(max_cases):
        onnx = onnx_parsed[index] if index < len(onnx_parsed) else None
        hm = hm_parsed[index] if index < len(hm_parsed) else None
        case_metrics, case_failures = _validate_eval_case(
            onnx, hm, names, thresholds, index
        )
        metrics.update(case_metrics)
        failures.extend(case_failures)
    return metrics, failures


def _validate_eval_case(onnx, hm, names, thresholds, index):
    """Validate one paired ONNX/HM evaluation case."""
    if onnx is None:
        return {}, [f"HM eval case {index} has no matching ONNX case"]
    if hm is None:
        return {}, [f"ONNX eval case {index} has no matching HM case"]
    metrics = {}
    failures = []
    for name in names:
        if name not in onnx or name not in hm:
            side = "ONNX" if name not in onnx else "HM"
            failures.append(f"{side} eval case {index} missing metric {name}")
            continue
        threshold = float(thresholds[name])
        if os.getenv("HOUMO_FULL_DATASET") is None:
            threshold *= PARTIAL_DATASET_THRESHOLD_FACTOR
        metrics[f"{name}.onnx.{index}"] = onnx[name]
        metrics[f"{name}.hm.{index}"] = hm[name]
        if hm[name] < threshold * onnx[name]:
            failures.append(
                f"eval {name} case {index}: hm={hm[name]} is below "
                f"onnx={onnx[name]} * threshold={threshold}"
            )
    return metrics, failures
