# Copyright (c) 2025 HOUMO AI
#
# File: framework_policies.py
# Description:
#  Code-Owned Policies and Ordering Rules for Model-Test Flows.
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

"""Centralize code-owned flow order, backend rules, and family policies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, TypeVar

from .flow_contracts import ModelFamily, ModelFlow


__all__ = [
    "BACKEND_POLICIES",
    "CV_FLOW_POLICY",
    "FLOW_DEPENDENCY_RULES",
    "FLOW_ORDER",
    "GET_MODEL_COMMAND_TIMEOUT_SECONDS",
    "LLM_FLOW_POLICY",
    "PARTIAL_DATASET_THRESHOLD_FACTOR",
    "BackendPolicy",
    "FamilyFlowPolicy",
    "allow_xh2_ncore4",
    "filter_xh2_ncore4",
    "hmatc_build_header",
    "release_source_allowed",
    "should_check_output_failure",
]


PARTIAL_DATASET_THRESHOLD_FACTOR = 0.5
GET_MODEL_COMMAND_TIMEOUT_SECONDS = 4 * 60 * 60

FLOW_ORDER = {
    ModelFlow.GET_MODEL: 0,
    ModelFlow.QUANT: 1,
    ModelFlow.COMPILE: 2,
    ModelFlow.DEMO: 3,
    ModelFlow.COMPARE: 4,
    ModelFlow.EVAL: 5,
    ModelFlow.PERF: 6,
}

FLOW_DEPENDENCY_RULES = {
    ModelFlow.GET_MODEL: (),
    ModelFlow.QUANT: (ModelFlow.GET_MODEL,),
    ModelFlow.COMPILE: (ModelFlow.QUANT, ModelFlow.GET_MODEL),
    ModelFlow.DEMO: (),
    ModelFlow.COMPARE: (),
    ModelFlow.EVAL: (),
    ModelFlow.PERF: (),
}


# A small number of legacy demos print words such as ``Fail`` as part of normal
# diagnostic output. Keep these exceptions in framework code instead of growing
# a model-JSON execution DSL.
OUTPUT_FAILURE_CHECK_OVERRIDES = {
    ("qwen2.5-vl", ModelFlow.DEMO): False,
    ("qwen2.5-vl", ModelFlow.PERF): False,
}


@dataclass(frozen=True)
class BackendPolicy:
    """Backend defaults for thresholds and platform-specific command flags."""
    compile_cosine_threshold: float
    compare_cosine_threshold: float
    add_skip_check_on_non_asic: bool = True
    filter_ncore_4_by_default: bool = False


BACKEND_POLICIES = {
    "xh1": BackendPolicy(0.99, 1.0),
    "xh2": BackendPolicy(0.90, 0.90, filter_ncore_4_by_default=True),
}


@dataclass(frozen=True)
class FamilyFlowPolicy:
    """Family-specific behavior shared by all handlers of a model family."""
    family: ModelFamily
    quant_requires_gpu: bool
    copy_raw_to_workspace: bool
    compile_skip_in_release: bool
    persist_workspace_for_separate: bool


CV_FLOW_POLICY = FamilyFlowPolicy(
    family=ModelFamily.CV,
    quant_requires_gpu=False,
    copy_raw_to_workspace=True,
    compile_skip_in_release=False,
    persist_workspace_for_separate=True,
)

LLM_FLOW_POLICY = FamilyFlowPolicy(
    family=ModelFamily.LLM,
    quant_requires_gpu=True,
    copy_raw_to_workspace=False,
    compile_skip_in_release=True,
    persist_workspace_for_separate=False,
)


def allow_xh2_ncore4() -> bool:
    """Return whether the model may run with four XH2 cores."""
    return os.getenv("IMODELZOO_ALLOW_XH2_NCORE4", "OFF").upper() == "ON"


T = TypeVar("T")


def filter_xh2_ncore4(cases: Iterable[T], get_ncore) -> tuple[T, ...]:
    """Filter unsupported four-core XH2 parameter cases."""
    if allow_xh2_ncore4():
        return tuple(cases)
    return tuple(case for case in cases if str(get_ncore(case)) != "4")


def release_source_allowed(file_type: str, source_type: str | None) -> bool:
    """Return whether a release source is allowed for the requested flow."""
    return file_type != "raw" and source_type != "modelscope"


def hmatc_build_header(backend: str, *, asic: bool) -> tuple[str, ...]:
    """Build the backend-specific HMATC compilation command prefix."""
    header = ["hmatc", "build", "--target", backend]
    if BACKEND_POLICIES[backend].add_skip_check_on_non_asic and not asic:
        header.append("--skip_check")
    return tuple(header)


def should_check_output_failure(model_name: str, flow: ModelFlow) -> bool:
    """Return whether textual failure markers must be checked."""
    return OUTPUT_FAILURE_CHECK_OVERRIDES.get((model_name, flow), True)
