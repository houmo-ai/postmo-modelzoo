# Copyright (c) 2025 HOUMO AI
#
# File: flow_registry.py
# Description:
#  Model-Flow Handler Registration and Runtime Service Composition.
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

"""Register flow handlers and assemble their shared runtime services."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...tests_utils.command_execution import CommandRunner
from ...tests_utils.workspace import WorkspaceManager
from ..model_workflow.artifact_cache_store import ArtifactCache
from ..model_workflow.backend_flow_policies import CV_FLOW_POLICY, LLM_FLOW_POLICY
from ..model_workflow.flow_contracts import (
    ConfigError,
    FlowHandler,
    ModelFamily,
    ModelFlow,
)
from .artifact_preparation import ArtifactPreparer
from .compare_flow import CompareFlowHandler
from .compile_flow import CompileFlowHandler
from .demo_flow import DemoFlowHandler
from .eval_flow import EvalFlowHandler
from .get_model_flow import GetModelFlowHandler
from .perf_flow import PerfFlowHandler
from .quant_flow import QuantFlowHandler

RegistryKey = tuple[ModelFamily, str, ModelFlow]


__all__ = ["FLOW_REGISTRY", "FlowRegistry", "FlowServices"]


@dataclass
class FlowRegistry:
    """Resolve a handler by model family, backend and flow."""

    _handlers: dict[RegistryKey, FlowHandler] = field(default_factory=dict)

    def register(
        self,
        family: ModelFamily,
        backend: str,
        flow: ModelFlow,
        handler: FlowHandler,
    ) -> None:
        """Register one handler for a family, backend, and flow combination."""
        key = (family, backend, flow)
        if key in self._handlers:
            raise ConfigError(f"Duplicate flow handler registration: {key}")
        self._handlers[key] = handler

    def resolve(self, family: ModelFamily, backend: str, flow: ModelFlow) -> FlowHandler:
        """Resolve the handler registered for a family, backend, and flow."""
        key = (family, backend, flow)
        try:
            return self._handlers[key]
        except KeyError as error:
            raise ConfigError(f"No flow handler registered for {key}") from error

    def keys(self) -> tuple[RegistryKey, ...]:
        """Return all registered family, backend, and flow keys."""
        return tuple(self._handlers)


# CV and LLM remain separate composition blocks even though their small registry
# modules are consolidated here. Business implementations stay in the flow files.
CV_HANDLERS = {
    "xh1": {
        ModelFlow.GET_MODEL: GetModelFlowHandler(ModelFamily.CV),
        ModelFlow.QUANT: QuantFlowHandler(CV_FLOW_POLICY),
        ModelFlow.COMPILE: CompileFlowHandler(CV_FLOW_POLICY),
        ModelFlow.DEMO: DemoFlowHandler(CV_FLOW_POLICY),
        ModelFlow.COMPARE: CompareFlowHandler(),
        ModelFlow.EVAL: EvalFlowHandler(),
        ModelFlow.PERF: PerfFlowHandler(CV_FLOW_POLICY),
    },
    "xh2": {
        ModelFlow.GET_MODEL: GetModelFlowHandler(ModelFamily.CV),
        ModelFlow.QUANT: QuantFlowHandler(CV_FLOW_POLICY),
        ModelFlow.COMPILE: CompileFlowHandler(CV_FLOW_POLICY),
        ModelFlow.DEMO: DemoFlowHandler(CV_FLOW_POLICY),
        ModelFlow.COMPARE: CompareFlowHandler(),
        ModelFlow.EVAL: EvalFlowHandler(),
        ModelFlow.PERF: PerfFlowHandler(CV_FLOW_POLICY),
    },
}


LLM_HANDLERS = {
    "xh1": {
        ModelFlow.GET_MODEL: GetModelFlowHandler(ModelFamily.LLM),
        ModelFlow.QUANT: QuantFlowHandler(LLM_FLOW_POLICY),
        ModelFlow.COMPILE: CompileFlowHandler(LLM_FLOW_POLICY),
        ModelFlow.DEMO: DemoFlowHandler(LLM_FLOW_POLICY),
        ModelFlow.PERF: PerfFlowHandler(LLM_FLOW_POLICY),
    },
    "xh2": {
        ModelFlow.GET_MODEL: GetModelFlowHandler(ModelFamily.LLM),
        ModelFlow.QUANT: QuantFlowHandler(LLM_FLOW_POLICY),
        ModelFlow.COMPILE: CompileFlowHandler(LLM_FLOW_POLICY),
        ModelFlow.DEMO: DemoFlowHandler(LLM_FLOW_POLICY),
        ModelFlow.PERF: PerfFlowHandler(LLM_FLOW_POLICY),
    },
}


@dataclass(frozen=True)
class FlowServices:
    """Concrete dependencies injected into handlers for one flow execution."""

    command_runner: CommandRunner = field(default_factory=CommandRunner)
    workspace_manager: WorkspaceManager = field(default_factory=WorkspaceManager)
    artifact_cache: ArtifactCache = field(default_factory=ArtifactCache)
    artifact_preparer: ArtifactPreparer = field(default_factory=ArtifactPreparer)


def _build_registry() -> FlowRegistry:
    """Build the complete model-flow handler registry."""
    registry = FlowRegistry()
    for backend in ("xh1", "xh2"):
        for flow, handler in CV_HANDLERS[backend].items():
            registry.register(ModelFamily.CV, backend, flow, handler)
        for flow, handler in LLM_HANDLERS[backend].items():
            registry.register(ModelFamily.LLM, backend, flow, handler)
    return registry


FLOW_REGISTRY = _build_registry()
