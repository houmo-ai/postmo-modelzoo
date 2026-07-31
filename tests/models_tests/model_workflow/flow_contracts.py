# Copyright (c) 2025 HOUMO AI
#
# File: flow_contracts.py
# Description:
#  Stable Data Contracts and Errors Shared by Model-Test Flows.
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

"""Define stable request, result, diagnostic, and error contracts for flows."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from ...tests_utils.command_execution import (
    CommandExecutionError as CrossSuiteCommandExecutionError,
    CommandResult,
    CommandSpec as CrossSuiteCommandSpec,
    OutputCaptureMode as CrossSuiteOutputCaptureMode,
)
from ...tests_utils.runtime_context import TCaseType

if TYPE_CHECKING:
    from ...tests_utils.command_execution import CommandRunner
    from ...tests_utils.workspace import WorkspaceManager
    from .artifact_cache_store import ArtifactCache
    from .model_config_repository import ModelConfig


__all__ = [
    "ArtifactResolutionError",
    "ArtifactValidationError",
    "CommandExecutionError",
    "CommandResult",
    "CommandSpec",
    "ConfigError",
    "DEFAULT_COMMAND_TIMEOUT_SECONDS",
    "DiagnosticContext",
    "FlowContext",
    "FlowDisposition",
    "FlowHandler",
    "FlowRequest",
    "FlowResult",
    "FlowServicesProtocol",
    "ModelFamily",
    "ModelFlow",
    "ModelTestError",
    "OutputCaptureMode",
    "PhaseResult",
    "ResultParseError",
    "ValidationResult",
]


class ModelFamily(str, Enum):
    """Supported model families used to select family policies."""

    CV = "cv"
    LLM = "llm"


class ModelFlow(str, Enum):
    """The seven pytest-visible model test flows."""

    GET_MODEL = "get_model"
    QUANT = "quant"
    COMPILE = "compile"
    DEMO = "demo"
    COMPARE = "compare"
    EVAL = "eval"
    PERF = "perf"


class FlowDisposition(str, Enum):
    """Whether a handler executed, prepared, or skipped its work."""

    EXECUTED = "executed"
    PREPARED_ONLY = "prepared_only"
    SKIPPED = "skipped"


DEFAULT_COMMAND_TIMEOUT_SECONDS = 8 * 60 * 60
CommandExecutionError = CrossSuiteCommandExecutionError
OutputCaptureMode = CrossSuiteOutputCaptureMode


@dataclass(frozen=True)
class DiagnosticContext:
    """Stable identifiers attached to logs and structured failures."""

    run_id: str
    model_name: str
    family: ModelFamily
    backend: str
    flow: ModelFlow
    case_id: str | None = None
    phase: str | None = None

    def as_fields(self) -> str:
        """Render the diagnostic context as stable key-value fields."""
        fields = self.as_mapping()
        return " ".join(f"{key}={value}" for key, value in fields.items() if value)

    def as_mapping(self) -> dict[str, object]:
        """Return generic diagnostic fields accepted by shared infrastructure."""
        return {
            "run_id": self.run_id,
            "model": self.model_name,
            "family": self.family.value,
            "backend": self.backend,
            "flow": self.flow.value,
            "case_id": self.case_id,
            "phase": self.phase,
        }

    def for_case(self, case_id: str | int, *, phase: str) -> "DiagnosticContext":
        """Derive command-level diagnostics without mutating the flow context."""
        return replace(self, case_id=str(case_id), phase=phase)


class ModelTestError(RuntimeError):
    """Base error converted to a pytest failure only at the orchestration boundary."""

    def __init__(
        self,
        message: str,
        *,
        context: DiagnosticContext | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the Model Test Error."""
        super().__init__(message)
        self.message = message
        self.context = context
        self.details = dict(details or {})

    def format_diagnostic(self) -> str:
        """Format the error, context, and details for pytest diagnostics."""
        lines = [self.__class__.__name__, self.message]
        if self.context is not None:
            lines.append(self.context.as_fields())
        lines.extend(f"{key}: {value}" for key, value in self.details.items())
        return "\n".join(lines)


class ConfigError(ModelTestError):
    """Raised when a model configuration violates a runtime contract."""

    pass


class ArtifactResolutionError(ModelTestError):
    """Raised when a required artifact cannot be located or mapped."""

    pass


class ArtifactValidationError(ModelTestError):
    """Raised when an artifact exists but fails content or identity checks."""

    pass


class ResultParseError(ModelTestError):
    """Raised when command output cannot be parsed into expected results."""

    pass


@dataclass(frozen=True)
class FlowContext:
    """Resolved runtime paths, platform facts, and diagnostic context for a flow."""

    diagnostic: DiagnosticContext
    platform: str | None
    test_type: TCaseType
    release: bool
    log_file: Path
    source_dir: Path
    model_cache_dir: Path
    result_cache_dir: Path
    ndevice_marker: str
    device_mem_marker: str


@dataclass(frozen=True)
class FlowRequest:
    """Immutable pair of normalized model configuration and runtime context."""

    context: FlowContext
    config: ModelConfig


@dataclass(frozen=True)
class CommandSpec(CrossSuiteCommandSpec):
    """Model command specification with model-suite execution defaults.

    Set ``IMODELZOO_MIRROR_COMMAND_OUTPUT=ON`` to copy subprocess stdout and
    stderr to the pytest process while they are still captured in the per-test
    log file. Pytest must also use ``-s``/``--capture=no`` or
    ``--capture=tee-sys`` for the copied output to be visible immediately.
    """

    timeout_seconds: float | None = DEFAULT_COMMAND_TIMEOUT_SECONDS
    mirror_to_console: bool = field(
        default_factory=lambda: os.getenv("IMODELZOO_MIRROR_COMMAND_OUTPUT", "ON").casefold()
        in {"1", "on", "true", "yes"}
    )


@dataclass(frozen=True)
class ValidationResult:
    """Outcome and metrics produced by a flow validation stage."""

    passed: bool
    summary: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class FlowResult:
    """Complete structured outcome returned by a flow handler."""

    disposition: FlowDisposition
    message: str = ""
    commands: tuple[CommandResult, ...] = ()
    validation: ValidationResult | None = None
    workspace_outputs: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PhaseResult:
    """Structured accounting for a reusable command-execution phase."""

    commands: tuple[CommandResult, ...] = ()
    failures: tuple[str, ...] = ()
    total_cases: int = 0
    executed_cases: int = 0
    reused_cases: int = 0
    filtered_cases: int = 0

    @property
    def completed_cases(self) -> int:
        """Return the number of executed, reused, or filtered cases."""
        return self.executed_cases + self.reused_cases + self.filtered_cases

    @property
    def all_filtered(self) -> bool:
        """Return whether every configured case was filtered out."""
        return self.total_cases > 0 and self.filtered_cases == self.total_cases

    @property
    def has_unaccounted_cases(self) -> bool:
        """Return whether any configured case lacks a terminal outcome."""
        return self.completed_cases < self.total_cases


class FlowServicesProtocol(Protocol):
    """Services a handler may use for commands, workspaces, and artifacts."""

    command_runner: CommandRunner
    workspace_manager: WorkspaceManager
    artifact_cache: ArtifactCache


class FlowHandler(Protocol):
    """Protocol implemented by each family/backend flow handler."""

    def run(self, request: FlowRequest, services: FlowServicesProtocol) -> FlowResult:
        """Execute the flow handler and return its structured result."""
        ...
