# Copyright (c) 2025 HOUMO AI
#
# File: get_model_flow.py
# Description:
#  Model Acquisition Flow and Family-Specific Release Filtering.
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

"""Download model inputs and publish validated raw, quant, or HMM artifacts."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ...tests_utils.resource_lock import ModelResourceLock
from ...tests_utils.runtime_context import TCaseType
from ..model_workflow.artifact_cache_store import (
    ArtifactManifest,
    ArtifactType,
    calculate_config_fingerprint,
)
from ..model_workflow.artifact_file_scanner import (
    build_required_file_roles,
    find_nonempty_artifact_files,
    find_nonempty_hmm_files,
)
from ..model_workflow.artifact_workspace import (
    persist_workspace_outputs,
    snapshot_workspace_files,
)
from ..model_workflow.cache_path_resolver import (
    cache_case_reference,
    get_model_case_artifact_id,
    resolve_case_paths,
)
from ..model_workflow.flow_contracts import (
    ArtifactValidationError,
    CommandSpec,
    FlowDisposition,
    FlowRequest,
    FlowResult,
    ModelFamily,
    ModelFlow,
    ValidationResult,
)
from ..model_workflow.backend_flow_policies import (
    GET_MODEL_COMMAND_TIMEOUT_SECONDS,
    release_source_allowed,
)
from ..model_workflow.parameter_matrix import (
    ParameterCase,
    ParameterMatrix,
    render_case_options,
)

logger = logging.getLogger(__name__)


__all__ = ["GetModelFlowHandler"]


@dataclass(frozen=True)
class GetModelFlowHandler:
    """Run model download cases while isolating family-specific source rules."""

    family: ModelFamily
    file_types: frozenset[str] | None = None
    case_ids: frozenset[str] | None = None
    config_paths: frozenset[str] | None = None

    def run(self, request: FlowRequest, services) -> FlowResult:
        """Execute the get model flow handler and return its structured result."""
        config = request.config
        context = request.context
        backend = context.diagnostic.backend

        if config.family != self.family:
            raise ArtifactValidationError(
                "Get-model handler received the wrong model family",
                context=context.diagnostic,
                details={"expected": self.family.value, "actual": config.family.value},
            )
        skip = self._skip_reason(request, backend)
        if skip:
            return FlowResult(FlowDisposition.SKIPPED, skip)
        if not context.source_dir.is_dir():
            raise ArtifactValidationError(
                "Model source directory does not exist",
                context=context.diagnostic,
                details={"source_dir": context.source_dir},
            )

        params = config.backend_section("get_model_params", backend)
        if params is None:
            return FlowResult(
                FlowDisposition.SKIPPED,
                f"{config.model_name} has no get_model_params for {backend}",
            )
        matrix = ParameterMatrix.from_columns(
            params,
            location=f"{config.model_name}.get_model_params.{backend}",
        )
        cases = tuple(
            case
            for case in matrix.cases
            if (self.file_types is None or str(case.values.get("type")) in self.file_types)
            and (self.case_ids is None or self.case_artifact_id(case) in self.case_ids)
            and (
                self.config_paths is None
                or self._normalized_config_path(case.values.get("config"))
                in self.config_paths
            )
            and self._release_case_allowed(case, release=context.release)
        )

        if not cases:
            # Every case was filtered out (release/file_types/case_ids). This is
            # "nothing to do", not success: returning EXECUTED would make
            # downstream consumers believe artifacts were produced and fail much
            # later with confusing missing-artifact errors. Report SKIPPED so
            # callers can distinguish it from a real get_model run.
            return FlowResult(
                FlowDisposition.SKIPPED,
                f"{config.model_name} has no eligible get_model cases",
            )
        return self._execute_cases(request, services, cases)

    def _skip_reason(self, request: FlowRequest, backend: str) -> str | None:
        """Return the reason get_model should not execute for this request."""
        config = request.config
        context = request.context
        if config.obsolete or not config.supports(backend, ModelFlow.GET_MODEL):
            return f"{config.model_name} does not support get_model on {backend}"
        if context.test_type == TCaseType.SEPARATE_INFER:
            return f"get_model for {config.model_name} already ran in separate no-infer"
        if context.platform is None:
            return f"{config.model_name} does not support the current platform"
        return None

    def _execute_cases(self, request: FlowRequest, services, cases) -> FlowResult:
        """Execute eligible get-model cases and publish their artifacts."""
        context = request.context
        command_results = []
        failures = []
        workspace_outputs: set[Path] = set()
        context.model_cache_dir.mkdir(parents=True, exist_ok=True)
        lock_file = context.model_cache_dir / "lock.lock"
        with services.workspace_manager.open(context.source_dir, phase="get_model") as workspace:
            with ModelResourceLock(
                str(lock_file),
                ModelResourceLock.LockMode.WRITE,
                "model downloading",
            ):
                for case in cases:
                    result, error = self._execute_case(
                        request, services, workspace, case, workspace_outputs
                    )
                    command_results.append(result)
                    if error:
                        failures.append(error)

        validation = ValidationResult(
            passed=not failures,
            summary=("get_model completed" if not failures else "get_model failed: " + "; ".join(failures)),
            failures=tuple(failures),
        )
        return FlowResult(
            FlowDisposition.EXECUTED,
            "no eligible release cases" if not cases else "get_model completed",
            commands=tuple(command_results),
            validation=validation,
            workspace_outputs=tuple(sorted(workspace_outputs)),
        )

    def _execute_case(self, request, services, workspace, case, workspace_outputs):
        """Run one get-model command and publish its produced artifact."""
        context = request.context
        workspace_snapshot = snapshot_workspace_files(workspace)
        case_id = self.case_artifact_id(case)
        case = self._with_demo_output(request, case, case_id)
        resolved_case = resolve_case_paths(
            case, context.model_cache_dir, context.result_cache_dir
        )
        file_type = str(resolved_case.values.get("type", ""))
        argv = ("python3", "get_model.py", *render_case_options(resolved_case))
        result = services.command_runner.run(
            CommandSpec(
                name=f"get_model[{case.index}]",
                argv=argv,
                cwd=workspace,
                allow_nonzero_exit=True,
                log_file=context.log_file,
                timeout_seconds=GET_MODEL_COMMAND_TIMEOUT_SECONDS,
            ),
            diagnostic_fields=context.diagnostic.for_case(
                case_id or case.index, phase="get-model"
            ).as_mapping(),
        )
        if not result.succeeded:
            return result, f"case {case.index} returned {result.return_code}: {argv}"
        try:
            persist = context.diagnostic.flow != ModelFlow.GET_MODEL
            if persist:
                workspace_outputs.update(
                    persist_workspace_outputs(
                        workspace, context.model_cache_dir, workspace_snapshot
                    )
                )
            artifact_directory = self._produced_artifact_directory(
                request, resolved_case, file_type, workspace, persist
            )
            self._publish_artifact(
                request,
                services,
                resolved_case,
                directory=artifact_directory,
                case_id=case_id,
            )
        except ArtifactValidationError as error:
            return result, error.message
        return result, None

    def _with_demo_output(
        self,
        request: FlowRequest,
        case: ParameterCase,
        case_id: str | None,
    ) -> ParameterCase:
        """Route an implicit download to its same-name JSON demo directory."""
        context = request.context
        if context.diagnostic.flow == ModelFlow.GET_MODEL or case_id is None:
            return case
        paths = self._demo_case_paths(request, case_id)
        if not paths:
            return case
        output = Path(os.path.commonpath(tuple(str(path) for path in paths)))
        if len(paths) == 1 and output.suffix:
            output = output.parent
        values = dict(case.values)
        keys = (
            "extract_dir",
            "quant_model_dir",
            "build_model_dir",
            "model_dir",
            "download_dir",
        )
        key = next((name for name in keys if values.get(name)), "extract_dir")
        values[key] = str(output)
        return ParameterCase(case.index, values)

    @staticmethod
    def _demo_case_paths(request: FlowRequest, case_id: str) -> set[Path]:
        """Return resolved demo paths belonging to one result-cache case."""
        backend = request.context.diagnostic.backend
        sections = (
            request.config.backend_section(name, backend) or {}
            for name in ("demo_params", "demo_multibatch_params")
        )
        values = (
            value
            for section in sections
            for column in section.values()
            if isinstance(column, list)
            for value in column
        )
        return {
            GetModelFlowHandler._resolve_demo_case_path(request, value)
            for value in values
            if GetModelFlowHandler._references_case(value, case_id)
        }

    @staticmethod
    def _references_case(value, case_id: str) -> bool:
        """Return whether one demo parameter references the requested cache case."""
        if not isinstance(value, str):
            return False
        reference = cache_case_reference(value)
        return reference is not None and reference[1] == case_id

    @staticmethod
    def _resolve_demo_case_path(request: FlowRequest, value: str) -> Path:
        """Resolve one cache-backed demo parameter into an absolute path."""
        resolved = resolve_case_paths(
            ParameterCase(0, {"path": value}),
            request.context.model_cache_dir,
            request.context.result_cache_dir,
        )
        return Path(resolved.values["path"])

    def _release_case_allowed(self, case: ParameterCase, *, release: bool) -> bool:
        """Return whether a download case satisfies release source policy."""
        if not release:
            return True
        file_type = str(case.values.get("type", ""))
        source_type = case.values.get("source_type")
        raw_allowed = self.family != ModelFamily.LLM or release_source_allowed(file_type, None)
        source_allowed = release_source_allowed("hmm", source_type)
        allowed = raw_allowed and source_allowed
        if not allowed:
            logger.info(
                "Release source policy filtered get_model case: family=%s " "case=%s type=%s source=%s",
                self.family.value,
                case.index,
                file_type,
                source_type,
            )
        return allowed

    @staticmethod
    def case_artifact_id(case: ParameterCase) -> str | None:
        """Build the stable artifact identifier for a download case."""
        return get_model_case_artifact_id(case)

    @staticmethod
    def _normalized_config_path(value) -> str | None:
        """Normalize an optional model-relative config path for case selection."""
        if not isinstance(value, str) or not value:
            return None
        return Path(value).as_posix().removeprefix("./")

    def _publish_artifact(
        self,
        request: FlowRequest,
        services,
        case: ParameterCase,
        *,
        directory: Path | None = None,
        case_id: str | None = None,
    ) -> None:
        """Publish artifact with ownership and manifest metadata."""
        file_type = str(case.values.get("type", ""))
        artifact_type = {
            "raw": ArtifactType.RAW_MODEL,
            "quant": ArtifactType.QUANT_MODEL,
            "hmm": ArtifactType.COMPILED_MODEL,
        }.get(file_type)
        if artifact_type is None:
            return
        directory = directory or self._artifact_directory(case, file_type)
        if directory is None or not directory.is_dir():
            raise ArtifactValidationError(
                "get_model command did not create its artifact directory",
                context=request.context.diagnostic,
                details={"file_type": file_type, "directory": directory},
            )

        required_files: dict[str, str] = {}
        if artifact_type == ArtifactType.COMPILED_MODEL:
            hmm_files = find_nonempty_hmm_files(directory)
            if not hmm_files:
                raise ArtifactValidationError(
                    "get_model reported success but produced no non-empty HMM files",
                    context=request.context.diagnostic,
                    details={"directory": directory, "case": case.index},
                )
            required_files = build_required_file_roles(directory, hmm_files, prefix="hmm")
        else:
            representative_files = find_nonempty_artifact_files(directory, limit=32)
            if not representative_files:
                raise ArtifactValidationError(
                    "get_model reported success but produced no non-empty artifact files",
                    context=request.context.diagnostic,
                    details={"directory": directory, "case": case.index},
                )
            required_files = build_required_file_roles(directory, representative_files, prefix="file")

        case_id = case_id or directory.name or f"case-{case.index}"
        fingerprint = calculate_config_fingerprint(
            {
                "model": request.config.model_name,
                "family": self.family.value,
                "backend": request.context.diagnostic.backend,
                "case_index": case.index,
                "parameters": dict(case.values),
            }
        )
        services.artifact_cache.write_manifest(
            directory,
            ArtifactManifest(
                schema_version=1,
                fingerprint_version=1,
                artifact_type=artifact_type.value,
                model_name=request.config.model_name,
                model_family=self.family.value,
                backend=request.context.diagnostic.backend,
                case_id=case_id,
                producer_flow=ModelFlow.GET_MODEL.value,
                source_type=str(case.values.get("source_type") or "download"),
                config_fingerprint=fingerprint,
                required_files=required_files,
            ),
        )

    def _produced_artifact_directory(
        self,
        request: FlowRequest,
        case: ParameterCase,
        file_type: str,
        workspace: Path,
        persisted_workspace_outputs: bool,
    ) -> Path | None:
        """Resolve the actual artifact directory created by get_model.

        Legacy CV download scripts use ``output/<backend>`` when no explicit
        build directory is supplied.  Standalone get-model validates that
        temporary output in place and then lets the workspace be removed.  A
        parent quant/compile/inference flow persists the same relative path into
        the model cache before validating it.
        """
        configured = self._artifact_directory(case, file_type)
        if file_type != "hmm":
            return configured
        default_output = workspace / "output" / request.context.diagnostic.backend
        if not find_nonempty_hmm_files(default_output):
            return configured
        if persisted_workspace_outputs:
            return request.context.model_cache_dir / "output" / request.context.diagnostic.backend
        return default_output

    @staticmethod
    def _artifact_directory(case: ParameterCase, file_type: str) -> Path | None:
        """Resolve the artifact directory produced by a get-model case."""
        if file_type == "raw":
            keys = ("download_dir", "model_dir")
        else:
            keys = (
                "extract_dir",
                "quant_model_dir",
                "build_model_dir",
                "model_dir",
                "download_dir",
            )
        for key in keys:
            value = case.values.get(key)
            if isinstance(value, str) and value:
                return Path(value)
        return None
