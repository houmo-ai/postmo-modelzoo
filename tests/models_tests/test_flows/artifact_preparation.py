# Copyright (c) 2026 HOUMO AI
#
# File: artifact_preparation.py
# Description:
#  Declarative Artifact Preparation and Upstream Flow Orchestration.
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

"""Resolve and prepare raw, quantized, and compiled flow artifacts.

Inference flows use this module to reuse valid cache entries or invoke the
producer flow needed to create a missing artifact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ...tests_utils.command_execution import CommandExecutionError
from ...tests_utils.runtime_context import TCaseType
from ..model_workflow.artifact_cache_store import ArtifactType, copy_cache_contents
from ..model_workflow.artifact_file_scanner import (
    find_nonempty_artifact_files,
)
from ..model_workflow.artifact_workspace import restore_workspace_outputs
from ..model_workflow.backend_flow_policies import FamilyFlowPolicy
from ..model_workflow.cache_path_resolver import (
    MODEL_CACHE_ROOT,
    cache_case_reference,
    cache_root_directory,
    get_model_case_artifact_id,
    resolve_cached_path,
)
from ..model_workflow.flow_contracts import (
    CommandResult,
    FlowDisposition,
    FlowRequest,
    FlowResult,
    ModelFlow,
    ModelTestError,
)
from ..model_workflow.parameter_matrix import ParameterMatrix

logger = logging.getLogger(__name__)


class ArtifactPurpose(str, Enum):
    """Describe why an artifact is required by the requesting flow."""

    FLOW_INPUT = "flow_input"
    INFERENCE = "inference"


@dataclass(frozen=True)
class ArtifactNeed:
    """Declare an artifact type and the purpose it must satisfy."""

    artifact_type: ArtifactType
    case_ids: frozenset[str] | None = None
    purpose: ArtifactPurpose = ArtifactPurpose.FLOW_INPUT

    @classmethod
    def raw_model(cls, case_ids: frozenset[str] | None = None) -> "ArtifactNeed":
        """Request raw model inputs and get-model workspace side effects."""
        return cls(ArtifactType.RAW_MODEL, case_ids)

    @classmethod
    def quant_model(cls) -> "ArtifactNeed":
        """Request quantized inputs referenced by compile parameters."""
        return cls(ArtifactType.QUANT_MODEL)

    @classmethod
    def inference_compiled_model(cls) -> "ArtifactNeed":
        """Request compiled artifacts required by an inference flow."""
        return cls(
            ArtifactType.COMPILED_MODEL,
            purpose=ArtifactPurpose.INFERENCE,
        )


@dataclass(frozen=True)
class PreparationReport:
    """Describe commands and terminal state produced while ensuring artifacts."""

    commands: tuple[CommandResult, ...] = ()
    failures: tuple[str, ...] = ()
    disposition: FlowDisposition = FlowDisposition.EXECUTED
    message: str = ""


@dataclass(frozen=True)
class _HmatcRawModelReference:
    """Map one HMATC config model path to cache and workspace locations."""

    config_path: Path
    configured_path: Path
    workspace_path: Path
    cache_path: Path | None


@dataclass(frozen=True)
class ArtifactPreparer:
    """Resolve artifact declarations and exclusively orchestrate upstream flows."""

    def ensure(
        self,
        request: FlowRequest,
        services,
        needs: Sequence[ArtifactNeed],
        *,
        workspace: Path | None = None,
        policy: FamilyFlowPolicy,
    ) -> PreparationReport:
        """Ensure each declared artifact need in dependency order."""
        commands: list[CommandResult] = []
        disposition = FlowDisposition.EXECUTED
        message = ""
        for need in needs:
            report = self._ensure_one(
                request,
                services,
                need,
                workspace=workspace,
                policy=policy,
            )
            commands.extend(report.commands)
            disposition = report.disposition
            message = report.message
            if report.failures or report.disposition == FlowDisposition.SKIPPED:
                return PreparationReport(
                    tuple(commands),
                    report.failures,
                    report.disposition,
                    report.message,
                )
        return PreparationReport(
            tuple(commands),
            disposition=disposition,
            message=message,
        )

    def ensure_inference(
        self,
        request: FlowRequest,
        services,
        workspace: Path,
        policy: FamilyFlowPolicy,
    ) -> PreparationReport:
        """Ensure the compiled artifacts required by an inference flow."""
        return self.ensure(
            request,
            services,
            (ArtifactNeed.inference_compiled_model(),),
            workspace=workspace,
            policy=policy,
        )

    def prepare_hmatc_v2_raw_models(
        self,
        request: FlowRequest,
        services,
        cases,
    ) -> PreparationReport:
        """Download and validate raw get-model cases referenced by HMATC v2 overrides.

        Always invoke ``get_model.py`` for the selected v2 configs.  ModelScope
        handles its own download cache, while running the command again ensures
        that a non-empty destination left by an older case is never mistaken for
        the raw model declared by the current configuration.
        """
        references = tuple(self._hmatc_v2_model_cache_directories(request, cases))
        if not references:
            return PreparationReport(message="HMATC v2 has no raw model cache references")

        from .get_model_flow import GetModelFlowHandler

        selected_configs = frozenset(case.config for case in cases)
        result = GetModelFlowHandler(
            request.config.family,
            file_types=frozenset({"raw"}),
            config_paths=selected_configs,
        ).run(request, services)
        report = self._report_from_flow(result)
        failures = list(report.failures)
        if result.disposition == FlowDisposition.SKIPPED:
            failures.append(result.message or "HMATC v2 raw get_model was skipped")
        unresolved = tuple(path for path in references if not self._hmatc_v2_cache_reference_exists(path))
        if unresolved:
            failures.append(
                "HMATC v2 raw model paths are missing after get_model: " + ", ".join(str(path) for path in unresolved)
            )
        return PreparationReport(
            commands=report.commands,
            failures=tuple(failures),
            disposition=report.disposition,
            message=report.message,
        )

    @staticmethod
    def _hmatc_v2_model_cache_directories(
        request: FlowRequest,
        cases,
    ) -> Iterable[Path]:
        """Yield unique cached_models directories from nested v2 overrides."""
        seen: set[Path] = set()
        for case in cases:
            for value in ArtifactPreparer._nested_strings(case.override):
                parts = Path(value.replace("\\", "/")).parts
                if MODEL_CACHE_ROOT not in parts:
                    continue
                resolved = Path(
                    resolve_cached_path(
                        value,
                        model_cache_dir=request.context.model_cache_dir,
                        result_cache_dir=request.context.result_cache_dir,
                    )
                )
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved

    @staticmethod
    def _hmatc_v2_cache_reference_exists(path: Path) -> bool:
        """Return whether a nested v2 cache reference is a usable file or directory."""
        try:
            if path.is_file():
                return path.stat().st_size > 0
        except OSError:
            return False
        return bool(find_nonempty_artifact_files(path, limit=1))

    @staticmethod
    def _nested_strings(value: Any) -> Iterable[str]:
        """Yield strings recursively from a JSON-like value."""
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            for nested in value.values():
                yield from ArtifactPreparer._nested_strings(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from ArtifactPreparer._nested_strings(nested)

    def backfill_referenced_demo_artifacts(self, request: FlowRequest, services) -> list[Path]:
        """Download referenced HMM directories not produced by compile cases."""
        from .get_model_flow import GetModelFlowHandler
        from .inference_flow_support import unproduced_demo_artifact_refs

        backend = request.context.diagnostic.backend
        references = unproduced_demo_artifact_refs(request, backend)
        if not references:
            return []
        get_params = request.config.backend_section("get_model_params", backend)
        if get_params is None:
            return []
        source_cases = {
            get_model_case_artifact_id(case): case
            for case in ParameterMatrix.from_columns(
                get_params,
                location=f"{request.config.model_name}.get_model_params.{backend}",
            ).cases
            if str(case.values.get("type")) == "hmm" and get_model_case_artifact_id(case)
        }
        missing: dict[str, str] = {}
        for root, case_id in sorted(references):
            if case_id not in source_cases:
                continue
            directory = (
                cache_root_directory(
                    root,
                    model_cache_dir=request.context.model_cache_dir,
                    result_cache_dir=request.context.result_cache_dir,
                )
                / case_id
            )
            if find_nonempty_artifact_files(directory, limit=1):
                continue
            missing[case_id] = root
        if not missing:
            return []

        logger.info(
            "Preparing demo artifact directories not produced by compile: %s",
            ", ".join(sorted(missing)),
        )
        prepare_context = replace(request.context, test_type=TCaseType.DEFAULT)
        try:
            result = GetModelFlowHandler(
                request.config.family,
                frozenset({"hmm"}),
                frozenset(missing),
            ).run(FlowRequest(prepare_context, request.config), services)
        except (ModelTestError, CommandExecutionError) as error:
            logger.warning("Failed to prepare demo artifact directories: %s", error)
            return []
        failures = self._flow_failures(result)
        if failures:
            logger.warning("Failed to prepare demo artifact directories: %s", "; ".join(failures))
        return [
            cache_root_directory(
                root,
                model_cache_dir=request.context.model_cache_dir,
                result_cache_dir=request.context.result_cache_dir,
            )
            / case_id
            for case_id, root in sorted(missing.items())
        ]

    def _ensure_one(
        self,
        request: FlowRequest,
        services,
        need: ArtifactNeed,
        *,
        workspace: Path | None,
        policy: FamilyFlowPolicy,
    ) -> PreparationReport:
        """Dispatch one declarative artifact need to its type-specific producer."""
        if need.artifact_type == ArtifactType.RAW_MODEL:
            return self._ensure_raw_model(request, services, need, workspace=workspace, policy=policy)
        if need.artifact_type == ArtifactType.QUANT_MODEL:
            return self._ensure_quant_model(request, services, policy=policy)
        if need.artifact_type == ArtifactType.COMPILED_MODEL and need.purpose == ArtifactPurpose.INFERENCE:
            if workspace is None:
                raise ValueError("Inference artifact preparation requires a workspace")
            return self._ensure_inference_compiled_model(request, services, workspace, policy)
        raise ValueError(f"Unsupported artifact need: {need}")

    def _ensure_raw_model(
        self,
        request: FlowRequest,
        services,
        need: ArtifactNeed,
        *,
        workspace: Path | None,
        policy: FamilyFlowPolicy,
    ) -> PreparationReport:
        """Ensure the raw model is available and restore it into the workspace."""
        from .get_model_flow import GetModelFlowHandler

        result = GetModelFlowHandler(
            policy.family,
            frozenset({"raw"}),
            need.case_ids,
        ).run(request, services)
        report = self._report_from_flow(result)
        if not report.failures and workspace is not None:
            restore_workspace_outputs(
                request.context.model_cache_dir,
                workspace,
                result.workspace_outputs,
            )
            if policy.copy_raw_to_workspace:
                copy_cache_contents(request.context.model_cache_dir, workspace)
        return report

    def _ensure_quant_model(self, request: FlowRequest, services, *, policy: FamilyFlowPolicy) -> PreparationReport:
        """Ensure compile inputs exist, running quantization when necessary."""
        from .quant_flow import QuantFlowHandler

        params = request.config.backend_section("compile_params", request.context.diagnostic.backend)
        if params is None:
            return PreparationReport(
                disposition=FlowDisposition.SKIPPED,
                message="missing compile_params",
            )
        matrix = ParameterMatrix.from_columns(
            params,
            location=(f"{request.config.model_name}.compile_params." f"{request.context.diagnostic.backend}"),
        )
        model_dirs = []
        for case in matrix.cases:
            value = case.values.get("model_dir")
            if not isinstance(value, str):
                continue
            model_dirs.append(
                Path(
                    resolve_cached_path(
                        value,
                        model_cache_dir=request.context.model_cache_dir,
                        result_cache_dir=request.context.result_cache_dir,
                    )
                )
            )
        if model_dirs and all(path.is_dir() and any(path.iterdir()) for path in model_dirs):
            return PreparationReport(message="quant inputs already exist")
        if not request.config.supports(request.context.diagnostic.backend, ModelFlow.QUANT):
            return PreparationReport(
                failures=(
                    f"missing compile model_dir: " f"{', '.join(str(path) for path in model_dirs) or '<unset>'}",
                ),
                message="compile input artifact is missing",
            )
        return self._report_from_flow(QuantFlowHandler(policy).run(request, services))

    def _ensure_inference_compiled_model(
        self,
        request: FlowRequest,
        services,
        workspace: Path,
        policy: FamilyFlowPolicy,
    ) -> PreparationReport:
        """Prepare the compiled artifact required by demo/compare/eval/perf."""
        from .compile_flow import CompileFlowHandler
        from .hmatc_flow_support import (
            restore_reusable_hmatc_inference_artifact,
            run_hmatc_inference_preparation,
        )
        from .inference_flow_support import validate_python_compiled_artifacts

        context = request.context
        if context.test_type == TCaseType.SEPARATE_INFER:
            return self._prepare_separate_infer_artifacts(request, services, workspace, policy)

        if context.platform == "aarch64":
            return self._prepare_aarch64_artifacts(request, services, workspace)

        if request.config.has_section("hmbuild_params"):
            return self._prepare_hmatc_artifacts(
                request,
                services,
                workspace,
                policy,
                restore_reusable_hmatc_inference_artifact,
                run_hmatc_inference_preparation,
            )

        return self._prepare_python_artifacts(
            request,
            services,
            policy,
            CompileFlowHandler,
            validate_python_compiled_artifacts,
        )

    def _prepare_separate_infer_artifacts(self, request, services, workspace, policy):
        """Prepare artifacts for the infer half of a separate-stage run."""
        from .inference_flow_support import validate_python_compiled_artifacts

        context = request.context
        commands: list[CommandResult] = []
        failures: list[str] = []
        if request.config.has_section("hmbuild_params"):
            raw_report = self._prepare_separate_infer_hmatc_raw_models(
                request,
                services,
                workspace,
                policy,
            )
            commands.extend(raw_report.commands)
            failures.extend(raw_report.failures)
        elif policy.copy_raw_to_workspace:
            # Legacy CV Python consumers may read raw get_model side effects
            # from source-relative locations. Restore the cache tree to retain
            # those paths when the model does not use HMATC build configs.
            copy_cache_contents(context.model_cache_dir, workspace)
        if policy.persist_workspace_for_separate:
            copy_cache_contents(context.result_cache_dir, workspace)
        if policy.family.value == "llm" and context.release:
            report = self._download_release_hmms(request, services)
            commands.extend(report.commands)
            failures.extend(report.failures)
        model_cache_report = self._prepare_referenced_model_cache_directories(request, services)
        commands.extend(model_cache_report.commands)
        failures.extend(model_cache_report.failures)
        self.backfill_referenced_demo_artifacts(request, services)
        if not request.config.has_section("hmbuild_params"):
            failures.extend(validate_python_compiled_artifacts(request, services))
        return PreparationReport(tuple(commands), tuple(failures))

    def _prepare_separate_infer_hmatc_raw_models(
        self,
        request: FlowRequest,
        services,
        workspace: Path,
        policy: FamilyFlowPolicy,
    ) -> PreparationReport:
        """Restore HMATC raw inputs, downloading them only when cache is missing."""
        references, reference_failures = self._hmatc_raw_model_references(
            request,
            workspace,
        )
        if reference_failures:
            return PreparationReport(
                failures=reference_failures,
                message="invalid HMATC raw model references",
            )

        copy_cache_contents(request.context.model_cache_dir, workspace)
        missing = self._missing_hmatc_raw_models(references)
        if not missing:
            return PreparationReport(message="HMATC raw models restored from cache")

        downloadable = tuple(
            reference
            for reference in missing
            if reference.cache_path is not None and not self._is_nonempty_file(reference.cache_path)
        )
        commands: list[CommandResult] = []
        failures: list[str] = []
        if downloadable:
            logger.info(
                "Preparing missing HMATC raw models for separate infer: %s",
                ", ".join(str(reference.cache_path) for reference in downloadable),
            )
            result = self._run_separate_infer_raw_get_model(
                request,
                services,
                policy,
            )
            report = self._report_from_flow(result)
            commands.extend(report.commands)
            failures.extend(report.failures)
            copy_cache_contents(request.context.model_cache_dir, workspace)

        unresolved = self._missing_hmatc_raw_models(references)
        if unresolved:
            failures.append(self._missing_hmatc_raw_model_message(unresolved))
        return PreparationReport(
            commands=tuple(commands),
            failures=tuple(failures),
            message=(
                "HMATC raw models prepared for separate infer" if not failures else "HMATC raw model preparation failed"
            ),
        )

    @staticmethod
    def _hmatc_raw_model_references(
        request: FlowRequest,
        workspace: Path,
    ) -> tuple[tuple[_HmatcRawModelReference, ...], tuple[str, ...]]:
        """Resolve model.model_path from every HMATC inference config."""
        from .hmatc_flow_support import hmatc_inference_config_paths

        try:
            config_paths = hmatc_inference_config_paths(request, workspace)
        except ModelTestError as error:
            return (), (error.message,)

        references: list[_HmatcRawModelReference] = []
        failures: list[str] = []
        seen: set[Path] = set()
        for config_path in config_paths:
            model_path, read_error = ArtifactPreparer._read_hmatc_model_path(config_path)
            if read_error:
                failures.append(read_error)
                continue
            reference, resolve_error = ArtifactPreparer._resolve_hmatc_raw_model_reference(
                request,
                workspace,
                config_path,
                model_path,
            )
            if resolve_error:
                failures.append(resolve_error)
            elif reference is not None and reference.workspace_path not in seen:
                seen.add(reference.workspace_path)
                references.append(reference)
        return tuple(references), tuple(failures)

    @staticmethod
    def _read_hmatc_model_path(config_path: Path) -> tuple[str | None, str | None]:
        """Read and validate model.model_path from one HMATC YAML config."""
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            return None, f"failed to read HMATC config {config_path}: {error}"
        model = config.get("model") if isinstance(config, Mapping) else None
        model_path = model.get("model_path") if isinstance(model, Mapping) else None
        if not isinstance(model_path, str) or not model_path:
            return None, f"HMATC config has no model.model_path: {config_path}"
        return model_path, None

    @staticmethod
    def _resolve_hmatc_raw_model_reference(
        request: FlowRequest,
        workspace: Path,
        config_path: Path,
        model_path: str | None,
    ) -> tuple[_HmatcRawModelReference | None, str | None]:
        """Map one YAML model path to workspace and model-cache locations."""
        if model_path is None:
            return None, f"HMATC config has no model.model_path: {config_path}"
        configured_path = Path(model_path)
        if configured_path.is_absolute():
            return (
                _HmatcRawModelReference(
                    config_path=config_path,
                    configured_path=configured_path,
                    workspace_path=configured_path,
                    cache_path=None,
                ),
                None,
            )

        resolved_workspace = workspace.resolve()
        workspace_path = (resolved_workspace / configured_path).resolve()
        if workspace_path != resolved_workspace and resolved_workspace not in workspace_path.parents:
            return None, (
                "HMATC model.model_path escapes the workspace: " f"config={config_path} model_path={model_path}"
            )
        relative = workspace_path.relative_to(resolved_workspace)
        return (
            _HmatcRawModelReference(
                config_path=config_path,
                configured_path=configured_path,
                workspace_path=workspace_path,
                cache_path=request.context.model_cache_dir / relative,
            ),
            None,
        )

    @classmethod
    def _missing_hmatc_raw_models(
        cls,
        references: tuple[_HmatcRawModelReference, ...],
    ) -> tuple[_HmatcRawModelReference, ...]:
        """Return HMATC model references absent from the inference workspace."""
        return tuple(reference for reference in references if not cls._is_nonempty_file(reference.workspace_path))

    @staticmethod
    def _is_nonempty_file(path: Path) -> bool:
        """Return whether a path is a regular non-empty file."""
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    @staticmethod
    def _run_separate_infer_raw_get_model(request, services, policy):
        """Run raw get_model with a non-infer context after a cache miss."""
        from .get_model_flow import GetModelFlowHandler

        prepare_context = replace(request.context, test_type=TCaseType.DEFAULT)
        return GetModelFlowHandler(
            policy.family,
            frozenset({"raw"}),
        ).run(FlowRequest(prepare_context, request.config), services)

    @staticmethod
    def _missing_hmatc_raw_model_message(
        references: tuple[_HmatcRawModelReference, ...],
    ) -> str:
        """Describe unresolved HMATC raw inputs before the tool is executed."""
        details = "; ".join(
            "config="
            f"{reference.config_path} model_path={reference.configured_path} "
            f"cache_path={reference.cache_path or '<absolute>'} "
            f"workspace_path={reference.workspace_path}"
            for reference in references
        )
        return f"HMATC raw model is missing for separate infer: {details}"

    def _prepare_referenced_model_cache_directories(
        self,
        request: FlowRequest,
        services,
    ) -> PreparationReport:
        """Prepare cached_models directories required by demo/perf commands.

        A separate-infer host may receive the persistent cached_results tree
        without tokenizer/model directories created as side effects of an HMM
        download. Match each demo parameter row to a declared get_model
        ``type=hmm`` case, run only the required cases, and then verify that all
        cached_models directories referenced by the command were created.
        """
        backend = request.context.diagnostic.backend
        get_params = request.config.backend_section("get_model_params", backend)
        if get_params is None:
            return PreparationReport(message="missing get_model_params")
        source_cases = self._get_model_hmm_source_cases(request, backend, get_params)
        if not source_cases:
            return PreparationReport(message="no get_model type=hmm cases")

        missing_directories, directory_sources = self._referenced_model_cache_plan(
            request,
            backend,
            source_cases,
        )
        if not missing_directories:
            return PreparationReport(message="referenced cached_models directories already exist")

        unmatched_directories = missing_directories - directory_sources.keys()
        if unmatched_directories:
            return self._unmatched_model_cache_report(unmatched_directories)

        selected_case_ids = frozenset(directory_sources.values())
        logger.info(
            "Preparing missing cached_models directories for separate infer: " "directories=%s get_model_hmm_cases=%s",
            ", ".join(str(path) for path in sorted(missing_directories)),
            ", ".join(sorted(selected_case_ids)),
        )
        result = self._run_model_cache_get_model_cases(
            request,
            services,
            selected_case_ids,
        )
        return self._model_cache_preparation_report(result, missing_directories)

    @staticmethod
    def _get_model_hmm_source_cases(request, backend, get_params):
        """Index declared get_model HMM cases by their artifact directory."""
        matrix = ParameterMatrix.from_columns(
            get_params,
            location=f"{request.config.model_name}.get_model_params.{backend}",
        )
        source_cases = {}
        for case in matrix.cases:
            case_id = get_model_case_artifact_id(case)
            if str(case.values.get("type")) == "hmm" and case_id is not None:
                source_cases[case_id] = case
        return source_cases

    def _referenced_model_cache_plan(self, request, backend, source_cases):
        """Map every missing cached_models reference to an HMM producer case."""
        from .inference_flow_support import best_matching_download_case

        missing_directories: set[Path] = set()
        directory_sources: dict[Path, str] = {}
        for section_name in ("demo_params", "demo_multibatch_params"):
            for case in self._section_parameter_cases(request, backend, section_name):
                case_missing = self._missing_model_cache_directories(request, case)
                if not case_missing:
                    continue
                missing_directories.update(case_missing)
                unassigned = case_missing - directory_sources.keys()
                if not unassigned:
                    continue
                selected = best_matching_download_case(
                    case,
                    source_cases,
                    allow_ambiguous=True,
                )
                if selected is not None:
                    directory_sources.update({directory: selected for directory in unassigned})
        return missing_directories, directory_sources

    @staticmethod
    def _section_parameter_cases(request, backend, section_name):
        """Return parameter cases from one optional inference section."""
        params = request.config.backend_section(section_name, backend)
        if params is None:
            return ()
        return ParameterMatrix.from_columns(
            params,
            location=f"{request.config.model_name}.{section_name}.{backend}",
        ).cases

    @staticmethod
    def _missing_model_cache_directories(request, case) -> set[Path]:
        """Return non-existent cached_models directories referenced by one case."""
        model_cache_dir = request.context.model_cache_dir
        missing = set()
        for value in case.values.values():
            if not isinstance(value, str):
                continue
            reference = cache_case_reference(value)
            if reference is None or reference[0] != MODEL_CACHE_ROOT:
                continue
            directory = model_cache_dir / reference[1]
            if not find_nonempty_artifact_files(directory, limit=1):
                missing.add(directory)
        return missing

    @staticmethod
    def _unmatched_model_cache_report(
        unmatched_directories: set[Path],
    ) -> PreparationReport:
        """Build the failure report for references without an HMM producer."""
        return PreparationReport(
            failures=(
                "no get_model type=hmm case matches missing cached_models "
                f"directories: {', '.join(str(path) for path in sorted(unmatched_directories))}",
            ),
            message="cached_models preparation has no matching HMM case",
        )

    @staticmethod
    def _run_model_cache_get_model_cases(request, services, selected_case_ids):
        """Run only HMM get_model cases selected for missing cache directories."""
        from .get_model_flow import GetModelFlowHandler

        prepare_context = replace(request.context, test_type=TCaseType.DEFAULT)
        return GetModelFlowHandler(
            request.config.family,
            frozenset({"hmm"}),
            selected_case_ids,
        ).run(FlowRequest(prepare_context, request.config), services)

    def _model_cache_preparation_report(self, result, missing_directories):
        """Verify downloaded directories and preserve the upstream flow report."""
        report = self._report_from_flow(result)
        failures = list(report.failures)
        if result.disposition == FlowDisposition.SKIPPED:
            failures.append(result.message or "missing cached_models directories could not be prepared")
        unresolved = {
            directory for directory in missing_directories if not find_nonempty_artifact_files(directory, limit=1)
        }
        if unresolved:
            failures.append(
                "get_model type=hmm did not create required cached_models "
                f"directories: {', '.join(str(path) for path in sorted(unresolved))}"
            )
        return PreparationReport(
            commands=report.commands,
            failures=tuple(failures),
            disposition=report.disposition,
            message=report.message,
        )

    def _prepare_aarch64_artifacts(self, request, services, workspace):
        """Prepare release artifacts for the aarch64 inference path."""
        report = self._download_release_hmms(request, services)
        self.backfill_referenced_demo_artifacts(request, services)
        copy_cache_contents(request.context.model_cache_dir, workspace)
        return PreparationReport(report.commands, report.failures)

    def _prepare_hmatc_artifacts(self, request, services, workspace, policy, restore_artifact, run_preparation):
        """Prepare raw model inputs, then reuse or build HMATC artifacts."""
        raw = self._ensure_raw_model(
            request,
            services,
            ArtifactNeed.raw_model(),
            workspace=workspace,
            policy=policy,
        )
        if raw.failures:
            return raw
        if restore_artifact(request, services, workspace):
            self.backfill_referenced_demo_artifacts(request, services)
            return PreparationReport(
                commands=raw.commands,
                disposition=raw.disposition,
                message="raw model prepared and compiled artifact already exists",
            )
        self.backfill_referenced_demo_artifacts(request, services)
        commands, failures = run_preparation(request, services, workspace)
        return PreparationReport(raw.commands + tuple(commands), raw.failures + tuple(failures))

    def _prepare_python_artifacts(self, request, services, policy, compile_handler, validate):
        """Reuse, compile, or download Python-generated inference artifacts."""
        backend = request.context.diagnostic.backend
        artifact_failures = validate(request, services)
        if request.config.backend_section("compile_params", backend) is not None and not artifact_failures:
            self.backfill_referenced_demo_artifacts(request, services)
            return PreparationReport(message="compiled artifacts already exist")
        compile_result = compile_handler(policy).run(request, services)
        commands = list(compile_result.commands)
        compile_failures = list(self._flow_failures(compile_result))
        self.backfill_referenced_demo_artifacts(request, services)
        artifact_failures = validate(request, services)
        failures = []
        if artifact_failures:
            downloaded = self._download_release_hmms(request, services)
            commands.extend(downloaded.commands)
            artifact_failures = validate(request, services)
            if artifact_failures:
                failures.extend(compile_failures)
                failures.extend(downloaded.failures)
                failures.extend(artifact_failures)
        return PreparationReport(tuple(commands), tuple(failures))

    def _download_release_hmms(self, request: FlowRequest, services) -> PreparationReport:
        """Download release HMM cases whose ids are referenced by demo JSON."""
        from .get_model_flow import GetModelFlowHandler
        from .inference_flow_support import release_hmm_case_ids

        prepare_context = replace(request.context, test_type=TCaseType.DEFAULT)
        case_ids = release_hmm_case_ids(request)
        if not case_ids:
            return PreparationReport(message="no matching release HMM cases")
        result = GetModelFlowHandler(
            request.config.family,
            frozenset({"hmm"}),
            case_ids,
        ).run(FlowRequest(prepare_context, request.config), services)
        return self._report_from_flow(result)

    @staticmethod
    def _flow_failures(result: FlowResult) -> tuple[str, ...]:
        """Extract normalized failure messages from a structured flow result."""
        if result.validation is None or result.validation.passed:
            return ()
        return result.validation.failures or (result.validation.summary,)

    @classmethod
    def _report_from_flow(cls, result: FlowResult) -> PreparationReport:
        """Convert a flow result into the artifact preparation report type."""
        return PreparationReport(
            commands=result.commands,
            failures=cls._flow_failures(result),
            disposition=result.disposition,
            message=result.message,
        )


def ensure_artifacts(
    request: FlowRequest,
    services,
    needs: Sequence[ArtifactNeed],
    *,
    workspace: Path | None = None,
    policy: FamilyFlowPolicy,
) -> PreparationReport:
    """Use the configured artifact preparer for declarative artifact needs."""
    preparer = getattr(services, "artifact_preparer", ArtifactPreparer())
    return preparer.ensure(
        request,
        services,
        needs,
        workspace=workspace,
        policy=policy,
    )


def ensure_inference_artifacts(
    request: FlowRequest,
    services,
    workspace: Path,
    policy: FamilyFlowPolicy,
) -> PreparationReport:
    """Ensure the compiled artifacts required by an inference flow."""
    return ensure_artifacts(
        request,
        services,
        (ArtifactNeed.inference_compiled_model(),),
        workspace=workspace,
        policy=policy,
    )


def prepare_hmatc_v2_raw_models(
    request: FlowRequest,
    services,
    cases,
) -> PreparationReport:
    """Use the configured artifact preparer for HMATC v2 raw references."""
    preparer = getattr(services, "artifact_preparer", ArtifactPreparer())
    return preparer.prepare_hmatc_v2_raw_models(request, services, cases)


__all__ = [
    "ArtifactNeed",
    "ArtifactPreparer",
    "ArtifactPurpose",
    "PreparationReport",
    "ensure_artifacts",
    "ensure_inference_artifacts",
    "prepare_hmatc_v2_raw_models",
]
