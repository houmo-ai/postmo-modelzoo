# Copyright (c) 2026 HOUMO AI
#
# File: hmatc_flow_support.py
# Description:
#  Shared HMATC Quantization and Inference Command Support.
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

"""Shared HMATC command runners used by multiple model-test flows."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Mapping

import yaml

from ...tests_utils.command_execution import output_reports_failure
from ...tests_utils.resource_lock import ModelResourceLock
from ..model_workflow.artifact_cache_store import (
    ArtifactManifest,
    ArtifactRequirement,
    ArtifactType,
    AtomicArtifactWriter,
    CacheStatus,
    calculate_config_fingerprint,
    copy_cache_contents,
)
from ..model_workflow.artifact_file_scanner import (
    build_required_file_roles,
    find_nonempty_artifact_files,
    find_nonempty_hmm_files,
    prune_compiler_intermediates,
)
from ..model_workflow.backend_flow_policies import FamilyFlowPolicy
from ..model_workflow.flow_contracts import (
    ArtifactValidationError,
    CommandResult,
    CommandSpec,
    FlowRequest,
    PhaseResult,
)
from ..model_workflow.parameter_matrix import ParameterMatrix, render_case_options

logger = logging.getLogger(__name__)


HMATC_INFERENCE_CASE_ID = "inference-default"
HMATC_INFERENCE_SOURCE_TYPE = "local_hmatc_inference"


def run_hmatc_quant_cases(request, services, workspace: Path) -> PhaseResult:
    """Run HMATC quant cases and return their structured execution outcome."""
    columns = request.config.hmatc_columns("hmquant_params")
    matrix = ParameterMatrix.from_columns(
        columns,
        location=f"{request.config.model_name}.hmquant_params",
    )
    results: list[CommandResult] = []
    failures: list[str] = []
    for case in matrix.cases:
        argv = (
            "hmatc",
            "quant",
            "--target",
            request.context.diagnostic.backend,
            *render_case_options(case, skipped_keys={"onnx"}),
        )
        result = services.command_runner.run(
            CommandSpec(
                name=f"hmquant[{case.index}]",
                argv=argv,
                cwd=workspace,
                allow_nonzero_exit=True,
                log_file=request.context.log_file,
            ),
            diagnostic_fields=request.context.diagnostic.for_case(case.index, phase="hmatc-quant").as_mapping(),
        )
        results.append(result)
        if not result.succeeded:
            failures.append(f"hmatc quant case {case.index} failed")
    return PhaseResult(
        commands=tuple(results),
        failures=tuple(failures),
        total_cases=len(matrix.cases),
        executed_cases=len(results),
    )


def run_hmatc_inference_preparation(request, services, workspace):
    """Prepare or restore the default HMATC artifact bundle for inference.

    The legacy implementation assumed one ``config.yml``.  A model may expose
    several HMATC components (for example SAM2's ``encoder.yml`` and
    ``decoder.yml``); all component configs are executed in one isolated
    workspace and published as one reusable inference bundle.
    """
    if restore_reusable_hmatc_inference_artifact(request, services, workspace):
        return [], []

    prepared_relative = hmatc_inference_artifact_relative_path(request, workspace)
    cached_directory = request.context.result_cache_dir / prepared_relative
    workspace_directory = workspace / prepared_relative
    config_paths = hmatc_inference_config_paths(request, workspace)
    fingerprint = _hmatc_inference_fingerprint(request, config_paths)
    lock_file = request.context.result_cache_dir / "lock.lock"
    request.context.result_cache_dir.mkdir(parents=True, exist_ok=True)

    with ModelResourceLock(str(lock_file), ModelResourceLock.LockMode.WRITE, "prepare hmatc inference"):
        if _restore_reusable_hmatc_inference_artifact_locked(
            request,
            services,
            cached_directory,
            workspace_directory,
            fingerprint,
            config_paths,
        ):
            return [], []

        results, failures = _execute_hmatc_inference_preparation(request, services, workspace)
        if failures:
            return results, failures
        _publish_hmatc_inference_artifact(
            request,
            services,
            workspace_directory,
            cached_directory,
            fingerprint,
        )
        copy_cache_contents(cached_directory, workspace_directory)
        return results, failures


def restore_reusable_hmatc_inference_artifact(request: FlowRequest, services, workspace: Path) -> bool:
    """Restore a valid default HMATC inference bundle without running tools."""
    prepared_relative = hmatc_inference_artifact_relative_path(request, workspace)
    cached_directory = request.context.result_cache_dir / prepared_relative
    workspace_directory = workspace / prepared_relative
    config_paths = hmatc_inference_config_paths(request, workspace)
    fingerprint = _hmatc_inference_fingerprint(request, config_paths)
    lock_file = request.context.result_cache_dir / "lock.lock"
    request.context.result_cache_dir.mkdir(parents=True, exist_ok=True)
    with ModelResourceLock(str(lock_file), ModelResourceLock.LockMode.WRITE, "reuse hmatc inference"):
        return _restore_reusable_hmatc_inference_artifact_locked(
            request,
            services,
            cached_directory,
            workspace_directory,
            fingerprint,
            config_paths,
        )


def _restore_reusable_hmatc_inference_artifact_locked(
    request,
    services,
    cached_directory: Path,
    workspace_directory: Path,
    fingerprint: str,
    config_paths: tuple[Path, ...],
) -> bool:
    """Inspect and restore one HMATC bundle while its cache lock is held."""
    if not _hmatc_inference_artifact_reusable(
        request,
        services,
        cached_directory,
        fingerprint,
        config_paths,
    ):
        return False
    workspace_directory.mkdir(parents=True, exist_ok=True)
    copy_cache_contents(cached_directory, workspace_directory)
    logger.info("Reuse HMATC inference artifact: %s", cached_directory)
    return True


def _execute_hmatc_inference_preparation(request, services, workspace):
    """Execute quant/build for every configured HMATC component."""
    results = []
    failures = []
    config_paths = hmatc_inference_config_paths(request, workspace)
    for phase, subcommand in (("quant", "quant"), ("build", "build")):
        for index, config_path in enumerate(config_paths):
            result = _execute_hmatc_inference_command(
                request, services, workspace, phase, subcommand, index, config_path, len(config_paths)
            )
            results.append(result)
            if not result.succeeded or output_reports_failure(result.combined_output):
                failures.append(f"hmatc inference {phase} config {config_path} failed")
                return results, failures
    return results, failures


def _execute_hmatc_inference_command(request, services, workspace, phase, subcommand, index, config_path, config_count):
    """Run one HMATC inference preparation command."""
    config_arg = config_path.relative_to(workspace).as_posix()
    if not config_arg.startswith("./"):
        config_arg = f"./{config_arg}"
    argv = [
        "hmatc",
        subcommand,
        "--target",
        request.context.diagnostic.backend,
        "--config",
        config_arg,
    ]
    if subcommand == "build":
        argv.insert(2, "--skip_check")
    command_name = f"hmatc-inference-{phase}"
    if config_count > 1:
        command_name += f"[{index}]"
    return services.command_runner.run(
        CommandSpec(
            name=command_name,
            argv=tuple(argv),
            cwd=workspace,
            allow_nonzero_exit=True,
            log_file=request.context.log_file,
        ),
        diagnostic_fields=request.context.diagnostic.for_case(
            f"{phase}-{index}", phase=f"hmatc-inference-{phase}"
        ).as_mapping(),
    )


def hmatc_inference_config_paths(request: FlowRequest, workspace: Path) -> tuple[Path, ...]:
    """Resolve unique HMATC config files used by the inference preparation.

    Configs are derived from the existing ``hmquant_params`` and
    ``hmbuild_params`` sections, so mixed-flow models need no new JSON field.
    A legacy model with neither section keeps the historical ``config.yml``
    fallback.
    """
    config_values: list[str] = []
    quant_columns = (
        request.config.hmatc_columns("hmquant_params") if request.config.has_section("hmquant_params") else {}
    )
    build_columns = (
        request.config.hmatc_columns("hmbuild_params", backend=request.context.diagnostic.backend)
        if request.config.backend_section("hmbuild_params", request.context.diagnostic.backend) is not None
        else {}
    )
    for columns in (quant_columns, build_columns):
        values = columns.get("config")
        if not isinstance(values, list):
            continue
        config_values.extend(str(value) for value in values if value not in (None, "default"))
    if not config_values:
        config_values = ["./config.yml"]

    paths: list[Path] = []
    seen: set[Path] = set()
    for value in config_values:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactValidationError(
                "HMATC inference config must stay inside the workspace",
                context=request.context.diagnostic,
                details={"config": value},
            )
        path = workspace / relative
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return tuple(paths)


def hmatc_inference_artifact_relative_path(request: FlowRequest, workspace: Path) -> Path:
    """Resolve the common HMATC output directory for all components."""
    config_paths = hmatc_inference_config_paths(request, workspace)
    save_dirs: set[str] = set()
    for config_path in config_paths:
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ArtifactValidationError(
                "Failed to read HMATC inference config",
                context=request.context.diagnostic,
                details={"config": config_path, "error": error},
            ) from error
        model = config.get("model") if isinstance(config, Mapping) else None
        save_dir = model.get("save_dir") if isinstance(model, Mapping) else None
        if not isinstance(save_dir, str) or not save_dir:
            raise ArtifactValidationError(
                "HMATC inference config has no model.save_dir",
                context=request.context.diagnostic,
                details={"config": config_path},
            )
        save_dirs.add(save_dir)
    if len(save_dirs) != 1:
        raise ArtifactValidationError(
            "HMATC inference components use different save_dir values",
            context=request.context.diagnostic,
            details={"save_dirs": sorted(save_dirs)},
        )
    relative = Path(next(iter(save_dirs))) / request.context.diagnostic.backend
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactValidationError(
            "HMATC model.save_dir must stay inside the workspace",
            context=request.context.diagnostic,
            details={"save_dir": save_dir},
        )
    return relative


def _hmatc_inference_fingerprint(request, config_paths: tuple[Path, ...]) -> str:
    """Fingerprint all HMATC component configs and their preparation commands."""
    if isinstance(config_paths, Path):
        config_paths = (config_paths,)
    # Use workspace-relative names in the fingerprint.  Basename-only keys
    # would make ``encoder/config.yml`` and ``decoder/config.yml`` collide.
    workspace = _common_config_root(config_paths)
    config_digests = {}
    for config_path in config_paths:
        config_name = _config_fingerprint_name(config_path, workspace)
        try:
            config_digests[config_name] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        except OSError as error:
            raise ArtifactValidationError(
                "Failed to fingerprint HMATC inference config",
                context=request.context.diagnostic,
                details={"config": config_path, "error": error},
            ) from error
    config_args = tuple(_config_fingerprint_name(path, workspace) for path in config_paths)
    return calculate_config_fingerprint(
        {
            "model": request.config.model_name,
            "family": request.config.family.value,
            "backend": request.context.diagnostic.backend,
            "case_id": HMATC_INFERENCE_CASE_ID,
            "configs": config_args,
            "argv": tuple(
                command
                for subcommand in ("quant", "build")
                for command in (f"hmatc {subcommand} --target <backend> --config {name}" for name in config_args)
            ),
            "config_sha256": config_digests,
        }
    )


def _config_fingerprint_name(path: Path, workspace: Path | None) -> str:
    """Return a normalized relative config name for cache fingerprints."""
    if workspace is not None:
        try:
            return path.relative_to(workspace).as_posix()
        except ValueError:
            pass
    return path.name


def _common_config_root(config_paths: tuple[Path, ...]) -> Path | None:
    """Find the shallowest common parent for component config paths."""
    if not config_paths:
        return None
    parents = [path.parent.resolve() for path in config_paths]
    common = parents[0]
    while not all(parent == common or common in parent.parents for parent in parents):
        if common.parent == common:
            return None
        common = common.parent
    return common


def _legacy_hmatc_inference_fingerprint(request, config_paths: tuple[Path, ...]) -> str | None:
    """Fingerprint a legacy cache when all persisted component configs exist."""
    if isinstance(config_paths, Path):
        config_paths = (config_paths,)
    root = _common_config_root(config_paths)
    cached_paths = tuple(
        request.context.result_cache_dir / _config_fingerprint_name(path, root) for path in config_paths
    )
    if not all(path.is_file() for path in cached_paths):
        return None
    return _hmatc_inference_fingerprint(request, cached_paths)


def _hmatc_inference_artifact_reusable(
    request,
    services,
    directory: Path,
    fingerprint: str,
    config_paths: tuple[Path, ...],
) -> bool:
    """Return whether a cached default HMATC inference artifact is reusable."""
    inspection = services.artifact_cache.inspect(
        directory,
        ArtifactRequirement(
            ArtifactType.COMPILED_MODEL,
            request.config.model_name,
            request.context.diagnostic.backend,
            HMATC_INFERENCE_CASE_ID,
        ),
        expected_fingerprint=fingerprint,
    )
    if inspection.status == CacheStatus.VALID:
        return True
    if inspection.status != CacheStatus.LEGACY:
        return False
    legacy_fingerprint = _legacy_hmatc_inference_fingerprint(request, config_paths)
    if legacy_fingerprint != fingerprint:
        logger.info(
            "Do not adopt legacy HMATC artifact with mismatched config: %s",
            directory,
        )
        return False
    try:
        _write_hmatc_inference_manifest(request, services, directory, fingerprint)
    except ArtifactValidationError:
        return False
    return True


def _publish_hmatc_inference_artifact(
    request,
    services,
    source: Path,
    destination: Path,
    fingerprint: str,
) -> None:
    """Atomically publish a newly prepared default HMATC artifact."""
    token = getattr(request.context.diagnostic, "run_id", "hmatc-inference")
    writer = AtomicArtifactWriter(
        destination,
        root=request.context.result_cache_dir,
        token=f"{token}-hmatc-inference",
    )
    prune_compiler_intermediates(source)
    with writer as staging:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        _write_hmatc_inference_manifest(request, services, staging, fingerprint)
        writer.commit()


def _write_hmatc_inference_manifest(request, services, directory: Path, fingerprint: str) -> None:
    """Validate and describe HMATC artifacts required by inference flows."""
    hmm_files = find_nonempty_hmm_files(directory)
    quant_files = find_nonempty_artifact_files(directory / "hmquant", patterns=("*with_act.onnx",))
    if not hmm_files or not quant_files:
        raise ArtifactValidationError(
            "HMATC inference preparation produced incomplete artifacts",
            context=request.context.diagnostic,
            details={
                "directory": directory,
                "hmm_count": len(hmm_files),
                "quant_onnx_count": len(quant_files),
            },
        )
    required_files = {
        **build_required_file_roles(directory, hmm_files, prefix="hmm"),
        **build_required_file_roles(directory, quant_files, prefix="quant_onnx"),
    }
    services.artifact_cache.write_manifest(
        directory,
        ArtifactManifest(
            schema_version=1,
            fingerprint_version=1,
            artifact_type=ArtifactType.COMPILED_MODEL.value,
            model_name=request.config.model_name,
            model_family=request.config.family.value,
            backend=request.context.diagnostic.backend,
            case_id=HMATC_INFERENCE_CASE_ID,
            producer_flow=request.context.diagnostic.flow.value,
            source_type=HMATC_INFERENCE_SOURCE_TYPE,
            config_fingerprint=fingerprint,
            required_files=required_files,
        ),
    )


def run_hmatc_cases(
    request,
    services,
    workspace,
    *,
    section_name,
    subcommand,
    extra_argv=(),
    output_validator=None,
    validation_description="output validation",
    environment=None,
):
    """Run HMATC cases and return its structured execution outcome."""
    section = request.config.section(section_name)
    if section is None:
        return [], [f"missing {section_name}"]
    params = section.get("params", section)
    required = params.get("required") if isinstance(params, Mapping) else None
    optional = params.get("optional") if isinstance(params, Mapping) else None
    if not isinstance(required, Mapping) or not isinstance(optional, Mapping):
        return [], [f"invalid {section_name}.params"]
    matrix = ParameterMatrix.from_columns(
        {**required, **optional},
        location=f"{request.config.model_name}.{section_name}",
    )
    results = []
    failures = []
    for case in matrix.cases:
        result = services.command_runner.run(
            CommandSpec(
                f"hmatc-{subcommand}[{case.index}]",
                (
                    "hmatc",
                    subcommand,
                    "--target",
                    request.context.diagnostic.backend,
                    *extra_argv,
                    *render_case_options(case, skipped_keys={"target", "onnx"}),
                ),
                cwd=workspace,
                allow_nonzero_exit=True,
                log_file=request.context.log_file,
                environment=environment or {},
            ),
            diagnostic_fields=request.context.diagnostic.for_case(case.index, phase=f"hmatc-{subcommand}").as_mapping(),
        )
        results.append(result)
        output = result.combined_output
        if not result.succeeded or output_reports_failure(output):
            failures.append(f"hmatc {subcommand} case {case.index} failed")
        elif output_validator is not None and not output_validator(output):
            failures.append(f"hmatc {subcommand} case {case.index} failed {validation_description}")
    return results, failures


def persist_separate_workspace(request: FlowRequest, workspace: Path, policy: FamilyFlowPolicy) -> None:
    """Persist only HMATC outputs and cache metadata for separate inference."""
    if not policy.persist_workspace_for_separate:
        return
    save_dir = hmatc_inference_artifact_relative_path(request, workspace).parts[0]
    config_paths = hmatc_inference_config_paths(request, workspace)
    request.context.result_cache_dir.mkdir(parents=True, exist_ok=True)
    # Preserve each config's relative directory, then copy the shared output
    # tree.  This keeps separate-infer restoration compatible with nested
    # component configurations while retaining legacy root-level behavior.
    for config_path in config_paths:
        relative = config_path.relative_to(workspace)
        destination = request.context.result_cache_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, destination)
    output_source = workspace / save_dir
    if output_source.is_dir():
        shutil.copytree(
            output_source,
            request.context.result_cache_dir / save_dir,
            dirs_exist_ok=True,
        )
    for entry in workspace.iterdir():
        if entry.name.startswith("artifact_manifest"):
            destination = request.context.result_cache_dir / entry.name
            if entry.is_file():
                shutil.copy2(entry, destination)


__all__ = [
    "hmatc_inference_artifact_relative_path",
    "persist_separate_workspace",
    "restore_reusable_hmatc_inference_artifact",
    "run_hmatc_cases",
    "run_hmatc_inference_preparation",
    "run_hmatc_quant_cases",
]
