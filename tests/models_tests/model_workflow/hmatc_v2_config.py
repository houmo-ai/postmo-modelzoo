# Copyright (c) 2026 HOUMO AI
#
# File: hmatc_v2_config.py
# Description:
#  HMATC v2 Case Parsing, YAML Materialization, and Stable Identity Helpers.
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

"""Parse and materialize the explicitly selected HMATC v2 flow schema."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .artifact_cache_store import calculate_config_fingerprint
from .cache_path_resolver import RESULT_CACHE_ROOT, resolve_cached_path
from .flow_contracts import ConfigError, ModelFlow

__all__ = [
    "HmatcV2Case",
    "deep_merge",
    "hmatc_v2_case_id",
    "hmatc_v2_fingerprint",
    "materialize_hmatc_v2_config",
    "parse_hmatc_v2_cases",
    "resolve_nested_cache_paths",
]


@dataclass(frozen=True)
class HmatcV2Case:
    """One row-style HMATC v2 quant or build case."""

    index: int
    config: str
    override: Mapping[str, Any]
    environment: Mapping[str, str]

    @property
    def logical_quant_output(self) -> str:
        """Return the logical quant artifact root declared by save_dir."""
        return str(self.override["save_dir"])

    @property
    def logical_build_output(self) -> str | None:
        """Return the logical build artifact root when this is a build case."""
        return self.environment.get("HMATC_BUILD_OUTPUT_DIR")

    def source_config_path(self, source_dir: Path) -> Path:
        """Resolve the case's source YAML inside a model workspace."""
        path = (source_dir / self.config).resolve()
        source_root = source_dir.resolve()
        if source_root not in path.parents:
            raise ConfigError(
                "HMATC v2 config escapes the model workspace",
                details={"config": self.config, "source_dir": source_dir},
            )
        return path


def parse_hmatc_v2_cases(config, section_name: str) -> tuple[HmatcV2Case, ...]:
    """Parse and validate one HMATC v2 list section."""
    section = config.raw.get(section_name)
    if not isinstance(section, list) or not section:
        raise ConfigError(
            f"{section_name} must be a non-empty list for hmatc_flow_version=2",
            details={"model": config.model_name, "config": config.path},
        )
    is_build = section_name == "hmbuild_params"
    cases = tuple(
        _parse_hmatc_v2_case(config, section_name, index, value, is_build=is_build)
        for index, value in enumerate(section)
    )
    _validate_unique_case_outputs(config, section_name, cases, is_build=is_build)
    return cases


def _parse_hmatc_v2_case(config, section_name, index, value, *, is_build):
    """Validate and normalize one HMATC v2 case object."""
    location = f"{config.model_name}.{section_name}[{index}]"
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} must be an object")
    unknown = set(value) - {"config", "override"} - {
        key for key in value if isinstance(key, str) and key.startswith("ENV_")
    }
    if unknown:
        raise ConfigError(f"Unknown HMATC v2 keys at {location}: {sorted(unknown)}")

    config_path = _normalize_config_path(value.get("config"), location)
    override = value.get("override")
    if not isinstance(override, Mapping):
        raise ConfigError(f"{location}.override must be an object")
    save_dir = override.get("save_dir")
    if not isinstance(save_dir, str) or not save_dir:
        raise ConfigError(f"{location}.override.save_dir must be a non-empty string")
    _validate_result_cache_path(save_dir, f"{location}.override.save_dir")

    environment = {
        str(key).removeprefix("ENV_"): _environment_value(
            raw_value, f"{location}.{key}"
        )
        for key, raw_value in value.items()
        if isinstance(key, str) and key.startswith("ENV_")
    }
    if is_build:
        output = environment.get("HMATC_BUILD_OUTPUT_DIR")
        if output is None:
            raise ConfigError(
                f"{location} requires ENV_HMATC_BUILD_OUTPUT_DIR"
            )
        _validate_result_cache_path(output, f"{location}.ENV_HMATC_BUILD_OUTPUT_DIR")
    elif environment:
        raise ConfigError(f"{location} quant case does not support ENV_* fields")

    _validate_source_config_exists(config, config_path, location)
    return HmatcV2Case(
        index=index,
        config=config_path,
        override=deepcopy(dict(override)),
        environment=environment,
    )


def _normalize_config_path(value: Any, location: str) -> str:
    """Normalize a repository-relative YAML path."""
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{location}.config must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"{location}.config must be a model-relative path")
    if path.suffix.lower() not in {".yml", ".yaml"}:
        raise ConfigError(f"{location}.config must end with .yml or .yaml")
    return path.as_posix().removeprefix("./")


def _validate_source_config_exists(config, relative_path: str, location: str) -> None:
    """Validate source YAML existence when the repository root is discoverable."""
    tests_dir = next((parent for parent in config.path.parents if parent.name == "tests"), None)
    if tests_dir is None:
        return
    source = tests_dir.parent / config.model_dir / relative_path
    if not source.is_file():
        raise ConfigError(
            f"{location}.config does not exist: {source}",
            details={"model": config.model_name, "config": config.path},
        )


def _environment_value(value: Any, location: str) -> str:
    """Normalize one scalar environment value to a string."""
    if isinstance(value, (Mapping, list, tuple, set)) or value is None:
        raise ConfigError(f"{location} must be a scalar value")
    return str(value)


def _validate_result_cache_path(value: str, location: str) -> None:
    """Require an artifact root below the logical cached_results directory."""
    parts = Path(value.replace("\\", "/")).parts
    if len(parts) < 2 or parts[0] != RESULT_CACHE_ROOT or ".." in parts:
        raise ConfigError(
            f"{location} must be below {RESULT_CACHE_ROOT}/<case>, got {value!r}"
        )


def _validate_unique_case_outputs(config, section_name, cases, *, is_build) -> None:
    """Reject ambiguous v2 cases that publish to the same logical directory."""
    outputs: dict[str, int] = {}
    for case in cases:
        output = case.logical_build_output if is_build else case.logical_quant_output
        assert output is not None
        previous = outputs.get(output)
        if previous is not None:
            raise ConfigError(
                f"Duplicate HMATC v2 output {output!r} in {section_name}",
                details={
                    "model": config.model_name,
                    "first_case": previous,
                    "second_case": case.index,
                },
            )
        outputs[output] = case.index


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive mapping merge without mutating either input."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_nested_cache_paths(
    value: Any,
    *,
    model_cache_dir: Path,
    result_cache_dir: Path,
) -> Any:
    """Resolve cache placeholders recursively through mappings and lists."""
    if isinstance(value, Mapping):
        return {
            key: resolve_nested_cache_paths(
                nested,
                model_cache_dir=model_cache_dir,
                result_cache_dir=result_cache_dir,
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            resolve_nested_cache_paths(
                nested,
                model_cache_dir=model_cache_dir,
                result_cache_dir=result_cache_dir,
            )
            for nested in value
        ]
    return resolve_cached_path(
        value,
        model_cache_dir=model_cache_dir,
        result_cache_dir=result_cache_dir,
    )


def hmatc_v2_case_id(case: HmatcV2Case) -> str:
    """Build a readable case id with a digest for non-path overrides."""
    semantic = _non_path_values(case.override)
    stem = Path(case.config).stem
    if not semantic:
        return stem
    digest = calculate_config_fingerprint({"override": semantic}).removeprefix(
        "sha256:"
    )[:8]
    return f"{stem}-{digest}"


def _non_path_values(value: Any, key: str | None = None) -> Any:
    """Remove path-valued fields before calculating the readable case id."""
    if key is not None and _is_path_key(key):
        return None
    if isinstance(value, Mapping):
        values = {
            nested_key: nested_value
            for nested_key, raw_value in value.items()
            for nested_value in (_non_path_values(raw_value, str(nested_key)),)
            if nested_value is not None and nested_value != {} and nested_value != []
        }
        return values
    if isinstance(value, list):
        return [_non_path_values(item) for item in value]
    return value


def _is_path_key(key: str) -> bool:
    """Return whether a YAML key conventionally carries a filesystem path."""
    normalized = key.replace("-", "_").lower()
    return normalized in {"save_dir", "model_path", "output"} or normalized.endswith(
        ("_dir", "_path")
    )


def hmatc_v2_fingerprint(
    case: HmatcV2Case,
    source_dir: Path,
    *,
    backend: str,
    flow: ModelFlow,
    upstream_quant_fingerprint: str | None = None,
) -> str:
    """Calculate a stable fingerprint from logical configuration inputs."""
    source = case.source_config_path(source_dir)
    try:
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise ConfigError(f"Failed to read HMATC v2 config {source}: {error}") from error
    return calculate_config_fingerprint(
        {
            "protocol": "hmatc_v2",
            "source_config": case.config,
            "source_digest": source_digest,
            "override": case.override,
            "environment": case.environment,
            "backend": backend,
            "flow": flow.value,
            "command": f"hmatc {flow.value if flow == ModelFlow.QUANT else 'build'} --target <backend> --config <effective-yaml>",
            "upstream_quant_fingerprint": upstream_quant_fingerprint,
        }
    )


def materialize_hmatc_v2_config(
    case: HmatcV2Case,
    source_dir: Path,
    staging_dir: Path,
    *,
    flow: ModelFlow,
    fingerprint: str,
    execution_override: Mapping[str, Any],
) -> Path:
    """Write one effective HMATC YAML directly into artifact staging."""
    source = case.source_config_path(source_dir)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"Failed to load HMATC v2 config {source}: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"HMATC v2 config must contain a top-level object: {source}")
    effective = deep_merge(loaded, execution_override)
    config_dir = staging_dir / ".imodelzoo_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    suffix = "hmquant" if flow == ModelFlow.QUANT else "hmbuild"
    target = config_dir / (
        f"{source.stem}.pytest-{suffix}-{fingerprint.removeprefix('sha256:')[:8]}"
        f"{source.suffix.lower()}"
    )
    try:
        target.write_text(
            yaml.safe_dump(effective, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"Failed to write HMATC v2 config {target}: {error}") from error
    return target
