# Copyright (c) 2025 HOUMO AI
#
# File: model_config.py
# Description:
#  Model Configuration Loading, Normalization, and Validation.
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

"""Load, normalize, and validate model JSON configuration contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .backend_flow_policies import BACKEND_POLICIES
from .cache_path_resolver import RESULT_CACHE_ROOT, cache_case_reference
from .flow_contracts import ConfigError, ModelFamily, ModelFlow
from .perf_metric_validation import (
    has_custom_perf_execution,
    normalize_perf_baseline,
    resolve_perf_behavior,
)

__all__ = [
    "ModelConfig",
    "ModelConfigRepository",
    "ResolvedValue",
    "validate_column_lengths",
]


@dataclass(frozen=True)
class ResolvedValue:
    """A resolved configuration value together with its source location."""

    value: Any
    source: str


@dataclass(frozen=True)
class ModelConfig:
    """Immutable normalized view of one model configuration file."""

    model_name: str
    path: Path
    family: ModelFamily
    model_dir: Path
    obsolete: bool
    dependencies: Mapping[str, Any]
    support_platform: tuple[str, ...]
    support_backend: tuple[str, ...]
    support_flow: Mapping[str, tuple[str, ...]]
    raw: Mapping[str, Any]

    def value(self, name: str, default: Any = None) -> Any:
        """Return a top-level compatibility value.

        Runtime handlers use this boundary instead of reaching into ``raw`` so
        legacy JSON migration remains isolated to the configuration module.
        """
        return self.raw.get(name, default)

    @property
    def hmatc_flow_version(self) -> int:
        """Return the explicitly selected HMATC flow protocol version."""
        value = self.raw.get("hmatc_flow_version", 1)
        if type(value) is not int or value not in (1, 2):
            raise ConfigError(f"hmatc_flow_version must be 1 or 2 in {self.path}, got {value!r}")
        return int(value)

    @property
    def uses_hmatc_v2(self) -> bool:
        """Return whether this model explicitly selects the HMATC v2 flow."""
        return self.hmatc_flow_version == 2

    def hmatc_v2_cases(self, name: str):
        """Parse one HMATC v2 case-list section."""
        if not self.uses_hmatc_v2:
            raise ConfigError(f"{name} requires hmatc_flow_version=2 in {self.path}")
        from .hmatc_v2_config import parse_hmatc_v2_cases

        return parse_hmatc_v2_cases(self, name)

    def has_hmatc_v2_cases(self, name: str) -> bool:
        """Return whether a non-empty HMATC v2 case-list exists."""
        return self.uses_hmatc_v2 and isinstance(self.raw.get(name), list) and bool(self.raw.get(name))

    def section(self, name: str) -> Mapping[str, Any] | None:
        """Return an optional raw configuration section."""
        value = self.raw.get(name)
        return value if isinstance(value, Mapping) else None

    def require_section(self, name: str) -> Mapping[str, Any]:
        """Return a required configuration section or raise a config error."""
        section = self.section(name)
        if section is None:
            raise ConfigError(f"Missing or invalid {name} in {self.path}")
        return section

    def has_section(self, name: str) -> bool:
        """Return whether a mapping configuration section exists."""
        return self.section(name) is not None

    def supports(self, backend: str, flow: ModelFlow) -> bool:
        """Return whether the model supports the backend and flow."""
        return backend in self.support_backend and flow.value in self.support_flow.get(backend, ())

    def backend_section(self, name: str, backend: str) -> Mapping[str, Any] | None:
        """Resolve a backend-specific configuration section."""
        section = self.section(name)
        if section is None:
            return None
        value = section.get(backend)
        return value if isinstance(value, Mapping) else None

    def hmatc_columns(self, name: str, *, backend: str | None = None) -> Mapping[str, Any]:
        """Return the merged required/optional columns for an hmatc section."""
        section = self.backend_section(name, backend) if backend is not None else self.require_section(name)
        if section is None:
            raise ConfigError(f"Missing {name}.{backend} in {self.path}")
        params = section.get("params", section)
        required = params.get("required") if isinstance(params, Mapping) else None
        optional = params.get("optional") if isinstance(params, Mapping) else None
        if not isinstance(required, Mapping) or not isinstance(optional, Mapping):
            location = f"{name}.{backend}" if backend is not None else name
            raise ConfigError(f"{location} requires required/optional objects in {self.path}")
        return {**required, **optional}

    def demo_enabled(self) -> bool:
        """Return whether Python demo execution is enabled."""
        return bool(self.value("enable_demo_test", True))

    def supported_core_count(self, backend: str) -> Any:
        """Return whether the requested core count is supported."""
        values = self.section("support_core_num")
        return values.get(backend) if values is not None else None

    def eval_thresholds(self) -> Mapping[str, Any] | None:
        """Return normalized evaluation thresholds."""
        return self.section("eval_threshold")

    def referenced_result_case_ids(self, backend: str) -> set[str]:
        """Return result-cache case identifiers referenced by configuration."""
        return {case_id for root, case_id in self.referenced_artifact_case_refs(backend) if root == RESULT_CACHE_ROOT}

    def referenced_artifact_case_refs(self, backend: str) -> set[tuple[str, str]]:
        """Return the cache root and case identifier pairs demos reference.

        Demo parameters may point at artifact directories under either cache
        root, including directories that no compile case produces.  Artifact
        preparation needs the cache root as well as the case identifier to
        locate those directories.
        """
        result: set[tuple[str, str]] = set()
        for section_name in ("demo_params", "demo_multibatch_params"):
            columns = self.backend_section(section_name, backend)
            result.update(self._section_artifact_refs(columns))
        return result

    @staticmethod
    def _section_artifact_refs(columns: Mapping[str, Any] | None) -> set[tuple[str, str]]:
        """Extract cache references from one backend parameter section."""
        if columns is None:
            return set()
        return {
            reference
            for values in columns.values()
            if isinstance(values, list)
            for value in values
            if isinstance(value, str)
            for reference in (cache_case_reference(value),)
            if reference is not None
        }

    def validation_threshold(self, backend: str, kind: str) -> ResolvedValue:
        """Resolve a validation threshold and record its source."""
        key = f"{kind}_cosine_threshold"
        validation = self.raw.get("validation")
        if isinstance(validation, Mapping):
            backend_validation = validation.get(backend)
            if isinstance(backend_validation, Mapping) and key in backend_validation:
                return ResolvedValue(
                    float(backend_validation[key]),
                    f"model.validation.{backend}.{key}",
                )
        try:
            policy = BACKEND_POLICIES[backend]
        except KeyError as error:
            raise ConfigError(f"Unsupported backend policy: {backend}") from error
        value = getattr(policy, key)
        return ResolvedValue(value, f"backend_defaults.{backend}.{key}")

    def perf_contract(self, backend: str, platform: str):
        """Build the normalized performance validation contract."""
        metrics = self.raw.get("perf_metrics")
        if not isinstance(metrics, Mapping):
            raise ConfigError(f"Missing perf_metrics in {self.path}")
        baseline = normalize_perf_baseline(metrics, backend, platform)
        default_runner = "demo" if self.raw.get("perf_params") == "demo" else "hmatc"
        behavior = resolve_perf_behavior(
            self.model_name,
            backend=backend,
            baseline_keys=tuple(baseline),
            default_runner=default_runner,
        )
        return baseline, behavior


class ModelConfigRepository:
    """Repository that discovers model JSON files and validates their schema."""

    def __init__(self, config_dir: Path) -> None:
        """Initialize the Model Config Repository."""
        self.config_dir = config_dir

    def resolve_path(self, model_name: str) -> Path:
        """Resolve a model configuration path by model name."""
        return self.config_dir / f"model_cfg_{model_name}.json"

    def load(self, model_name: str) -> ModelConfig:
        """Load, normalize, and validate one model configuration."""
        path = self.resolve_path(model_name)
        if not path.is_file():
            raise ConfigError(
                f"Model configuration does not exist: {path}",
                details={"model_name": model_name, "config": path},
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigError(
                f"Failed to load model configuration: {path}",
                details={"error": error},
            ) from error
        if not isinstance(raw, dict):
            raise ConfigError(f"Model configuration must be a JSON object: {path}")
        return self._normalize(model_name, path, raw)

    def iter_configs(self, *, include_obsolete: bool = False) -> Iterable[ModelConfig]:
        """Iterate over normalized model configurations in stable order."""
        for path in sorted(self.config_dir.glob("model_cfg_*.json")):
            if "template" in path.name:
                continue
            model_name = path.stem.removeprefix("model_cfg_")
            config = self.load(model_name)
            if include_obsolete or not config.obsolete:
                yield config

    def _normalize(self, model_name: str, path: Path, raw: Mapping[str, Any]) -> ModelConfig:
        """Normalize raw JSON data into the immutable model configuration."""
        try:
            family = ModelFamily(str(raw.get("model_type", "cv")))
        except ValueError as error:
            raise ConfigError(f"Unsupported model_type in {path}: {raw.get('model_type')}") from error

        model_dir = raw.get("model_dir")
        if not isinstance(model_dir, str) or not model_dir:
            raise ConfigError(f"Missing or invalid model_dir in {path}")
        model_dir_path = Path(model_dir)
        if model_dir_path.is_absolute() or ".." in model_dir_path.parts:
            raise ConfigError(f"model_dir must be a repository-relative path in {path}")

        support_backend = raw.get("support_backend", ())
        support_platform = raw.get("support_platform", ())
        support_flow = raw.get("support_flow", {})
        if not isinstance(support_backend, list) or not all(isinstance(item, str) for item in support_backend):
            raise ConfigError(f"Invalid support_backend in {path}")
        if not isinstance(support_platform, list) or not all(isinstance(item, str) for item in support_platform):
            raise ConfigError(f"Invalid support_platform in {path}")
        if not isinstance(support_flow, Mapping):
            raise ConfigError(f"Invalid support_flow in {path}")

        normalized_flows: dict[str, tuple[str, ...]] = {}
        for backend, flows in support_flow.items():
            if not isinstance(flows, list) or not all(isinstance(item, str) for item in flows):
                raise ConfigError(f"Invalid support_flow.{backend} in {path}")
            normalized_flows[str(backend)] = tuple(flows)

        config = ModelConfig(
            model_name=model_name,
            path=path,
            family=family,
            model_dir=model_dir_path,
            obsolete=bool(raw.get("obsolete", False)),
            dependencies=raw.get("dependencies", {}),
            support_platform=tuple(support_platform),
            support_backend=tuple(support_backend),
            support_flow=normalized_flows,
            raw=raw,
        )
        if not config.obsolete:
            self._validate_active(config)
        return config

    def _validate_active(self, config: ModelConfig) -> None:
        """Validate invariants required by every active model configuration."""
        self._validate_required_fields(config)
        self._validate_backend_flows(config)
        self._validate_parameter_sections(config)
        self._validate_hmatc_sections(config)
        self._validate_flow_sections(config)
        self._validate_perf_contracts(config)
        self._validate_thresholds(config)

    @staticmethod
    def _validate_required_fields(config: ModelConfig) -> None:
        """Validate fields shared by every active model configuration."""
        if not config.support_backend:
            raise ConfigError(f"support_backend cannot be empty in {config.path}")
        if not config.support_platform:
            raise ConfigError(f"support_platform cannot be empty in {config.path}")
        if not isinstance(config.dependencies, Mapping):
            raise ConfigError(f"dependencies must be an object in {config.path}")

    @staticmethod
    def _validate_backend_flows(config: ModelConfig) -> None:
        """Validate backend policies and flow names declared by a model."""
        known_flows = {flow.value for flow in ModelFlow}
        for backend in config.support_backend:
            if backend not in BACKEND_POLICIES:
                raise ConfigError(
                    f"Unsupported backend in active config: {backend}",
                    details={"model": config.model_name, "config": config.path},
                )
            if backend not in config.support_flow:
                raise ConfigError(
                    f"support_backend contains {backend} but support_flow is missing it",
                    details={"model": config.model_name, "config": config.path},
                )
            unknown = set(config.support_flow[backend]) - known_flows - {"demo_multibatch"}
            if unknown:
                raise ConfigError(
                    f"Unknown flows in support_flow.{backend}: {sorted(unknown)}",
                    details={"model": config.model_name, "config": config.path},
                )
        ModelConfigRepository._validate_static_dependency_topology(config)

    @staticmethod
    def _validate_parameter_sections(config: ModelConfig) -> None:
        """Validate column lengths for standard flow parameter sections."""
        for section_name in (
            "get_model_params",
            "quant_params",
            "compile_params",
            "demo_params",
            "demo_multibatch_params",
        ):
            section = config.raw.get(section_name)
            if not isinstance(section, Mapping):
                continue
            for backend, columns in section.items():
                if isinstance(columns, Mapping):
                    validate_column_lengths(
                        columns,
                        location=f"{config.model_name}.{section_name}.{backend}",
                        ignored_keys={"prerequisites"},
                    )

    @staticmethod
    def _validate_perf_contracts(config: ModelConfig) -> None:
        """Validate perf contracts for every supported backend and platform."""
        for backend in config.support_backend:
            if ModelFlow.PERF.value not in config.support_flow.get(backend, ()):
                continue
            for platform in config.support_platform:
                config.perf_contract(backend, platform)

    @staticmethod
    def _validate_thresholds(config: ModelConfig) -> None:
        """Validate optional cosine-threshold overrides."""
        validation = config.raw.get("validation")
        if validation is None:
            return
        if not isinstance(validation, Mapping):
            raise ConfigError(f"validation must be an object in {config.path}")
        for backend, values in validation.items():
            ModelConfigRepository._validate_threshold_backend(config, backend, values)

    @staticmethod
    def _validate_threshold_backend(config: ModelConfig, backend: str, values: Any) -> None:
        """Validate one backend's optional threshold overrides."""
        if backend not in config.support_backend or not isinstance(values, Mapping):
            raise ConfigError(f"Invalid validation.{backend} in {config.path}")
        allowed = {"compile_cosine_threshold", "compare_cosine_threshold"}
        unknown = set(values) - allowed
        if unknown:
            raise ConfigError(f"Unknown validation keys in {config.path}: {sorted(unknown)}")
        for key, value in values.items():
            if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                raise ConfigError(f"validation.{backend}.{key} must be between 0 and 1")

    @staticmethod
    def _validate_static_dependency_topology(config: ModelConfig) -> None:
        """One generated node must have the same prerequisite on every backend."""
        quant_dependencies: set[str | None] = set()
        compile_dependencies: set[str | None] = set()
        for backend in config.support_backend:
            flows = set(config.support_flow.get(backend, ()))
            if ModelFlow.QUANT.value in flows:
                quant_dependencies.add(ModelFlow.GET_MODEL.value if ModelFlow.GET_MODEL.value in flows else None)
            if ModelFlow.COMPILE.value in flows:
                dependency = None
                if ModelFlow.QUANT.value in flows:
                    dependency = ModelFlow.QUANT.value
                elif ModelFlow.GET_MODEL.value in flows:
                    dependency = ModelFlow.GET_MODEL.value
                compile_dependencies.add(dependency)
        if len(quant_dependencies) > 1 or len(compile_dependencies) > 1:
            raise ConfigError(
                "Flow prerequisites differ across backends but tests use one static node",
                details={
                    "model": config.model_name,
                    "quant_dependencies": sorted(map(str, quant_dependencies)),
                    "compile_dependencies": sorted(map(str, compile_dependencies)),
                },
            )

    @staticmethod
    def _validate_hmatc_sections(config: ModelConfig) -> None:
        """Validate HMATC sections and raise a structured error when invalid."""
        if config.uses_hmatc_v2:
            ModelConfigRepository._validate_hmatc_v2_sections(config)
            return
        for section_name in (
            "hmquant_params",
            "hmdemo_params",
            "hmcompare_params",
            "hmeval_params",
            "hmperf_params",
        ):
            ModelConfigRepository._validate_hmatc_section(config, section_name)
        ModelConfigRepository._validate_hmatc_build_sections(config)

    @staticmethod
    def _validate_hmatc_v2_sections(config: ModelConfig) -> None:
        """Validate the independent HMATC v2 quant/build schema."""
        unsupported = tuple(
            name
            for name in (
                "hmdemo_params",
                "hmcompare_params",
                "hmeval_params",
                "hmperf_params",
            )
            if config.raw.get(name) is not None
        )
        if unsupported:
            raise ConfigError(
                "hmatc_flow_version=2 supports only hmquant_params and "
                f"hmbuild_params, found {unsupported} in {config.path}"
            )
        for section_name in ("hmquant_params", "hmbuild_params"):
            if section_name in config.raw:
                config.hmatc_v2_cases(section_name)

    @staticmethod
    def _validate_hmatc_section(config: ModelConfig, section_name: str) -> None:
        """Validate one shared HMATC parameter section."""
        section = config.raw.get(section_name)
        if not isinstance(section, Mapping):
            return
        params = section.get("params")
        ModelConfigRepository._validate_required_optional_params(
            config,
            params,
            location=f"{config.model_name}.{section_name}.params",
            error_location=f"{section_name}.params",
        )

    @staticmethod
    def _validate_hmatc_build_sections(config: ModelConfig) -> None:
        """Validate backend-specific HMATC build sections."""
        build = config.raw.get("hmbuild_params")
        if not isinstance(build, Mapping):
            return
        for backend, params in build.items():
            ModelConfigRepository._validate_required_optional_params(
                config,
                params,
                location=f"{config.model_name}.hmbuild_params.{backend}",
                error_location=f"hmbuild_params.{backend}",
            )

    @staticmethod
    def _validate_required_optional_params(
        config: ModelConfig,
        params: Any,
        *,
        location: str,
        error_location: str,
    ) -> None:
        """Validate required/optional parameter columns at one config location."""
        if not isinstance(params, Mapping):
            required = optional = None
        else:
            required = params.get("required")
            optional = params.get("optional")
        if not isinstance(required, Mapping) or not isinstance(optional, Mapping):
            raise ConfigError(
                f"{error_location} requires required/optional objects",
                details={"model": config.model_name, "config": config.path},
            )
        validate_column_lengths({**required, **optional}, location=location)

    @staticmethod
    def _validate_flow_sections(config: ModelConfig) -> None:
        """Validate flow sections and raise a structured error when invalid."""
        for backend in config.support_backend:
            flows = set(config.support_flow[backend])
            requirements = ModelConfigRepository._flow_requirements(config, backend)
            missing = [flow for flow, present in requirements.items() if flow in flows and not present]
            if missing:
                raise ConfigError(
                    f"Supported flows are missing parameter sections: {missing}",
                    details={
                        "model": config.model_name,
                        "backend": backend,
                        "config": config.path,
                    },
                )
            ModelConfigRepository._validate_flow_family(config, backend, flows)
            ModelConfigRepository._validate_multibatch(config, backend, flows)

    @staticmethod
    def _flow_requirements(config: ModelConfig, backend: str) -> dict[str, bool]:
        """Return whether each flow has a usable parameter section."""
        hmbuild = config.raw.get("hmbuild_params")
        v2_quant = config.has_hmatc_v2_cases("hmquant_params")
        v2_build = config.has_hmatc_v2_cases("hmbuild_params")
        return {
            ModelFlow.GET_MODEL.value: config.backend_section("get_model_params", backend) is not None,
            ModelFlow.QUANT.value: v2_quant
            or isinstance(config.raw.get("hmquant_params"), Mapping)
            or config.backend_section("quant_params", backend) is not None,
            ModelFlow.COMPILE.value: v2_build
            or (isinstance(hmbuild, Mapping) and isinstance(hmbuild.get(backend), Mapping))
            or config.backend_section("compile_params", backend) is not None,
            ModelFlow.DEMO.value: isinstance(config.raw.get("hmdemo_params"), Mapping)
            or config.backend_section("demo_params", backend) is not None,
            ModelFlow.COMPARE.value: isinstance(config.raw.get("hmcompare_params"), Mapping),
            ModelFlow.EVAL.value: isinstance(config.raw.get("hmeval_params"), Mapping)
            and isinstance(config.raw.get("eval_threshold"), Mapping),
            ModelFlow.PERF.value: isinstance(config.raw.get("hmperf_params"), Mapping)
            or config.raw.get("perf_params") == "demo"
            or has_custom_perf_execution(config.model_name, backend),
        }

    @staticmethod
    def _validate_flow_family(config: ModelConfig, backend: str, flows: set[str]) -> None:
        """Reject flow combinations that have no registered handler."""
        unsupported = {"compare", "eval"} & flows
        if config.family == ModelFamily.LLM and unsupported:
            raise ConfigError(
                f"LLM models do not support {sorted(unsupported)} flows",
                details={"model": config.model_name, "backend": backend, "config": config.path},
            )

    @staticmethod
    def _validate_multibatch(config: ModelConfig, backend: str, flows: set[str]) -> None:
        """Ensure demo_multibatch has its matching parameter section."""
        if "demo_multibatch" in flows and config.backend_section("demo_multibatch_params", backend) is None:
            raise ConfigError(
                "demo_multibatch requires demo_multibatch_params",
                details={"model": config.model_name, "backend": backend, "config": config.path},
            )


def validate_column_lengths(
    columns: Mapping[str, Any],
    *,
    location: str,
    ignored_keys: set[str] | None = None,
) -> None:
    """Validate that parameter columns have compatible lengths."""
    ignored_keys = ignored_keys or set()
    lengths = {key: len(value) for key, value in columns.items() if key not in ignored_keys and isinstance(value, list)}
    if not lengths:
        return
    expected = next(iter(lengths.values()))
    if expected == 0:
        raise ConfigError(
            f"Parameter matrix cannot be empty at {location}",
            details={"lengths": lengths},
        )
    mismatched = {key: value for key, value in lengths.items() if value != expected}
    if mismatched:
        raise ConfigError(
            f"Parameter columns must have equal lengths at {location}",
            details={"lengths": lengths},
        )
