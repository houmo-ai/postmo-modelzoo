# Copyright (c) 2025 HOUMO AI
#
# File: perf_validation.py
# Description:
#  Performance Metric Extraction, Behavior Selection, and Validation.
#  Model JSON stores baselines and command parameters only. Most demo/hmatc output
#    uses the default key-value parser; the small number of exceptional formats are
#    registered explicitly in :data:`PERF_BEHAVIOR_OVERRIDES`.
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

"""Resolve performance runners and parse/validate their reported metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from .flow_contracts import ConfigError, ResultParseError, ValidationResult

__all__ = [
    "CustomScriptSpec",
    "PerfBehavior",
    "PerfBehaviorOverride",
    "PerfMetricSpec",
    "extract_perf_metrics",
    "has_custom_perf_execution",
    "normalize_perf_baseline",
    "resolve_perf_behavior",
    "validate_perf_metrics",
]

# Number (supports optional sign, decimal, scientific notation), reused by multiple extractors
_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class PerfMetricSpec:
    """Parser and comparison rules for one performance metric."""

    name: str
    key: str
    extractor: str = "key_value"
    pattern: str | None = None
    group: str | int = "value"
    direction: str = "higher"
    aggregation: str = "max"


@dataclass(frozen=True)
class PerfBehavior:
    """Resolved runner, output source, and metric rules for a perf flow."""

    runner: str
    source: str
    metrics: Mapping[str, PerfMetricSpec]
    custom_script: "CustomScriptSpec | None" = None


@dataclass(frozen=True)
class CustomScriptSpec:
    """Code-owned specification for a non-standard performance script."""

    script: str
    params_section: str
    parameter_keys: tuple[str, ...]
    case_index: int = 0


@dataclass(frozen=True)
class PerfBehaviorOverride:
    """Optional model/backend override layered over default perf behavior."""

    runner: str | None = None
    source: str | None = None
    metrics: Mapping[str, PerfMetricSpec] | None = None
    custom_script: CustomScriptSpec | None = None


PERF_BEHAVIOR_OVERRIDES: Mapping[tuple[str, str], PerfBehaviorOverride] = {
    ("sdxl", "xh1"): PerfBehaviorOverride(
        metrics={
            "avg_cost": PerfMetricSpec(
                name="avg_cost",
                key="avg_cost",
                extractor="regex",
                pattern=r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*ms,\s*average",
                direction="lower",
                aggregation="min",
            )
        }
    ),
    # Wenet is currently obsolete, but keeping its former behavior here makes a
    # future re-enable explicit without putting execution/parser DSL in JSON.
    ("wenet", "xh1"): PerfBehaviorOverride(
        runner="custom_script",
        source="stdout",
        custom_script=CustomScriptSpec(
            script="build.py",
            params_section="compile_params",
            parameter_keys=("model_dir",),
            case_index=0,
        ),
        metrics={
            "qps": PerfMetricSpec(
                name="qps",
                key="qps",
                extractor="regex",
                pattern=(r"infer completed.*\s(?P<value>[0-9]+(?:\.[0-9]+)?)\s+\S+\s*$"),
                aggregation="first",
            )
        },
    ),
}


def has_custom_perf_execution(model_name: str, backend: str) -> bool:
    """Return whether a custom performance runner is configured."""
    override = PERF_BEHAVIOR_OVERRIDES.get((model_name, backend))
    return override is not None and override.runner == "custom_script"


def normalize_perf_baseline(perf_metrics: Mapping[str, Any], backend: str, platform: str) -> dict[str, float]:
    """Normalize legacy scalar/platform baselines into a metric dictionary."""
    backend_value = perf_metrics.get(backend)
    if not isinstance(backend_value, Mapping):
        raise ConfigError(f"Missing perf_metrics.{backend}")
    selected: Any = backend_value.get(platform, backend_value)
    if isinstance(selected, (int, float)):
        return {"qps": float(selected)}
    if not isinstance(selected, Mapping):
        raise ConfigError(f"Invalid perf baseline for {backend}/{platform}")
    result = {str(key): float(value) for key, value in selected.items() if isinstance(value, (int, float))}
    if not result:
        raise ConfigError(f"Empty perf baseline for {backend}/{platform}")
    return result


def resolve_perf_behavior(
    model_name: str,
    *,
    backend: str,
    baseline_keys: Sequence[str],
    default_runner: str,
    default_source: str = "log_file",
) -> PerfBehavior:
    """Resolve code-owned parsing and validation behavior for a model."""
    override = PERF_BEHAVIOR_OVERRIDES.get((model_name, backend))
    runner = override.runner if override and override.runner else default_runner
    source = override.source if override and override.source else default_source
    metric_overrides = dict(override.metrics or {}) if override else {}
    if set(metric_overrides) - set(baseline_keys):
        raise ConfigError(
            "Code-side perf override contains metrics missing from perf_metrics",
            details={
                "model": model_name,
                "override": sorted(metric_overrides),
                "baseline": sorted(baseline_keys),
            },
        )
    metrics: dict[str, PerfMetricSpec] = {}
    for name in baseline_keys:
        metrics[name] = metric_overrides.get(
            name,
            PerfMetricSpec(name=name, key=_default_key(name)),
        )
    behavior = PerfBehavior(
        runner=runner,
        source=source,
        metrics=metrics,
        custom_script=override.custom_script if override else None,
    )
    _validate_code_owned_behavior(model_name, behavior)
    return behavior


def _validate_code_owned_behavior(model_name: str, behavior: PerfBehavior) -> None:
    """Validate the code-owned parsing contract for a model override."""
    _validate_runner_and_source(model_name, behavior)
    _validate_custom_script(model_name, behavior)
    for spec in behavior.metrics.values():
        _validate_metric_spec(spec, behavior.runner)


def _validate_runner_and_source(model_name: str, behavior: PerfBehavior) -> None:
    """Validate the runner and output source selected for a model."""
    if behavior.runner not in {"hmatc", "demo", "custom_script"}:
        raise ConfigError(f"Unsupported perf runner for {model_name}: {behavior.runner}")
    if behavior.source not in {"stdout", "stderr", "combined_output", "log_file"}:
        raise ConfigError(f"Unsupported perf source for {model_name}: {behavior.source}")


def _validate_custom_script(model_name: str, behavior: PerfBehavior) -> None:
    """Ensure custom script metadata matches the selected runner."""
    if behavior.runner == "custom_script" and behavior.custom_script is None:
        raise ConfigError(f"Custom perf runner for {model_name} has no script spec")
    if behavior.runner != "custom_script" and behavior.custom_script is not None:
        raise ConfigError(f"Non-custom perf runner for {model_name} has a script spec")


def _validate_metric_spec(spec: PerfMetricSpec, runner: str) -> None:
    """Validate one metric's extraction, aggregation, and regex contract."""
    if spec.extractor not in {"key_value", "regex"}:
        raise ConfigError(f"Unsupported perf extractor: {spec.extractor}")
    if spec.direction not in {"higher", "lower"}:
        raise ConfigError(f"Unsupported perf direction: {spec.direction}")
    if spec.aggregation not in {"max", "min", "first", "last", "average"}:
        raise ConfigError(f"Unsupported perf aggregation: {spec.aggregation}")
    if runner == "hmatc" and spec.name == "qps" and spec.aggregation != "max":
        raise ConfigError("hmatc qps aggregation is fixed to max")
    if spec.extractor == "regex":
        _validate_regex_metric(spec)


def _validate_regex_metric(spec: PerfMetricSpec) -> None:
    """Validate regex and capture-group settings for one metric."""
    if not spec.pattern:
        raise ConfigError(f"regex perf metric {spec.name} requires pattern")
    compiled = re.compile(spec.pattern)
    if isinstance(spec.group, str) and spec.group not in compiled.groupindex:
        raise ConfigError(f"Perf regex for {spec.name} has no group named {spec.group}")
    if isinstance(spec.group, int) and spec.group > compiled.groups:
        raise ConfigError(f"Perf regex for {spec.name} has no group {spec.group}")


def extract_perf_metrics(text: str, behavior: PerfBehavior) -> dict[str, float]:
    """Extract normalized performance metrics from command output."""
    result: dict[str, float] = {}
    for name, spec in behavior.metrics.items():
        values = _extract_values(text, spec)
        if not values:
            raise ResultParseError(
                f"Failed to extract performance metric: {name}",
                details={"key": spec.key, "extractor": spec.extractor},
            )
        result[name] = _aggregate(values, spec.aggregation)
    return result


def validate_perf_metrics(
    actual: Mapping[str, float],
    baseline: Mapping[str, float],
    behavior: PerfBehavior,
    *,
    minimum_ratio: float,
) -> ValidationResult:
    """Validate measured performance metrics against configured baselines."""
    failures: list[str] = []
    for name, expected in baseline.items():
        measured = actual.get(name)
        if measured is None:
            failures.append(f"{name}: metric is missing")
            continue
        spec = behavior.metrics[name]
        if spec.direction == "higher":
            limit = expected * minimum_ratio
            passed = measured >= limit
            relation = ">="
        else:
            limit = expected * (2.0 - minimum_ratio)
            passed = measured <= limit
            relation = "<="
        if not passed:
            failures.append(
                f"{name}: actual={measured} expected {relation} {limit} "
                f"(baseline={expected}, direction={spec.direction})"
            )
    return ValidationResult(
        passed=not failures,
        summary="performance validation passed" if not failures else "; ".join(failures),
        metrics=dict(actual),
        failures=tuple(failures),
    )


def _default_key(name: str) -> str:
    """Derive the default output key for a performance metric."""
    return {
        "qps": "[Throughput] qps",
        "prefill": "Prefill Speed",
        "decode": "Decode Speed",
        "end2end": "E2E TPS",
    }.get(name, name)


def _extract_values(text: str, spec: PerfMetricSpec) -> list[float]:
    """Extract values from normalized command output."""
    if spec.extractor == "regex":
        assert spec.pattern is not None
        return [float(match.group(spec.group)) for match in re.finditer(spec.pattern, text)]
    values = _extract_key_value_values(text, spec.key)
    if values:
        return values
    return _extract_structured_values(text, spec.name)


def _extract_key_value_values(text: str, key: str) -> list[float]:
    """Extract the legacy ``key: value`` performance representation."""
    values: list[float] = []
    number = re.compile(_NUMBER_RE)
    for line in text.splitlines():
        if key not in line:
            continue
        suffix = line.split(key, 1)[1].lstrip(" :=")
        match = number.search(suffix)
        if match:
            values.append(float(match.group(0)))
    return values


def _extract_structured_values(text: str, metric_name: str) -> list[float]:
    """Extract metrics from the newer Timing and Overall Metrics sections."""
    if metric_name not in {"end2end", "ttft", "e2e_latency", "tpot"}:
        return _extract_timing_speed(text, metric_name)
    return _extract_overall_metric(text, metric_name)


def _extract_timing_speed(text: str, scope: str) -> list[float]:
    """Read the ``infer`` speed from a Timing scope and its child rows.

    The aggregate scope speed can vary substantially with machine setup.  When
    available, the parser therefore uses the indented ``infer`` row belonging
    to the requested scope (for example ``prefill -> infer``).  The aggregate
    row remains a compatibility fallback for logs that do not report children.
    """
    aggregate: list[float] = []
    infer_values: list[float] = []
    active_indent: int | None = None
    for indent, columns in _iter_timing_rows(text):
        if active_indent is not None and indent <= active_indent and columns[0] != scope:
            active_indent = None
        if columns[0] == scope:
            active_indent = indent
        speed = _parse_timing_speed(columns)
        if speed is None:
            continue
        if columns[0] == scope:
            aggregate.append(speed)
        elif columns[0] == "infer" and active_indent is not None and indent > active_indent:
            infer_values.append(speed)
    return infer_values or aggregate


def _iter_timing_rows(text: str) -> Iterator[tuple[int, list[str]]]:
    """Yield indented, tokenized rows between the Timing section markers."""
    in_timing = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "Timing":
            in_timing = True
            continue
        if not in_timing:
            continue
        if stripped == "Overall Performance Metrics":
            break
        columns = stripped.split()
        if len(columns) >= 3:
            yield len(line) - len(line.lstrip()), columns


def _parse_timing_speed(columns: Sequence[str]) -> float | None:
    """Parse a numeric Speed column from a Timing row."""
    if columns[-1] not in {"tokens/s", "images/s"}:
        return None
    number = re.compile(_NUMBER_RE)
    match = number.fullmatch(columns[-2])
    return float(match.group(0)) if match else None


def _extract_overall_metric(text: str, metric_name: str) -> list[float]:
    """Read a scalar from the Overall Performance Metrics section."""
    labels = {
        "end2end": r"E2E\s+TPS(?:\s*\([^)]*\))?",
        "ttft": r"TTFT(?:\s*\([^)]*\))?",
        "e2e_latency": r"E2E\s+Latency(?:\s*\([^)]*\))?",
        "tpot": r"TPOT(?:\s*\([^)]*\))?",
    }
    label = labels.get(metric_name)
    if label is None:
        return []
    pattern = re.compile(rf"^\s*{label}\s*:\s*(?P<value>{_NUMBER_RE})\b", re.IGNORECASE)
    return [float(match.group("value")) for match in pattern.finditer(text)]


def _aggregate(values: Sequence[float], aggregation: str) -> float:
    """Aggregate repeated metric samples with the configured strategy."""
    if aggregation == "max":
        return max(values)
    if aggregation == "min":
        return min(values)
    if aggregation == "first":
        return values[0]
    if aggregation == "last":
        return values[-1]
    return sum(values) / len(values)
