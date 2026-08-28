# Copyright (c) 2026 HOUMO AI
#
# File: dumper.py
# Description:
#   Deterministic YAML export for PostMo performance reports.
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

"""Structured, deterministic YAML export for PostMo performance reports."""

import os
import tempfile
from pathlib import Path
from typing import Any

from .stats import PerfReport, ScopeStats

_RUNTIME_OPERATIONS = (
    "prefill_model",
    "decode_model",
    "set_input",
    "run",
    "get_output",
)
_OPERATION_ORDER = {name: index for index, name in enumerate(_RUNTIME_OPERATIONS)}
_DERIVED_KEYS = ("ttft_ms", "e2e_ms")


def _validate_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty trimmed string")
    if model_name != model_name.strip():
        raise ValueError("model_name must be a non-empty trimmed string")
    return model_name


def _stats_dict(stats: ScopeStats) -> dict[str, Any]:
    return {
        "count": stats.count,
        "total_ms": stats.total_ms,
        "min_ms": (
            None
            if stats.count is None
            else stats.min_ms if stats.count else 0.0
        ),
        "max_ms": stats.max_ms if stats.count is not None else None,
        "avg_ms": stats.avg_ms,
    }


def _report_data(report: PerfReport, model_name: str) -> dict[str, Any]:
    model_operations: list[dict[str, Any]] = []
    custom_timings: list[dict[str, Any]] = []

    for path, stats in report.scopes.items():
        parts = path.split(".")
        operation = parts[-1]
        if operation in _RUNTIME_OPERATIONS:
            if len(parts) != 3:
                raise ValueError(
                    "runtime operation path must have exactly three "
                    f"components: {path!r}"
                )
            category, role = parts[:2]
            model_operations.append(
                {
                    "model_category": category,
                    "model_name": model_name,
                    "model_role": role,
                    "operation": operation,
                    **_stats_dict(stats),
                }
            )
        else:
            custom_timings.append({"name": path, "model_name": model_name, **_stats_dict(stats)})

    model_operations.sort(
        key=lambda item: (
            item["model_category"],
            item["model_role"],
            _OPERATION_ORDER[item["operation"]],
        )
    )
    custom_timings.sort(key=lambda item: item["name"])

    derived: list[dict[str, Any]] = []
    for category, values in sorted(report.derived.items()):
        selected = {key: values[key] for key in _DERIVED_KEYS if key in values}
        if selected:
            derived.append({"model_category": category, "model_name": model_name, **selected})

    return {
        "schema_version": 1,
        "time_unit": "ms",
        "model_name": model_name,
        "model_operations": model_operations,
        "custom_timings": custom_timings,
        "derived": derived,
    }


def dumps_yaml(report: PerfReport, *, model_name: str) -> str:
    """Serialize a report using safe, deterministic YAML output."""
    import yaml

    model_name = _validate_model_name(model_name)
    return yaml.safe_dump(
        _report_data(report, model_name),
        sort_keys=False,
        allow_unicode=True,
    )


def dump_yaml(
    report: PerfReport,
    path: str | os.PathLike[str],
    *,
    model_name: str,
    overwrite: bool = True,
) -> None:
    """Atomically publish a YAML report to an existing parent directory."""
    target = Path(path)
    parent = target.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"parent directory does not exist: {parent}")
    content = dumps_yaml(report, model_name=model_name).encode("utf-8")
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
            temporary = None
        else:
            os.link(temporary, target)
            os.unlink(temporary)
            temporary = None
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


__all__ = ["dump_yaml", "dumps_yaml"]
