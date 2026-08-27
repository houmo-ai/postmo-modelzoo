# Copyright (c) 2026 HOUMO AI
#
# File: formatter.py
# Description:
#   Human-readable formatting for PostMo performance summaries.
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

"""Human-readable fixed-length performance summary."""

from .result import CaseResult


_RUNTIME_ORDER = {name: index for index, name in enumerate(("load_model", "set_input", "run", "get_output"))}


def _stats_line(name: str, stats) -> str:
    return (
        f"{name:<28} {stats.count:>5} {stats.total_ms:>12.3f} "
        f"{stats.avg_ms:>10.3f} {stats.min_ms:>10.3f} {stats.max_ms:>10.3f}"
    )


def _runtime_lines(report) -> list[str]:
    rows = []
    for path, stats in report.scopes.items():
        parts = path.split(".")
        if len(parts) == 3 and parts[-1] in _RUNTIME_ORDER:
            rows.append((parts[0], parts[1], parts[2], stats))
    rows.sort(key=lambda row: (row[0], row[1], _RUNTIME_ORDER[row[2]]))
    if not rows:
        return []
    lines = [
        "",
        "Runtime Operation",
        f"{'Model/Role/Operation':<28} {'Count':>5} {'Total(ms)':>12} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}",
        "-" * 86,
    ]
    lines.extend(_stats_line(f"{category}.{role}.{operation}", stats) for category, role, operation, stats in rows)
    return lines


def _stage_lines(report) -> list[str]:
    rows = []
    for path in ("postmo.prefill", "postmo.decode", "postmo.e2e"):
        stats = report.scopes.get(path)
        if stats is not None:
            rows.append((path, stats))
    if not rows:
        return []
    lines = [
        "",
        "Stage Timing",
        f"{'Stage':<28} {'Count':>5} {'Total(ms)':>12} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}",
        "-" * 86,
    ]
    lines.extend(_stats_line(name, stats) for name, stats in rows)
    return lines


def format_case(result: CaseResult) -> str:
    report = result.average
    lines = [
        f"Model: {result.model_name}",
        f"Input tokens: {result.input_tokens}",
        f"Decode tokens: {result.output_tokens}",
        f"Warmup: {result.warmup}",
        f"Loops: {len(result.loops)}",
    ]
    lines.extend(_stage_lines(report))
    lines.extend(_runtime_lines(report))
    derived = report.derived.get("llm", {})
    if derived:
        lines.extend(("", "Derived Metrics"))
    for name in (
        "prefill_tps",
        "prefill_runtime_tps",
        "decode_tps",
        "decode_runtime_tps",
        "ttft_ms",
        "tpot_ms",
        "e2e_tps",
    ):
        if name in derived:
            lines.append(f"{name}: {derived[name]:.3f}")
    for path in ("llm.prefill.run", "llm.decode.run"):
        if path in report.scopes:
            lines.append(f"{path}.count: {report.scopes[path].count}")
    return "\n".join(lines)
