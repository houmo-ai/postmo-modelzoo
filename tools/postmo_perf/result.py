# Copyright (c) 2026 HOUMO AI
#
# File: result.py
# Description:
#   Performance loop results and formal-loop average aggregation.
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

"""Result objects and report aggregation."""

from dataclasses import dataclass
from typing import Any

from postmo_engine.perf import PerfReport, ScopeStats, aggregate_parent_scopes


@dataclass(frozen=True)
class LoopResult:
    loop: int
    report: PerfReport


@dataclass(frozen=True)
class CaseResult:
    model_name: str
    input_tokens: int
    output_tokens: int
    warmup: int
    loops: tuple[LoopResult, ...]
    average: PerfReport
    prefill_path: str = ""
    decode_path: str = ""
    embedding_path: str = ""
    visual_path: str = ""
    devices: tuple[int, ...] = (0,)
    batch: int = 1
    lazy_mode: bool = False
    skip_perf: bool = False
    monitor_interval: int = 0
    perf_case_index: int = 1
    perf_case_total: int = 1


def average_reports(reports: list[PerfReport]) -> PerfReport:
    if not reports:
        raise ValueError("at least one report is required")
    scopes: dict[str, ScopeStats] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for report in reports:
        for path, value in report.scopes.items():
            target = scopes.get(path)
            if target is None:
                scopes[path] = value.copy()
                continue
            target.total_ms += value.total_ms
            if target.count is None or value.count is None:
                # Automatically aggregated parents have no comparable sample
                # count; preserve that meaning across loop aggregation.
                target.count = None
                target.min_ms = None
                target.max_ms = None
            else:
                target.count += value.count
                target.min_ms = min(target.min_ms, value.min_ms)
                target.max_ms = max(target.max_ms, value.max_ms)
        for path, values in report.metrics.items():
            metrics.setdefault(path, {}).update(values)
    aggregate = PerfReport(scopes=scopes, metrics=metrics)
    aggregate_parent_scopes(aggregate.scopes)
    from postmo_engine.perf.metrics import derive_metrics, derive_speeds

    aggregate.derived = derive_metrics(aggregate)
    aggregate.speeds = derive_speeds(aggregate)
    llm = aggregate.derived.setdefault("llm", {})
    prefill = aggregate.scopes.get("postmo.prefill")
    decode = aggregate.scopes.get("postmo.decode")
    e2e = aggregate.scopes.get("postmo.e2e")
    input_tokens = aggregate.metrics.get("llm", {}).get("input_tokens")
    decode_tokens = aggregate.metrics.get("llm", {}).get("decode_tokens")
    if prefill and prefill.avg_ms > 0 and input_tokens:
        llm["prefill_tps"] = input_tokens * 1000 / prefill.avg_ms
        llm["ttft_ms"] = prefill.avg_ms
    if decode and decode.avg_ms > 0 and decode_tokens:
        llm["decode_tps"] = decode_tokens * 1000 / decode.avg_ms
        llm["tpot_ms"] = decode.avg_ms / decode_tokens
    if e2e and e2e.avg_ms > 0 and decode_tokens:
        llm["e2e_ms"] = e2e.avg_ms
        llm["e2e_tps"] = decode_tokens * 1000 / e2e.avg_ms
    prefill_run = aggregate.scopes.get("llm.prefill.run")
    decode_run = aggregate.scopes.get("llm.decode.run")
    if prefill_run and prefill_run.count is not None and prefill_run.count > 0 and input_tokens:
        run_ms = prefill_run.total_ms / len(reports)
        if run_ms > 0:
            llm["prefill_runtime_tps"] = input_tokens * 1000 / run_ms
    if decode_run and decode_run.count is not None and decode_run.count > 0 and decode_tokens:
        run_ms = decode_run.total_ms / len(reports)
        if run_ms > 0:
            llm["decode_runtime_tps"] = decode_tokens * 1000 / run_ms
    return aggregate


def add_initialization_scopes(report: PerfReport, initialization: PerfReport) -> None:
    """Add one-time model-load scopes without treating them as loop samples."""
    for path, stats in initialization.scopes.items():
        if path == "llm.load" or path.startswith("llm.load."):
            report.scopes[path] = stats.copy()
