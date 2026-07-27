# Copyright (c) 2026 HOUMO AI
#
# File: metrics.py
# Description:
#   Derived performance metrics and speed calculations.
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

from typing import Any

from .stats import PerfReport


def _root_metrics(report: PerfReport, root: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for path, values in report.metrics.items():
        if path == root or path.startswith(f"{root}."):
            metrics.update(values)
    return metrics


def _scope_ms(report: PerfReport, path: str) -> float | None:
    stats = report.scopes.get(path)
    if stats is None or stats.count == 0:
        return None
    return stats.total_ms


def _positive_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def derive_metrics(report: PerfReport) -> dict[str, dict[str, float]]:
    roots = {
        path.split(".", 1)[0]
        for path in report.scopes.keys() | report.metrics.keys()
    }
    derived: dict[str, dict[str, float]] = {}

    for root in roots:
        if root not in {"llm", "llm_mtp", "asr", "tts"}:
            continue
        metrics = _root_metrics(report, root)
        values: dict[str, float] = {}

        input_tokens = _positive_number(metrics.get("input_tokens"))
        if input_tokens is not None:
            values["input_tokens"] = input_tokens

        output_tokens = _positive_number(metrics.get("output_tokens"))
        if output_tokens is not None:
            values["output_tokens"] = output_tokens

        ttft_ms = _scope_ms(report, f"{root}.ttft")
        if ttft_ms is not None:
            values["ttft_ms"] = ttft_ms

        e2e_ms = _scope_ms(report, f"{root}.e2e")
        if e2e_ms is not None:
            values["e2e_ms"] = e2e_ms

        if root == "llm":
            if e2e_ms is not None and e2e_ms > 0 and output_tokens is not None:
                values["e2e_tps"] = output_tokens * 1000 / e2e_ms

            decode_tokens = _positive_number(metrics.get("decode_tokens"))
            decode_ms = _scope_ms(report, f"{root}.decode")
            if decode_ms is not None and decode_tokens is not None:
                values["tpot_ms"] = decode_ms / decode_tokens

        if root == "llm_mtp":
            if e2e_ms is not None and e2e_ms > 0 and output_tokens is not None:
                values["e2e_tps"] = output_tokens * 1000 / e2e_ms

            decode_tokens = _positive_number(metrics.get("decode_tokens"))
            if (
                e2e_ms is not None
                and ttft_ms is not None
                and e2e_ms > ttft_ms
            ):
                decode_active_ms = e2e_ms - ttft_ms
                values["decode_active_ms"] = decode_active_ms
                if decode_tokens is not None:
                    values["decode_active_tps"] = (
                        decode_tokens * 1000 / decode_active_ms
                    )
                    values["tpot_ms"] = decode_active_ms / decode_tokens

            draft_tokens = _positive_number(metrics.get("draft_tokens"))
            accepted_draft_tokens = _positive_number(
                metrics.get("accepted_draft_tokens")
            )
            speculative_rounds = _positive_number(
                metrics.get("speculative_rounds")
            )
            if draft_tokens is not None and accepted_draft_tokens is not None:
                values["mtp_acceptance_rate"] = (
                    accepted_draft_tokens / draft_tokens
                )
            if (
                speculative_rounds is not None
                and accepted_draft_tokens is not None
            ):
                values["mtp_accepted_per_round"] = (
                    accepted_draft_tokens / speculative_rounds
                )

        audio_length_s = _positive_number(metrics.get("audio_length_s"))
        if root == "asr" and audio_length_s is not None:
            values["audio_length_s"] = audio_length_s
            audio_length_ms = audio_length_s * 1000
            if e2e_ms is not None:
                values["overall_rtf"] = e2e_ms / audio_length_ms

            infer_ms = sum(
                stats.total_ms
                for path, stats in report.scopes.items()
                if path.startswith(f"{root}.")
                and path.endswith(".infer")
                and stats.count > 0
            )
            if infer_ms > 0:
                values["inference_rtf"] = infer_ms / audio_length_ms

        if values:
            derived[root] = values

    return derived


def derive_speeds(report: PerfReport) -> dict[str, tuple[float, str]]:
    speeds: dict[str, tuple[float, str]] = {}
    for path, stats in report.scopes.items():
        if stats.total_ms <= 0:
            continue
        segments = path.split(".")
        if len(segments) < 2:
            continue
        root = segments[0]
        stage = segments[1]
        is_stage = len(segments) == 2
        is_infer = path.endswith(".infer")
        if not is_stage and not is_infer:
            continue

        metrics = _root_metrics(report, root)
        amount = None
        unit = None
        if root == "llm" and "prefill" in stage:
            amount = _positive_number(metrics.get("input_tokens"))
            unit = "tokens/s"
        elif root == "llm" and "decode" in stage:
            amount = _positive_number(metrics.get("decode_tokens"))
            unit = "tokens/s"
        elif root == "llm" and stage == "vision":
            amount = _positive_number(metrics.get("num_images"))
            unit = "images/s"
        elif root == "llm_mtp" and stage == "mtp_prefill":
            amount = _positive_number(metrics.get("mtp_prefill_tokens"))
            unit = "tokens/s"
        elif root == "llm_mtp" and stage == "prefill":
            amount = _positive_number(metrics.get("input_tokens"))
            unit = "tokens/s"
        elif root == "llm_mtp" and stage == "draft":
            amount = _positive_number(metrics.get("draft_tokens"))
            unit = "tokens/s"
        elif root == "llm_mtp" and stage == "verify":
            amount = _positive_number(metrics.get("verify_tokens"))
            unit = "tokens/s"
        elif root == "asr" and stage == "encode":
            speeds[path] = (stats.avg_ms, "ms/chunk")
            continue
        elif root == "asr" and stage == "prefill":
            amount = _positive_number(metrics.get("prefill_tokens"))
            unit = "tokens/s"
        elif root == "asr" and stage == "decode":
            amount = _positive_number(metrics.get("decode_tokens"))
            unit = "tokens/s"

        if amount is not None and unit is not None:
            speeds[path] = (amount * 1000 / stats.total_ms, unit)
    return speeds
