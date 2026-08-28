# Copyright (c) 2026 HOUMO AI
#
# File: metrics.py
# Description:
#   Derived performance metric and speed calculations.
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

_TOKENS_PER_SECOND = "tokens/s"


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


def _base_metrics(
    report: PerfReport, root: str, metrics: dict[str, Any]
) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in ("input_tokens", "output_tokens"):
        value = _positive_number(metrics.get(name))
        if value is not None:
            values[name] = value

    speech_tokens = _positive_number(metrics.get("speech_tokens"))
    if root == "lalm" and speech_tokens is not None:
        values["speech_tokens"] = speech_tokens

    for name, scope in (("ttft_ms", "ttft"), ("e2e_ms", "e2e")):
        elapsed_ms = _scope_ms(report, f"{root}.{scope}")
        if elapsed_ms is not None:
            values[name] = elapsed_ms
    return values


def _derive_lalm_metrics(
    report: PerfReport,
    metrics: dict[str, Any],
    values: dict[str, float],
) -> None:
    output_tokens = _positive_number(metrics.get("output_tokens"))
    speech_tokens = _positive_number(metrics.get("speech_tokens"))
    s2t_ms = _scope_ms(report, "lalm.e2e_s2t")
    s2s_ms = _scope_ms(report, "lalm.e2e_s2s")
    token2wav_ms = _scope_ms(report, "lalm.e2e_token2wav")

    if s2t_ms is not None:
        values["e2e_ms"] = s2t_ms
        if output_tokens is not None:
            values["e2e_tps"] = output_tokens * 1000 / s2t_ms
    if s2s_ms is not None:
        values["s2s_e2e_ms"] = s2s_ms
        if output_tokens is not None:
            total_tokens = output_tokens + (speech_tokens or 0.0)
            values["s2s_e2e_tps"] = total_tokens * 1000 / s2s_ms
    if token2wav_ms is not None:
        values["token2wav_e2e_ms"] = token2wav_ms
        audio_length_s = _positive_number(metrics.get("output_audio_length_s"))
        if audio_length_s is not None:
            values["output_audio_length_s"] = audio_length_s
            values["token2wav_rtf"] = token2wav_ms / (audio_length_s * 1000)

    decode_tokens = _positive_number(metrics.get("decode_tokens"))
    decode_ms = _scope_ms(report, "lalm.decode")
    if decode_ms is not None and decode_tokens is not None:
        values["tpot_ms"] = decode_ms / decode_tokens


def _derive_llm_metrics(
    report: PerfReport,
    metrics: dict[str, Any],
    values: dict[str, float],
) -> None:
    output_tokens = _positive_number(metrics.get("output_tokens"))
    e2e_ms = values.get("e2e_ms")
    if e2e_ms is not None and e2e_ms > 0 and output_tokens is not None:
        values["e2e_tps"] = output_tokens * 1000 / e2e_ms

    decode_tokens = _positive_number(metrics.get("decode_tokens"))
    decode_ms = _scope_ms(report, "llm.decode")
    if decode_ms is not None and decode_tokens is not None:
        values["tpot_ms"] = decode_ms / decode_tokens


def _derive_llm_mtp_metrics(
    report: PerfReport,
    metrics: dict[str, Any],
    values: dict[str, float],
) -> None:
    output_tokens = _positive_number(metrics.get("output_tokens"))
    e2e_ms = values.get("e2e_ms")
    if e2e_ms is not None and e2e_ms > 0 and output_tokens is not None:
        values["e2e_tps"] = output_tokens * 1000 / e2e_ms

    decode_tokens = _positive_number(metrics.get("decode_tokens"))
    ttft_ms = values.get("ttft_ms")
    if e2e_ms is not None and ttft_ms is not None and e2e_ms > ttft_ms:
        decode_active_ms = e2e_ms - ttft_ms
        values["decode_active_ms"] = decode_active_ms
        if decode_tokens is not None:
            values["decode_active_tps"] = decode_tokens * 1000 / decode_active_ms
            values["tpot_ms"] = decode_active_ms / decode_tokens

    draft_tokens = _positive_number(metrics.get("draft_tokens"))
    accepted_tokens = _positive_number(metrics.get("accepted_draft_tokens"))
    rounds = _positive_number(metrics.get("speculative_rounds"))
    if draft_tokens is not None and accepted_tokens is not None:
        values["mtp_acceptance_rate"] = accepted_tokens / draft_tokens
    if rounds is not None and accepted_tokens is not None:
        values["mtp_accepted_per_round"] = accepted_tokens / rounds


def _derive_asr_metrics(
    report: PerfReport,
    root: str,
    metrics: dict[str, Any],
    values: dict[str, float],
) -> None:
    audio_length_s = _positive_number(metrics.get("audio_length_s"))
    if audio_length_s is None:
        return

    values["audio_length_s"] = audio_length_s
    audio_length_ms = audio_length_s * 1000
    e2e_ms = values.get("e2e_ms")
    if e2e_ms is not None:
        values["overall_rtf"] = e2e_ms / audio_length_ms

    infer_ms = sum(
        stats.total_ms
        for path, stats in report.scopes.items()
        if path.startswith(f"{root}.")
        and path.endswith(".infer")
        and stats.count is not None
        and stats.count > 0
    )
    if infer_ms > 0:
        values["inference_rtf"] = infer_ms / audio_length_ms


def derive_metrics(report: PerfReport) -> dict[str, dict[str, float]]:
    roots = {
        path.split(".", 1)[0]
        for path in report.scopes.keys() | report.metrics.keys()
    }
    derived: dict[str, dict[str, float]] = {}

    for root in roots:
        if root not in {"llm", "llm_mtp", "asr", "tts", "lalm"}:
            continue
        metrics = _root_metrics(report, root)
        values = _base_metrics(report, root, metrics)
        if root == "lalm":
            _derive_lalm_metrics(report, metrics, values)
        elif root == "llm":
            _derive_llm_metrics(report, metrics, values)
        elif root == "llm_mtp":
            _derive_llm_mtp_metrics(report, metrics, values)
        elif root == "asr":
            _derive_asr_metrics(report, root, metrics, values)

        if values:
            derived[root] = values

    return derived


def derive_speeds(report: PerfReport) -> dict[str, tuple[float, str]]:
    speeds: dict[str, tuple[float, str]] = {}
    for path, stats in report.scopes.items():
        if stats.total_ms <= 0 or stats.count is None:
            continue
        segments = path.split(".")
        if len(segments) < 2:
            continue
        root = segments[0]
        stage = segments[1]
        is_stage = len(segments) == 2
        is_runtime_run = path.endswith(".run")
        is_infer = path.endswith(".infer")
        if not is_stage and not is_infer and not is_runtime_run:
            continue

        metrics = _root_metrics(report, root)
        amount = None
        unit = None
        if (root == "llm" and "prefill" in stage) or (
            root == "lalm" and stage == "prefill"
        ):
            amount = _positive_number(metrics.get("input_tokens"))
            unit = _TOKENS_PER_SECOND
        elif (root == "llm" and "decode" in stage) or (
            root == "lalm" and stage == "decode"
        ):
            amount = _positive_number(metrics.get("decode_tokens"))
            unit = _TOKENS_PER_SECOND
        elif root == "llm" and stage == "vision":
            amount = _positive_number(metrics.get("num_images"))
            unit = "images/s"
        elif root == "llm_mtp" and stage == "mtp_prefill":
            amount = _positive_number(metrics.get("mtp_prefill_tokens"))
            unit = _TOKENS_PER_SECOND
        elif root == "llm_mtp" and stage == "prefill":
            amount = _positive_number(metrics.get("input_tokens"))
            unit = _TOKENS_PER_SECOND
        elif root == "llm_mtp" and stage == "draft":
            amount = _positive_number(metrics.get("draft_tokens"))
            unit = _TOKENS_PER_SECOND
        elif root == "llm_mtp" and stage == "verify":
            amount = _positive_number(metrics.get("verify_tokens"))
            unit = _TOKENS_PER_SECOND
        elif root == "asr" and stage == "encode":
            speeds[path] = (stats.avg_ms, "ms/chunk")
            continue
        elif root == "asr" and stage == "prefill":
            amount = _positive_number(metrics.get("prefill_tokens"))
            unit = _TOKENS_PER_SECOND
        elif root == "asr" and stage == "decode":
            amount = _positive_number(metrics.get("decode_tokens"))
            unit = _TOKENS_PER_SECOND

        if amount is not None and unit is not None:
            speeds[path] = (amount * 1000 / stats.total_ms, unit)
    return speeds
