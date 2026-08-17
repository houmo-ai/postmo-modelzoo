# Copyright (c) 2026 HOUMO AI
#
# File: formatter.py
# Description:
#   Performance report formatting utilities.
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

from .stats import PerfReport

_TOKENS_PER_SECOND = "tokens/s"

_PHASE_ORDER = {
    "init": 0,
    "encode": 1,
    "vision": 2,
    "prefill": 3,
    "mtp_prefill": 4,
    "decode": 5,
    "draft": 6,
    "verify": 7,
    "set_input": 8,
    "infer": 9,
    "get_output": 10,
    "ttft": 11,
    "e2e": 12,
}


def _path_sort_key(path: str) -> tuple:
    segments = path.split(".")[1:]
    return tuple((_PHASE_ORDER.get(segment, 100), segment) for segment in segments)


def _timing_rows(report: PerfReport) -> list[tuple[str, ...]]:
    """Build formatted rows for scope timing statistics."""
    rows = []
    for path, stats in sorted(report.scopes.items(), key=lambda item: _path_sort_key(item[0])):
        if path.endswith((".e2e", ".ttft")) or path.startswith("lalm.e2e_"):
            continue
        display_path = _display_path(path, report.scopes)
        minimum = stats.min_ms if stats.count else 0.0
        speed = report.speeds.get(path)
        speed_text = f"{speed[0]:.2f} {speed[1]}" if speed is not None else "-"
        rows.append(
            (
                display_path,
                str(stats.count),
                f"{stats.total_ms:.3f}",
                f"{stats.avg_ms:.3f}",
                f"{minimum:.3f}",
                f"{stats.max_ms:.3f}",
                speed_text,
            )
        )
    return rows


def _display_path(path: str, scopes: dict) -> str:
    """Render a scope path with indentation for nested timing scopes."""
    segments = path.split(".")
    parent_path = path.rsplit(".", 1)[0] if "." in path else ""
    if len(segments) <= 2:
        return segments[-1]
    if parent_path in scopes:
        return "  " * (len(segments) - 2) + segments[-1]
    return ".".join(segments[1:])


def _timing_lines(timing_rows: list[tuple[str, ...]]) -> list[str]:
    """Render the timing table, including dynamically sized columns."""

    headers = (
        "Scope",
        "Count",
        "Total(ms)",
        "Avg(ms)",
        "Min(ms)",
        "Max(ms)",
        "Speed",
    )
    widths = [len(header) for header in headers]
    for row in timing_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def timing_line(row: tuple[str, ...]) -> str:
        """Render one timing table row using the calculated column widths."""
        values = (f"{row[0]:<{widths[0]}}",) + tuple(
            f"{value:>{widths[index]}}" for index, value in enumerate(row[1:], start=1)
        )
        return "  ".join(values)

    return [
        "",
        "Timing",
        timing_line(headers),
        "  ".join("-" * width for width in widths),
        *(timing_line(row) for row in timing_rows),
    ]

def _derived_lines(report: PerfReport) -> list[str]:
    """Render the derived performance metrics section."""
    labels = {
        "input_tokens": ("Input Tokens", ""),
        "output_tokens": ("Output Tokens", ""),
        "speech_tokens": ("Speech Tokens", ""),
        "audio_length_s": ("Audio Length", "s"),
        "ttft_ms": ("TTFT (Time To First Token)", "ms"),
        "tpot_ms": ("TPOT (Time Per Output Token)", "ms/token"),
        "e2e_ms": ("E2E Latency (End-to-End)", "ms"),
        "e2e_tps": ("E2E TPS (Throughput)", _TOKENS_PER_SECOND),
        "s2s_e2e_ms": ("S2S E2E Latency", "ms"),
        "s2s_e2e_tps": ("S2S E2E TPS", _TOKENS_PER_SECOND),
        "token2wav_e2e_ms": ("Token2Wav Latency", "ms"),
        "output_audio_length_s": ("Output Audio Length", "s"),
        "token2wav_rtf": ("Token2Wav RTF", ""),
        "overall_rtf": ("Overall RTF (Real-Time Factor)", ""),
        "inference_rtf": ("Inference RTF (Real-Time Factor)", ""),
        "mtp_acceptance_rate": ("MTP Acceptance Rate", ""),
        "mtp_accepted_per_round": ("MTP Accepted Per Round", "tokens/round"),
        "decode_active_ms": ("Decode Active Time", "ms"),
        "decode_active_tps": ("Decode Active Speed", _TOKENS_PER_SECOND),
    }
    lines = ["", "Overall Performance Metrics"]
    for root, metrics in sorted(report.derived.items()):
        if len(report.derived) > 1:
            lines.append(f"[{root}]")
        for name, value in metrics.items():
            label, unit = labels[name]
            if root == "lalm" and name == "output_tokens":
                label = "Text Output Tokens"
            suffix = f" {unit}" if unit else ""
            value_text = _format_metric_value(root, name, value, metrics)
            lines.append(f"{label}: {value_text}{suffix}")
    return lines


def _format_metric_value(root: str, name: str, value: float, metrics: dict) -> str:
    """Format one derived metric according to its semantic type."""
    if name in {"input_tokens", "output_tokens", "speech_tokens"}:
        return str(int(value))
    if root == "lalm" and name == "s2s_e2e_tps":
        text_tokens = int(metrics.get("output_tokens", 0))
        speech_tokens = int(metrics.get("speech_tokens", 0))
        return f"{value:.2f} ({text_tokens} text + {speech_tokens} speech)"
    if name == "mtp_acceptance_rate":
        return f"{value:.2%}"
    return f"{value:.2f}"


def format_report(report: PerfReport) -> str:
    """Render a performance report as plain text."""
    roots = sorted({path.split(".", 1)[0] for path in report.scopes | report.metrics})
    title = "Performance Summary" if len(roots) != 1 else f"Performance Summary: {roots[0]}"
    lines = [title]
    timing_rows = _timing_rows(report)
    if timing_rows:
        lines.extend(_timing_lines(timing_rows))
    if report.derived:
        lines.extend(_derived_lines(report))

    return "\n".join(lines)
