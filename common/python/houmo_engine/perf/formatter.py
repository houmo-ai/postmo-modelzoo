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


def format_report(report: PerfReport) -> str:
    roots = sorted({path.split(".", 1)[0] for path in report.scopes | report.metrics})
    title = "Performance Summary"
    if len(roots) == 1:
        title = f"{title}: {roots[0]}"

    timing_rows = []
    for path, stats in sorted(report.scopes.items(), key=lambda item: _path_sort_key(item[0])):
        segments = path.split(".")
        display_path = "  " * max(len(segments) - 2, 0) + segments[-1]
        minimum = stats.min_ms if stats.count else 0.0
        speed = report.speeds.get(path)
        speed_text = f"{speed[0]:.2f} {speed[1]}" if speed is not None else "-"
        timing_rows.append(
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
        return (
            f"{row[0]:<{widths[0]}}  {row[1]:>{widths[1]}}  "
            f"{row[2]:>{widths[2]}}  {row[3]:>{widths[3]}}  "
            f"{row[4]:>{widths[4]}}  {row[5]:>{widths[5]}}  "
            f"{row[6]:>{widths[6]}}"
        )

    lines = [title]
    if timing_rows:
        lines.extend(("", "Timing", timing_line(headers)))
        lines.append("  ".join("-" * width for width in widths))
        lines.extend(timing_line(row) for row in timing_rows)

    if report.derived:
        lines.extend(("", "Overall Performance Metrics"))
        labels = {
            "input_tokens": ("Input Tokens", ""),
            "output_tokens": ("Output Tokens", ""),
            "audio_length_s": ("Audio Length", "s"),
            "ttft_ms": ("TTFT (Time To First Token)", "ms"),
            "tpot_ms": ("TPOT (Time Per Output Token)", "ms/token"),
            "e2e_ms": ("E2E Latency (End-to-End)", "ms"),
            "e2e_tps": ("E2E TPS (Throughput)", "tokens/s"),
            "overall_rtf": ("Overall RTF (Real-Time Factor)", ""),
            "inference_rtf": ("Inference RTF (Real-Time Factor)", ""),
            "mtp_acceptance_rate": ("MTP Acceptance Rate", ""),
            "mtp_accepted_per_round": ("MTP Accepted Per Round", "tokens/round"),
            "decode_active_ms": ("Decode Active Time", "ms"),
            "decode_active_tps": ("Decode Active Speed", "tokens/s"),
        }
        for root, metrics in sorted(report.derived.items()):
            if len(report.derived) > 1:
                lines.append(f"[{root}]")
            for name, value in metrics.items():
                label, unit = labels[name]
                suffix = f" {unit}" if unit else ""
                if name in {"input_tokens", "output_tokens"}:
                    value_text = str(int(value))
                elif name == "mtp_acceptance_rate":
                    value_text = f"{value:.2%}"
                else:
                    value_text = f"{value:.2f}"
                lines.append(f"{label}: {value_text}{suffix}")

    return "\n".join(lines)
