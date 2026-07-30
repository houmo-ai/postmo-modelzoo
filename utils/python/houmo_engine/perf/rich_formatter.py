# Copyright (c) 2026 HOUMO AI
#
# File: rich_formatter.py
# Description:
#   Optional Rich performance report formatter.
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

"""Optional Rich rendering for :class:`~houmo_engine.perf.PerfReport`.

This module is intentionally not imported by the default performance path.
Applications that want Rich output can call :func:`print_rich_report` or use
the returned text from :func:`format_rich_report`.
"""

from io import StringIO

from .stats import PerfReport

_RICH_REQUIRED_MESSAGE = "Rich performance output requires the optional 'rich' package"

_METRIC_LABELS = {
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


_PHASE_ORDER = {
    "init": 0,
    "encode": 1,
    "vision": 2,
    "embedding": 3,
    "frame_prepare": 4,
    "talker": 5,
    "code_predictor": 6,
    "speech_tokenizer": 7,
    "stateful_decoder": 8,
    "postprocess": 9,
    "prefill": 10,
    "decode": 11,
    "prepare": 12,
    "sampling": 13,
    "set_input": 14,
    "infer": 15,
    "get_output": 16,
    "ttft": 17,
    "e2e": 18,
}


def _path_sort_key(path: str) -> tuple:
    segments = path.split(".")[1:]
    return tuple((_PHASE_ORDER.get(segment, 100), segment) for segment in segments)


def _format_metric(name: str, value: float) -> str:
    label, unit = _METRIC_LABELS.get(name, (name, ""))
    if name in {"input_tokens", "output_tokens"}:
        rendered = str(int(value))
    elif name == "mtp_acceptance_rate":
        rendered = f"{value:.2%}"
    else:
        rendered = f"{value:.2f}"
    return f"{label}: {rendered}{f' {unit}' if unit else ''}"


def _scope_label(path: str, scopes: dict, text_type):
    segments = path.split(".")
    parent_path = path.rsplit(".", 1)[0] if "." in path else ""
    if len(segments) <= 2:
        depth = 0
        label = segments[-1]
    elif parent_path in scopes:
        depth = len(segments) - 2
        label = segments[-1]
    else:
        depth = 0
        label = ".".join(segments[1:])
    scope_text = text_type("  " * depth + label)
    if depth == 0:
        scope_text.stylize("bold cyan")
    return scope_text


def _render_timing_table(report: PerfReport, console, table_type, text_type) -> None:
    table = table_type(title="Timing", expand=False, show_lines=False)
    table.add_column("Scope", min_width=28)
    table.add_column("Count", justify="right")
    table.add_column("Total(ms)", justify="right")
    table.add_column("Avg(ms)", justify="right")
    table.add_column("Min(ms)", justify="right")
    table.add_column("Max(ms)", justify="right")
    table.add_column("Speed", justify="right")

    ordered = sorted(report.scopes.items(), key=lambda item: _path_sort_key(item[0]))
    for path, stats in ordered:
        speed = report.speeds.get(path)
        speed_text = f"{speed[0]:.2f} {speed[1]}" if speed else "-"
        table.add_row(
            _scope_label(path, report.scopes, text_type),
            str(stats.count),
            f"{stats.total_ms:.3f}",
            f"{stats.avg_ms:.3f}",
            f"{stats.min_ms if stats.count else 0.0:.3f}",
            f"{stats.max_ms:.3f}",
            speed_text,
        )
    console.print(table)


def _render_metrics_table(report: PerfReport, console, table_type) -> None:
    table = table_type(title="Overall Performance Metrics", expand=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for root, metrics in sorted(report.derived.items()):
        if len(report.derived) > 1:
            table.add_row(f"[{root}]", "")
        for name, value in metrics.items():
            label = _METRIC_LABELS.get(name, (name, ""))[0]
            rendered = _format_metric(name, value).removeprefix(f"{label}: ")
            table.add_row(label, rendered)
    console.print(table)


def _render_rich_report(report: PerfReport, console) -> None:
    try:
        from rich.table import Table
        from rich.text import Text
    except ImportError as error:
        raise RuntimeError(_RICH_REQUIRED_MESSAGE) from error
    roots = sorted({path.split(".", 1)[0] for path in report.scopes | report.metrics})
    title = "Performance Summary"
    if len(roots) == 1:
        title = f"{title}: {roots[0]}"
    console.print(f"[bold green]{title}[/bold green]")

    if report.scopes:
        _render_timing_table(report, console, Table, Text)

    if report.derived:
        _render_metrics_table(report, console, Table)


def format_rich_report(report: PerfReport, *, color: bool = False) -> str:
    """Render a Rich report and return captured terminal text.

    Rich is imported lazily, so the default Houmo Engine import path does not
    require the optional dependency. Plain text is the default for safe use in
    log files; pass ``color=True`` when ANSI output is explicitly required.
    """

    try:
        from rich.console import Console
    except ImportError as error:
        raise RuntimeError(_RICH_REQUIRED_MESSAGE) from error

    output = StringIO()
    console = Console(
        file=output,
        color_system="standard" if color else None,
        force_terminal=color,
        width=160,
    )
    _render_rich_report(report, console)

    return output.getvalue().rstrip("\n")


def print_rich_report(report: PerfReport, *, color: bool = True) -> None:
    """Print a report using the optional Rich formatter."""

    try:
        from rich.console import Console
    except ImportError as error:
        raise RuntimeError(_RICH_REQUIRED_MESSAGE) from error

    console = Console(
        color_system="auto" if color else None,
        force_terminal=None,
        width=160,
    )
    _render_rich_report(report, console)


__all__ = ["format_rich_report", "print_rich_report"]
