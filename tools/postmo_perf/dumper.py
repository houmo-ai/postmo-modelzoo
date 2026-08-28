# Copyright (c) 2026 HOUMO AI
#
# File: dumper.py
# Description:
#   C++ llm_perf-compatible YAML export for average performance results.
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

"""Strict YAML compatibility exporter for the C++ ``llm_perf`` schema."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .result import CaseResult


def _fixed(value: float, precision: int = 2) -> str:
    return f"{float(value):.{precision}f}"


def _scope_ms(result: CaseResult, path: str, *, per_loop: bool = False) -> float:
    stats = result.average.scopes.get(path)
    if stats is None:
        return 0.0
    value = stats.total_ms
    if per_loop:
        loop_count = len(result.loops)
        return value / loop_count if loop_count else 0.0
    return value


def _build_perf_settings(result: CaseResult) -> dict[str, Any]:
    return {
        "ModelName": result.model_name,
        "prefill": result.prefill_path,
        "decode": result.decode_path,
        "visual": result.visual_path,
        "embedding": result.embedding_path,
        "input": result.input_tokens,
        "output": result.output_tokens,
        "devices": list(result.devices),
        "batch": result.batch,
        "loop": len(result.loops),
        "LazyMode": result.lazy_mode,
        "warm_up": result.warmup > 0,
        "warm_up_input": result.input_tokens,
        "warm_up_output": result.output_tokens,
        "skip_perf": result.skip_perf,
        "monitor_interval": result.monitor_interval,
        "perf_case_index": result.perf_case_index,
        "perf_case_total": result.perf_case_total,
    }


def _build_perf_results(result: CaseResult) -> dict[str, Any]:
    prefill_ms = _scope_ms(result, "postmo.prefill", per_loop=True)
    decode_ms = _scope_ms(result, "postmo.decode", per_loop=True)
    e2e_ms = _scope_ms(result, "postmo.e2e", per_loop=True)
    prefill_infer_ms = _scope_ms(result, "llm.prefill.run", per_loop=True)
    decode_infer_ms = _scope_ms(result, "llm.decode.run", per_loop=True)
    e2e_seconds = e2e_ms / 1000.0

    def speed(tokens: int, elapsed_ms: float) -> str:
        return _fixed(tokens * 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0)

    return {
        "input_token": result.input_tokens,
        "output_token": result.output_tokens,
        "prefill_load_time": _fixed(_scope_ms(result, "llm.load.prefill_model")),
        "decode_load_time": _fixed(_scope_ms(result, "llm.load.decode_model")),
        "vision_load_time": _fixed(0.0),
        "prefill_time": _fixed(prefill_ms),
        "decode_time": _fixed(decode_ms),
        "vision_time": _fixed(0.0),
        "prefill_speed": speed(result.input_tokens, prefill_ms),
        "decode_speed": speed(result.output_tokens, decode_ms),
        "vision_speed": _fixed(0.0),
        "prefill_infer_time_avg": _fixed(prefill_infer_ms / result.input_tokens if result.input_tokens else 0.0),
        "decode_infer_time_avg": _fixed(decode_infer_ms / result.output_tokens if result.output_tokens else 0.0),
        "vision_infer_time_avg": _fixed(0.0),
        "prefill_infer_speed_avg": speed(result.input_tokens, prefill_infer_ms),
        "decode_infer_speed_avg": speed(result.output_tokens, decode_infer_ms),
        "vision_infer_speed_avg": _fixed(0.0),
        "prefill_embedding_time": _fixed(0.0),
        "decode_embedding_time": _fixed(0.0),
        "kvcache_mem": _fixed(0.0),
        "TTFT": _fixed(prefill_ms),
        "TPOT": _fixed(decode_ms / result.output_tokens if result.output_tokens else 0.0),
        "e2e_latency": _fixed(e2e_seconds),
        "e2e_tps": _fixed(result.output_tokens / e2e_seconds if e2e_seconds > 0 else 0.0),
    }


def _build_host_monitor() -> dict[str, str]:
    return {
        "physical_memory": "",
        "virtual_memory": "",
        "max_physical_memory": "",
        "max_virtual_memory": "",
    }


def _build_device_monitor(devices: tuple[int, ...]) -> dict[str, dict[str, str]]:
    fields = (
        "ipu_freq_max", "ipu_freq_min", "ipu_freq_avg",
        "temperature_max", "temperature_min", "temperature_avg",
        "power_max", "power_min", "power_avg", "mem_total", "mem_used",
        "mem_used_max", "mem_used_min", "mem_used_avg",
    )
    return {str(device): {field: "" for field in fields} for device in devices}


def _build_model_load_memory(devices: tuple[int, ...]) -> dict[str, dict[str, str]]:
    return {str(device): {"mem_total": "", "mem_used": ""} for device in devices}


def build_llm_perf_case(result: CaseResult) -> dict[str, Any]:
    return {
        "PerfSettings": _build_perf_settings(result),
        "PerfResults": _build_perf_results(result),
        "HostMonitor": _build_host_monitor(),
        "DeviceMonitor": _build_device_monitor(result.devices),
        "ModelLoadMemory": _build_model_load_memory(result.devices),
    }


def dumps_llm_perf_results(results: tuple[CaseResult, ...]) -> str:
    if not results:
        raise ValueError("at least one case result is required")
    return yaml.safe_dump(
        {"PerfMetrics": [build_llm_perf_case(result) for result in results]},
        sort_keys=False,
        allow_unicode=True,
    )


def dump_llm_perf_results(results: tuple[CaseResult, ...], path: str | Path) -> None:
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"parent directory does not exist: {target.parent}")
    content = dumps_llm_perf_results(results).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


# Keep the old names as an internal compatibility alias; the CLI uses the
# strict C++-compatible exporter above.
dump_results = dump_llm_perf_results
dumps_results = dumps_llm_perf_results

__all__ = [
    "build_llm_perf_case",
    "dump_llm_perf_results",
    "dumps_llm_perf_results",
    "dump_results",
    "dumps_results",
]
