# Copyright (c) 2026 HOUMO AI
#
# File: config.py
# Description:
#   Configuration models and YAML loading for performance test cases.
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

"""Configuration and validation for fixed-length performance cases."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class PerfCase:
    model_dir: Path
    input_tokens: int
    output_tokens: int
    loop: int = 1
    warmup: int = 1
    seed: int = 0
    model_name: str = "qwen3.5"
    prefill: Path | None = None
    decode: Path | None = None
    embedding: Path | None = None
    visual: Path | None = None
    devices: tuple[int, ...] = (0,)
    batch: int = 1
    lazy_mode: bool = False
    skip_perf: bool = False
    monitor_interval: int = 0
    perf_case_index: int = 1
    perf_case_total: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.model_dir, Path):
            raise TypeError("model_dir must be a Path")
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"model directory does not exist: {self.model_dir}")
        for name in ("input_tokens", "output_tokens", "loop"):
            _positive_int(getattr(self, name), name)
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ValueError("warmup must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(self.devices, tuple) or not self.devices:
            raise ValueError("devices must be a non-empty tuple")
        if any(isinstance(device, bool) or not isinstance(device, int) or device < 0 for device in self.devices):
            raise ValueError("devices must contain non-negative integers")
        if self.batch != 1:
            raise ValueError("only batch=1 is supported")
        if self.devices != (0,):
            raise ValueError("only device 0 is supported")
        if isinstance(self.monitor_interval, bool) or not isinstance(self.monitor_interval, int) or self.monitor_interval < 0:
            raise ValueError("monitor_interval must be a non-negative integer")
        if (self.prefill is None) != (self.decode is None):
            raise ValueError("prefill and decode must be provided together")
        for name in ("prefill", "decode"):
            path = getattr(self, name)
            if path is not None:
                if not isinstance(path, Path):
                    raise TypeError(f"{name} must be a Path or None")
                if not path.is_file():
                    raise FileNotFoundError(f"{name} model does not exist: {path}")
        for name in ("embedding", "visual"):
            path = getattr(self, name)
            if path is not None and not isinstance(path, Path):
                raise TypeError(f"{name} must be a Path or None")


@dataclass(frozen=True)
class PerfSettings:
    cases: tuple[PerfCase, ...]
    dump_file: Path | None = None

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("at least one performance case is required")
        if self.dump_file is not None and not isinstance(self.dump_file, Path):
            raise TypeError("dump_file must be a Path or None")


def _case(data: dict[str, Any], *, base_dir: Path) -> PerfCase:
    unsupported = {"batch", "batch_size", "device", "device_id", "devices"} & data.keys()
    if unsupported:
        raise ValueError(f"unsupported fixed-length setting(s): {', '.join(sorted(unsupported))}")
    required = ("model_dir", "input_tokens", "output_tokens")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"missing case setting(s): {', '.join(missing)}")
    values = dict(data)
    model_dir = Path(values["model_dir"])
    values["model_dir"] = model_dir if model_dir.is_absolute() else base_dir / model_dir
    for name in ("prefill", "decode"):
        if values.get(name):
            path = Path(values[name])
            values[name] = path if path.is_absolute() else base_dir / path
    if values.get("embedding"):
        path = Path(values["embedding"])
        values["embedding"] = path if path.is_absolute() else base_dir / path
    if values.get("visual"):
        path = Path(values["visual"])
        values["visual"] = path if path.is_absolute() else base_dir / path
    if "devices" in values:
        values["devices"] = tuple(values["devices"]) if isinstance(values["devices"], list) else tuple(values["devices"])
    return PerfCase(**values)


def load_config(path: str | Path) -> PerfSettings:
    """Load a YAML file containing either one case or a ``cases`` list."""
    import yaml

    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("performance config must be a mapping")
    raw_cases = data.get("cases")
    if raw_cases is None:
        raw_case = dict(data)
        raw_case.pop("dump_file", None)
        raw_cases = [raw_case]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must be a non-empty list")
    if not all(isinstance(item, dict) for item in raw_cases):
        raise ValueError("each case must be a mapping")
    dump_file = Path(data["dump_file"]) if data.get("dump_file") else None
    if dump_file is not None and not dump_file.is_absolute():
        dump_file = source.parent / dump_file
    cases = tuple(_case(item, base_dir=source.parent) for item in raw_cases)
    cases = tuple(
        replace(case, perf_case_index=index, perf_case_total=len(cases))
        for index, case in enumerate(cases, start=1)
    )
    return PerfSettings(cases, dump_file)
