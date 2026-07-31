# Copyright (c) 2025 HOUMO AI
#
# File: platform_device.py
# Description:
#  Host Platform and Houmo Device Detection Shared by Test Suites.
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

from __future__ import annotations

import logging
import platform
import socket
import subprocess
from pathlib import Path
from typing import Sequence

from .command_execution import CommandExecutionError, CommandRunner, CommandSpec

logger = logging.getLogger(__name__)


def is_asic_platform(*, hostname: str | None = None, device_root: Path = Path("/dev")) -> bool:
    """Detect ASIC hosts while preserving the legacy hostname heuristics."""
    resolved_hostname = hostname
    if resolved_hostname is None:
        try:
            resolved_hostname = socket.gethostname()
        except OSError as error:
            logger.warning("Failed to get hostname: %s", error)
            resolved_hostname = ""
    if "smoke" in resolved_hostname:
        return True
    if "nj-gpu01" in resolved_hostname:
        return False
    try:
        return any(path.name.startswith("xh2a_") for path in device_root.iterdir())
    except OSError as error:
        logger.warning("Failed to inspect %s: %s", device_root, error)
        return False


def get_platform(support_list: Sequence[str]) -> str | None:
    """Return the current Linux machine architecture when supported."""
    system = platform.system()
    machine = platform.machine()
    logger.info("Only supports Linux system, current system is %s.", system)
    return machine if system == "Linux" and machine in support_list else None


def check_gpu() -> dict[str, object]:
    """Return NVIDIA GPU availability using the established nvidia-smi probe."""
    result: dict[str, object] = {"has_gpu": False, "gpu_info": []}
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return result
    gpu_info = [f"NVIDIA (nvidia-smi): {line.strip()}" for line in output.splitlines() if line.strip()]
    return {"has_gpu": bool(gpu_info), "gpu_info": gpu_info}


def _hm_smi_output(runner: CommandRunner) -> str | None:
    try:
        result = runner.run(
            CommandSpec(
                "hm-smi",
                ("hm_smi", "-a"),
                allow_nonzero_exit=True,
            )
        )
    except CommandExecutionError as error:
        logger.error("Failed to probe Houmo device: %s", error)
        return None
    return result.stdout if result.succeeded else None


def check_device_info(
    support_list: Sequence[int] | None,
    *,
    backend: str,
    runner: CommandRunner,
) -> bool:
    """Check whether all visible devices expose a supported core count."""
    if not support_list:
        logger.error("No supported HMM core count is configured.")
        return False
    output = _hm_smi_output(runner)
    if output is None:
        return False
    core_label = "Core_Num" if backend == "xh2" else "Core Num"
    values = [line.split(":", 1)[-1].strip() for line in output.splitlines() if core_label in line]
    if not values or len(set(values)) != 1:
        logger.error("Unsupported device core information: %s", values)
        return False
    try:
        device_core_num = int(values[0])
    except ValueError:
        logger.error("Invalid device core count: %s", values[0])
        return False
    supported = device_core_num in support_list or any(device_core_num % core_num == 0 for core_num in support_list)
    if not supported:
        logger.error(
            "Unsupported device core num %s, expected one of %s",
            device_core_num,
            support_list,
        )
    return supported


def check_vpu_status(*, backend: str, runner: CommandRunner) -> bool:
    """Return whether xh1 reports VPU memory usage above the legacy threshold."""
    if backend != "xh1":
        return False
    output = _hm_smi_output(runner)
    if output is None:
        return False
    values = [line.split(":", 1)[-1].strip() for line in output.splitlines() if "Used" in line]
    used_mem = 0.0
    if values and values[0].endswith("MB"):
        try:
            used_mem = float(values[0][:-2])
        except ValueError:
            used_mem = 0.0
    if used_mem > 2000:
        logger.info("Device 0 is using the VPU driver, memory: %s", values)
        return True
    logger.info("Device 0 is not using the VPU driver, memory: %s", values)
    return False


def reset_chips(*, backend: str, runner: CommandRunner) -> None:
    """Explicitly reset chips for operator-requested recovery only."""
    xh2_reset = Path("/usr/local/houmo-sdk/hal/utility/ipu_reset")
    command = (
        str(xh2_reset) if backend == "xh2" and xh2_reset.exists() else "/usr/local/houmo-sdk/scripts/reset_aicore.sh"
    )
    runner.run(CommandSpec("reset-chips", (command,)))


__all__ = [
    "check_device_info",
    "check_gpu",
    "check_vpu_status",
    "get_platform",
    "is_asic_platform",
    "reset_chips",
]
