# Copyright 2025 HOUMO AI
#
# File: get_hm_devices.py
# Description:
#   This file contains functions for getting available HM devices.
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
import os
from typing import List

try:
    import hmatc.python.smi as smi
except ImportError:
    smi = None


def _get_default_devices(ndevices: int = 1) -> List[int]:
    """Return the default device IDs, respecting HOUMO_VISIBLE_DEVICES."""
    DEFAULT_RUN_DEVEVICES = [i for i in range(ndevices)]
    if not os.getenv("HOUMO_VISIBLE_DEVICES"):
        return DEFAULT_RUN_DEVEVICES

    env_devices = os.getenv("HOUMO_VISIBLE_DEVICES").split(",")
    env_devices = [int(dev.strip()) for dev in env_devices]
    assert (
        len(env_devices) >= ndevices
    ), f"Not enough devices specified in HOUMO_VISIBLE_DEVICES. Required: {ndevices}, Provided: {len(env_devices)}"

    dev_start_idx = sorted(env_devices)[0]
    return [dev_start_idx + i for i in range(ndevices)]


def get_hm_devices(ndevices=1) -> list:
    """
    Get a list of HM devices available on the system.

    Returns:
        List[HMDevice]: A list of HMDevice objects representing the available HM devices.
    """
    DEFAULT_RUN_DEVEVICES = _get_default_devices(ndevices)
    if smi is not None:
        assert smi.device_ctc_check(DEFAULT_RUN_DEVEVICES)
    return DEFAULT_RUN_DEVEVICES
