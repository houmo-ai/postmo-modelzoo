# Copyright (c) 2025 HOUMO AI
#
# File: pytest_support.py
# Description:
#  Pure Pytest Marker Parsing Shared by Test Suites.
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

from dataclasses import dataclass
from typing import Iterable

NDEVICE_MARKER = "ndevice"
DEVICE_MEM_MARKER = "dev_mem"


class MarkerConfigurationError(ValueError):
    """Report missing or malformed device markers without controlling pytest."""


@dataclass(frozen=True)
class DeviceMarkers:
    """Retain marker tokens and their normalized values."""

    ndevice_token: str
    device_mem_token: str
    ndevice: int
    device_mem: str

    @property
    def result_directory_name(self) -> str:
        return f"{self.ndevice_token}_{self.device_mem_token}"

    def as_legacy_mapping(self) -> dict[str, str]:
        return {
            NDEVICE_MARKER: self.ndevice_token,
            DEVICE_MEM_MARKER: self.device_mem_token,
        }


def parse_device_markers(marker_names: Iterable[str]) -> DeviceMarkers:
    """Parse exactly one ndevice and one device-memory marker."""
    names = tuple(marker_names)
    ndevice_tokens = tuple(name for name in names if name.startswith("ndevice_"))
    memory_tokens = tuple(name for name in names if name.startswith("dev_mem_"))
    if len(ndevice_tokens) != 1 or len(memory_tokens) != 1:
        raise MarkerConfigurationError(
            "Expected exactly one ndevice_* and one dev_mem_* marker; "
            f"got ndevice={ndevice_tokens}, dev_mem={memory_tokens}"
        )
    try:
        ndevice = int(ndevice_tokens[0].removeprefix("ndevice_"))
    except ValueError as error:
        raise MarkerConfigurationError(f"Invalid device-count marker: {ndevice_tokens[0]}") from error
    memory = memory_tokens[0].removeprefix("dev_mem_")
    if not memory:
        raise MarkerConfigurationError(f"Invalid device-memory marker: {memory_tokens[0]}")
    return DeviceMarkers(ndevice_tokens[0], memory_tokens[0], ndevice, memory)


def device_markers_from_request(pytest_request) -> DeviceMarkers:
    """Extract marker names from a pytest request without skipping the test."""
    return parse_device_markers(marker.name for marker in pytest_request.node.own_markers)


__all__ = [
    "DEVICE_MEM_MARKER",
    "DeviceMarkers",
    "MarkerConfigurationError",
    "NDEVICE_MARKER",
    "device_markers_from_request",
    "parse_device_markers",
]
