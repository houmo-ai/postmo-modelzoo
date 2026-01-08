# Copyright 2025 HOUMO AI
#
# File: monitor_device_mem.py
# Description:
#   Python wrapper for device memory monitoring functionality.
#   This module provides functions to query device memory usage by calling
#   the underlying C++ executable and parsing its output.
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
import re
import argparse
import subprocess


def parse_args():
    """Parse command line arguments for the device memory monitor."""
    parser = argparse.ArgumentParser(description="Device Memory Monitor")
    parser.add_argument(
        "-d",
        "--device_id",
        type=int,
        choices=[0, 1],
        help="device id.",
    )

    args = parser.parse_args()
    return args


def _run_command(command):
    """
    Execute a system command and return the output and exit code.

    Args:
        command (str): Command to execute

    Returns:
        tuple: (output, return_code) Output result and exit code
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout, result.returncode
    except Exception as e:
        return f"Failed to execute command: {str(e)}", -1


def get_device_mem(device_id: int = None) -> dict:
    """Get device memory information for specified device(s).

    Calls the underlying C++ executable to retrieve memory statistics
    for one or more devices and parses the output into a structured format.

    Args:
        device_id (int, optional): Specific device ID to query. If None, queries all devices.

    Returns:
        dict: Dictionary containing memory information for each device,
              with device IDs as keys and memory stats as values
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chip_list = [0, 1]
    if device_id:
        chip_list = [device_id]

    result = {}
    for chip_id in chip_list:
        cmd = f"{script_dir}/bin/dev_monitor -d {chip_id}"
        opt, ret = _run_command(cmd)
        if ret != 0:
            continue

        # Define regex pattern to parse the output
        pattern = r"device_id: (?P<device_id>\d+), time: (?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), mem_total: (?P<mem_total>\d+), mem_used: (?P<mem_used>\d+), mem_avail: (?P<mem_avail>\d+)"
        for line in opt.split("\n"):
            if "device_id:" in line:
                match = re.match(pattern, line.strip())
                if match:
                    tmp_result = match.groupdict()
                    result[int(tmp_result["device_id"])] = tmp_result

    return result
