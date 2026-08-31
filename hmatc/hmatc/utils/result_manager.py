# Copyright 2025 HOUMO AI
#
# File: result_manager.py
# Description:
#   Result management for HMATC tool - handles result.yml generation and updates
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
"""
Result Manager for HMATC Tool

Provides standardized result collection and YAML output for all commands.

Result Structure:
-----------------
result:
  meta:
    model_name: str
    target: str (xh2)
    created_at: str (ISO format)
    updated_at: str (ISO format)
    hmatc_version: str
    commit: str

  quant:
    success: bool
    time: str (seconds, 6 decimal places)
    quant_type: str
    hmonnx: str (path)

  build:
    success: bool
    time: str (seconds, 6 decimal places)
    ncore: int
    opt_level: int
    batch: int
    hmm: str (path)
    outputs:
      <output_name>:
        cosine_dist: str (6 decimal places)
        md5: str
        golden_md5: str

  check:
    success: bool
    time: str (seconds, 6 decimal places)
    outputs:
      <output_name>:
        cosine_dist: str (6 decimal places)
        md5: str
        golden_md5: str

  compare:
    success: bool
    data_path: str
    outputs:
      <output_name>:
        onnx_vs_hmquant: str (6 decimal places, omitted when HMONNX is unavailable)
        onnx_vs_xh2: str (6 decimal places)
        hmquant_vs_xh2: str (6 decimal places, omitted when HMONNX is unavailable)

  perf:
    success: bool
    params:
      hmm_path: str
      warmup_num: int
      sample_num: int
      thread_num: int
      loop_num: int
      stream_num: int
      devices: list
    perf_info:
      avg_cost: str (6 decimal places)
      infer_avg_latency: str (6 decimal places)
      qps: str (6 decimal places)

  demo:
    success: bool
    backend: str (onnx/hmonnx/xh2)
    data_dir: str
    num: int
    error: str (if failed)

  eval:
    <backend>:
      success: bool
      data_dir: str
      num: int
      results: dict
      error: str (if failed)
"""

from datetime import datetime
from typing import Any, Dict, Optional

from .._version import __build_time__, __commit__, __version__
from ..utils import logger


def format_float(value: float) -> str:
    """Format float to 6 decimal places as string.

    Args:
        value: Float value to format

    Returns:
        Formatted string with 6 decimal places
    """
    return f"{value:.6f}"


def format_dict_floats(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively format all float values in a dictionary to 6 decimal places.

    Args:
        d: Dictionary to format

    Returns:
        Dictionary with floats formatted as strings
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, float):
            result[key] = format_float(value)
        elif isinstance(value, dict):
            result[key] = format_dict_floats(value)
        else:
            result[key] = value
    return result


def save_result(
    result_path: str, res_info: Dict[str, Any], model_name: str = "", target: str = ""
) -> None:
    """Save result to YAML file with overwrite mode.

    Args:
        result_path: Path to save the result YAML file
        res_info: Result information dictionary
        model_name: Model name for metadata
        target: Target platform for metadata
    """
    from ..utils.utils import save_dict_to_yaml
    import os

    formatted = {"result": {}}

    # Load existing result if file exists
    if os.path.exists(result_path):
        from ..utils.utils import read_yaml_to_dict

        existing = read_yaml_to_dict(result_path)
        if "result" in existing:
            formatted["result"] = existing["result"]
            # Preserve created_at from existing file
            if "meta" in existing["result"]:
                formatted["result"]["meta"]["updated_at"] = datetime.now().isoformat()

    # Ensure meta exists
    if "meta" not in formatted["result"]:
        formatted["result"]["meta"] = {
            "model_name": model_name,
            "target": target,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "hmatc_version": __version__,
            "commit": __commit__,
        }

    # Format floats and overwrite results
    for key, value in res_info.items():
        if key == "eval":
            # eval uses backend as sub-key: {backend: {...}}
            if key not in formatted["result"]:
                formatted["result"][key] = {}
            if isinstance(value, dict):
                formatted["result"][key].update(format_dict_floats(value))
        else:
            # All other results: direct overwrite
            formatted["result"][key] = format_dict_floats(value)

    save_dict_to_yaml(formatted, result_path)
    logger.info(f"Result saved to: {result_path}")
