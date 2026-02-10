# Copyright 2025 HOUMO AI
#
# File: execute_demo.py
# Description:
#   Execute LLM model Demo.
#
#   This script processes performance test results from LLM models and generates Excel reports
#   with detailed performance metrics.
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
import json
import argparse
import logging
from compiler_utils import setup_logging, execute_cmd
import pandas as pd
from typing import Dict

script_dir = os.path.dirname(os.path.abspath(__file__))

MODELZOO_FOLDER = "/data02/modelzoo"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify LLMs")
    parser.add_argument(
        "--perf_cfg",
        type=str,
        help="the path of perf_file.",
    )
    parser.add_argument(
        "-log",
        "--log_file",
        type=str,
        default="",
        help="the path of log.",
    )

    args = parser.parse_args()
    return args


def _load_config_file(cfg_path: str, default_data: Dict = None) -> Dict:
    """
    Load configuration file with fallback to default data.

    Args:
        cfg_path (str): Path to the configuration file
        default_data (Dict): Default data to return if file doesn't exist or loading fails

    Returns:
        Dict: Loaded configuration data
    """
    default_data = default_data or {"Streams": []}
    if not os.path.exists(cfg_path):
        return default_data
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Load config {cfg_path} failed: {e}, use default data")
        return default_data


if __name__ == "__main__":
    args = parse_args()

    setup_logging(log_file=args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("Perf cfg path: %s", args.perf_cfg)

    cfg_data = _load_config_file(args.perf_cfg)
    model_info = _load_config_file(f"{script_dir}/supported_models.json")

    os.environ["HDPL_PLATFORM"] = "ASIC"

    verify_dict = {}
    # Execute demo command
    for demo_md in cfg_data["Streams"]:
        os.chdir(script_dir)

        model_key_str = demo_md["ModelName"]
        model_name = demo_md["model_name"]
        model_size = demo_md["model_size"]

        verify_dict[model_name] = False

        model_raw_name = model_info[model_name][model_size].get("raw", "")
        model_folder = model_info[model_name][model_size].get("path", "")

        if not model_raw_name or not model_folder:
            logger.warning(f"Skip to verify model {model_key_str}.")
            continue

        os.chdir(f"{script_dir}/../../{model_folder}")

        embedding_path = demo_md["embedding"].replace(".bin", ".pt")
        demo_cmd = [
            "python3",
            "demo.py",
            "--tokenizer_dir",
            f"{MODELZOO_FOLDER}/{model_folder}/{model_raw_name}",
            "--embedding_path",
            embedding_path,
            "--prefill_path",
            demo_md["prefill"],
            "--decode_path",
            demo_md["decode"],
            "--ndevice",
            str(demo_md["ndevices"]),
            "--question",
            "你是谁",
        ]
        logger.info(f"[Verfiy Demo] execute cmd: {demo_cmd}, folder: {os.getcwd()}")
        try:
            ret, outputs = execute_cmd(demo_cmd, args.log_file, get_outputs=True)
        except Exception as e:
            logger.error(f"[Verfiy Demo] {model_key_str} failed: {e}")
            continue

        if ret:
            target_keywords = ["通义千问", "DeepSeek", "ChatGPT"]
            keyword_count = 0
            garbled_count = 0
            for line in outputs:
                line_stripped = line.strip()

                for target_keyword in target_keywords:
                    if target_keyword in line_stripped:
                        keyword_count += line_stripped.count(target_keyword)
                        break

                if "�" in line_stripped:
                    garbled_count += 1

            if keyword_count > 5:
                logger.error(
                    f"[Verfiy Demo] {model_key_str} failed: Duplication occurs."
                )
                continue
            if garbled_count > 0:
                logger.error(
                    f"[Verfiy Demo] {model_key_str} failed: Garbled characters occur, num: {garbled_count}."
                )
                continue

            verify_dict[model_name] = True

    logger.info(f"[Verfiy Demo] Final Results: {verify_dict}")
