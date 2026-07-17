#!/usr/bin/env python3
# Copyright 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download SigLIP2 raw or compiled model artifacts.
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
"""Download SigLIP2 raw or compiled model artifacts."""

import argparse
import os
from hmatc.utils.utils import (
    first_not_none,
    get_houmo_version,
    get_model_configs,
    hmatc_get_file,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument(
        "--type",
        dest="file_type",
        default="hmm",
        choices=["raw", "hmm"],
        help="resource type to download",
    )
    parser.add_argument(
        "--download_dir",
        default=".",
        help="directory for downloaded files",
    )
    parser.add_argument(
        "--extract_dir",
        default=None,
        help="directory for extracted files",
    )
    parser.add_argument(
        "--source_type",
        default="jfrog",
        choices=["jfrog", "modelscope"],
        help="model download source",
    )
    parser.add_argument("--model_name", default=None, help="model name")
    parser.add_argument(
        "--model_size",
        default=None,
        choices=["large-patch16-256"],
        help="model size",
    )
    parser.add_argument("--ndevice", type=int, default=None, help="device number")
    parser.add_argument("--ncore", type=int, default=None, help="core number")
    parser.add_argument(
        "--quant_type",
        dest="quant_type",
        type=str,
        default=None,
        help="quantization type",
    )
    return parser.parse_args()


def main():
    args = get_args()
    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    model_name = first_not_none(args.model_name, default_model_name)
    model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(model_name, {}).get(model_size, {})
    ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    ncore = first_not_none(args.ncore, model_config.get("ncore", 1))
    quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "wmix_amix")
    )

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": model_name,
        "model_info": {
            "model_size": model_size,
            "ncore": ncore,
            "ndevice": ndevice,
            "quant_type": quant_type,
        },
    }
    if args.file_type == "raw":
        model_cfgs["modelscope_repo"] = {
            "repo_ids": model_config.get("modelscope_repo", [])
        }

    _, ret_dict = hmatc_get_file(
        model_cfgs,
        args.file_type,
        args.download_dir,
        args.extract_dir,
        args.source_type,
    )
    if not ret_dict.get("ret", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
