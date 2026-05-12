# Copyright (c) 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download Qwen3.5 / Qwen3.6 models for text generation tasks.
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
import argparse
from hmatc.utils.utils import hmatc_get_file, get_houmo_version

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        dest="file_type",
        type=str,
        default="hmm",
        choices=["raw", "hmm"],
        help="which resource to get, choice in [raw, hmm]",
    )
    parser.add_argument(
        "--download_dir",
        dest="download_dir",
        type=str,
        default=".",
        help="where to save downloaded model",
    )
    parser.add_argument(
        "--extract_dir",
        dest="extract_dir",
        type=str,
        default=None,
        help="where to save extracted files",
    )
    parser.add_argument(
        "--source_type",
        dest="source_type",
        type=str,
        default="jfrog",
        choices=["jfrog", "modelscope"],
        help="download the model from which source",
    )
    parser.add_argument(
        "--context_length",
        dest="context_length",
        type=str,
        default="256k",
        help="context length",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=1,
        help="device number",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="qwen3.6",
        choices=["qwen3.5", "qwen3.6"],
        help="model name: qwen3.5 or qwen3.6",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default="35b-a3b",
        choices=["0.8b", "2b", "4b", "9b", "27b", "35b-a3b"],
        help="model size: 0.8b, 2b, 4b, 9b, 27b, 35b-a3b",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Model configurations based on size
    model_configs = {
        "qwen3.5": {
            "0.8b": {
                "ncore": 2,
                "modelscope_repo": ["Qwen/Qwen3.5-0.8B"],
            },
            "2b": {
                "ncore": 2,
                "modelscope_repo": ["Qwen/Qwen3.5-2B"],
            },
            "4b": {
                "ncore": 2,
                "modelscope_repo": ["Qwen/Qwen3.5-4B"],
            },
            "9b": {
                "ncore": 2,
                "modelscope_repo": ["Qwen/Qwen3.5-9B"],
            },
            "27b": {
                "ncore": 2,
                "modelscope_repo": ["Qwen/Qwen3.5-27B"],
            },
            "35b-a3b": {
                "ncore": 2,
                "modelscope_repo": ["Qwen/Qwen3.5-35B-A3B"],
            },
        },
        "qwen3.6": {
            "35b-a3b": {
                "ncore": 2,
                "modelscope_repo": ["Qwen/Qwen3.6-35B-A3B"],
            },
            "27b": {
                "ncore": 2,
                "modelscope_repo": ["Qwen/Qwen3.6-27B"],
            },
        },
    }

    config = model_configs[args.model_name][args.model_size]

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": args.model_name,
        "model_info": {
            "model_size": args.model_size,
            "ncore": config["ncore"],
            "ndevice": args.ndevice,
            "context_len": args.context_length,
            "prefill_len": 256,
            "batch": args.batch,
        },
        "raw_files": {"raw_path": "3rdparty/wikitext-2-raw-v1.zip"},
        "modelscope_repo": {"repo_ids": config["modelscope_repo"]},
    }

    _, ret_dict = hmatc_get_file(
        model_cfgs,
        args.file_type,
        args.download_dir,
        args.extract_dir,
        args.source_type,
    )
    if ret_dict.get("ret", False) is False:
        exit(1)
