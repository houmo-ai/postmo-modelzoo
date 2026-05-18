# Copyright 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download Qwen3 model (8B/14B) for text generation tasks.
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
from hmatc.utils.utils import hmatc_get_file, get_houmo_version, get_model_configs

HOUMO_CORE_NUM = os.getenv("HOUMO_CORE_NUM", 2)
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default="./config.yaml",
        help="path to config.yaml",
    )
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
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        help="model name",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size: 0.6b, 1.7b, 8b, or 14b",
    )
    parser.add_argument(
        "--context_length",
        dest="context_length",
        type=str,
        default="",
        help="context length",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=None,
        help="batch size",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number",
    )
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=None,
        help="number of cores",
    )
    parser.add_argument(
        "--prefill_length",
        dest="prefill_length",
        type=int,
        default=None,
        help="prefill length",
    )
    parser.add_argument(
        "--quant_type",
        dest="quant_type",
        type=str,
        default=None,
        help="quantization type",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    default_model_size, model_configs = get_model_configs(args.config_path)
    # Get model configs
    model_size = args.model_size if args.model_size is not None else default_model_size
    model_config = model_configs.get(model_size, {})

    ncore = (
        args.ncore
        if args.ncore is not None
        else model_config.get("ncore", HOUMO_CORE_NUM)
    )
    ndevice = (
        args.ndevice if args.ndevice is not None else model_config.get("ndevice", 1)
    )
    batch = args.batch if args.batch is not None else model_config.get("batch", 1)
    prefill_length = (
        args.prefill_length
        if args.prefill_length is not None
        else model_config.get("prefill_length", 256)
    )
    context_length = (
        args.context_length
        if args.context_length
        else model_config.get("context_length", "32k")
    )
    model_name = (
        args.model_name
        if args.model_name is not None
        else model_config.get("model_name", "qwen3")
    )

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": model_name,
        "model_info": {
            "model_size": model_config.get("model_size", model_size),
            "ncore": ncore,
            "ndevice": ndevice,
            "context_len": context_length,
            "prefill_len": prefill_length,
            "batch": batch,
        },
        "raw_files": {"raw_path": "3rdparty/wikitext-2-raw-v1.zip"},
        "modelscope_repo": {"repo_ids": model_config.get("modelscope_repo", [])},
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
