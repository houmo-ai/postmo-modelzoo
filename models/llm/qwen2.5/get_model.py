# Copyright 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download Qwen2.5 model for text generation tasks.
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
        help="which resource to get, choise in [raw, hmm]",
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
        default="32k",
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": "qwen2.5",
        "model_info": {
            "model_size": "7b",
            "ncore": 2,
            "ndevice": args.ndevice,
            "context_len": args.context_length,
            "prefill_len": 256,
            "batch": args.batch,
        },
        "raw_files": {"raw_path": "3rdparty/wikitext-2-raw-v1.zip"},
        "modelscope_repo": {"repo_ids": ["qwen/qwen2.5-7b"]},
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
