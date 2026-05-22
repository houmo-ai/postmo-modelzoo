# Copyright (c) 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Model Download Tool - Python script for downloading qwen3 ASR
# models.
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
import shutil
import argparse
from hmatc.utils.utils import hmatc_get_file, get_houmo_version


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
HOUMO_EXAMPLES_PATH = os.getenv("HOUMO_EXAMPLES_PATH")


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
        "--ndevice",
        dest="ndevice",
        type=int,
        default=1,
        help="device number",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default="0.6b",
        help="device number",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="qwen3_asr",
        choices=["qwen3_asr", "qwen3_forcealigner"],
        help="device number",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    shutil.copy2(
        os.path.join(HOUMO_EXAMPLES_PATH, "data", "audio", "audio.mp3"), "./audio.mp3"
    )

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": args.model_name,
        "model_info": {
            "model_size": args.model_size,
            "ncore": 2,
            "ndevice": args.ndevice,
        },
        "modelscope_repo": {
            "repo_ids": ["Qwen/Qwen3-ASR-0.6B"],
            "ignore_patterns": ["*.bin", "*.h5", "*.msgpack", "*.safetensors"],
        },
    }

    if args.model_name == "qwen3_asr" and args.model_size == "1.7b":
        model_cfgs["modelscope_repo"]["repo_ids"][0] = "Qwen/Qwen3-ASR-1.7B"

    if args.model_name == "qwen3_forcealigner":
        model_cfgs["modelscope_repo"]["repo_ids"][0] = "Qwen/Qwen3-ForcedAligner-0.6B"

    _, ret_dict = hmatc_get_file(
        model_cfgs,
        args.file_type,
        args.download_dir,
        args.extract_dir,
        args.source_type,
    )
    if ret_dict.get("ret", False) is False:
        exit(1)
