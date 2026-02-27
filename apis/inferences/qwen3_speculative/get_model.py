# Copyright 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download Qwen3 Speculative model for text generation tasks.
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
import sys
import argparse
from modelscope import snapshot_download

HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../../..")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/hmatc")
from hmatc.utils.utils import get_file_from_jfrog

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default="",
        help="where to save downloaded model",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    _ = snapshot_download(
        model_id="qwen/qwen3-8b",
        cache_dir="./qwen3-8b",
        ignore_file_pattern=["*.safetensors", "*.bin"],
    )

    model_dir = (
        os.path.join(HOUMO_EXAMPLES_PATH, "apis/models")
        if not args.model_dir
        else args.model_dir
    )
    hmm_path = "models/apis/qwen3/hmm_xh2_qwen3_speculative_b1_2core_1device.zip"

    get_file_from_jfrog(hmm_path, model_dir, os.path.join("output", HOUMO_TARGET))
