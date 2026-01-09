# Copyright 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download Qwen2 2 Blocks model.
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
import onnx
import argparse
from hmatc.utils.utils import get_file_from_jfrog

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        dest="model_type",
        type=str,
        default="all",
        help="which model type to get, choise in [raw, quant, all]",
    )
    parser.add_argument(
        "--quant_model_dir",
        dest="quant_model_dir",
        type=str,
        default="./",
        help="where to save quant_model",
    )
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
    quant_model_dir = args.quant_model_dir
    model_type = args.model_type
    model_dir = args.model_dir
    raw_path = "models/qwen2/qwen2.onnx"
    quant_path_1 = "http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/qwen2_prefill_2block.zip"
    quant_path_2 = "http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/qwen2_decode_2block.zip"

    if model_type == "raw" or model_type == "all":
        # get_file_from_jfrog(raw_path, model_dir)
        os.system(
            "huggingface-cli download --resume-download Qwen/Qwen2-7B-Instruct --local-dir qwen2-7b-instruct-hf"
        )

    if model_type == "quant" or model_type == "all":
        file_path_1 = get_file_from_jfrog(quant_path_1, model_dir)
        file_path_2 = get_file_from_jfrog(quant_path_2, model_dir)
        os.system("wget " + file_path_1)
        os.system("wget " + file_path_2)
        os.system("mkdir -p " + quant_model_dir)
        os.system("unzip -o -d " + quant_model_dir + " " + file_path_1)
        os.system("unzip -o -d " + quant_model_dir + " " + file_path_2)
