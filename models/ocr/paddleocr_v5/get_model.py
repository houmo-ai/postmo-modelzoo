# Copyright 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download PPOCRv5 det, rec, and HMM models for recognition tasks.
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

import argparse
import os
import sys

from hmatc.utils.utils import get_file_from_jfrog, get_houmo_version

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        dest="model_type",
        type=str,
        default="hmm",
        choices=["raw", "hmm"],
        help="which model type to get, choise in [raw, hmm]",
    )
    parser.add_argument(
        "--build_model_dir",
        dest="build_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
        help="where to save build models",
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=".",
        help="where to save downloaded model",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    build_model_dir = args.build_model_dir
    model_type = args.model_type
    model_dir = args.model_dir

    model_name = "paddleocr_v5"
    target = HOUMO_TARGET.lower()
    det_raw_path = "models/raw/onnx/paddleocr_v5_det_sim.onnx"
    rec_raw_path = "models/raw/onnx/paddleocr_v5_rec_sim.onnx"
    det_input_path = "http://10.10.1.53:8082/artifactory/toolchain/support/models/paddleocr_v5/paddleocr_v5_det_input.npy"
    batch = 1
    ncore = 1
    opt_level = "O2"
    version = get_houmo_version()
    hmm_path = (
        f"models/{target}-{version}/{model_name}/"
        f"{model_name}_{HOUMO_TARGET}_b{batch}_{ncore}core_{opt_level}_{version}.tar.xz"
    )

    if model_type in ["raw"]:
        get_file_from_jfrog(det_raw_path, model_dir)
        get_file_from_jfrog(rec_raw_path, model_dir)
        get_file_from_jfrog(det_input_path, model_dir)

    if model_type in ["hmm"] and not get_file_from_jfrog(
        hmm_path, model_dir, build_model_dir
    ):
        sys.exit(1)
