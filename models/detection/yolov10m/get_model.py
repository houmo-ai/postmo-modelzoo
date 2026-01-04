# Copyright 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download YOLOv10m model for image detection tasks.
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
import onnx
import argparse
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
        help="where to save build_model",
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
    build_model_dir = args.build_model_dir
    model_type = args.model_type
    model_dir = args.model_dir

    model_name = "yolov10m"
    ncore = 1
    batch = 1
    opt_level = "O2"
    version = get_houmo_version()
    target = HOUMO_TARGET
    raw_path = "models/raw/onnx/yolov10m.onnx"
    build_path = f"models/{target.lower()}-{version}/{model_name}/{model_name}_{target}_b{batch}_{ncore}core_{opt_level}_{version}.tar.xz"

    if model_type in ["raw"]:
        file_path = get_file_from_jfrog(raw_path, model_dir)
        if file_path:
            extract_path = os.path.join(os.path.dirname(file_path), "yolov10m_opt.onnx")
            onnx.utils.extract_model(
                file_path,
                extract_path,
                input_names=["images"],
                output_names=[
                    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
                    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
                    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
                    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
                    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
                    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
                ],
                check_model=True,
            )
        else:
            sys.exit(1)

    if model_type in ["hmm"] and not get_file_from_jfrog(
        build_path, model_dir, build_model_dir
    ):
        sys.exit(1)
