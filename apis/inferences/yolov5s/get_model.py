# Copyright 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download Yolov5s model for image detection tasks.
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
import platform
import argparse

HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../../..")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/hmatc")
from hmatc.utils.utils import get_file_from_jfrog, get_houmo_version

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline arguments for model download configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default="",
        help="where to save downloaded model",
    )
    parser.add_argument(
        "--enable_ort",
        action="store_true",
        help="install onnxruntime environment to support post-processing model inference.",
    )
    args = parser.parse_args()
    return args


def execute_cmd(cmd, shell=False):
    """Execute a command in the shell and return the result.

    Args:
        cmd: Command to execute
        shell (bool): Whether to run the command in shell mode

    Returns:
        str: Output from the command execution

    Raises:
        Exception: If command execution fails
    """
    import subprocess

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, shell=shell
        )
        return result.stdout
    except Exception as e:
        print(f"Error occured: {e}")
        raise


def install_ort_env(third_party_dir: str, ort_pkg_name: str) -> None:
    """Install onnxruntime environment to support post-processing model inference.

    This function configures the ORT C++ environment by renaming the downloaded
    package directory to 'onnxruntime'.

    Args:
        third_party_dir (str): Path to the third party directory
        ort_pkg_name (str): Name of the ORT package to be renamed

    Returns:
        None
    """
    try:
        print("configure the ORT C++ environment...")
        os.rename(f"{third_party_dir}/{ort_pkg_name}", f"{third_party_dir}/onnxruntime")
    except Exception as e:
        print(f"Failed to configure the ORT C++ environment, error: {e}")


if __name__ == "__main__":
    args = get_args()

    # Determine the model directory to save the downloaded model
    model_dir = (
        os.path.join(HOUMO_EXAMPLES_PATH, "apis/models")
        if not args.model_dir
        else args.model_dir
    )
    # Path to the YOLOv5s HMM model in the repository
    model_name = "yolov5s"
    ncore = 1
    batch = 1
    opt_level = "O2"
    version = get_houmo_version()
    target = HOUMO_TARGET
    hmm_path = f"models/{target.lower()}-{version}/{model_name}_api/{model_name}_{target}_b{batch}_{ncore}core_{opt_level}_{version}.tar.xz"
    get_file_from_jfrog(hmm_path, model_dir, "./")

    # If ONNX Runtime support is enabled and the system is Linux
    if args.enable_ort and platform.system() == "Linux":
        # Download yolov5s post-processing onnx model
        onnx_path = "models/raw/onnx/yolov5s_640x640_postprocess.onnx"
        get_file_from_jfrog(onnx_path, "./")

        platform_name = platform.machine()
        # Determine the appropriate ORT package based on architecture
        if platform_name == "x86_64":
            ort_env_str = "x64"
        elif platform_name == "aarch64":
            ort_env_str = "aarch64"
        else:
            print(
                f"Current platform is {platform_name} and does not support onnxruntime c++ env."
            )
            exit(0)

        third_party_dir = os.path.join(HOUMO_EXAMPLES_PATH, "apis/models/3rdparty")
        ort_pkg_name = "onnxruntime-linux-" + ort_env_str + "-1.22.0"
        ort_pkg_path = "3rdparty/" + ort_pkg_name + ".tgz"
        get_file_from_jfrog(ort_pkg_path, third_party_dir, third_party_dir)

        # Install ORT environment if it doesn't already exist
        if not os.path.exists(third_party_dir + "/onnxruntime"):
            install_ort_env(third_party_dir, ort_pkg_name)
