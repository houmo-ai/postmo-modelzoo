# Copyright (c) 2026 HOUMO AI
#
# File: get_3rdparty.py
# Description:
#   Download 3rd party dependencies.
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
from hmatc.utils.utils import get_file_from_jfrog

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    googletest_path = "3rdparty/googletest.zip"
    target_dir = "./3rdparty"
    save_path = get_file_from_jfrog(googletest_path, target_dir, target_dir)
    print(f"GoogleTest downloaded to {save_path} and extracted to: {target_dir}")

    ctest_data = "3rdparty/ctest_data.zip"
    target_dir = "./tests"
    save_path = get_file_from_jfrog(ctest_data, target_dir, target_dir)
    print(f"CTest data downloaded to {save_path} and extracted to: {target_dir}")
