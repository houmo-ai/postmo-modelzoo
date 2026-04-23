# Copyright 2025 HOUMO AI
#
# File: get_3rdparty.py
# Description:
#   Download third-party dependencies for CosyVoice3 model Cpp example on Android.
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
from hmatc.utils.utils import get_file_from_jfrog


if __name__ == "__main__":
    tokenizer_path = "3rdparty/cosyvoice-audio-cpp.zip"
    target_dir = "./3rdparty"
    save_path = get_file_from_jfrog(tokenizer_path, target_dir, target_dir)
    print(f"Audio third-party dependencies downloaded to {save_path} and extracted to: {target_dir}")


    tokenizer_path = "3rdparty/cpp_3rdparty_source.zip"
    target_dir = os.path.join(os.getenv("HOUMO_EXAMPLES_PATH"), "apis/3rdparty")
    save_path = get_file_from_jfrog(tokenizer_path, target_dir, target_dir)
    print(f"Audio third-party dependencies downloaded to {save_path} and extracted to: {target_dir}")
