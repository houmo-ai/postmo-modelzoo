# Copyright 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download ResNet50 model for image classification tasks.
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

HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../../..")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/hmatc")
from hmatc.utils.utils import get_file_from_jfrog


if __name__ == "__main__":
    model_dir = os.path.join(HOUMO_EXAMPLES_PATH, "apis/models")
    raw_path = "models/raw/onnx/resnet50.onnx"
    get_file_from_jfrog(raw_path, model_dir)
