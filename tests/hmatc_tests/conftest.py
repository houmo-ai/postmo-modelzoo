# Copyright 2025 HOUMO AI
#
# File: conftest.py
# Description:
#   Configuration file for HMATC tests.
#   This file sets up the testing environment for HMATC tests using the pytest framework.
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

script_dir = os.path.dirname(os.path.abspath(__file__))


def pytest_configure(config):
    """
    Configure pytest with custom markers for HMATC tests.

    Args:
        config: pytest configuration object
    """
    # Define markers for HMATC test types
    apis_type_markers = ["hmatc"]
    # Define markers for different model types
    md_markers = [
        "resnet50",
        "yolov5s",
    ]
    for markers in apis_type_markers:
        config.addinivalue_line("markers", markers)
    for markers in md_markers:
        config.addinivalue_line("markers", markers)
