# Copyright 2025 HOUMO AI
#
# File: conftest.py
# Description:
#   Configuration file for model tests using pytest framework.
#   This file sets up the testing environment for model tests, including test markers
#   definition and test execution ordering.
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

import pytest
import os


def pytest_configure(config):
    """
    Configure pytest with custom markers for model tests.

    Args:
        config: pytest configuration object
    """
    # Define markers for different test flow phases
    test_flow_markers = [
        "get_model",
        "quant",
        "compile",
        "demo",
        "compare",
        "eval",
        "perf",
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_markers = []

    # Read model names from the text file
    with open(f"{script_dir}/model_names.txt", "r", encoding="utf-8") as f:
        for line in f:
            model_name = line.strip()
            if model_name:
                md_markers.append(model_name)
    print("Supported model names:", md_markers)

    for markers in test_flow_markers:
        config.addinivalue_line("markers", markers)
    for markers in md_markers:
        config.addinivalue_line("markers", markers)


def pytest_collection_modifyitems(session, config, items):
    """
    Modify the order of collected test items to ensure proper execution sequence.

    Args:
        session: pytest session object
        config: pytest configuration object
        items: List of collected test items to be modified in place
    """
    # Define the preferred execution order for test files
    file_order = [
        "models_tests/test_get_models.py",
        "models_tests/test_quant_models.py",
        "models_tests/test_compile_models.py",
        "models_tests/test_demo_models.py",
        "models_tests/test_compare_models.py",
        "models_tests/test_eval_models.py",
        "models_tests/test_perf_models.py",
    ]

    def get_sort_key(item):
        """
        Get the sort key for a test item based on its file location.

        Args:
            item: A pytest test item

        Returns:
            int: Index in the file_order list or length of list for files not in order
        """
        # Get the test file path from the item location
        filename = item.location[0]
        try:
            return file_order.index(filename)
        except ValueError:
            return len(file_order)

    # Sort the test items based on the defined order
    items.sort(key=get_sort_key)
