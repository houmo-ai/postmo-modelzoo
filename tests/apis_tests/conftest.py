# Copyright 2025 HOUMO AI
#
# File: conftest.py
# Description:
#   Configuration file for API tests using pytest framework.
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
os.environ.setdefault("HOUMO_EXAMPLES_PATH", os.path.abspath(f"{script_dir}/../.."))


def pytest_configure(config):
    """
    Configure pytest with custom markers for API tests.

    Args:
        config: pytest configuration object
    """
    # Define markers for different API test types
    apis_type_markers = [
        "apis",
        "inference",
        "multistreams",
        "pipeline",
        "multibatch",
        "video_detect",
    ]
    # Define markers for different model types
    md_markers = ["qwen3", "resnet50", "yolov5s", "qwen3_speculative", "qwen3_multibatch"]
    for markers in apis_type_markers:
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
        "apis_tests/test_inferences_apis.py",
        "apis_tests/test_scenes_apis.py",
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
            # Files not in the file_order list are placed at the end
            return len(file_order)

    # Sort the test items based on the defined order
    items.sort(key=get_sort_key)
