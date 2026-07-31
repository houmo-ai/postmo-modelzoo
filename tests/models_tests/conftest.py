# Copyright (c) 2025 HOUMO AI
#
# File: conftest.py
# Description:
#  Pytest Configuration and Execution Ordering for Model Tests.
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
from pathlib import Path

# Historical snapshots retain their original ``test_*.py`` filenames for
# comparison only and must not participate in the active pytest suite.
collect_ignore = ["history_codes"]


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

    names_file = Path(__file__).resolve().parent / "model_names.txt"
    if not names_file.is_file():
        raise pytest.UsageError(
            f"{names_file.name} is missing; run " f"'python -m tests.models_tests.update_test_py' to regenerate it"
        )
    md_markers = [line.strip() for line in names_file.read_text(encoding="utf-8").splitlines() if line.strip()]

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
    worker_count = getattr(config.option, "numprocesses", None)
    if worker_count not in (None, 0, 1, "0", "1"):
        raise pytest.UsageError(
            "models_tests has cross-flow artifact dependencies and currently runs "
            "serially; pytest-xdist with more than one worker is not supported"
        )

    file_order = {
        "test_get_models.py": 0,
        "test_quant_models.py": 1,
        "test_compile_models.py": 2,
        "test_demo_models.py": 3,
        "test_compare_models.py": 4,
        "test_eval_models.py": 5,
        "test_perf_models.py": 6,
    }

    def get_sort_key(item):
        """
        Get the sort key for a test item based on its file location.

        Args:
            item: A pytest test item

        Returns:
            int: Index in the file_order list or length of list for files not in order
        """
        # Get the test file path from the item location
        filename = Path(item.location[0]).name
        try:
            return file_order[filename]
        except KeyError:
            return len(file_order)

    # Sort the test items based on the defined order
    items.sort(key=get_sort_key)
