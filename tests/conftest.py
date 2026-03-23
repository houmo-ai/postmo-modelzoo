# Copyright 2025 HOUMO AI
#
# File: conftest.py
# Description:
#   Configuration file for model tests using pytest framework.
#   This file sets up the testing environment for model tests, including environment variable
#   configuration, test markers definition, and logging setup for individual test cases.
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
import logging
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

# Download models from local server
os.environ["HOUMO_MODELZOO_URL"] = "http://artifactory.houmo.ai/artifactory/Dadao"

# Set up library paths
ori_ld = os.getenv("LD_LIBRARY_PATH", "")
append_ld = f"/opt/venv/houmo/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/venv/houmo/lib/python3.12/site-packages/torch/lib:{script_dir}/../apis/models/3rdparty/onnxruntime/lib:"
os.environ["LD_LIBRARY_PATH"] = f"{ori_ld}:{append_ld}" if ori_ld else append_ld

# Set up dataset path
os.environ["HOUMO_DATASETS_PATH"] = f"{script_dir}/../data/datasets/"
if os.getenv("HOUMO_EXAMPLES_PATH", None) is None or not os.getenv(
    "HOUMO_EXAMPLES_PATH"
):
    os.environ["HOUMO_EXAMPLES_PATH"] = os.path.abspath(f"{script_dir}/../")

# Set default HOUMO version if not already set
if os.getenv("HOUMO_VERSION", None) is None:
    os.environ["HOUMO_VERSION"] = "2.4.2"
# os.environ["IMODELZOO_MODELS_PATH"] = f"{script_dir}/../../modelzoo/"
# os.environ["IMODELZOO_MODELS_PATH"] = f"/develop02/modelzoo/"

# Create models directory if it doesn't exist
# os.makedirs(f"{script_dir}/models/", exist_ok=True)


def pytest_configure(config):
    """
    Configure pytest with shared markers for all tests.

    Args:
        config: pytest configuration object
    """
    shared_markers = [
        "imodelzoo",
        "ndevice_1",
        "ndevice_2",
        "ndevice_4",
        "dev_mem_12g",
        "dev_mem_24g",
        "dev_mem_48g",
    ]
    for markers in shared_markers:
        config.addinivalue_line("markers", markers)


@pytest.fixture(autouse=True)
def setup_logging(request):
    """
    Create an independent log file for each test case and return the log path.

    Args:
        request: Pytest request object containing information about the test

    Yields:
        str: Path to the log file for the current test case
    """
    current_date = datetime.now().strftime("%Y%m%d")

    # Create log folder
    logs_dir = os.path.join(script_dir + "/", f"test_logs/{current_date}/")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    # Generate a log file name based on test name, module name, and timestamp
    test_name = request.node.name
    module_name = request.module.__name__ if request.module else "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{module_name}_{test_name}_{timestamp}.log")

    # Setup logging with file handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Clear existing handlers and add the new one
    logger.handlers.clear()
    logger.addHandler(file_handler)

    # Pass the log path to the test case
    yield (log_file, request)

    # Remove handler after test completion to prevent duplicate logs
    logger.removeHandler(file_handler)
