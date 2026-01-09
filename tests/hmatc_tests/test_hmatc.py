# Copyright 2025 HOUMO AI
#
# File: test_hmatc.py
# Description:
#   HMATC test module.
#   This module contains test functions for HMATC testing across different models.
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
import logging
from .test_hmatc_utils import *

logger = logging.getLogger(__name__)


def _hmatc_func(model_name: str, log_file: str) -> None:
    """
    Execute HMATC test for a specific model.

    Args:
        model_name (str): Name of the model to test
        log_file (str): Path to the log file for test output
    """
    logger.info("===> TEST START: test_hmatc_%s", model_name)
    execute_hmatc_cmd(model_name, log_file)


@pytest.mark.hmatc
@pytest.mark.resnet50
def test_hmatc_resnet50(setup_logging) -> None:
    model_name = "resnet50"
    _hmatc_func(model_name, setup_logging)


@pytest.mark.hmatc
@pytest.mark.yolov5s
def test_hmatc_yolov5s(setup_logging) -> None:
    model_name = "yolov5s"
    _hmatc_func(model_name, setup_logging)
