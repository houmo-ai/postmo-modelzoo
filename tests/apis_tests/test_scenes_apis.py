# Copyright 2025 HOUMO AI
#
# File: test_scenes_apis.py
# Description:
#   APIs scene tests module.
#   This module contains test functions for API-based scene scenarios,
#   such as video detection using multiple models.
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
from .test_apis_utils import *

logger = logging.getLogger(__name__)


def _scene_func(example_name: str, log_file: str) -> None:
    """
    Execute API scene test for a specific example.

    Args:
        example_name (str): Name of the example to test
        log_file (str): Path to the log file for test output
    """
    logger.info("===> TEST START: test_apis_scenes_%s", example_name)
    execute_apis_examples(example_name, log_file)


@pytest.mark.apis
@pytest.mark.video_detect
@pytest.mark.yolov5s
@pytest.mark.resnet50
def test_apis_scenes_video_detect(setup_logging) -> None:
    example_name = "video_detect"
    _scene_func(example_name, setup_logging)
