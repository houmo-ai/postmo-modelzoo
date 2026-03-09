# Copyright 2025 HOUMO AI
#
# File: test_inferences_apis.py
# Description:
#   APIs inference tests module.
#   This module contains test functions for API-based inference scenarios using various models
#   such as ResNet50, YOLOv5s, and Qwen3.
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


def _inference_func(example_name: str, setup_logging) -> None:
    """
    Execute API inference test for a specific example.

    Args:
        example_name (str): Name of the example to test
        setup_logging: pytest fixture for setting up logging configuration
    """
    logger.info("===> TEST START: test_apis_inferences_%s", example_name)
    execute_apis_examples(example_name, setup_logging)


@pytest.mark.apis
@pytest.mark.resnet50
@pytest.mark.inference
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_apis_inferences_resnet50(setup_logging) -> None:
    example_name = "resnet50"
    _inference_func(example_name, setup_logging)


@pytest.mark.apis
@pytest.mark.yolov5s
@pytest.mark.inference
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_apis_inferences_yolov5s(setup_logging) -> None:
    example_name = "yolov5s"
    _inference_func(example_name, setup_logging)


@pytest.mark.apis
@pytest.mark.qwen3
@pytest.mark.inference
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_apis_inferences_qwen3(setup_logging) -> None:
    example_name = "qwen3"
    _inference_func(example_name, setup_logging)


@pytest.mark.apis
@pytest.mark.resnet50
@pytest.mark.multistreams
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_apis_inferences_resnet50_multistreams(setup_logging) -> None:
    example_name = "resnet50_multistreams"
    _inference_func(example_name, setup_logging)


@pytest.mark.apis
@pytest.mark.resnet50
@pytest.mark.pipeline
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_apis_inferences_resnet50_pipeline(setup_logging) -> None:
    example_name = "resnet50_pipeline"
    _inference_func(example_name, setup_logging)


@pytest.mark.apis
@pytest.mark.resnet50
@pytest.mark.yolov5s
@pytest.mark.multibatch
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_apis_inferences_yolov5s_resnet50_multibatch(setup_logging) -> None:
    example_name = "yolov5s_resnet50_multibatch"
    _inference_func(example_name, setup_logging)
