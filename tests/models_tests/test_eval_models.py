# Copyright 2025 HOUMO AI
#
# File: test_eval_models.py
# Description:
#   Model evaluation tests module.
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
from .test_models_utils import *

logger = logging.getLogger(__name__)


def _eval_func(model_name: str, log_file: str) -> None:
    """
    Execute model evaluation test for a specific model.

    Args:
        model_name (str): Name of the model to evaluate
        log_file (str): Path to the log file for test output
    """
    logger.info("===> TEST START: test_%s_eval", model_name)
    execute_eval_flow(model_name, log_file)


@pytest.mark.efficientnet
@pytest.mark.eval
def test_backbone_efficientnet_eval(setup_logging) -> None:
    model_name = "efficientnet"
    _eval_func(model_name, setup_logging)


@pytest.mark.mobilenetv2
@pytest.mark.eval
def test_backbone_mobilenetv2_eval(setup_logging) -> None:
    model_name = "mobilenetv2"
    _eval_func(model_name, setup_logging)


@pytest.mark.resnet50
@pytest.mark.eval
def test_backbone_resnet50_eval(setup_logging) -> None:
    model_name = "resnet50"
    _eval_func(model_name, setup_logging)


@pytest.mark.vit
@pytest.mark.eval
def test_backbone_vit_eval(setup_logging) -> None:
    model_name = "vit"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolov3
@pytest.mark.eval
def test_detection_yolov3_eval(setup_logging) -> None:
    model_name = "yolov3"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolov5s
@pytest.mark.eval
def test_detection_yolov5s_eval(setup_logging) -> None:
    model_name = "yolov5s"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolov5s_feature
@pytest.mark.eval
def test_detection_yolov5s_feature_eval(setup_logging) -> None:
    model_name = "yolov5s_feature"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolov8m
@pytest.mark.eval
def test_detection_yolov8m_eval(setup_logging) -> None:
    model_name = "yolov8m"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolo12m
@pytest.mark.eval
def test_detection_yolo12m_eval(setup_logging) -> None:
    model_name = "yolo12m"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolov8m_pose
@pytest.mark.eval
def test_estimation_yolov8m_pose_eval(setup_logging) -> None:
    model_name = "yolov8m-pose"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolov8m_seg
@pytest.mark.eval
def test_segmentation_yolov8m_seg_eval(setup_logging) -> None:
    model_name = "yolov8m-seg"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolov7
@pytest.mark.eval
def test_detection_yolov7_eval(setup_logging) -> None:
    model_name = "yolov7"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolov5m_face
@pytest.mark.eval
def test_detection_yolov5m_face_eval(setup_logging) -> None:
    model_name = "yolov5m_face"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolox
@pytest.mark.eval
def test_detection_yolox_eval(setup_logging) -> None:
    """test_detection_yolox_eval"""
    model_name = "yolox"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolov10m
@pytest.mark.eval
def test_detection_yolov10m_eval(setup_logging) -> None:
    """test_detection_yolov10m_eval"""
    model_name = "yolov10m"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolo11m
@pytest.mark.eval
def test_detection_yolo11m_eval(setup_logging) -> None:
    """test_detection_yolo11m_eval"""
    model_name = "yolo11m"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolov9m
@pytest.mark.eval
def test_detection_yolov9m_eval(setup_logging) -> None:
    """test_detection_yolov9m_eval"""
    model_name = "yolov9m"
    _eval_func(model_name, setup_logging)


@pytest.mark.yolov8m_cls
@pytest.mark.eval
def test_backbone_yolov8m_cls_eval(setup_logging) -> None:
    """test_backbone_yolov8m_cls_eval"""
    model_name = "yolov8m-cls"
    _eval_func(model_name, setup_logging)
