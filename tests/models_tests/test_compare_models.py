# Copyright 2025 HOUMO AI
#
# File: test_compare_models.py
# Description:
#   Model comparison tests module.
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


def _compare_func(model_name: str, log_file: str) -> None:
    """
    Execute model comparison test for a specific model.

    Args:
        model_name (str): Name of the model to compare
        log_file (str): Path to the log file for test output
    """
    logger.info("===> TEST START: test_%s_compare", model_name)
    execute_compare_flow(model_name, log_file)


@pytest.mark.yolop
@pytest.mark.compare
def test_autodrive_yolop_compare(setup_logging) -> None:
    model_name = "yolop"
    _compare_func(model_name, setup_logging)


@pytest.mark.efficientnet
@pytest.mark.compare
def test_backbone_efficientnet_compare(setup_logging) -> None:
    model_name = "efficientnet"
    _compare_func(model_name, setup_logging)


@pytest.mark.mobilenetv2
@pytest.mark.compare
def test_backbone_mobilenetv2_compare(setup_logging) -> None:
    model_name = "mobilenetv2"
    _compare_func(model_name, setup_logging)


@pytest.mark.resnet50
@pytest.mark.compare
def test_backbone_resnet50_compare(setup_logging) -> None:
    model_name = "resnet50"
    _compare_func(model_name, setup_logging)


@pytest.mark.vit
@pytest.mark.compare
def test_backbone_vit_compare(setup_logging) -> None:
    model_name = "vit"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolov3
@pytest.mark.compare
def test_detection_yolov3_compare(setup_logging) -> None:
    model_name = "yolov3"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolov5s
@pytest.mark.compare
def test_detection_yolov5s_compare(setup_logging) -> None:
    model_name = "yolov5s"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolov5s_feature
@pytest.mark.compare
def test_detection_yolov5s_feature_compare(setup_logging) -> None:
    model_name = "yolov5s_feature"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolov8m
@pytest.mark.compare
def test_detection_yolov8m_compare(setup_logging) -> None:
    model_name = "yolov8m"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolo12m
@pytest.mark.compare
def test_detection_yolo12m_compare(setup_logging) -> None:
    model_name = "yolo12m"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolov8m_pose
@pytest.mark.compare
def test_estimation_yolov8m_pose_compare(setup_logging) -> None:
    model_name = "yolov8m-pose"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolov8m_seg
@pytest.mark.compare
def test_segmentation_yolov8m_seg_compare(setup_logging) -> None:
    model_name = "yolov8m-seg"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolov7
@pytest.mark.compare
def test_detection_yolov7_compare(setup_logging) -> None:
    model_name = "yolov7"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolov5m_face
@pytest.mark.compare
def test_detection_yolov5m_face_compare(setup_logging) -> None:
    model_name = "yolov5m_face"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolox
@pytest.mark.compare
def test_detection_yolox_compare(setup_logging) -> None:
    model_name = "yolox"
    _compare_func(model_name, setup_logging)


@pytest.mark.lprnet
@pytest.mark.compare
def test_ocr_lprnet_compare(setup_logging) -> None:
    """test_ocr_lprnet_compare"""
    model_name = "lprnet"
    _compare_func(model_name, setup_logging)


@pytest.mark.ppocrv3_det
@pytest.mark.compare
def test_ocr_ppocrv3_det_compare(setup_logging) -> None:
    """test_ocr_ppocrv3_det_compare"""
    model_name = "ppocrv3_det"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolov10m
@pytest.mark.compare
def test_detection_yolov10m_compare(setup_logging) -> None:
    """test_detection_yolov10m_compare"""
    model_name = "yolov10m"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolo11m
@pytest.mark.compare
def test_detection_yolo11m_compare(setup_logging) -> None:
    """test_detection_yolo11m_compare"""
    model_name = "yolo11m"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolov9m
@pytest.mark.compare
def test_detection_yolov9m_compare(setup_logging) -> None:
    """test_detection_yolov9m_compare"""
    model_name = "yolov9m"
    _compare_func(model_name, setup_logging)


@pytest.mark.yolov8m_cls
@pytest.mark.compare
def test_backbone_yolov8m_cls_compare(setup_logging) -> None:
    """test_backbone_yolov8m_cls_compare"""
    model_name = "yolov8m-cls"
    _compare_func(model_name, setup_logging)
