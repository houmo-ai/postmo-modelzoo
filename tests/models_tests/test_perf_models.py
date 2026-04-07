# Copyright 2025 HOUMO AI
#
# File: test_perf_models.py
# Description:
#   Model performance tests module.
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


def _perf_func(model_name: str, setup_logging) -> None:
    """
    Execute model performance test for a specific model.

    Args:
        model_name (str): Name of the model to performance test
        setup_logging: Fixture of setup_logging
    """
    logger.info("===> TEST START: test_%s_perf", model_name)
    execute_perf_flow(model_name, setup_logging)


@pytest.mark.yolop
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_autodrive_yolop_perf(setup_logging) -> None:
    model_name = "yolop"
    _perf_func(model_name, setup_logging)


@pytest.mark.efficientnet
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_backbone_efficientnet_perf(setup_logging) -> None:
    model_name = "efficientnet"
    _perf_func(model_name, setup_logging)


@pytest.mark.mobilenetv2
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_backbone_mobilenetv2_perf(setup_logging) -> None:
    model_name = "mobilenetv2"
    _perf_func(model_name, setup_logging)


@pytest.mark.resnet50
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_backbone_resnet50_perf(setup_logging) -> None:
    model_name = "resnet50"
    _perf_func(model_name, setup_logging)


@pytest.mark.vit
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_backbone_vit_perf(setup_logging) -> None:
    model_name = "vit"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov3
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_detection_yolov3_perf(setup_logging) -> None:
    model_name = "yolov3"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov5s
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_detection_yolov5s_perf(setup_logging) -> None:
    model_name = "yolov5s"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov5s_feature
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_detection_yolov5s_feature_perf(setup_logging) -> None:
    model_name = "yolov5s_feature"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov8m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_detection_yolov8m_perf(setup_logging) -> None:
    model_name = "yolov8m"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolo12m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_detection_yolo12m_perf(setup_logging) -> None:
    model_name = "yolo12m"
    _perf_func(model_name, setup_logging)


@pytest.mark.deepseek
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_llm_deepseek_perf(setup_logging) -> None:
    model_name = "deepseek"
    _perf_func(model_name, setup_logging)


@pytest.mark.deepseek_r1_qwen3_8b
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_llm_deepseek_r1_qwen3_8b_perf(setup_logging) -> None:
    model_name = "deepseek-r1-qwen3-8b"
    _perf_func(model_name, setup_logging)


@pytest.mark.qwen3
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_llm_qwen3_perf(setup_logging) -> None:
    model_name = "qwen3"
    _perf_func(model_name, setup_logging)


@pytest.mark.qwen3_14b
@pytest.mark.ndevice_2
@pytest.mark.dev_mem_24g
@pytest.mark.perf
def test_llm_qwen3_14b_perf(setup_logging) -> None:
    model_name = "qwen3-14b"
    _perf_func(model_name, setup_logging)


@pytest.mark.qwen2dot5
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_llm_qwen2dot5_perf(setup_logging) -> None:
    model_name = "qwen2.5"
    _perf_func(model_name, setup_logging)


@pytest.mark.qwen2dot5_vl
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_vlm_qwen2dot5_vl_perf(setup_logging) -> None:
    model_name = "qwen2.5-vl"
    _perf_func(model_name, setup_logging)


@pytest.mark.wenet
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_asr_wenet_perf(setup_logging) -> None:
    model_name = "wenet"
    _perf_func(model_name, setup_logging)


@pytest.mark.sdxl
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_diffusion_sdxl_perf(setup_logging) -> None:
    model_name = "sdxl"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov8m_pose
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_estimation_yolov8m_pose_perf(setup_logging) -> None:
    model_name = "yolov8m-pose"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov8m_seg
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_segmentation_yolov8m_seg_perf(setup_logging) -> None:
    model_name = "yolov8m-seg"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov7
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_detection_yolov7_perf(setup_logging) -> None:
    model_name = "yolov7"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov5m_face
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_detection_yolov5m_face_perf(setup_logging) -> None:
    model_name = "yolov5m_face"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolox
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_detection_yolox_perf(setup_logging) -> None:
    model_name = "yolox"
    _perf_func(model_name, setup_logging)


@pytest.mark.lprnet
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_ocr_lprnet_perf(setup_logging) -> None:
    """test_ocr_lprnet_perf"""
    model_name = "lprnet"
    _perf_func(model_name, setup_logging)


@pytest.mark.ppocrv3_det
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_ocr_ppocrv3_det_perf(setup_logging) -> None:
    """test_ocr_ppocrv3_det_perf"""
    model_name = "ppocrv3_det"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov10m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_detection_yolov10m_perf(setup_logging) -> None:
    """test_detection_yolov10m_perf"""
    model_name = "yolov10m"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolo11m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_detection_yolo11m_perf(setup_logging) -> None:
    """test_detection_yolo11m_perf"""
    model_name = "yolo11m"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov9m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_detection_yolov9m_perf(setup_logging) -> None:
    """test_detection_yolov9m_perf"""
    model_name = "yolov9m"
    _perf_func(model_name, setup_logging)


@pytest.mark.qwen3_30b_a3b
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
@pytest.mark.perf
def test_llm_qwen3_30b_a3b_perf(setup_logging) -> None:
    """test_llm_qwen3_30b_a3b_perf"""
    model_name = "qwen3-30b-a3b"
    _perf_func(model_name, setup_logging)


@pytest.mark.qwen3_vl
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_vlm_qwen3_vl_perf(setup_logging) -> None:
    """test_vlm_qwen3_vl_perf"""
    model_name = "qwen3-vl"
    _perf_func(model_name, setup_logging)


@pytest.mark.whisper
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_asr_whisper_perf(setup_logging) -> None:
    """test_asr_whisper_perf"""
    model_name = "whisper"
    _perf_func(model_name, setup_logging)


@pytest.mark.gte
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_embedding_gte_perf(setup_logging) -> None:
    """test_embedding_gte_perf"""
    model_name = "gte"
    _perf_func(model_name, setup_logging)


@pytest.mark.gpt_oss
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
@pytest.mark.perf
def test_llm_gpt_oss_perf(setup_logging) -> None:
    """test_llm_gpt_oss_perf"""
    model_name = "gpt-oss"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov8m_cls
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_backbone_yolov8m_cls_perf(setup_logging) -> None:
    """test_backbone_yolov8m_cls_perf"""
    model_name = "yolov8m-cls"
    _perf_func(model_name, setup_logging)


@pytest.mark.yolo26m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.perf
def test_detection_yolo26m_perf(setup_logging) -> None:
    """test_detection_yolo26m_perf"""
    model_name = 'yolo26m'
    _perf_func(model_name, setup_logging)
