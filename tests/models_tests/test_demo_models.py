# Copyright 2025 HOUMO AI
#
# File: test_demo_models.py
# Description:
#   Model demo tests module.
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


def _demo_func(model_name: str, setup_logging) -> None:
    """
    Execute model demo test for a specific model.

    Args:
        model_name (str): Name of the model to demo
        setup_logging: Fixture of setup_logging
    """
    logger.info("===> TEST START: test_%s_demo", model_name)
    execute_demo_flow(model_name, setup_logging)


@pytest.mark.wenet
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_asr_wenet_demo(setup_logging) -> None:
    model_name = "wenet"
    _demo_func(model_name, setup_logging)


@pytest.mark.qwen3_asr
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_asr_qwen3_asr_demo(setup_logging) -> None:
    """test_asr_qwen3_asr_demo"""
    model_name = "qwen3-asr"
    _demo_func(model_name, setup_logging)


@pytest.mark.qwen3_forcealigner
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_asr_qwen3_forcealigner_demo(setup_logging) -> None:
    """test_asr_qwen3_forcealigner_demo"""
    model_name = "qwen3-forcealigner"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolop
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_autodrive_yolop_demo(setup_logging) -> None:
    model_name = "yolop"
    _demo_func(model_name, setup_logging)


@pytest.mark.efficientnet
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_backbone_efficientnet_demo(setup_logging) -> None:
    model_name = "efficientnet"
    _demo_func(model_name, setup_logging)


@pytest.mark.mobilenetv2
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_backbone_mobilenetv2_demo(setup_logging) -> None:
    model_name = "mobilenetv2"
    _demo_func(model_name, setup_logging)


@pytest.mark.resnet50
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_backbone_resnet50_demo(setup_logging) -> None:
    model_name = "resnet50"
    _demo_func(model_name, setup_logging)


@pytest.mark.vit
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_backbone_vit_demo(setup_logging) -> None:
    model_name = "vit"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolov3
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_detection_yolov3_demo(setup_logging) -> None:
    model_name = "yolov3"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolov5s
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_detection_yolov5s_demo(setup_logging) -> None:
    model_name = "yolov5s"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolov5s_feature
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_detection_yolov5s_feature_demo(setup_logging) -> None:
    model_name = "yolov5s_feature"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolov8m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_detection_yolov8m_demo(setup_logging) -> None:
    model_name = "yolov8m"
    _demo_func(model_name, setup_logging)


@pytest.mark.sdxl
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_diffusion_sdxl_demo(setup_logging) -> None:
    model_name = "sdxl"
    _demo_func(model_name, setup_logging)


@pytest.mark.qwen2dot5
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_llm_qwen2dot5_demo(setup_logging) -> None:
    model_name = "qwen2.5"
    _demo_func(model_name, setup_logging)


@pytest.mark.qwen3
@pytest.mark.ndevice_2
@pytest.mark.dev_mem_24g
@pytest.mark.demo
def test_llm_qwen3_demo(setup_logging) -> None:
    model_name = "qwen3"
    _demo_func(model_name, setup_logging)


@pytest.mark.qwen3dot5
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
@pytest.mark.demo
def test_llm_qwen3dot5_demo(setup_logging) -> None:
    """test_llm_qwen3dot5_demo"""
    model_name = "qwen3.5"
    _demo_func(model_name, setup_logging)


@pytest.mark.deepseek
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_llm_deepseek_demo(setup_logging) -> None:
    model_name = "deepseek"
    _demo_func(model_name, setup_logging)


@pytest.mark.deepseek_r1_qwen3_8b
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_llm_deepseek_r1_qwen3_8b_demo(setup_logging) -> None:
    model_name = "deepseek-r1-qwen3-8b"
    _demo_func(model_name, setup_logging)


@pytest.mark.qwen2dot5_vl
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_vlm_qwen2dot5_vl_demo(setup_logging) -> None:
    model_name = "qwen2.5-vl"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolo12m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_detection_yolo12m_demo(setup_logging) -> None:
    model_name = "yolo12m"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolov8m_pose
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_estimation_yolov8m_pose_demo(setup_logging) -> None:
    model_name = "yolov8m-pose"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolov8m_seg
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_segmentation_yolov8m_seg_demo(setup_logging) -> None:
    model_name = "yolov8m-seg"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolov7
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_detection_yolov7_demo(setup_logging) -> None:
    model_name = "yolov7"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolov5m_face
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_detection_yolov5m_face_demo(setup_logging) -> None:
    model_name = "yolov5m_face"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolox
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_detection_yolox_demo(setup_logging) -> None:
    model_name = "yolox"
    _demo_func(model_name, setup_logging)


@pytest.mark.lprnet
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_ocr_lprnet_demo(setup_logging) -> None:
    """test_ocr_lprnet_demo"""
    model_name = "lprnet"
    _demo_func(model_name, setup_logging)


@pytest.mark.ppocrv3_det
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_ocr_ppocrv3_det_demo(setup_logging) -> None:
    """test_ocr_ppocrv3_det_demo"""
    model_name = "ppocrv3_det"
    _demo_func(model_name, setup_logging)


@pytest.mark.ppocrv3_rec
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_ocr_ppocrv3_rec_demo(setup_logging) -> None:
    """test_ocr_ppocrv3_rec_demo"""
    model_name = "ppocrv3_rec"
    _demo_func(model_name, setup_logging)


@pytest.mark.bge
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_embedding_bge_demo(setup_logging) -> None:
    """test_embedding_bge_demo"""
    model_name = "bge"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolov10m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_detection_yolov10m_demo(setup_logging) -> None:
    """test_detection_yolov10m_demo"""
    model_name = "yolov10m"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolo11m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_detection_yolo11m_demo(setup_logging) -> None:
    """test_detection_yolo11m_demo"""
    model_name = "yolo11m"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolov9m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_detection_yolov9m_demo(setup_logging) -> None:
    """test_detection_yolov9m_demo"""
    model_name = "yolov9m"
    _demo_func(model_name, setup_logging)


@pytest.mark.minicpmo
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_omni_minicpmo_demo(setup_logging) -> None:
    """test_omni_minicpmo_demo"""
    model_name = "minicpmo"
    _demo_func(model_name, setup_logging)


@pytest.mark.qwen3_30b_a3b
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
@pytest.mark.demo
def test_llm_qwen3_30b_a3b_demo(setup_logging) -> None:
    """test_llm_qwen3_30b_a3b_demo"""
    model_name = "qwen3-30b-a3b"
    _demo_func(model_name, setup_logging)


@pytest.mark.qwen3_vl
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_vlm_qwen3_vl_demo(setup_logging) -> None:
    """test_vlm_qwen3_vl_demo"""
    model_name = "qwen3-vl"
    _demo_func(model_name, setup_logging)


@pytest.mark.whisper
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_asr_whisper_demo(setup_logging) -> None:
    """test_asr_whisper_demo"""
    model_name = "whisper"
    _demo_func(model_name, setup_logging)


@pytest.mark.sensevoice
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_asr_sensevoice_demo(setup_logging) -> None:
    """test_asr_sensevoice_demo"""
    model_name = "sensevoice"
    _demo_func(model_name, setup_logging)


@pytest.mark.glm_asr
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_asr_glm_asr_demo(setup_logging) -> None:
    """test_asr_glm_asr_demo"""
    model_name = "glm-asr"
    _demo_func(model_name, setup_logging)


@pytest.mark.whisper_turbo
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_asr_whisper_turbo_demo(setup_logging) -> None:
    """test_asr_whisper_turbo_demo"""
    model_name = "whisper-turbo"
    _demo_func(model_name, setup_logging)


@pytest.mark.gte
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_embedding_gte_demo(setup_logging) -> None:
    """test_embedding_gte_demo"""
    model_name = "gte"
    _demo_func(model_name, setup_logging)


@pytest.mark.qwen3_embedding
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_embedding_qwen3_embedding_demo(setup_logging) -> None:
    """test_embedding_qwen3_embedding_demo"""
    model_name = "qwen3-embedding"
    _demo_func(model_name, setup_logging)


@pytest.mark.gpt_oss
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
@pytest.mark.demo
def test_llm_gpt_oss_demo(setup_logging) -> None:
    """test_llm_gpt_oss_demo"""
    model_name = "gpt-oss"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolov8m_cls
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_backbone_yolov8m_cls_demo(setup_logging) -> None:
    """test_backbone_yolov8m_cls_demo"""
    model_name = "yolov8m-cls"
    _demo_func(model_name, setup_logging)


@pytest.mark.cosyvoice3
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_tts_cosyvoice3_demo(setup_logging) -> None:
    """test_tts_cosyvoice3_demo"""
    model_name = "cosyvoice3"
    _demo_func(model_name, setup_logging)


@pytest.mark.yolo26m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_detection_yolo26m_demo(setup_logging) -> None:
    """test_detection_yolo26m_demo"""
    model_name = "yolo26m"
    _demo_func(model_name, setup_logging)


@pytest.mark.glm_ocr
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_ocr_glm_ocr_demo(setup_logging) -> None:
    """test_ocr_glm_ocr_demo"""
    model_name = "glm-ocr"
    _demo_func(model_name, setup_logging)


@pytest.mark.qwen3_reranker
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.demo
def test_reranker_qwen3_reranker_demo(setup_logging) -> None:
    """test_reranker_qwen3_reranker_demo"""
    model_name = "qwen3-reranker"
    _demo_func(model_name, setup_logging)


@pytest.mark.gemma4_vl
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
@pytest.mark.demo
def test_vlm_gemma4_vl_demo(setup_logging) -> None:
    """test_vlm_gemma4_vl_demo"""
    model_name = "gemma4-vl"
    _demo_func(model_name, setup_logging)
