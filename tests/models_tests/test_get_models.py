# Copyright 2025 HOUMO AI
#
# File: test_get_models.py
# Description:
#   Model downloading tests module.
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


def _get_model_func(model_name: str, setup_logging) -> None:
    """
    Execute model downloading test for a specific model.

    Args:
        model_name (str): Name of the model to download
        setup_logging: Fixture of setup_logging
    """
    logger.info("===> TEST START: test_%s_get_model", model_name)
    execute_get_model_flow(model_name, setup_logging)


@pytest.mark.wenet
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_asr_wenet_get_model")
def test_asr_wenet_get_model(setup_logging) -> None:
    model_name = "wenet"
    _get_model_func(model_name, setup_logging)


@pytest.mark.qwen3_asr
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_asr_qwen3_asr_get_model")
def test_asr_qwen3_asr_get_model(setup_logging) -> None:
    """test_asr_qwen3_asr_get_model"""
    model_name = "qwen3-asr"
    _get_model_func(model_name, setup_logging)


@pytest.mark.qwen3_forcealigner
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_asr_qwen3_forcealigner_get_model")
def test_asr_qwen3_forcealigner_get_model(setup_logging) -> None:
    """test_asr_qwen3_forcealigner_get_model"""
    model_name = "qwen3-forcealigner"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolop
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_autodrive_yolop_get_model")
def test_autodrive_yolop_get_model(setup_logging) -> None:
    model_name = "yolop"
    _get_model_func(model_name, setup_logging)


@pytest.mark.efficientnet
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_backbone_efficientnet_get_model")
def test_backbone_efficientnet_get_model(setup_logging) -> None:
    model_name = "efficientnet"
    _get_model_func(model_name, setup_logging)


@pytest.mark.mobilenetv2
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_backbone_mobilenetv2_get_model")
def test_backbone_mobilenetv2_get_model(setup_logging) -> None:
    model_name = "mobilenetv2"
    _get_model_func(model_name, setup_logging)


@pytest.mark.resnet50
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_backbone_resnet50_get_model")
def test_backbone_resnet50_get_model(setup_logging) -> None:
    model_name = "resnet50"
    _get_model_func(model_name, setup_logging)


@pytest.mark.vit
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_backbone_vit_get_model")
def test_backbone_vit_get_model(setup_logging) -> None:
    model_name = "vit"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolov3
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolov3_get_model")
def test_detection_yolov3_get_model(setup_logging) -> None:
    model_name = "yolov3"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolov5s
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolov5s_get_model")
def test_detection_yolov5s_get_model(setup_logging) -> None:
    model_name = "yolov5s"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolov5s_feature
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolov5s_feature_get_model")
def test_detection_yolov5s_feature_get_model(setup_logging) -> None:
    model_name = "yolov5s_feature"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolov8m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolov8m_get_model")
def test_detection_yolov8m_get_model(setup_logging) -> None:
    model_name = "yolov8m"
    _get_model_func(model_name, setup_logging)


@pytest.mark.sdxl
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_diffusion_sdxl_get_model")
def test_diffusion_sdxl_get_model(setup_logging) -> None:
    model_name = "sdxl"
    _get_model_func(model_name, setup_logging)


@pytest.mark.qwen2dot5
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_qwen2dot5_get_model")
def test_llm_qwen2dot5_get_model(setup_logging) -> None:
    model_name = "qwen2.5"
    _get_model_func(model_name, setup_logging)


@pytest.mark.qwen3
@pytest.mark.ndevice_2
@pytest.mark.dev_mem_24g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_qwen3_get_model")
def test_llm_qwen3_get_model(setup_logging) -> None:
    model_name = "qwen3"
    _get_model_func(model_name, setup_logging)


@pytest.mark.qwen3dot5
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_qwen3dot5_get_model")
def test_llm_qwen3dot5_get_model(setup_logging) -> None:
    """test_llm_qwen3dot5_get_model"""
    model_name = "qwen3.5"
    _get_model_func(model_name, setup_logging)


@pytest.mark.deepseek
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_deepseek_get_model")
def test_llm_deepseek_get_model(setup_logging) -> None:
    model_name = "deepseek"
    _get_model_func(model_name, setup_logging)


@pytest.mark.deepseek_r1_qwen3_8b
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_deepseek_r1_qwen3_8b_get_model")
def test_llm_deepseek_r1_qwen3_8b_get_model(setup_logging) -> None:
    model_name = "deepseek-r1-qwen3-8b"
    _get_model_func(model_name, setup_logging)


@pytest.mark.qwen2dot5_vl
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_vlm_qwen2dot5_vl_get_model")
def test_vlm_qwen2dot5_vl_get_model(setup_logging) -> None:
    model_name = "qwen2.5-vl"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolo12m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolo12m_get_model")
def test_detection_yolo12m_get_model(setup_logging) -> None:
    model_name = "yolo12m"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolov8m_pose
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_estimation_yolov8m_pose_get_model")
def test_estimation_yolov8m_pose_get_model(setup_logging) -> None:
    model_name = "yolov8m-pose"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolov8m_seg
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_segmentation_yolov8m_seg_get_model")
def test_segmentation_yolov8m_seg_get_model(setup_logging) -> None:
    model_name = "yolov8m-seg"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolov7
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolov7_get_model")
def test_detection_yolov7_get_model(setup_logging) -> None:
    model_name = "yolov7"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolov5m_face
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolov5m_face_get_model")
def test_detection_yolov5m_face_get_model(setup_logging) -> None:
    model_name = "yolov5m_face"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolox
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolox_get_model")
def test_detection_yolox_get_model(setup_logging) -> None:
    model_name = "yolox"
    _get_model_func(model_name, setup_logging)


@pytest.mark.lprnet
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_ocr_lprnet_get_model")
def test_ocr_lprnet_get_model(setup_logging) -> None:
    """test_ocr_lprnet_get_model"""
    model_name = "lprnet"
    _get_model_func(model_name, setup_logging)


@pytest.mark.ppocrv3_det
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_ocr_ppocrv3_det_get_model")
def test_ocr_ppocrv3_det_get_model(setup_logging) -> None:
    """test_ocr_ppocrv3_det_get_model"""
    model_name = "ppocrv3_det"
    _get_model_func(model_name, setup_logging)


@pytest.mark.ppocrv3_rec
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_ocr_ppocrv3_rec_get_model")
def test_ocr_ppocrv3_rec_get_model(setup_logging) -> None:
    """test_ocr_ppocrv3_rec_get_model"""
    model_name = "ppocrv3_rec"
    _get_model_func(model_name, setup_logging)


@pytest.mark.bge
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_embedding_bge_get_model")
def test_embedding_bge_get_model(setup_logging) -> None:
    """test_embedding_bge_get_model"""
    model_name = "bge"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolov10m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolov10m_get_model")
def test_detection_yolov10m_get_model(setup_logging) -> None:
    """test_detection_yolov10m_get_model"""
    model_name = "yolov10m"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolo11m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolo11m_get_model")
def test_detection_yolo11m_get_model(setup_logging) -> None:
    """test_detection_yolo11m_get_model"""
    model_name = "yolo11m"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolov9m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolov9m_get_model")
def test_detection_yolov9m_get_model(setup_logging) -> None:
    """test_detection_yolov9m_get_model"""
    model_name = "yolov9m"
    _get_model_func(model_name, setup_logging)


@pytest.mark.minicpmo
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_omni_minicpmo_get_model")
def test_omni_minicpmo_get_model(setup_logging) -> None:
    """test_omni_minicpmo_get_model"""
    model_name = "minicpmo"
    _get_model_func(model_name, setup_logging)


@pytest.mark.qwen3_30b_a3b
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_qwen3_30b_a3b_get_model")
def test_llm_qwen3_30b_a3b_get_model(setup_logging) -> None:
    """test_llm_qwen3_30b_a3b_get_model"""
    model_name = "qwen3-30b-a3b"
    _get_model_func(model_name, setup_logging)


@pytest.mark.copaw_flash
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_copaw_flash_get_model")
def test_llm_copaw_flash_get_model(setup_logging) -> None:
    """test_llm_copaw_flash_get_model"""
    model_name = "copaw-flash"
    _get_model_func(model_name, setup_logging)


@pytest.mark.qwen3_vl
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_vlm_qwen3_vl_get_model")
def test_vlm_qwen3_vl_get_model(setup_logging) -> None:
    """test_vlm_qwen3_vl_get_model"""
    model_name = "qwen3-vl"
    _get_model_func(model_name, setup_logging)


@pytest.mark.whisper
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_asr_whisper_get_model")
def test_asr_whisper_get_model(setup_logging) -> None:
    """test_asr_whisper_get_model"""
    model_name = "whisper"
    _get_model_func(model_name, setup_logging)


@pytest.mark.sensevoice
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_asr_sensevoice_get_model")
def test_asr_sensevoice_get_model(setup_logging) -> None:
    """test_asr_sensevoice_get_model"""
    model_name = "sensevoice"
    _get_model_func(model_name, setup_logging)


@pytest.mark.glm_asr
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_asr_glm_asr_get_model")
def test_asr_glm_asr_get_model(setup_logging) -> None:
    """test_asr_glm_asr_get_model"""
    model_name = "glm-asr"
    _get_model_func(model_name, setup_logging)


@pytest.mark.whisper_turbo
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_asr_whisper_turbo_get_model")
def test_asr_whisper_turbo_get_model(setup_logging) -> None:
    """test_asr_whisper_turbo_get_model"""
    model_name = "whisper-turbo"
    _get_model_func(model_name, setup_logging)


@pytest.mark.gte
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_embedding_gte_get_model")
def test_embedding_gte_get_model(setup_logging) -> None:
    """test_embedding_gte_get_model"""
    model_name = "gte"
    _get_model_func(model_name, setup_logging)


@pytest.mark.qwen3_embedding
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_embedding_qwen3_embedding_get_model")
def test_embedding_qwen3_embedding_get_model(setup_logging) -> None:
    """test_embedding_qwen3_embedding_get_model"""
    model_name = "qwen3-embedding"
    _get_model_func(model_name, setup_logging)


@pytest.mark.gpt_oss
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_gpt_oss_get_model")
def test_llm_gpt_oss_get_model(setup_logging) -> None:
    """test_llm_gpt_oss_get_model"""
    model_name = "gpt-oss"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolov8m_cls
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_backbone_yolov8m_cls_get_model")
def test_backbone_yolov8m_cls_get_model(setup_logging) -> None:
    """test_backbone_yolov8m_cls_get_model"""
    model_name = "yolov8m-cls"
    _get_model_func(model_name, setup_logging)


@pytest.mark.cosyvoice3
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_tts_cosyvoice3_get_model")
def test_tts_cosyvoice3_get_model(setup_logging) -> None:
    """test_tts_cosyvoice3_get_model"""
    model_name = "cosyvoice3"
    _get_model_func(model_name, setup_logging)


@pytest.mark.yolo26m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolo26m_get_model")
def test_detection_yolo26m_get_model(setup_logging) -> None:
    """test_detection_yolo26m_get_model"""
    model_name = "yolo26m"
    _get_model_func(model_name, setup_logging)


@pytest.mark.glm_ocr
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_ocr_glm_ocr_get_model")
def test_ocr_glm_ocr_get_model(setup_logging) -> None:
    """test_ocr_glm_ocr_get_model"""
    model_name = "glm-ocr"
    _get_model_func(model_name, setup_logging)


@pytest.mark.qwen3_reranker
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_reranker_qwen3_reranker_get_model")
def test_reranker_qwen3_reranker_get_model(setup_logging) -> None:
    """test_reranker_qwen3_reranker_get_model"""
    model_name = "qwen3-reranker"
    _get_model_func(model_name, setup_logging)


@pytest.mark.gemma4
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
@pytest.mark.get_model
@pytest.mark.dependency(name="test_vlm_gemma4_get_model")
def test_vlm_gemma4_get_model(setup_logging) -> None:
    """test_vlm_gemma4_get_model"""
    model_name = "gemma4"
    _get_model_func(model_name, setup_logging)
