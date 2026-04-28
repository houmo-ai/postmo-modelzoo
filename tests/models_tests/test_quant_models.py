# Copyright 2025 HOUMO AI
#
# File: test_quant_models.py
# Description:
#   Model quantization tests module.
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


def _quant_func(model_name: str, setup_logging) -> None:
    """
    Execute model quantization test for a specific model.

    Args:
        model_name (str): Name of the model to quantize
        setup_logging: Fixture of setup_logging
    """
    logger.info("===> TEST START: test_%s_quant", model_name)
    execute_quant_flow(model_name, setup_logging)


@pytest.mark.yolop
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_autodrive_yolop_quant",
    depends_on=["test_get_models.py::test_autodrive_yolop_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_autodrive_yolop_quant(setup_logging) -> None:
    """test_autodrive_yolop_quant"""
    model_name = "yolop"
    _quant_func(model_name, setup_logging)


@pytest.mark.qwen3_asr
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_asr_qwen3_asr_quant",
    depends_on=["test_get_models.py::test_asr_qwen3_asr_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_asr_qwen3_asr_quant(setup_logging) -> None:
    """test_asr_qwen3_asr_quant"""
    model_name = "qwen3-asr"
    _quant_func(model_name, setup_logging)


@pytest.mark.qwen3_forcealigner
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_asr_qwen3_forcealigner_quant",
    depends_on=["test_get_models.py::test_asr_qwen3_forcealigner_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_asr_qwen3_forcealigner_quant(setup_logging) -> None:
    """test_asr_qwen3_forcealigner_quant"""
    model_name = "qwen3-forcealigner"
    _quant_func(model_name, setup_logging)


@pytest.mark.efficientnet
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_backbone_efficientnet_quant",
    depends_on=["test_get_models.py::test_backbone_efficientnet_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_backbone_efficientnet_quant(setup_logging) -> None:
    """test_backbone_efficientnet_quant"""
    model_name = "efficientnet"
    _quant_func(model_name, setup_logging)


@pytest.mark.mobilenetv2
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_backbone_mobilenetv2_quant",
    depends_on=["test_get_models.py::test_backbone_mobilenetv2_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_backbone_mobilenetv2_quant(setup_logging) -> None:
    """test_backbone_efficientnet_quant"""
    model_name = "mobilenetv2"
    _quant_func(model_name, setup_logging)


@pytest.mark.resnet50
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_backbone_resnet50_quant",
    depends_on=["test_get_models.py::test_backbone_resnet50_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_backbone_resnet50_quant(setup_logging) -> None:
    """test_backbone_resnet50_quant"""
    model_name = "resnet50"
    _quant_func(model_name, setup_logging)


@pytest.mark.vit
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_backbone_vit_quant",
    depends_on=["test_get_models.py::test_backbone_vit_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_backbone_vit_quant(setup_logging) -> None:
    """test_backbone_vit_quant"""
    model_name = "vit"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolov3
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_detection_yolov3_quant",
    depends_on=["test_get_models.py::test_detection_yolov3_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov3_quant(setup_logging) -> None:
    """test_detection_yolov3_quant"""
    model_name = "yolov3"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolov5s
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_detection_yolov5s_quant",
    depends_on=["test_get_models.py::test_detection_yolov5s_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov5s_quant(setup_logging) -> None:
    """test_detection_yolov5s_quant"""
    model_name = "yolov5s"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolov5s_feature
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_detection_yolov5s_feature_quant",
    depends_on=["test_get_models.py::test_detection_yolov5s_feature_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov5s_feature_quant(setup_logging) -> None:
    """test_detection_yolov5s_feature_quant"""
    model_name = "yolov5s_feature"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolov8m
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_detection_yolov8m_quant",
    depends_on=["test_get_models.py::test_detection_yolov8m_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov8m_quant(setup_logging) -> None:
    """test_detection_yolov8m_quant"""
    model_name = "yolov8m"
    _quant_func(model_name, setup_logging)


@pytest.mark.qwen2dot5
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_llm_qwen2dot5_quant",
    depends_on=["test_get_models.py::test_llm_qwen2dot5_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_llm_qwen2dot5_quant(setup_logging) -> None:
    """test_llm_qwen2dot5_quant"""
    model_name = "qwen2.5"
    _quant_func(model_name, setup_logging)


@pytest.mark.qwen3
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_llm_qwen3_quant",
    depends_on=["test_get_models.py::test_llm_qwen3_get_model"],
)
@pytest.mark.ndevice_2
@pytest.mark.dev_mem_24g
def test_llm_qwen3_quant(setup_logging) -> None:
    """test_llm_qwen3_quant"""
    model_name = "qwen3"
    _quant_func(model_name, setup_logging)


@pytest.mark.qwen3_30b_a3b
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_llm_qwen3_30b_a3b_quant",
    depends_on=["test_get_models.py::test_llm_qwen3_30b_a3b_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
def test_llm_qwen3_30b_a3b_quant(setup_logging) -> None:
    """test_llm_qwen3_30b_a3b_quant"""
    model_name = "qwen3-30b-a3b"
    _quant_func(model_name, setup_logging)


@pytest.mark.qwen3dot5
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_llm_qwen3dot5_quant",
    depends_on=["test_get_models.py::test_llm_qwen3dot5_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
def test_llm_qwen3dot5_quant(setup_logging) -> None:
    """test_llm_qwen3dot5_quant"""
    model_name = "qwen3.5"
    _quant_func(model_name, setup_logging)


@pytest.mark.deepseek
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_llm_deepseek_quant",
    depends_on=["test_get_models.py::test_llm_deepseek_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_llm_deepseek_quant(setup_logging) -> None:
    """test_llm_deepseek_quant"""
    model_name = "deepseek"
    _quant_func(model_name, setup_logging)


@pytest.mark.deepseek_r1_qwen3_8b
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_llm_deepseek_r1_qwen3_8b_quant",
    depends_on=["test_get_models.py::test_llm_deepseek_r1_qwen3_8b_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_llm_deepseek_r1_qwen3_8b_quant(setup_logging) -> None:
    """test_llm_deepseek_r1_qwen3_8b_quant"""
    model_name = "deepseek-r1-qwen3-8b"
    _quant_func(model_name, setup_logging)


@pytest.mark.qwen2dot5_vl
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_vlm_qwen2dot5_vl_quant",
    depends_on=["test_get_models.py::test_vlm_qwen2dot5_vl_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_vlm_qwen2dot5_vl_quant(setup_logging) -> None:
    """test_vlm_qwen2dot5_vl_quant"""
    model_name = "qwen2.5-vl"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolo12m
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_detection_yolo12m_quant",
    depends_on=["test_get_models.py::test_detection_yolo12m_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolo12m_quant(setup_logging) -> None:
    """test_detection_yolo12m_quant"""
    model_name = "yolo12m"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolov8m_pose
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_estimation_yolov8m_pose_quant",
    depends_on=["test_get_models.py::test_estimation_yolov8m_pose_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_estimation_yolov8m_pose_quant(setup_logging) -> None:
    """test_estimation_yolov8m_pose_quant"""
    model_name = "yolov8m-pose"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolov8m_seg
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_segmentation_yolov8m_seg_quant",
    depends_on=["test_get_models.py::test_segmentation_yolov8m_seg_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_segmentation_yolov8m_seg_quant(setup_logging) -> None:
    """test_segmentation_yolov8m_seg_quant"""
    model_name = "yolov8m-seg"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolov7
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_detection_yolov7_quant",
    depends_on=["test_get_models.py::test_detection_yolov7_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov7_quant(setup_logging) -> None:
    """test_detection_yolov7_quant"""
    model_name = "yolov7"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolov5m_face
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_detection_yolov5m_face_quant",
    depends_on=["test_get_models.py::test_detection_yolov5m_face_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov5m_face_quant(setup_logging) -> None:
    """test_detection_yolov5m_face_quant"""
    model_name = "yolov5m_face"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolox
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_detection_yolox_quant",
    depends_on=["test_get_models.py::test_detection_yolox_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolox_quant(setup_logging) -> None:
    """test_detection_yolox_quant"""
    model_name = "yolox"
    _quant_func(model_name, setup_logging)


@pytest.mark.lprnet
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_ocr_lprnet_quant",
    depends_on=["test_get_models.py::test_ocr_lprnet_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_ocr_lprnet_quant(setup_logging) -> None:
    """test_ocr_lprnet_quant"""
    model_name = "lprnet"
    _quant_func(model_name, setup_logging)


@pytest.mark.ppocrv3_det
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_ocr_ppocrv3_det_quant",
    depends_on=["test_get_models.py::test_ocr_ppocrv3_det_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_ocr_ppocrv3_det_quant(setup_logging) -> None:
    """test_ocr_ppocrv3_det_quant"""
    model_name = "ppocrv3_det"
    _quant_func(model_name, setup_logging)


@pytest.mark.ppocrv3_rec
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_ocr_ppocrv3_rec_quant",
    depends_on=["test_get_models.py::test_ocr_ppocrv3_rec_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_ocr_ppocrv3_rec_quant(setup_logging) -> None:
    """test_ocr_ppocrv3_rec_quant"""
    model_name = "ppocrv3_rec"
    _quant_func(model_name, setup_logging)


@pytest.mark.bge
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_embedding_bge_quant",
    depends_on=["test_get_models.py::test_embedding_bge_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_embedding_bge_quant(setup_logging) -> None:
    """test_embedding_bge_quant"""
    model_name = "bge"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolov10m
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_detection_yolov10m_quant",
    depends_on=["test_get_models.py::test_detection_yolov10m_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov10m_quant(setup_logging) -> None:
    """test_detection_yolov10m_quant"""
    model_name = "yolov10m"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolo11m
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_detection_yolo11m_quant",
    depends_on=["test_get_models.py::test_detection_yolo11m_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolo11m_quant(setup_logging) -> None:
    """test_detection_yolo11m_quant"""
    model_name = "yolo11m"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolov9m
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_detection_yolov9m_quant",
    depends_on=["test_get_models.py::test_detection_yolov9m_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov9m_quant(setup_logging) -> None:
    """test_detection_yolov9m_quant"""
    model_name = "yolov9m"
    _quant_func(model_name, setup_logging)


@pytest.mark.minicpmo
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_omni_minicpmo_quant",
    depends_on=["test_get_models.py::test_omni_minicpmo_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_omni_minicpmo_quant(setup_logging) -> None:
    """test_omni_minicpmo_quant"""
    model_name = "minicpmo"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolov8m_cls
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_backbone_yolov8m_cls_quant",
    depends_on=["test_get_models.py::test_backbone_yolov8m_cls_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_backbone_yolov8m_cls_quant(setup_logging) -> None:
    """test_backbone_yolov8m_cls_quant"""
    model_name = "yolov8m-cls"
    _quant_func(model_name, setup_logging)


@pytest.mark.gte
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_embedding_gte_quant",
    depends_on=["test_get_models.py::test_embedding_gte_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_embedding_gte_quant(setup_logging) -> None:
    """test_embedding_gte_quant"""
    model_name = "gte"
    _quant_func(model_name, setup_logging)


@pytest.mark.qwen3_embedding
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_embedding_qwen3_embedding_quant",
    depends_on=["test_get_models.py::test_embedding_qwen3_embedding_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_embedding_qwen3_embedding_quant(setup_logging) -> None:
    """test_embedding_qwen3_embedding_quant"""
    model_name = "qwen3-embedding"
    _quant_func(model_name, setup_logging)


@pytest.mark.cosyvoice3
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_tts_cosyvoice3_quant",
    depends_on=["test_get_models.py::test_tts_cosyvoice3_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_tts_cosyvoice3_quant(setup_logging) -> None:
    """test_tts_cosyvoice3_quant"""
    model_name = "cosyvoice3"
    _quant_func(model_name, setup_logging)


@pytest.mark.sensevoice
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_asr_sensevoice_quant",
    depends_on=["test_get_models.py::test_asr_sensevoice_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_asr_sensevoice_quant(setup_logging) -> None:
    """test_asr_sensevoice_quant"""
    model_name = "sensevoice"
    _quant_func(model_name, setup_logging)


@pytest.mark.glm_asr
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_asr_glm_asr_quant",
    depends_on=["test_get_models.py::test_asr_glm_asr_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_asr_glm_asr_quant(setup_logging) -> None:
    """test_asr_glm_asr_quant"""
    model_name = "glm-asr"
    _quant_func(model_name, setup_logging)


@pytest.mark.whisper_turbo
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_asr_whisper_turbo_quant",
    depends_on=["test_get_models.py::test_asr_whisper_turbo_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_asr_whisper_turbo_quant(setup_logging) -> None:
    """test_asr_whisper_turbo_quant"""
    model_name = "whisper-turbo"
    _quant_func(model_name, setup_logging)


@pytest.mark.yolo26m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_detection_yolo26m_quant",
    depends_on=["test_get_models.py::test_detection_yolo26m_get_model"],
)
def test_detection_yolo26m_quant(setup_logging) -> None:
    """test_detection_yolo26m_quant"""
    model_name = "yolo26m"
    _quant_func(model_name, setup_logging)


@pytest.mark.glm_ocr
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_ocr_glm_ocr_quant",
    depends_on=["test_get_models.py::test_ocr_glm_ocr_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_ocr_glm_ocr_quant(setup_logging) -> None:
    """test_ocr_glm_ocr_quant"""
    model_name = "glm-ocr"
    _quant_func(model_name, setup_logging)


@pytest.mark.qwen3_reranker
@pytest.mark.quant
@pytest.mark.dependency(
    name="test_reranker_qwen3_reranker_quant",
    depends_on=["test_get_models.py::test_reranker_qwen3_reranker_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_reranker_qwen3_reranker_quant(setup_logging) -> None:
    """test_reranker_qwen3_reranker_quant"""
    model_name = "qwen3-reranker"
    _quant_func(model_name, setup_logging)
