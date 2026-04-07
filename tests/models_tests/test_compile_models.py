# Copyright 2025 HOUMO AI
#
# File: test_compile_models.py
# Description:
#   Model compilation tests module.
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


def _compile_func(model_name: str, setup_logging) -> None:
    """
    Execute model compilation test for a specific model.

    Args:
        model_name (str): Name of the model to compile
        setup_logging: Fixture of setup_logging
    """
    logger.info("===> TEST START: test_%s_compile", model_name)
    execute_compile_flow(model_name, setup_logging)


@pytest.mark.wenet
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_asr_wenet_compile",
    depends_on=["test_get_models.py::test_asr_wenet_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_asr_wenet_compile(setup_logging) -> None:
    """test_asr_wenet_compile"""
    model_name = "wenet"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolop
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_autodrive_yolop_compile",
    depends_on=["test_quant_models.py::test_autodrive_yolop_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_autodrive_yolop_compile(setup_logging) -> None:
    """test_autodrive_yolop_compile"""
    model_name = "yolop"
    _compile_func(model_name, setup_logging)


@pytest.mark.efficientnet
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_backbone_efficientnet_compile",
    depends_on=["test_quant_models.py::test_backbone_efficientnet_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_backbone_efficientnet_compile(setup_logging) -> None:
    """test_backbone_efficientnet_compile"""
    model_name = "efficientnet"
    _compile_func(model_name, setup_logging)


@pytest.mark.mobilenetv2
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_backbone_mobilenetv2_compile",
    depends_on=["test_quant_models.py::test_backbone_mobilenetv2_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_backbone_mobilenetv2_compile(setup_logging) -> None:
    """test_backbone_mobilenetv2_compile"""
    model_name = "mobilenetv2"
    _compile_func(model_name, setup_logging)


@pytest.mark.resnet50
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_backbone_resnet50_compile",
    depends_on=["test_quant_models.py::test_backbone_resnet50_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_backbone_resnet50_compile(setup_logging) -> None:
    """test_backbone_resnet50_compile"""
    model_name = "resnet50"
    _compile_func(model_name, setup_logging)


@pytest.mark.vit
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_backbone_vit_compile",
    depends_on=["test_quant_models.py::test_backbone_vit_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_backbone_vit_compile(setup_logging) -> None:
    """test_backbone_vit_compile"""
    model_name = "vit"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolov3
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolov3_compile",
    depends_on=["test_quant_models.py::test_detection_yolov3_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov3_compile(setup_logging) -> None:
    """test_detection_yolov3_compile"""
    model_name = "yolov3"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolov5s
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolov5s_compile",
    depends_on=["test_quant_models.py::test_detection_yolov5s_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov5s_compile(setup_logging) -> None:
    """test_detection_yolov5s_compile"""
    model_name = "yolov5s"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolov5s_feature
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolov5s_feature_compile",
    depends_on=["test_quant_models.py::test_detection_yolov5s_feature_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov5s_feature_compile(setup_logging) -> None:
    """test_detection_yolov5s_feature_compile"""
    model_name = "yolov5s_feature"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolov8m
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolov8m_compile",
    depends_on=["test_quant_models.py::test_detection_yolov8m_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov8m_compile(setup_logging) -> None:
    """test_detection_yolov8m_compile"""
    model_name = "yolov8m"
    _compile_func(model_name, setup_logging)


@pytest.mark.sdxl
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_diffusion_sdxl_compile",
    depends_on=["test_get_models.py::test_diffusion_sdxl_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_diffusion_sdxl_compile(setup_logging) -> None:
    """test_diffusion_sdxl_compile"""
    model_name = "sdxl"
    _compile_func(model_name, setup_logging)


@pytest.mark.qwen2dot5
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_llm_qwen2dot5_compile",
    depends_on=["test_quant_models.py::test_llm_qwen2dot5_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_llm_qwen2dot5_compile(setup_logging) -> None:
    """test_llm_qwen2dot5_compile"""
    model_name = "qwen2.5"
    _compile_func(model_name, setup_logging)


@pytest.mark.qwen3
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_llm_qwen3_compile",
    depends_on=["test_quant_models.py::test_llm_qwen3_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_llm_qwen3_compile(setup_logging) -> None:
    """test_llm_qwen3_compile"""
    model_name = "qwen3"
    _compile_func(model_name, setup_logging)


@pytest.mark.qwen3_14b
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_llm_qwen3_14b_compile",
    depends_on=["test_quant_models.py::test_llm_qwen3_14b_quant"],
)
@pytest.mark.ndevice_2
@pytest.mark.dev_mem_24g
def test_llm_qwen3_14b_compile(setup_logging) -> None:
    """test_llm_qwen3_14b_compile"""
    model_name = "qwen3-14b"
    _compile_func(model_name, setup_logging)


@pytest.mark.deepseek
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_llm_deepseek_compile",
    depends_on=["test_quant_models.py::test_llm_deepseek_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_llm_deepseek_compile(setup_logging) -> None:
    """test_llm_deepseek_compile"""
    model_name = "deepseek"
    _compile_func(model_name, setup_logging)


@pytest.mark.deepseek_r1_qwen3_8b
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_llm_deepseek_r1_qwen3_8b_compile",
    depends_on=["test_quant_models.py::test_llm_deepseek_r1_qwen3_8b_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_llm_deepseek_r1_qwen3_8b_compile(setup_logging) -> None:
    """test_llm_deepseek_r1_qwen3_8b_compile"""
    model_name = "deepseek-r1-qwen3-8b"
    _compile_func(model_name, setup_logging)


@pytest.mark.qwen2dot5_vl
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_vlm_qwen2dot5_vl_compile",
    depends_on=["test_quant_models.py::test_vlm_qwen2dot5_vl_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_vlm_qwen2dot5_vl_compile(setup_logging) -> None:
    """test_vlm_qwen2dot5_vl_compile"""
    model_name = "qwen2.5-vl"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolo12m
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolo12m_compile",
    depends_on=["test_quant_models.py::test_detection_yolo12m_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolo12m_compile(setup_logging) -> None:
    """test_detection_yolo12m_compile"""
    model_name = "yolo12m"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolov8m_pose
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_estimation_yolov8m_pose_compile",
    depends_on=["test_quant_models.py::test_estimation_yolov8m_pose_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_estimation_yolov8m_pose_compile(setup_logging) -> None:
    """test_estimation_yolov8m_pose_compile"""
    model_name = "yolov8m-pose"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolov8m_seg
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_segmentation_yolov8m_seg_compile",
    depends_on=["test_quant_models.py::test_segmentation_yolov8m_seg_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_segmentation_yolov8m_seg_compile(setup_logging) -> None:
    """test_segmentation_yolov8m_seg_compile"""
    model_name = "yolov8m-seg"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolov7
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolov7_compile",
    depends_on=["test_quant_models.py::test_detection_yolov7_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov7_compile(setup_logging) -> None:
    """test_detection_yolov7_compile"""
    model_name = "yolov7"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolov5m_face
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolov5m_face_compile",
    depends_on=["test_quant_models.py::test_detection_yolov5m_face_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov5m_face_compile(setup_logging) -> None:
    """test_detection_yolov5m_face_compile"""
    model_name = "yolov5m_face"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolox
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolox_compile",
    depends_on=["test_quant_models.py::test_detection_yolox_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolox_compile(setup_logging) -> None:
    """test_detection_yolox_compile"""
    model_name = "yolox"
    _compile_func(model_name, setup_logging)


@pytest.mark.lprnet
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_ocr_lprnet_compile",
    depends_on=["test_quant_models.py::test_ocr_lprnet_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_ocr_lprnet_compile(setup_logging) -> None:
    """test_ocr_lprnet_compile"""
    model_name = "lprnet"
    _compile_func(model_name, setup_logging)


@pytest.mark.ppocrv3_det
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_ocr_ppocrv3_det_compile",
    depends_on=["test_quant_models.py::test_ocr_ppocrv3_det_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_ocr_ppocrv3_det_compile(setup_logging) -> None:
    """test_ocr_ppocrv3_det_compile"""
    model_name = "ppocrv3_det"
    _compile_func(model_name, setup_logging)


@pytest.mark.bge
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_embedding_bge_compile",
    depends_on=["test_quant_models.py::test_embedding_bge_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_embedding_bge_compile(setup_logging) -> None:
    """test_embedding_bge_compile"""
    model_name = "bge"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolov10m
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolov10m_compile",
    depends_on=["test_quant_models.py::test_detection_yolov10m_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov10m_compile(setup_logging) -> None:
    """test_detection_yolov10m_compile"""
    model_name = "yolov10m"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolo11m
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolo11m_compile",
    depends_on=["test_quant_models.py::test_detection_yolo11m_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolo11m_compile(setup_logging) -> None:
    """test_detection_yolo11m_compile"""
    model_name = "yolo11m"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolov9m
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolov9m_compile",
    depends_on=["test_quant_models.py::test_detection_yolov9m_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_detection_yolov9m_compile(setup_logging) -> None:
    """test_detection_yolov9m_compile"""
    model_name = "yolov9m"
    _compile_func(model_name, setup_logging)


@pytest.mark.minicpmo
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_omni_minicpmo_compile",
    depends_on=["test_quant_models.py::test_omni_minicpmo_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_omni_minicpmo_compile(setup_logging) -> None:
    """test_omni_minicpmo_compile"""
    model_name = "minicpmo"
    _compile_func(model_name, setup_logging)


@pytest.mark.qwen3_30b_a3b
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_llm_qwen3_30b_a3b_compile",
    depends_on=["test_get_models.py::test_llm_qwen3_30b_a3b_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
def test_llm_qwen3_30b_a3b_compile(setup_logging) -> None:
    """test_llm_qwen3_30b_a3b_compile"""
    model_name = "qwen3-30b-a3b"
    _compile_func(model_name, setup_logging)


@pytest.mark.qwen3_vl
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_vlm_qwen3_vl_compile",
    depends_on=["test_get_models.py::test_vlm_qwen3_vl_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_vlm_qwen3_vl_compile(setup_logging) -> None:
    """test_vlm_qwen3_vl_compile"""
    model_name = "qwen3-vl"
    _compile_func(model_name, setup_logging)


@pytest.mark.whisper
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_asr_whisper_compile",
    depends_on=["test_get_models.py::test_asr_whisper_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_asr_whisper_compile(setup_logging) -> None:
    """test_asr_whisper_compile"""
    model_name = "whisper"
    _compile_func(model_name, setup_logging)

@pytest.mark.sensevoice
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_asr_sensevoice_compile",
    depends_on=["test_quant_models.py::test_asr_sensevoice_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_asr_sensevoice_compile(setup_logging) -> None:
    """test_asr_sensevoice_compile"""
    model_name = "sensevoice"
    _compile_func(model_name, setup_logging)

@pytest.mark.gte
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_embedding_gte_compile",
    depends_on=["test_get_models.py::test_embedding_gte_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_embedding_gte_compile(setup_logging) -> None:
    """test_embedding_gte_compile"""
    model_name = "gte"
    _compile_func(model_name, setup_logging)


@pytest.mark.gpt_oss
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_llm_gpt_oss_compile",
    depends_on=["test_get_models.py::test_llm_gpt_oss_get_model"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_24g
def test_llm_gpt_oss_compile(setup_logging) -> None:
    """test_llm_gpt_oss_compile"""
    model_name = "gpt-oss"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolov8m_cls
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_backbone_yolov8m_cls_compile",
    depends_on=["test_quant_models.py::test_backbone_yolov8m_cls_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_backbone_yolov8m_cls_compile(setup_logging) -> None:
    """test_backbone_yolov8m_cls_compile"""
    model_name = "yolov8m-cls"
    _compile_func(model_name, setup_logging)


@pytest.mark.cosyvoice3
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_tts_cosyvoice3_compile",
    depends_on=["test_quant_models.py::test_tts_cosyvoice3_quant"],
)
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
def test_tts_cosyvoice3_compile(setup_logging) -> None:
    """test_tts_cosyvoice3_compile"""
    model_name = "cosyvoice3"
    _compile_func(model_name, setup_logging)


@pytest.mark.yolo26m
@pytest.mark.ndevice_1
@pytest.mark.dev_mem_12g
@pytest.mark.compile
@pytest.mark.dependency(name='test_detection_yolo26m_compile', depends_on=['test_quant_models.py::test_detection_yolo26m_quant'])
def test_detection_yolo26m_compile(setup_logging) -> None:
    """test_detection_yolo26m_compile"""
    model_name = 'yolo26m'
    _compile_func(model_name, setup_logging)
