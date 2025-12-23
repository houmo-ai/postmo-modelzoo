import pytest
import logging
from .test_models_utils import *

logger = logging.getLogger(__name__)


def _perf_func(model_name, log_file):
    logger.info("===> TEST START: test_%s_perf", model_name)
    execute_perf_flow(model_name, log_file)


@pytest.mark.yolop
@pytest.mark.perf
def test_autodrive_yolop_perf(setup_logging):
    model_name = "yolop"
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.efficientnet
@pytest.mark.perf
def test_backbone_efficientnet_perf(setup_logging):
    model_name = "efficientnet"
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.mobilenetv2
@pytest.mark.perf
def test_backbone_mobilenetv2_perf(setup_logging):
    model_name = "mobilenetv2"
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.resnet50
@pytest.mark.perf
def test_backbone_resnet50_perf(setup_logging):
    model_name = "resnet50"
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.vit
@pytest.mark.perf
def test_backbone_vit_perf(setup_logging):
    model_name = "vit"
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov3
@pytest.mark.perf
def test_detection_yolov3_perf(setup_logging):
    model_name = "yolov3"
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5s
@pytest.mark.perf
def test_detection_yolov5s_perf(setup_logging):
    model_name = "yolov5s"
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5s_feature
@pytest.mark.perf
def test_detection_yolov5s_feature_perf(setup_logging):
    model_name = "yolov5s_feature"
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m
@pytest.mark.perf
def test_detection_yolov8m_perf(setup_logging):
    model_name = "yolov8m"
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.yolo12m
@pytest.mark.perf
def test_detection_yolo12m_perf(setup_logging):
    model_name = 'yolo12m'
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.deepseek
@pytest.mark.perf
def test_llm_deepseek_perf(setup_logging):
    model_name = 'deepseek'
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.deepseek_r1_qwen3_8b
@pytest.mark.perf
def test_llm_deepseek_r1_qwen3_8b_perf(setup_logging):
    model_name = "deepseek-r1-qwen3-8b"
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen3
@pytest.mark.perf
def test_llm_qwen3_perf(setup_logging):
    model_name = 'qwen3'
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen3_14b
@pytest.mark.perf
def test_llm_qwen3_14b_perf(setup_logging):
    model_name = "qwen3-14b"
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen2dot5
@pytest.mark.perf
def test_llm_qwen2dot5_perf(setup_logging):
    model_name = 'qwen2.5'
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen2dot5_vl
@pytest.mark.perf
def test_vllm_qwen2dot5_vl_perf(setup_logging):
    model_name = 'qwen2.5-vl'
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.wenet
@pytest.mark.perf
def test_asr_wenet_perf(setup_logging):
    model_name = 'wenet'
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.sdxl
@pytest.mark.perf
def test_diffusion_sdxl_perf(setup_logging):
    model_name = 'sdxl'
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m_pose
@pytest.mark.perf
def test_estimation_yolov8m_pose_perf(setup_logging):
    model_name = 'yolov8m-pose'
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m_seg
@pytest.mark.perf
def test_segmentation_yolov8m_seg_perf(setup_logging):
    model_name = 'yolov8m-seg'
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov7
@pytest.mark.perf
def test_detection_yolov7_perf(setup_logging):
    model_name = 'yolov7'
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5m_face
@pytest.mark.perf
def test_detection_yolov5m_face_perf(setup_logging):
    model_name = 'yolov5m_face'
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.yolox
@pytest.mark.perf
def test_detection_yolox_perf(setup_logging):
    model_name = 'yolox'
    _perf_func(model_name, setup_logging)
    assert True


@pytest.mark.lprnet
@pytest.mark.perf
def test_ocr_lprnet_perf(setup_logging: type(print)) -> None:
    """test_ocr_lprnet_perf"""
    model_name = 'lprnet'
    _perf_func(model_name, setup_logging)


@pytest.mark.ppocrv3_det
@pytest.mark.perf
def test_ocr_ppocrv3_det_perf(setup_logging: type(print)) -> None:
    """test_ocr_ppocrv3_det_perf"""
    model_name = 'ppocrv3_det'
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov8m_cls
@pytest.mark.perf
def test_segmentation_yolov8m_cls_perf(setup_logging: type(print)) -> None:
    """test_segmentation_yolov8m_cls_perf"""
    model_name = 'yolov8m-cls'
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov10m
@pytest.mark.perf
def test_detection_yolov10m_perf(setup_logging: type(print)) -> None:
    """test_detection_yolov10m_perf"""
    model_name = 'yolov10m'
    _perf_func(model_name, setup_logging)


@pytest.mark.yolo11m
@pytest.mark.perf
def test_detection_yolo11m_perf(setup_logging: type(print)) -> None:
    """test_detection_yolo11m_perf"""
    model_name = 'yolo11m'
    _perf_func(model_name, setup_logging)


@pytest.mark.yolov9m
@pytest.mark.perf
def test_detection_yolov9m_perf(setup_logging: type(print)) -> None:
    """test_detection_yolov9m_perf"""
    model_name = 'yolov9m'
    _perf_func(model_name, setup_logging)


@pytest.mark.qwen3_30b_a3b
@pytest.mark.perf
def test_llm_qwen3_30b_a3b_perf(setup_logging: type(print)) -> None:
    """test_llm_qwen3_30b_a3b_perf"""
    model_name = 'qwen3-30b-a3b'
    _perf_func(model_name, setup_logging)


@pytest.mark.qwen3_vl
@pytest.mark.perf
def test_vllm_qwen3_vl_perf(setup_logging: type(print)) -> None:
    """test_vllm_qwen3_vl_perf"""
    model_name = 'qwen3-vl'
    _perf_func(model_name, setup_logging)


@pytest.mark.whisper
@pytest.mark.perf
def test_asr_whisper_perf(setup_logging: type(print)) -> None:
    """test_asr_whisper_perf"""
    model_name = 'whisper'
    _perf_func(model_name, setup_logging)


@pytest.mark.gte
@pytest.mark.perf
def test_embedding_gte_perf(setup_logging: type(print)) -> None:
    """test_embedding_gte_perf"""
    model_name = 'gte'
    _perf_func(model_name, setup_logging)


@pytest.mark.gpt_oss
@pytest.mark.perf
def test_llm_gpt_oss_perf(setup_logging: type(print)) -> None:
    """test_llm_gpt_oss_perf"""
    model_name = 'gpt-oss'
    _perf_func(model_name, setup_logging)
