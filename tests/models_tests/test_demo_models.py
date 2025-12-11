import pytest
import logging
from .test_models_utils import *

logger = logging.getLogger(__name__)


def _demo_func(model_name, log_file):
    logger.info("===> TEST START: test_%s_demo", model_name)
    execute_demo_flow(model_name, log_file)


@pytest.mark.wenet
@pytest.mark.demo
def test_asr_wenet_demo(setup_logging):
    model_name = "wenet"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.yolop
@pytest.mark.demo
def test_autodrive_yolop_demo(setup_logging):
    model_name = "yolop"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.efficientnet
@pytest.mark.demo
def test_backbone_efficientnet_demo(setup_logging):
    model_name = "efficientnet"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.mobilenetv2
@pytest.mark.demo
def test_backbone_mobilenetv2_demo(setup_logging):
    model_name = "mobilenetv2"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.resnet50
@pytest.mark.demo
def test_backbone_resnet50_demo(setup_logging):
    model_name = "resnet50"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.vit
@pytest.mark.demo
def test_backbone_vit_demo(setup_logging):
    model_name = "vit"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov3
@pytest.mark.demo
def test_detection_yolov3_demo(setup_logging):
    model_name = "yolov3"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5s
@pytest.mark.demo
def test_detection_yolov5s_demo(setup_logging):
    model_name = "yolov5s"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5s_feature
@pytest.mark.demo
def test_detection_yolov5s_feature_demo(setup_logging):
    model_name = "yolov5s_feature"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m
@pytest.mark.demo
def test_detection_yolov8m_demo(setup_logging):
    model_name = "yolov8m"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.sdxl
@pytest.mark.demo
def test_diffusion_sdxl_demo(setup_logging):
    model_name = "sdxl"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen2dot5
@pytest.mark.demo
def test_llm_qwen2dot5_demo(setup_logging):
    model_name = "qwen2.5"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen3
@pytest.mark.demo
def test_llm_qwen3_demo(setup_logging):
    model_name = "qwen3"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen3_14b
@pytest.mark.demo
def test_llm_qwen3_14b_demo(setup_logging):
    model_name = "qwen3-14b"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.deepseek
@pytest.mark.demo
def test_llm_deepseek_demo(setup_logging):
    model_name = "deepseek"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.deepseek_r1_qwen3_8b
@pytest.mark.demo
def test_llm_deepseek_r1_qwen3_8b_demo(setup_logging):
    model_name = "deepseek-r1-qwen3-8b"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen2dot5_vl
@pytest.mark.demo
def test_vllm_qwen2dot5_vl_demo(setup_logging):
    model_name = "qwen2.5-vl"
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.yolo12m
@pytest.mark.demo
def test_detection_yolo12m_demo(setup_logging):
    model_name = 'yolo12m'
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m_pose
@pytest.mark.demo
def test_estimation_yolov8m_pose_demo(setup_logging):
    model_name = 'yolov8m-pose'
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m_seg
@pytest.mark.demo
def test_segmentation_yolov8m_seg_demo(setup_logging):
    model_name = 'yolov8m-seg'
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov7
@pytest.mark.demo
def test_detection_yolov7_demo(setup_logging):
    model_name = 'yolov7'
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5m_face
@pytest.mark.demo
def test_detection_yolov5m_face_demo(setup_logging):
    model_name = 'yolov5m_face'
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.yolox
@pytest.mark.demo
def test_detection_yolox_demo(setup_logging):
    model_name = 'yolox'
    _demo_func(model_name, setup_logging)
    assert True


@pytest.mark.lprnet
@pytest.mark.demo
def test_ocr_lprnet_demo(setup_logging: type(print)) -> None:
    """test_ocr_lprnet_demo"""
    model_name = 'lprnet'
    _demo_func(model_name, setup_logging)


@pytest.mark.ppocrv3_det
@pytest.mark.demo
def test_ocr_ppocrv3_det_demo(setup_logging: type(print)) -> None:
    """test_ocr_ppocrv3_det_demo"""
    model_name = 'ppocrv3_det'
    _demo_func(model_name, setup_logging)


@pytest.mark.bge
@pytest.mark.demo
def test_embedding_bge_demo(setup_logging: type(print)) -> None:
    """test_embedding_bge_demo"""
    model_name = 'bge'
    _demo_func(model_name, setup_logging)
