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
