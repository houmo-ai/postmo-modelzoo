import pytest
import logging
from .test_models_utils import *

logger = logging.getLogger(__name__)


def _compile_func(model_name, log_file):
    logger.info("===> TEST START: test_%s_compile", model_name)
    execute_compile_flow(model_name, log_file, False)


@pytest.mark.wenet
@pytest.mark.compile
def test_asr_wenet_compile(setup_logging):
    model_name = "wenet"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolop
@pytest.mark.compile
def test_autodrive_yolop_compile(setup_logging):
    model_name = "yolop"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.efficientnet
@pytest.mark.compile
def test_backbone_efficientnet_compile(setup_logging):
    model_name = "efficientnet"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.mobilenetv2
@pytest.mark.compile
def test_backbone_mobilenetv2_compile(setup_logging):
    model_name = "mobilenetv2"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.resnet50
@pytest.mark.compile
def test_backbone_resnet50_compile(setup_logging):
    model_name = "resnet50"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov3
@pytest.mark.compile
def test_detection_yolov3_compile(setup_logging):
    model_name = "yolov3"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5s
@pytest.mark.compile
def test_detection_yolov5s_compile(setup_logging):
    model_name = "yolov5s"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m
@pytest.mark.compile
def test_detection_yolov8m_compile(setup_logging):
    model_name = "yolov8m"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.sdxl
@pytest.mark.compile
def test_diffusion_sdxl_compile(setup_logging):
    model_name = "sdxl"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen2dot5
@pytest.mark.compile
def test_llm_qwen2dot5_compile(setup_logging):
    model_name = "qwen2.5"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen3
@pytest.mark.compile
def test_llm_qwen3_compile(setup_logging):
    model_name = "qwen3"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen3_14b
@pytest.mark.compile
def test_llm_qwen3_14b_compile(setup_logging):
    model_name = "qwen3-14b"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.deepseek
@pytest.mark.compile
def test_llm_deepseek_compile(setup_logging):
    model_name = "deepseek"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.deepseek_r1_qwen3_8b
@pytest.mark.compile
def test_llm_deepseek_r1_qwen3_8b_compile(setup_logging):
    model_name = "deepseek-r1-qwen3-8b"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen2dot5_vl
@pytest.mark.compile
def test_vllm_qwen2dot5_vl_compile(setup_logging):
    model_name = "qwen2.5-vl"
    _compile_func(model_name, setup_logging)
    assert True
