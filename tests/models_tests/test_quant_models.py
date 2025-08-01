import pytest
import logging
from .test_models_utils import *

logger = logging.getLogger(__name__)


def _quant_func(model_name, log_file):
    logger.info("===> TEST START: test_%s_quant", model_name)
    execute_quant_flow(model_name, log_file)


@pytest.mark.yolop
@pytest.mark.quant
def test_autodrive_yolop_quant(setup_logging):
    model_name = "yolop"
    _quant_func(model_name, setup_logging)
    assert True


@pytest.mark.efficientnet
@pytest.mark.quant
def test_backbone_efficientnet_quant(setup_logging):
    model_name = "efficientnet"
    _quant_func(model_name, setup_logging)
    assert True


@pytest.mark.mobilenetv2
@pytest.mark.quant
def test_backbone_mobilenetv2_quant(setup_logging):
    model_name = "mobilenetv2"
    _quant_func(model_name, setup_logging)
    assert True


@pytest.mark.resnet50
@pytest.mark.quant
def test_backbone_resnet50_quant(setup_logging):
    model_name = "resnet50"
    _quant_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov3
@pytest.mark.quant
def test_detection_yolov3_quant(setup_logging):
    model_name = "yolov3"
    _quant_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5s
@pytest.mark.quant
def test_detection_yolov5s_quant(setup_logging):
    model_name = "yolov5s"
    _quant_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m
@pytest.mark.quant
def test_detection_yolov8m_quant(setup_logging):
    model_name = "yolov8m"
    _quant_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen2dot5
@pytest.mark.quant
def test_llm_qwen2dot5_quant(setup_logging):
    model_name = "qwen2.5"
    _quant_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen3
@pytest.mark.quant
def test_llm_qwen3_quant(setup_logging):
    model_name = "qwen3"
    _quant_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen3_14b
@pytest.mark.quant
def test_llm_qwen3_14b_quant(setup_logging):
    model_name = "qwen3-14b"
    _quant_func(model_name, setup_logging)
    assert True


@pytest.mark.deepseek
@pytest.mark.quant
def test_llm_deepseek_quant(setup_logging):
    model_name = "deepseek"
    _quant_func(model_name, setup_logging)
    assert True


@pytest.mark.deepseek_r1_qwen3_8b
@pytest.mark.quant
def test_llm_deepseek_r1_qwen3_8b_quant(setup_logging):
    model_name = "deepseek-r1-qwen3-8b"
    _quant_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen2dot5_vl
@pytest.mark.quant
def test_vllm_qwen2dot5_vl_quant(setup_logging):
    model_name = "qwen2.5-vl"
    _quant_func(model_name, setup_logging)
    assert True
