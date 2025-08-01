import pytest
import logging
from .test_models_utils import *

logger = logging.getLogger(__name__)


def _compare_func(model_name, log_file):
    logger.info("===> TEST START: test_%s_compare", model_name)
    execute_compare_flow(model_name, log_file)


@pytest.mark.yolop
@pytest.mark.compare
def test_autodrive_yolop_compare(setup_logging):
    model_name = "yolop"
    _compare_func(model_name, setup_logging)
    assert True


@pytest.mark.efficientnet
@pytest.mark.compare
def test_backbone_efficientnet_compare(setup_logging):
    model_name = "efficientnet"
    _compare_func(model_name, setup_logging)
    assert True


@pytest.mark.mobilenetv2
@pytest.mark.compare
def test_backbone_mobilenetv2_compare(setup_logging):
    model_name = "mobilenetv2"
    _compare_func(model_name, setup_logging)
    assert True


@pytest.mark.resnet50
@pytest.mark.compare
def test_backbone_resnet50_compare(setup_logging):
    model_name = "resnet50"
    _compare_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov3
@pytest.mark.compare
def test_detection_yolov3_compare(setup_logging):
    model_name = "yolov3"
    _compare_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5s
@pytest.mark.compare
def test_detection_yolov5s_compare(setup_logging):
    model_name = "yolov5s"
    _compare_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m
@pytest.mark.compare
def test_detection_yolov8m_compare(setup_logging):
    model_name = "yolov8m"
    _compare_func(model_name, setup_logging)
    assert True
