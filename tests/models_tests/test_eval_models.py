import pytest
import logging
from .test_models_utils import *

logger = logging.getLogger(__name__)


def _eval_func(model_name, log_file):
    logger.info("===> TEST START: test_%s_eval", model_name)
    execute_eval_flow(model_name, log_file)


@pytest.mark.efficientnet
@pytest.mark.eval
def test_backbone_efficientnet_eval(setup_logging):
    model_name = "efficientnet"
    _eval_func(model_name, setup_logging)
    assert True


@pytest.mark.mobilenetv2
@pytest.mark.eval
def test_backbone_mobilenetv2_eval(setup_logging):
    model_name = "mobilenetv2"
    _eval_func(model_name, setup_logging)
    assert True


@pytest.mark.resnet50
@pytest.mark.eval
def test_backbone_resnet50_eval(setup_logging):
    model_name = "resnet50"
    _eval_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov3
@pytest.mark.eval
def test_detection_yolov3_eval(setup_logging):
    model_name = "yolov3"
    _eval_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5s
@pytest.mark.eval
def test_detection_yolov5s_eval(setup_logging):
    model_name = "yolov5s"
    _eval_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m
@pytest.mark.eval
def test_detection_yolov8m_eval(setup_logging):
    model_name = "yolov8m"
    _eval_func(model_name, setup_logging)
    assert True
