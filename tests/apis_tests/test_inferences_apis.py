import pytest
import logging
from .test_apis_utils import *

logger = logging.getLogger(__name__)


def _inference_func(example_name, log_file):
    logger.info("===> TEST START: test_apis_inferences_%s", example_name)
    execute_apis_examples(example_name, log_file)


@pytest.mark.apis
@pytest.mark.resnet50
@pytest.mark.inference
def test_apis_inferences_resnet50(setup_logging):
    example_name = "resnet50"
    _inference_func(example_name, setup_logging)
    assert True


@pytest.mark.apis
@pytest.mark.yolov5s
@pytest.mark.inference
def test_apis_inferences_yolov5s(setup_logging):
    example_name = "yolov5s"
    _inference_func(example_name, setup_logging)
    assert True


@pytest.mark.apis
@pytest.mark.qwen3
@pytest.mark.inference
def test_apis_inferences_qwen3(setup_logging):
    example_name = "qwen3"
    _inference_func(example_name, setup_logging)
    assert True


@pytest.mark.apis
@pytest.mark.resnet50
@pytest.mark.multistreams
def test_apis_inferences_resnet50_multistreams(setup_logging):
    example_name = "resnet50_multistreams"
    _inference_func(example_name, setup_logging)
    assert True


@pytest.mark.apis
@pytest.mark.resnet50
@pytest.mark.pipeline
def test_apis_inferences_resnet50_pipeline(setup_logging):
    example_name = "resnet50_pipeline"
    _inference_func(example_name, setup_logging)
    assert True


@pytest.mark.apis
@pytest.mark.resnet50
@pytest.mark.yolov5s
@pytest.mark.multibatch
def test_apis_inferences_yolov5s_resnet50_multibatch(setup_logging):
    example_name = "yolov5s_resnet50_multibatch"
    _inference_func(example_name, setup_logging)
    assert True
