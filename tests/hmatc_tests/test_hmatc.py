import pytest
import logging
from .test_hmatc_utils import *

logger = logging.getLogger(__name__)


def _hmatc_func(model_name, log_file):
    logger.info("===> TEST START: test_hmatc_%s", model_name)
    execute_hmatc_cmd(model_name, log_file)


@pytest.mark.hmatc
@pytest.mark.resnet50
def test_hmatc_resnet50(setup_logging):
    model_name = "resnet50"
    _hmatc_func(model_name, setup_logging)
    assert True


@pytest.mark.hmatc
@pytest.mark.yolov5s
def test_hmatc_yolov5s(setup_logging):
    model_name = "yolov5s"
    _hmatc_func(model_name, setup_logging)
    assert True
