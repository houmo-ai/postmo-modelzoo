import pytest
import logging
from .test_apis_utils import *

logger = logging.getLogger(__name__)


def _scene_func(example_name, log_file):
    logger.info("===> TEST START: test_apis_scenes_%s", example_name)
    execute_apis_examples(example_name, log_file)


@pytest.mark.apis
@pytest.mark.video_detect
@pytest.mark.yolov5s
@pytest.mark.resnet50
def test_apis_scenes_video_detect(setup_logging):
    example_name = "video_detect"
    _scene_func(example_name, setup_logging)
    assert True
