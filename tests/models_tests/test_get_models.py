import pytest
import logging
from .test_models_utils import *

logger = logging.getLogger(__name__)


def _get_model_func(model_name, log_file):
    logger.info("===> TEST START: test_%s_get_model", model_name)
    execute_get_model_flow(model_name, log_file)


@pytest.mark.wenet
@pytest.mark.get_model
@pytest.mark.dependency(name="test_asr_wenet_get_model")
def test_asr_wenet_get_model(setup_logging):
    model_name = "wenet"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.yolop
@pytest.mark.get_model
@pytest.mark.dependency(name="test_autodrive_yolop_get_model")
def test_autodrive_yolop_get_model(setup_logging):
    model_name = "yolop"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.efficientnet
@pytest.mark.get_model
@pytest.mark.dependency(name="test_backbone_efficientnet_get_model")
def test_backbone_efficientnet_get_model(setup_logging):
    model_name = "efficientnet"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.mobilenetv2
@pytest.mark.get_model
@pytest.mark.dependency(name="test_backbone_mobilenetv2_get_model")
def test_backbone_mobilenetv2_get_model(setup_logging):
    model_name = "mobilenetv2"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.resnet50
@pytest.mark.get_model
@pytest.mark.dependency(name="test_backbone_resnet50_get_model")
def test_backbone_resnet50_get_model(setup_logging):
    model_name = "resnet50"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.vit
@pytest.mark.get_model
@pytest.mark.dependency(name="test_backbone_vit_get_model")
def test_backbone_vit_get_model(setup_logging):
    model_name = "vit"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov3
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolov3_get_model")
def test_detection_yolov3_get_model(setup_logging):
    model_name = "yolov3"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5s
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolov5s_get_model")
def test_detection_yolov5s_get_model(setup_logging):
    model_name = "yolov5s"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5s_feature
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolov5s_feature_get_model")
def test_detection_yolov5s_feature_get_model(setup_logging):
    model_name = "yolov5s_feature"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m
@pytest.mark.get_model
@pytest.mark.dependency(name="test_detection_yolov8m_get_model")
def test_detection_yolov8m_get_model(setup_logging):
    model_name = "yolov8m"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.sd3
@pytest.mark.get_model
@pytest.mark.dependency(name="test_diffusion_sd3_get_model")
def test_diffusion_sd3_get_model(setup_logging):
    model_name = "sd3"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.sdxl
@pytest.mark.get_model
@pytest.mark.dependency(name="test_diffusion_sdxl_get_model")
def test_diffusion_sdxl_get_model(setup_logging):
    model_name = "sdxl"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen2dot5
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_qwen2dot5_get_model")
def test_llm_qwen2dot5_get_model(setup_logging):
    model_name = "qwen2.5"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen3
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_qwen3_get_model")
def test_llm_qwen3_get_model(setup_logging):
    model_name = "qwen3"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen3_14b
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_qwen3_14b_get_model")
def test_llm_qwen3_14b_get_model(setup_logging):
    model_name = "qwen3-14b"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.deepseek
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_deepseek_get_model")
def test_llm_deepseek_get_model(setup_logging):
    model_name = "deepseek"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.deepseek_r1_qwen3_8b
@pytest.mark.get_model
@pytest.mark.dependency(name="test_llm_deepseek_r1_qwen3_8b_get_model")
def test_llm_deepseek_r1_qwen3_8b_get_model(setup_logging):
    model_name = "deepseek-r1-qwen3-8b"
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen2dot5_vl
@pytest.mark.get_model
@pytest.mark.dependency(name="test_vllm_qwen2dot5_vl_get_model")
def test_vllm_qwen2dot5_vl_get_model(setup_logging):
    model_name = 'qwen2.5-vl'
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.yolo12m
@pytest.mark.get_model
@pytest.mark.dependency(name='test_detection_yolo12m_get_model')
def test_detection_yolo12m_get_model(setup_logging):
    model_name = 'yolo12m'
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m_pose
@pytest.mark.get_model
@pytest.mark.dependency(name='test_estimation_yolov8m_pose_get_model')
def test_estimation_yolov8m_pose_get_model(setup_logging):
    model_name = 'yolov8m-pose'
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m_seg
@pytest.mark.get_model
@pytest.mark.dependency(name='test_segmentation_yolov8m_seg_get_model')
def test_segmentation_yolov8m_seg_get_model(setup_logging):
    model_name = 'yolov8m-seg'
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov7
@pytest.mark.get_model
@pytest.mark.dependency(name='test_detection_yolov7_get_model')
def test_detection_yolov7_get_model(setup_logging):
    model_name = 'yolov7'
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5m_face
@pytest.mark.get_model
@pytest.mark.dependency(name='test_detection_yolov5m_face_get_model')
def test_detection_yolov5m_face_get_model(setup_logging):
    model_name = 'yolov5m_face'
    _get_model_func(model_name, setup_logging)
    assert True


@pytest.mark.yolox
@pytest.mark.get_model
@pytest.mark.dependency(name='test_detection_yolox_get_model')
def test_detection_yolox_get_model(setup_logging):
    model_name = 'yolox'
    _get_model_func(model_name, setup_logging)
    assert True
