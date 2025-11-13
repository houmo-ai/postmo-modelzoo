import pytest
import logging
from .test_models_utils import *

logger = logging.getLogger(__name__)


def _compile_func(model_name, log_file):
    logger.info("===> TEST START: test_%s_compile", model_name)
    execute_compile_flow(model_name, log_file, False)


@pytest.mark.wenet
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_asr_wenet_compile",
    depends_on=["test_get_models.py::test_asr_wenet_get_model"],
)
def test_asr_wenet_compile(setup_logging):
    model_name = "wenet"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolop
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_autodrive_yolop_compile",
    depends_on=["test_quant_models.py::test_autodrive_yolop_quant"],
)
def test_autodrive_yolop_compile(setup_logging):
    model_name = "yolop"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.efficientnet
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_backbone_efficientnet_compile",
    depends_on=["test_quant_models.py::test_backbone_efficientnet_quant"],
)
def test_backbone_efficientnet_compile(setup_logging):
    model_name = "efficientnet"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.mobilenetv2
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_backbone_mobilenetv2_compile",
    depends_on=["test_quant_models.py::test_backbone_mobilenetv2_quant"],
)
def test_backbone_mobilenetv2_compile(setup_logging):
    model_name = "mobilenetv2"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.resnet50
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_backbone_resnet50_compile",
    depends_on=["test_quant_models.py::test_backbone_resnet50_quant"],
)
def test_backbone_resnet50_compile(setup_logging):
    model_name = "resnet50"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.vit
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_backbone_vit_compile",
    depends_on=["test_quant_models.py::test_backbone_vit_quant"],
)
def test_backbone_vit_compile(setup_logging):
    model_name = "vit"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov3
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolov3_compile",
    depends_on=["test_quant_models.py::test_detection_yolov3_quant"],
)
def test_detection_yolov3_compile(setup_logging):
    model_name = "yolov3"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5s
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolov5s_compile",
    depends_on=["test_quant_models.py::test_detection_yolov5s_quant"],
)
def test_detection_yolov5s_compile(setup_logging):
    model_name = "yolov5s"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5s_feature
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolov5s_feature_compile",
    depends_on=["test_quant_models.py::test_detection_yolov5s_feature_quant"],
)
def test_detection_yolov5s_feature_compile(setup_logging):
    model_name = "yolov5s_feature"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_detection_yolov8m_compile",
    depends_on=["test_quant_models.py::test_detection_yolov8m_quant"],
)
def test_detection_yolov8m_compile(setup_logging):
    model_name = "yolov8m"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.sdxl
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_diffusion_sdxl_compile",
    depends_on=["test_get_models.py::test_diffusion_sdxl_get_model"],
)
def test_diffusion_sdxl_compile(setup_logging):
    model_name = "sdxl"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen2dot5
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_llm_qwen2dot5_compile",
    depends_on=["test_quant_models.py::test_llm_qwen2dot5_quant"],
)
def test_llm_qwen2dot5_compile(setup_logging):
    model_name = "qwen2.5"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen3
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_llm_qwen3_compile",
    depends_on=["test_quant_models.py::test_llm_qwen3_quant"],
)
def test_llm_qwen3_compile(setup_logging):
    model_name = "qwen3"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen3_14b
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_llm_qwen3_14b_compile",
    depends_on=["test_quant_models.py::test_llm_qwen3_14b_quant"],
)
def test_llm_qwen3_14b_compile(setup_logging):
    model_name = "qwen3-14b"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.deepseek
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_llm_deepseek_compile",
    depends_on=["test_quant_models.py::test_llm_deepseek_quant"],
)
def test_llm_deepseek_compile(setup_logging):
    model_name = "deepseek"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.deepseek_r1_qwen3_8b
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_llm_deepseek_r1_qwen3_8b_compile",
    depends_on=["test_quant_models.py::test_llm_deepseek_r1_qwen3_8b_quant"],
)
def test_llm_deepseek_r1_qwen3_8b_compile(setup_logging):
    model_name = "deepseek-r1-qwen3-8b"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.qwen2dot5_vl
@pytest.mark.compile
@pytest.mark.dependency(
    name="test_vllm_qwen2dot5_vl_compile",
    depends_on=["test_quant_models.py::test_vllm_qwen2dot5_vl_quant"],
)
def test_vllm_qwen2dot5_vl_compile(setup_logging):
    model_name = "qwen2.5-vl"
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolo12m
@pytest.mark.compile
@pytest.mark.dependency(name='test_detection_yolo12m_compile', depends_on=['test_quant_models.py::test_detection_yolo12m_quant'])
def test_detection_yolo12m_compile(setup_logging):
    model_name = 'yolo12m'
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m_pose
@pytest.mark.compile
@pytest.mark.dependency(name='test_estimation_yolov8m_pose_compile', depends_on=['test_quant_models.py::test_estimation_yolov8m_pose_quant'])
def test_estimation_yolov8m_pose_compile(setup_logging):
    model_name = 'yolov8m-pose'
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov8m_seg
@pytest.mark.compile
@pytest.mark.dependency(name='test_segmentation_yolov8m_seg_compile', depends_on=['test_quant_models.py::test_segmentation_yolov8m_seg_quant'])
def test_segmentation_yolov8m_seg_compile(setup_logging):
    model_name = 'yolov8m-seg'
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov7
@pytest.mark.compile
@pytest.mark.dependency(name='test_detection_yolov7_compile', depends_on=['test_quant_models.py::test_detection_yolov7_quant'])
def test_detection_yolov7_compile(setup_logging):
    model_name = 'yolov7'
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolov5m_face
@pytest.mark.compile
@pytest.mark.dependency(name='test_detection_yolov5m_face_compile', depends_on=['test_quant_models.py::test_detection_yolov5m_face_quant'])
def test_detection_yolov5m_face_compile(setup_logging):
    model_name = 'yolov5m_face'
    _compile_func(model_name, setup_logging)
    assert True


@pytest.mark.yolox
@pytest.mark.compile
@pytest.mark.dependency(name='test_detection_yolox_compile', depends_on=['test_quant_models.py::test_detection_yolox_quant'])
def test_detection_yolox_compile(setup_logging):
    model_name = 'yolox'
    _compile_func(model_name, setup_logging)
    assert True