#from ..onnxBaseOpt.onnxBaseOptimizer import OnnxBaseOptimizer
from ..onnxBaseOpt.onnxOptimizerManager import OnnxOptimizerManager
from .onnxNpuPlatformOptimizerManager import NpuOptimizerManager
from ..onnxBaseOpt.onnxConfigController import OnnxCfg

from ...utils import logger

@OnnxOptimizerManager("general_opt")
class OnnxGeneralOptimizer(object):

    @classmethod
    def opt(cls, onnx_model):
        platform = OnnxCfg.get_val("platform", None)
        if platform is None:
            logger.warning(f"Not support platform {platform}!")
            return onnx_model
        domain = f"houmo.{platform}"
        optimizer = NpuOptimizerManager.get_npu_optimizer(domain)
        onnx_model = optimizer.opt(onnx_model)
        return onnx_model