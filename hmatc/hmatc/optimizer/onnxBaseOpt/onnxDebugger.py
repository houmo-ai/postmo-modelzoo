import os
import onnx
import logging
from functools import wraps
from typing import List, Dict, Optional

from .onnxConfigController import OnnxCfg
from ..onnxUtils.generalClassUtils import Singleton
from ...utils import logger

TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]

class OnnxDebugger(metaclass=Singleton):
    work_mode = "debug"

    @staticmethod
    def getWorkMode():
        return OnnxDebugger.work_mode

    @staticmethod
    def set_work_mode():
        OnnxDebugger.work_mode = OnnxCfg.get_val("work_mode", "debug")

    @staticmethod
    def set_logging():
        LOG_FORMAT = "%(asctime)s - [HMAPPOPT] - %(levelname)s - %(message)s"
        log_level = logging.ERROR
        if OnnxDebugger.work_mode == "debug":
            log_level = logging.DEBUG
        elif OnnxDebugger.work_mode in ["release", "product"]:
            log_level = logging.INFO
        OnnxDebugger.reset_logging(log_level, LOG_FORMAT)
        logging.info("\033[1;32m============== Start Houmo Application Optimization ==============\033[0m")
        logging.info(f"Working mode:{OnnxDebugger.work_mode}")

    @staticmethod
    def reset_logging(log_level=logging.INFO, log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
        #root_logger = logging.getLogger()
        for h in logger.handlers:
            logger.removeHandler(h)
        logging.basicConfig(level=log_level, format=log_format)

    @staticmethod
    def onnx_opt_func_debug_wrapper(func):
        @wraps(func)
        def catcher(*arg, **kwargs):
            return_model = arg[0] if isinstance(arg[0], onnx.ModelProto) else arg[1]  # for class method
            function_mode = OnnxCfg.get_val(func.__name__, True)
            if function_mode is False:
                return return_model
            if OnnxDebugger.work_mode == "product":
                try:
                    # logger.info(f"Pre Execute:{func.__name__}")
                    func_return = func(*arg, **kwargs)
                    return_model, activated = func_return if isinstance(func_return, tuple) else (func_return, False)
                    if activated:
                        logger.info(f"Execute: {func.__name__}\n")
                except Exception as e:
                    logger.error(f"Failing to execute:{func.__name__}")
                    logger.error(e)
                    save_path = f'{OnnxCfg.get_val("out_path")}/{OnnxCfg.get_val("model_name")}-opt-debug.onnx'
                    onnx.save(return_model, save_path)
                    raise ValueError(f"Failing to execute:{func.__name__}")
            elif OnnxDebugger.work_mode in ["release", "debug"]:
                try:
                    # logger.info(f"Pre Execute:{func.__name__}")
                    func_return = func(*arg, **kwargs)
                    return_model, activated = func_return if isinstance(func_return, tuple) else (func_return, False)
                    if activated:
                        logger.info(f"Execute: {func.__name__}\n")
                except Exception as e:
                    logger.error(f"Failing to execute:{func.__name__}")
                    logger.error(e)
            else:
                logger.error("Please check work mode")
                raise ValueError("invalid work mode")

            if function_mode == "dump_model":
                dump_dir = OnnxCfg.get_val("dump_model_dir", "test_workspace")
                model_name = f"dump_model_{func.__name__}.onnx"
                dump_path = os.path.join(dump_dir, model_name)
                onnx.save_model(return_model, dump_path)
            return return_model

        return catcher