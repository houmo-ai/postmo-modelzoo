import os
import time
from abc import ABC

import onnx
import onnxruntime as ort
from onnx import mapping

from ..base.base_infer import BaseInfer
from ..utils import logger


class OnnxInfer(BaseInfer, ABC):
    def __init__(self):
        super().__init__()
        self.backend = "onnx"
        self.model_ext = ".onnx"
        self.output_names = list()
        self.inputs_batch = dict()
        
    def load(self, model_path):
        if not os.path.exists(model_path):
            logger.error(f"model path: {model_path} not exists.")
            exit(-1)
        self.engine = ort.InferenceSession(model_path)
        logger.info("load onnx model successfully.")
        for idx, tensor in enumerate(self.engine.get_inputs()):
            logger.info(f"[onnx] input{idx}, name: {tensor.name}, shape={tensor.shape}, dtype={self.onnx_type_to_numpy(tensor.type)}")
            bs = tensor.shape[0]
            # 动态就是1
            self.inputs_batch[tensor.name] = 1 if (not isinstance(bs, int) or bs < 0) else bs
        for idx, tensor in enumerate(self.engine.get_outputs()):
            logger.info(f"[onnx] output{idx}, name: {tensor.name}, shape={tensor.shape}, dtype={self.onnx_type_to_numpy(tensor.type)}")
            self.output_names.append(tensor.name)
            
    def run(self, in_datas: dict, to_file=False):
        self.total += 1
        t_start = time.time()
        outputs = self.engine.run(None, in_datas)
        self.time_span += (time.time() - t_start) * 1000
        res = dict()
        for idx in range(len(outputs)):
            res[self.output_names[idx]] = outputs[idx]
        return res

    def unload(self):
        pass
    
    @staticmethod
    def onnx_type_to_numpy(tensor_type_str):
        try:
            elem_type_str = tensor_type_str.split('(')[-1].split(')')[0].upper()
            onnx_dtype = onnx.TensorProto.DataType.Value(elem_type_str)
            return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]
        except Exception as e:
            raise ValueError(f"Unsupported ONNX type: {tensor_type_str}") from e