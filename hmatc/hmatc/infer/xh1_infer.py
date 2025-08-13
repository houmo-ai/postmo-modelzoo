import os
import time
from abc import ABC

import numpy as np
import tcim_lite

from ..base.base_infer import BaseInfer
from ..utils import logger


class Xh1Infer(BaseInfer, ABC):
    def __init__(self):
        super().__init__()
        self.backend = "xh1"
        self.model_ext = ".hmm"
        self.inputs_info = dict()
        self.inputs_batch = dict()

    def load(self, model_path):
        if not os.path.exists(model_path):
            logger.error(f"model path: {model_path} not exists.")
            exit(-1)
        logger.info(f"load model from {model_path}")
        self.engine = tcim_lite.runtime.load(model_path)
        logger.info("load xh1 model successfully.")
        # 获取模型输入输出信息
        input_num = self.engine.get_num_inputs()
        for idx in range(input_num):
            input_name = self.engine.get_input_name(idx)
            input_info = self.engine.get_input_info(input_name)
            shape = list(input_info.shape)
            dtype = np.dtype(input_info.dtype).name
            fmt = input_info.format.name
            self.inputs_info[input_name] = input_info
            self.inputs_batch[input_name] = 1 if input_name.startswith("resizer_crop_") and len(shape) == 1 else shape[0]
            logger.info(f"[{self.backend}] input[{input_name}] shape = {shape}, dtype = {dtype}, format = {fmt}")
        output_num = self.engine.get_num_outputs()
        for idx in range(output_num):
            output_name = self.engine.get_output_name(idx)
            output_info = self.engine.get_output_info(output_name)
            shape = list(output_info.shape)
            dtype = np.dtype(output_info.dtype).name
            fmt = output_info.format.name
            logger.info(f"[{self.backend}] output[{output_name}] shape = {shape}, dtype = {dtype}, format = {fmt}")
            
    def run(self, in_datas: dict):
        # set input
        for input_name in in_datas:
            self.engine.set_input(input_name, in_datas[input_name])
        self.total += 1
        t_start = time.time()
        self.engine.run()
        self.engine.sync()
        self.time_span += (time.time() - t_start) * 1000
        output_num = self.engine.get_num_outputs()
        outputs = dict()
        outputs_dequanted = dict()
        for idx in range(output_num):
            output_name = self.engine.get_output_name(idx)
            output_info = self.engine.get_output_info(output_name)
            output_data = self.engine.get_output(output_name)
            dequanted_data = np.ascontiguousarray(output_data.cast(np.float32).numpy())
            outputs[output_name] = np.ascontiguousarray(output_data.numpy())
            outputs_dequanted[output_name] = dequanted_data
        return outputs, outputs_dequanted

    def unload(self):
        pass
    
    def quantize(self, input_name: str, in_data: np.ndarray) -> np.ndarray:
        # 若是非图像输入，需要对输入数据进行量化
        input_info = self.inputs_info[input_name]
        input_info_dequanted = input_info.astype(np.dtype(in_data.dtype).type)
        in_tensor_dequanted = tcim_lite.runtime.Tensor(input_info_dequanted, in_data)
        in_tensor = tcim_lite.runtime.Tensor(input_info).to_host(to_contiguous=True)
        in_tensor_dequanted.cast_to(in_tensor)
        return in_tensor.numpy()
    
    def dequantize(self, output_name: str, out_data: np.ndarray) -> np.ndarray:
        output_info = self.engine.get_output_info(output_name)
        output_info_dequanted = output_info.astype(np.float32)
        out_tensor_quanted = tcim_lite.runtime.Tensor(output_info, out_data)
        out_tensor = tcim_lite.runtime.Tensor(output_info_dequanted).to_host(to_contiguous=True)
        out_tensor_quanted.cast_to(out_tensor)
        return out_tensor.numpy()
        
