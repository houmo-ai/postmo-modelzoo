import os
import time
from abc import ABC
from typing import Dict

import numpy as np
import torch
from xhquant.api import HMONNXInference, xhquant_init

from ..base.base_infer import BaseInfer
from ..utils import logger
from ..utils.utils import torch_to_numpy_dtype


class Xh2HmQuantInfer(BaseInfer, ABC):
    def __init__(self):
        super().__init__()
        self.backend = "Xh2Hmquant"
        self.model_ext = ".onnx"
        self.input_names = list()
        self.output_names = list()
        xhquant_init(None, debug=False)

    def load(self, model_path, device_id=0):
        if not os.path.exists(model_path):
            logger.error(f"model path: {model_path} not exists.")
            exit(-1)
        self.engine = HMONNXInference(model_path)
        # self.engine.to_fast_mode()
        self.engine.to(torch.device(self.device))
        self.input_names = self.engine.get_input_names()
        self.output_names = self.engine.get_output_names()
        logger.info("load Xh2Hmquant model successfully.")

        for idx, name in enumerate(self.input_names):
            info = self.engine.get_input(name)
            logger.info(
                f"[Xh2Hmquant] input{info.name} shape = {list(info.shape)}  dtype = {torch_to_numpy_dtype[info.dtype]}"
            )

        for idx, name in enumerate(self.output_names):
            info = self.engine.get_output(name)
            logger.info(
                f"[Xh2Hmquant] output{info.name} shape = {list(info.shape)} dtype = {torch_to_numpy_dtype[info.dtype]}"
            )

    def run(self, in_datas: dict) -> Dict[str, np.ndarray]:
        self.total += 1
        t_start = time.time()
        outputs = self.engine.run(in_datas)
        self.time_span += (time.time() - t_start) * 1000
        if len(self.output_names) == 1:
            outputs = {
                self.output_names[0]: outputs.detach().cpu().numpy().astype(np.float32)
            }
            return outputs
        return {
            output_name: outputs[idx].detach().cpu().numpy().astype(np.float32)
            for idx, output_name in enumerate(self.output_names)
        }

    def unload(self):
        pass
