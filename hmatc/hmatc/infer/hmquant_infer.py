import os
import pickle
import time
import torch
from abc import ABC
from typing import Dict
from ..base.base_infer import BaseInfer
from ..utils import logger


class HmQuantInfer(BaseInfer, ABC):
    def __init__(self):
        super().__init__()
        self.backend = "hmquant"
        self.model_ext = ".pkl"
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        logger.info(f"Using device: {self.device}")

    def load(self, model_path):
        if not os.path.exists(model_path):
            logger.error(f"model path: {model_path} not exists.")
            exit(-1)
        from hmquant.api import quant_single_onnx_network

        with open(model_path, "rb") as f:
            self.engine = pickle.load(f)
        # self.engine.set_device(self.device)
        self.engine.set_ops_mode("hardware_forward")  # quant_forward or raw
        logger.info("load Xh1Hmquant model successfully.")
        graph_input_nodes = self.engine.graph_input_nodes
        graph_output_nodes = self.engine.graph_output_nodes

    def run(self, in_datas: Dict[str, torch.Tensor], to_file=False):
        # in_datas = {k: v.to(self.device) for k, v in in_datas.items()}
        self.total += 1
        t_start = time.time()
        outputs = self.engine.forward(in_datas, get_output_dict=True)
        self.time_span += (time.time() - t_start) * 1000
        return {
            output_name: outputs[output_name].detach().cpu().numpy()
            for output_name in outputs
        }

    def unload(self):
        pass
