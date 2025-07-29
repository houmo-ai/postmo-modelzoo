import time
import os
import pickle
import onnxruntime as ort
from abc import ABC
from ..base.base_infer import BaseInfer
from ..utils import logger


class HmQuantInfer(BaseInfer, ABC):
    def __init__(self):
        super().__init__()
        self.backend = "hmquant"
        self.model_ext = ".pkl"

    def load(self, model_path):
        assert os.path.exists(model_path)
        from hmquant.api import quant_single_onnx_network
        with open(model_path, "rb") as f:
            self.engine = pickle.load(f)
        # self.engine.set_ops_mode("quant_forward")
        self.engine.set_ops_mode("hardware_forward")
        logger.info("load Xh1Hmquant model successfully.")
        graph_input_nodes = self.engine.graph_input_nodes
        graph_output_nodes = self.engine.graph_output_nodes
                
    def run(self, in_datas: dict, to_file=False):
        self.total += 1
        t_start = time.time()
        outputs = self.engine.forward(in_datas, get_output_dict=True) 
        self.time_span += (time.time() - t_start) * 1000
        return {output_name: outputs[output_name].detach().cpu().numpy() for output_name in outputs}

    def unload(self):
        pass
