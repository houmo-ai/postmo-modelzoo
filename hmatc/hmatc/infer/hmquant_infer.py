# Copyright 2025 HOUMO AI
#
# File: hmquant_infer.py
# Description:
#   XH1 HMQuant Inference
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import os
import pickle
import time
import torch
from abc import ABC
from typing import Dict
from ..base.base_infer import BaseInfer
from ..utils import logger


class HmQuantInfer(BaseInfer, ABC):
    """
    Inference class for HmQuant models.
    Handles loading, running and unloading of quantized models using the hmquant backend.
    """

    def __init__(self):
        """
        Initialize the HmQuantInfer instance.
        Sets up the backend, model extension, and device configuration.
        """
        super().__init__()
        self.backend = "hmquant"
        self.model_ext = ".pkl"
        if torch.cuda.is_available():
            self.device = "cuda"
        logger.info(f"Using device: {self.device}")

    def load(self, model_path, device_id=0):
        """
        Load the quantized model from the specified path.

        Args:
            model_path (str): Path to the quantized model file (.pkl)
            device_id (int): Device ID for inference (not used in CPU mode)
        """
        if not os.path.exists(model_path):
            logger.error(f"model path: {model_path} not exists.")
            exit(-1)
        try:
            # from hmquant import set_external_logger

            # set_external_logger(logger)
            from hmquant.api import quant_single_onnx_network
        except ImportError:
            logger.error("Not found hmquant module, and please install hmquant first.")
            exit(-1)

        with open(model_path, "rb") as f:
            self.engine = pickle.load(f)
        # self.engine.set_device(self.device)
        self.engine.set_ops_mode("hardware_forward")  # quant_forward or raw
        logger.info("load Xh1Hmquant model successfully.")
        graph_input_nodes = self.engine.graph_input_nodes
        graph_output_nodes = self.engine.graph_output_nodes

    def run(self, in_datas: Dict[str, torch.Tensor], to_file=False):
        """
        Run inference on the loaded model with the provided input data.

        Args:
            in_datas (Dict[str, torch.Tensor]): Dictionary of input data where keys are
                input names and values are torch tensors
            to_file (bool): Whether to save output to file (not implemented)

        Returns:
            dict: Dictionary of output data where keys are output names and values are
                numpy arrays containing the inference results
        """
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
        """
        Unload the model from memory.
        Currently not implemented (no-op).
        """
        pass
