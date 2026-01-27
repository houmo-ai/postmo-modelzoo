# Copyright 2025 HOUMO AI
#
# File: xhquant_infer.py
# Description:
#   XH2 HmQuant inference script using xhquant backend.
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
import time
import numpy as np
import torch
from abc import ABC
from typing import Dict
from ..base.base_infer import BaseInfer
from ..utils import logger
from ..utils.utils import torch_to_numpy_dtype


class Xh2HmQuantInfer(BaseInfer, ABC):
    """
    Inference class for XH2 HmQuant models.
    Handles loading, running and unloading of quantized models using the xhquant backend for XH2 hardware.
    """

    def __init__(self):
        """
        Initialize the Xh2HmQuantInfer instance.
        Sets up the backend, model extension, and initializes input/output tracking.
        Initializes the xhquant runtime.
        """
        super().__init__()
        self.backend = "hmonnx"
        self.model_ext = ".onnx"
        self.input_names = list()
        self.output_names = list()
        try:
            from xhquant.api import xhquant_init
        except ImportError:
            logger.error("Please install xhquant first.")
            exit(-1)
        xhquant_init(None, debug=False)

    def load(self, model_path, device_id=0):
        """
        Load the XH2 HmQuant model from the specified path.

        Args:
            model_path (str): Path to the HMONNX model file (.onnx)
            device_id (int): Device ID for inference (not used in this implementation)
        """
        if not os.path.exists(model_path):
            logger.error(f"model path: {model_path} not exists.")
            exit(-1)
        try:
            from xhquant.api import HMONNXInference
        except ImportError:
            logger.error("Please install xhquant first.")
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
                f"[Xh2Hmquant] input[{info.name}], shape = {list(info.shape)}, dtype = {torch_to_numpy_dtype[info.dtype]}"
            )
            self.inputs_batch[name] = info.shape[0]

        for idx, name in enumerate(self.output_names):
            info = self.engine.get_output(name)
            logger.info(
                f"[Xh2Hmquant] output[{info.name}], shape = {list(info.shape)}, dtype = {torch_to_numpy_dtype[info.dtype]}"
            )

    def run(self, in_datas: dict) -> Dict[str, np.ndarray]:
        """
        Run inference on the loaded XH2 HmQuant model with the provided input data.

        Args:
            in_datas (dict): Dictionary of input data where keys are input names
                and values are torch tensors

        Returns:
            Dict[str, np.ndarray]: Dictionary of output data where keys are output names
                and values are numpy arrays containing the inference results as float32
        """
        for key in in_datas:
            in_data = in_datas[key]
            if isinstance(in_data, np.ndarray):
                in_datas[key] = torch.from_numpy(in_data)
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
        """
        Unload the model from memory.
        Currently not implemented (no-op).
        """
        pass
