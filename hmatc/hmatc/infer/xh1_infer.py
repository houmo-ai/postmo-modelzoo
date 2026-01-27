# Copyright 2025 HOUMO AI
#
# File: xh1_infer.py
# Description:
#   XH1 inference script using TCIM Lite.
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
from abc import ABC
from ..base.base_infer import BaseInfer
from ..utils import logger

try:
    import tcim_lite
except ImportError:
    logger.error("Not found tcim_lite module, and please install tcim_lite first.")
    exit(-1)


class Xh1Infer(BaseInfer, ABC):
    """
    Inference class for XH1 hardware models.
    Handles loading, running and unloading of HMM models for XH1 hardware inference.
    """

    def __init__(self):
        """
        Initialize the Xh1Infer instance.
        Sets up the backend, model extension, and initializes input/output tracking.
        """
        super().__init__()
        self.backend = "xh1"
        self.model_ext = ".hmm"
        self.inputs_info = dict()
        self.inputs_format = dict()

    def load(self, model_path, device_id=0):
        """
        Load the XH1 model from the specified path.

        Args:
            model_path (str): Path to the HMM model file (.hmm)
            device_id (int): Device ID for inference on XH1 hardware
        """
        if not os.path.exists(model_path):
            logger.error(f"model path: {model_path} not exists.")
            exit(-1)

        if device_id >= tcim_lite.runtime.get_device_num():
            logger.error(f"device_id: {device_id} out of range")
            exit(-1)
        logger.info(f"load model from {model_path}")
        wm = tcim_lite.runtime.WeightManager(device_id)
        option = tcim_lite.runtime.Option(wm)
        self.engine = tcim_lite.runtime.load(model_path, option=option)
        logger.info(f"load {self.backend} model successfully.")
        # Get model input/output information
        input_num = self.engine.get_num_inputs()
        for idx in range(input_num):
            input_name = self.engine.get_input_name(idx)
            input_info = self.engine.get_input_info(input_name)
            shape = list(input_info.shape)
            dtype = np.dtype(input_info.dtype).name
            fmt = input_info.format.name
            self.inputs_format[input_name] = fmt
            self.inputs_info[input_name] = input_info
            self.inputs_batch[input_name] = (
                1
                if input_name.startswith("resizer_crop_") and len(shape) == 1
                else shape[0]
            )
            logger.info(
                f"[{self.backend}] input[{input_name}] shape = {shape}, dtype = {dtype}, format = {fmt}"
            )
        output_num = self.engine.get_num_outputs()
        for idx in range(output_num):
            output_name = self.engine.get_output_name(idx)
            output_info = self.engine.get_output_info(output_name)
            shape = list(output_info.shape)
            dtype = np.dtype(output_info.dtype).name
            fmt = output_info.format.name
            logger.info(
                f"[{self.backend}] output[{output_name}] shape = {shape}, dtype = {dtype}, format = {fmt}"
            )

    def run(self, in_datas: dict):
        """
        Run inference on the loaded XH1 model with the provided input data.

        Args:
            in_datas (dict): Dictionary of input data where keys are input names
                and values are numpy arrays

        Returns:
            tuple: A tuple containing:
                - outputs (dict): Dictionary of output data where keys are output names
                  and values are numpy arrays containing the quantized inference results
                - outputs_dequanted (dict): Dictionary of dequantized output data where
                  keys are output names and values are numpy arrays containing the
                  dequantized inference results
        """
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
            outputs[output_name] = np.ascontiguousarray(output_data.numpy())
            if output_name in ["auto_profile_data.bin", "primitive_profile_data.bin"]:
                continue
            dequanted_data = np.ascontiguousarray(output_data.cast(np.float32).numpy())
            outputs_dequanted[output_name] = dequanted_data
        return outputs, outputs_dequanted

    def unload(self):
        """
        Unload the model from memory.
        Currently not implemented (no-op).
        """
        pass

    def quantize(self, input_name: str, in_data: np.ndarray) -> np.ndarray:
        """
        Quantize input data for the specified input.

        Args:
            input_name (str): Name of the input
            in_data (np.ndarray): Input data to be quantized

        Returns:
            np.ndarray: Quantized input data
        """
        # If the input is non-image input, the input data needs to be quantized
        input_info = self.inputs_info[input_name]
        input_info_dequanted = input_info.astype(np.dtype(in_data.dtype).type)
        in_tensor_dequanted = tcim_lite.runtime.Tensor(input_info_dequanted, in_data)
        in_tensor = tcim_lite.runtime.Tensor(input_info).to_host(to_contiguous=True)
        in_tensor_dequanted.cast_to(in_tensor)
        return in_tensor.numpy()

    def dequantize(self, output_name: str, out_data: np.ndarray) -> np.ndarray:
        """
        Dequantize output data for the specified output.

        Args:
            output_name (str): Name of the output
            out_data (np.ndarray): Output data to be dequantized

        Returns:
            np.ndarray: Dequantized output data
        """
        output_info = self.engine.get_output_info(output_name)
        output_info_dequanted = output_info.astype(np.float32)
        out_tensor_quanted = tcim_lite.runtime.Tensor(output_info, out_data)
        out_tensor = tcim_lite.runtime.Tensor(output_info_dequanted).to_host(
            to_contiguous=True
        )
        out_tensor_quanted.cast_to(out_tensor)
        return out_tensor.numpy()
