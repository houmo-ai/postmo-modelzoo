# Copyright 2025 HOUMO AI
#
# File: hmonnx_infer.py
# Description:
#   HMONNX inference script using xhquant backend.
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
import threading
import time
import numpy as np
import torch
from abc import ABC
from typing import Dict
from ..base.base_infer import BaseInfer
from ..utils import logger
from ..utils.utils import torch_to_numpy_dtype, gen_random_data

_xhquant_init_lock = threading.Lock()
_xhquant_initialized_pid = None


def ensure_xhquant_initialized():
    """Initialize xhquant at most once in the current process."""
    global _xhquant_initialized_pid

    current_pid = os.getpid()
    if _xhquant_initialized_pid == current_pid:
        return

    with _xhquant_init_lock:
        if _xhquant_initialized_pid == current_pid:
            return
        from xhquant.api import xhquant_init

        xhquant_init(logger=logger)
        _xhquant_initialized_pid = current_pid


def is_hmonnx_available():
    """Return whether the optional HMONNX inference backend is installed."""
    try:
        from xhquant.api import HMONNXInference  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


class HmonnxInfer(BaseInfer, ABC):
    """
    Inference class for HMONNX models.
    Handles loading, running and unloading of quantized models using the xhquant backend for HMONNX hardware.
    """

    def __init__(self):
        """
        Initialize the HmonnxInfer instance.
        Sets up the backend, model extension, and initializes input/output tracking.
        Initializes the xhquant runtime.
        """
        super().__init__()
        self.backend = "hmonnx"
        self.model_ext = ".onnx"
        self.input_names = list()
        self.output_names = list()
        self.inputs_info = dict()
        self.outputs_info = dict()
        if torch.cuda.is_available():
            self.device = "cuda"

    def load(self, model_path, device_id=0):
        """
        Load the HMONNX model from the specified path.

        Args:
            model_path (str): Path to the HMONNX model file (.onnx)
            device_id (int): Device ID for inference (not used in this implementation)
        """
        if not os.path.exists(model_path):
            logger.fatal(f"model path: {model_path} not exists.")
        try:
            ensure_xhquant_initialized()
            from xhquant.api import HMONNXInference
        except ImportError:
            logger.fatal("Please install xhquant first.")
        self.engine = HMONNXInference(model_path)
        # self.engine.to_fast_mode()
        self.engine.to(torch.device(self.device))
        self.input_names = self.engine.get_input_names()
        self.output_names = self.engine.get_output_names()
        logger.info("load HMONNX model successfully.")

        for idx, name in enumerate(self.input_names):
            info = self.engine.get_input(name)
            logger.info(
                f"[HMONNX] input[{info.name}], shape = {list(info.shape)}, dtype = {torch_to_numpy_dtype[info.dtype]}"
            )
            self.inputs_batch[name] = info.shape[0]
            self.inputs_info[name] = {"shape": info.shape, "dtype": info.dtype}

        for idx, name in enumerate(self.output_names):
            info = self.engine.get_output(name)
            logger.info(
                f"[HMONNX] output[{info.name}], shape = {list(info.shape)}, dtype = {torch_to_numpy_dtype[info.dtype]}"
            )
            self.outputs_info[name] = {"shape": info.shape, "dtype": info.dtype}
            self.outputs_batch[name] = info.shape[0]

    def run(self, in_datas: dict, dequant=True, to_numpy=True) -> Dict[str, np.ndarray]:
        """
        Run inference on the loaded HMONNX model with the provided input data.

        Args:
            in_datas (dict): Dictionary of input data where keys are input names
                and values are torch tensors

            dequant (bool): Whether to dequantize the output data (default: True)
            to_numpy (bool): Whether to convert the output data to numpy arrays (default: True)

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
            data = outputs
            if dequant:
                data = data.to(torch.float32)
            if to_numpy:
                data = data.detach().cpu().numpy()
            outputs = {self.output_names[0]: data}
            return outputs
        out_datas = dict()
        for idx, name in enumerate(self.output_names):
            data = outputs[idx]
            if dequant:
                data = data.to(torch.float32)
            if to_numpy:
                data = data.detach().cpu().numpy()
            out_datas[name] = data
        return out_datas

    def unload(self):
        """
        Unload the model from memory.
        Currently not implemented (no-op).
        """
        pass

    @property
    def has_dynamic_resizer(self):
        for name in self.input_names:
            shape = self.inputs_info[name]["shape"]
            dtype = self.inputs_info[name]["dtype"]
            if (
                name.startswith("resizer_crop_")
                and len(shape) == 2
                and shape[1] in [4, 10]
                and dtype == torch.int32
            ):
                return True
        return False

    def get_random_input_data(self, name: str) -> np.ndarray:
        shape = self.inputs_info[name]["shape"]
        dtype = self.inputs_info[name]["dtype"]
        if (
            name.startswith("resizer_crop_")
            and len(shape) == 2
            and shape[1] in [4, 10]
            and dtype == torch.int32
        ):
            raise NotImplementedError("dynamic resizer input is not supported.")
        else:
            return gen_random_data(shape, torch_to_numpy_dtype[dtype])

    def get_input_name(self, idx: int) -> str:
        return self.input_names[idx]

    def get_output_name(self, idx: int) -> str:
        return self.output_names[idx]

    def get_num_inputs(self):
        return len(self.input_names)

    def get_num_outputs(self):
        return len(self.output_names)

    def get_input_info(self, name: str):
        return self.inputs_info[name]

    def get_output_info(self, name: str):
        return self.outputs_info[name]
