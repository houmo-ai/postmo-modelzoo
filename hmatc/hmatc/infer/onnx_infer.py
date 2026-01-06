# Copyright 2025 HOUMO AI
#
# File: onnx_infer.py
# Description:
#   ONNX inference script using ONNX Runtime.
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
import onnx
import onnxruntime as ort
from abc import ABC
from onnx import mapping
from ..base.base_infer import BaseInfer
from ..utils import logger


class OnnxInfer(BaseInfer, ABC):
    """
    Inference class for ONNX models.
    Handles loading, running and unloading of ONNX models using ONNX Runtime.
    """

    def __init__(self):
        """
        Initialize the OnnxInfer instance.
        Sets up the backend, model extension, and initializes input/output tracking.
        """
        super().__init__()
        self.backend = "onnx"
        self.model_ext = ".onnx"
        self.output_names = list()
        self.inputs_batch = dict()

    def load(self, model_path, device_id=0):
        """
        Load the ONNX model from the specified path.

        Args:
            model_path (str): Path to the ONNX model file (.onnx)
            device_id (int): Device ID for inference (not used in ONNX Runtime)
        """
        if not os.path.exists(model_path):
            logger.error(f"model path: {model_path} not exists.")
            exit(-1)
        self.engine = ort.InferenceSession(model_path)
        logger.info("load onnx model successfully.")
        for idx, tensor in enumerate(self.engine.get_inputs()):
            logger.info(
                f"[onnx] input{idx}, name: {tensor.name}, shape={tensor.shape}, dtype={self.onnx_type_to_numpy(tensor.type)}"
            )
            bs = tensor.shape[0]
            # Dynamic is 1
            self.inputs_batch[tensor.name] = (
                1 if (not isinstance(bs, int) or bs < 0) else bs
            )
        for idx, tensor in enumerate(self.engine.get_outputs()):
            logger.info(
                f"[onnx] output{idx}, name: {tensor.name}, shape={tensor.shape}, dtype={self.onnx_type_to_numpy(tensor.type)}"
            )
            self.output_names.append(tensor.name)

    def run(self, in_datas: dict, to_file=False):
        """
        Run inference on the loaded ONNX model with the provided input data.

        Args:
            in_datas (dict): Dictionary of input data where keys are input names
                and values are input tensors
            to_file (bool): Whether to save output to file (not implemented)

        Returns:
            dict: Dictionary of output data where keys are output names and values are
                numpy arrays containing the inference results
        """
        self.total += 1
        t_start = time.time()
        outputs = self.engine.run(None, in_datas)
        self.time_span += (time.time() - t_start) * 1000
        res = dict()
        for idx in range(len(outputs)):
            res[self.output_names[idx]] = outputs[idx]
        return res

    def unload(self):
        """
        Unload the model from memory.
        Currently not implemented (no-op).
        """
        pass

    @staticmethod
    def onnx_type_to_numpy(tensor_type_str):
        """
        Convert ONNX tensor type string to numpy data type.

        Args:
            tensor_type_str (str): ONNX tensor type string

        Returns:
            numpy.dtype: Corresponding numpy data type

        Raises:
            ValueError: If the ONNX type is not supported
        """
        try:
            elem_type_str = tensor_type_str.split("(")[-1].split(")")[0].upper()
            onnx_dtype = onnx.TensorProto.DataType.Value(elem_type_str)
            return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]
        except Exception as e:
            raise ValueError(f"Unsupported ONNX type: {tensor_type_str}") from e
