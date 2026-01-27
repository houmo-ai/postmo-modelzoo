# Copyright 2025 HOUMO AI
#
# File: base_model.py
# Description:
#   Base model for all models.
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
import abc
import time
import torch
import numpy as np
from typing import Dict, Any
from ..utils import logger
from ..utils.utils import SUPPORT_BACKEND
from ..utils.preprocess import xh1_preprocess as resizer_preprocess
from ..infer.xh1_infer import Xh1Infer
from ..infer.xh2_infer import Xh2Infer
from ..infer.onnx_infer import OnnxInfer
from ..infer.xhquant_infer import Xh2HmQuantInfer


COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (169, 169, 169),
    (0, 0, 139),
    (0, 69, 255),
    (30, 105, 210),
    (10, 215, 255),
    (0, 255, 255),
    (0, 128, 128),
    (144, 238, 144),
    (139, 139, 0),
    (230, 216, 173),
    (130, 0, 75),
    (128, 0, 128),
    (203, 192, 255),
    (147, 20, 255),
    (238, 130, 238),
]


class BaseModel(object, metaclass=abc.ABCMeta):
    """Base model class for model inference and processing operations.

    This abstract class provides a unified interface for model inference across different
    backends (ONNX, XH1, XH2), handling preprocessing, inference execution, and postprocessing
    with support for various input formats and resizer modes.
    """

    def __init__(self, **kwargs):
        """Initialize the model instance with configuration parameters.

        Args:
            **kwargs: Keyword arguments including:
                - inputs_cfg (dict): Configuration for model inputs
                - is_image_single_input (bool): Whether input is single image
                - resizer_mode (int): Mode for resizing (default: 0)
                - roi_num (int): Number of regions of interest (default: 1)
                - backend (str): Backend type (onnx/xh1/xh2/hmonnx)
        """
        self.time_span = 0  # Total time span for inference operations
        self.total = 0  # Total number of inference operations performed
        self.engine = None  # Inference engine instance
        self.inputs_cfg = kwargs["inputs_cfg"]  # Configuration for model inputs
        self.inputs_name = list(self.inputs_cfg.keys())  # List of input names
        self.is_image_single_input = kwargs[
            "is_image_single_input"
        ]  # Whether input is single image
        self.resizer_mode = kwargs.get("resizer_mode", 0)  # Mode for resizing
        self.roi_num = kwargs.get("roi_num", 1)  # Number of regions of interest
        self.backend = kwargs["backend"]  # Backend type: onnx/xh1/xh2
        if self.backend not in SUPPORT_BACKEND:
            logger.error(f"backend not in {SUPPORT_BACKEND}")
            exit(-1)
        if self.backend == "onnx":
            self.engine = OnnxInfer()
        elif self.backend == "xh1":
            self.engine = Xh1Infer()
        elif self.backend == "xh2":
            self.engine = Xh2Infer()
        elif self.backend == "hmonnx":
            self.engine = Xh2HmQuantInfer()
        else:
            logger.error(f"Not support backend: {self.backend}")
            exit(-1)

    def load(self, model_path: str, device_id=0):
        """Load the model for inference.

        Args:
            model_path (str): Path to the model file to be loaded.
            device_id (int): Device ID to load the model on (default: 0).
        """
        model_name = os.path.basename(model_path)
        _, ext = os.path.splitext(model_name)
        if ext != self.engine.model_ext:
            logger.error(f"{model_name} is not {self.engine.model_ext}")
            exit(-1)
        self.engine.load(model_path, device_id=device_id)

    def preprocess(self, in_datas: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Preprocess input data for the model.

        Args:
            in_datas (dict): Dictionary containing input data arrays.

        Returns:
            dict: Dictionary containing preprocessed input data arrays.
        """
        if not self.is_image_single_input:
            # Single input non-image or multi-input, input data is preprocessed externally
            if self.backend == "onnx":
                return in_datas
            elif self.backend in ["xh1"]:
                for input_name in in_datas:
                    in_data = in_datas[input_name]
                    in_datas[input_name] = self.engine.quantize(input_name, in_data)
                return in_datas
            else:
                logger.error(f"Not support backend: {self.backend}")
                exit(-1)
        else:
            new_datas = dict()
            # Single input image, can be supported by internal preprocessing
            in_name = list(in_datas.keys())[0]
            cv_image = in_datas[in_name]
            input_cfg = self.inputs_cfg[in_name]
            input_shape = input_cfg["shape"]
            data_format = input_cfg["data_format"]
            mean = input_cfg.get("mean")
            std = input_cfg.get("std")
            resize_type = input_cfg["resize_type"]
            padding_mode = input_cfg.get("padding_mode")
            padding_values = input_cfg.get("padding_values")
            N, C, H, W = input_shape
            resizer_cfg = input_cfg.get("resizer", dict())
            toYUV_format = resizer_cfg.get("toYUV_format", None)
            max_input_size = resizer_cfg.get("max_input_size", (H, W))
            im, dyn_info = resizer_preprocess(
                cv_image,
                input_shape,
                max_input_size,
                mean=mean,
                std=std,
                use_resize=self.resizer_mode in [0, 3] or self.backend == "onnx",
                use_norm=self.resizer_mode == 0 or self.backend == "onnx",
                use_rgb=data_format == "RGB"
                and (self.resizer_mode == 0 or self.backend == "onnx"),
                resize_type=resize_type,
                padding_mode=padding_mode,
                padding_values=padding_values,
                is_onnx=self.resizer_mode == 0
                or self.backend
                == "onnx",  # Static resizer, need to convert to YUV in non-quantization stage, cannot set is_onnx=True
                to_YUV=self.resizer_mode in [1, 2, 3],
                fmt=toYUV_format,
                return_dynamic_v1_format=self.backend in ["xh2", "hmonnx"]
                and self.resizer_mode in [1, 2],
            )
            if self.backend == "onnx":
                new_datas[in_name] = im.detach().cpu().numpy()
            elif self.backend in ["xh2", "hmonnx"] and self.resizer_mode == 0:
                new_datas[in_name] = im.detach().cpu().numpy().astype(np.float16)
            elif self.backend == "xh1" and self.resizer_mode == 0:
                new_datas[in_name] = self.engine.quantize(
                    in_name, im.detach().cpu().numpy()
                )
            elif self.backend == "hmonnx" and self.resizer_mode in [1, 2, 3]:
                yuv_pad = im.detach().cpu()
                h, w, c = yuv_pad.shape
                new_datas[in_name] = yuv_pad.view(1, c, h, w).contiguous().numpy()
            elif self.resizer_mode in [1, 2, 3]:
                yuv_pad = im.detach().cpu().numpy().flatten()
                if toYUV_format == "YUV420SP":
                    valid_len = yuv_pad.size // 2
                elif toYUV_format == "YUV422SP":
                    valid_len = yuv_pad.size * 2 // 3
                elif toYUV_format in ["YUV444SP", "YUV400"]:
                    valid_len = yuv_pad.size
                yuv = yuv_pad[:valid_len].copy().reshape(1, -1)
                new_datas[in_name] = np.ascontiguousarray(yuv)
            if self.resizer_mode in [1, 2] and self.backend in ["xh1", "xh2"]:
                dyn_info = dyn_info.detach().cpu().numpy()
                new_datas[f"resizer_crop_{in_name}"] = dyn_info
            return new_datas

    def run(self, in_datas: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute model inference.

        Args:
            in_datas (dict): Dictionary containing input data arrays.

        Returns:
            dict: Dictionary containing output data arrays after inference.
        """
        prerpcessed_in_datas = self.preprocess(in_datas)
        # For multi-batch, directly duplicate data, and subsequent resizer information duplication
        for name in prerpcessed_in_datas:
            in_data = prerpcessed_in_datas[name]
            if (
                name.startswith("resizer_crop_")
                and self.roi_num > 1
                and self.backend in ["xh1", "xh2"]
            ):
                prerpcessed_in_datas[name] = np.repeat(
                    in_data, repeats=self.roi_num, axis=0
                )
                continue
            batch = self.engine.get_input_batch_size(name)
            prerpcessed_in_datas[name] = np.repeat(in_data, repeats=batch, axis=0)
        t = time.time()
        # Inference
        outs = self.engine.run(prerpcessed_in_datas)
        self.time_span += time.time() - t
        # XH1 outputs both quantized and dequantized results, only take the dequantized one
        if isinstance(outs, tuple):
            outs = outs[1]
        # Before postprocessing, only take batch 0
        for name in outs:
            out = outs[name][0:1, ...]
            outs[name] = out.copy()
        outs = self.postprocess(outs, in_datas)
        self.total += 1
        return outs

    @abc.abstractmethod
    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """Postprocess model outputs.

        Args:
            outs (dict): Dictionary containing raw output data arrays.
            in_datas (dict): Dictionary containing original input data arrays.

        Returns:
            Processed output results based on the specific implementation.
        """
        pass

    def unload(self):
        """Unload the model and clean up resources."""
        pass

    @abc.abstractmethod
    def demo(self, filepaths: list):
        """Run model demonstration.

        Args:
            filepaths (list): List of file paths for the demo.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, dataset, num=0):
        """Evaluate the model performance.

        Args:
            dataset: Dataset object for evaluation.
            num (int): Number of samples to evaluate (default: 0, meaning all).

        Returns:
            Evaluation results based on the specific implementation.
        """
        pass

    @property
    def ave_latency_ms(self):
        """float: Average latency in milliseconds for inference operations.

        Returns 0 if no inference operations have been performed.
        """
        if self.total == 0:
            return 0
        return (self.time_span / self.total) * 1000
