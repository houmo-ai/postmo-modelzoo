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
import numpy as np
from typing import Dict, Any
from ..utils import logger
from ..utils.utils import SUPPORT_BACKEND
from ..dataloaders.loaders import validate_sample
from ..infer.xh2_infer import Xh2Infer
from ..infer.onnx_infer import OnnxInfer
from ..infer.hmonnx_infer import HmonnxInfer

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
    """Base model class for DataLoader-driven inference."""

    def __init__(self, **kwargs):
        """Initialize the model instance with configuration parameters."""
        self.time_span = 0
        self.total = 0
        self.engine = None
        self.inputs_cfg = kwargs["inputs_cfg"]
        self.inputs_name = list(self.inputs_cfg.keys())
        self.resizer_modes = kwargs.get("resizer_modes", {})
        self.roi_num = kwargs.get("roi_num", 1)

        self.backend = kwargs["backend"]
        if self.backend not in SUPPORT_BACKEND:
            logger.fatal(f"backend not in {SUPPORT_BACKEND}")
        if self.backend == "onnx":
            self.engine = OnnxInfer()
        elif self.backend == "xh2":
            self.engine = Xh2Infer()
        elif self.backend == "hmonnx":
            self.engine = HmonnxInfer()
        else:
            logger.fatal(f"Not support backend: {self.backend}")

    def load(self, model_path: str, device_id=0):
        """Load the model for inference."""
        model_name = os.path.basename(model_path)
        _, ext = os.path.splitext(model_name)
        if ext != self.engine.model_ext:
            logger.fatal(f"{model_name} is not {self.engine.model_ext}")

        self.engine.load(model_path, device_id=device_id)

    def run(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Execute model inference with one standard DataLoader sample."""
        sample = validate_sample(sample, self.inputs_cfg)
        runtime_inputs = self._build_runtime_inputs(sample)
        runtime_inputs = self._repeat_runtime_inputs(runtime_inputs)

        t = time.time()
        outs = self.engine.run(runtime_inputs)
        self.time_span += time.time() - t

        # xh2 returns (quantized, dequantized); model postprocess uses dequantized data.
        if isinstance(outs, tuple):
            outs = outs[1]

        for name in outs:
            outs[name] = outs[name][0:1, ...].copy()

        outs = self.postprocess(outs, self._get_postprocess_inputs(sample))
        self.total += 1
        return outs

    def _build_runtime_inputs(self, sample):
        if self.backend == "onnx":
            return {name: sample["inputs"][name] for name in self.inputs_name}
        if self.backend == "hmonnx":
            return self._build_hmonnx_inputs(sample)
        if self.backend == "xh2":
            return self._build_xh2_inputs(sample)
        logger.fatal(f"Not support backend: {self.backend}")

    def _build_hmonnx_inputs(self, sample):
        inputs = {}
        hmonnx_inputs = sample["hmonnx_inputs"]
        dyn_info = sample.get("meta", {}).get("dyn_info", {}) or {}

        for input_name in self.inputs_name:
            data = hmonnx_inputs[input_name]
            if self.resizer_modes.get(input_name, 0) == 0:
                data = self._cast_runtime_input(data)
            elif data.dtype == np.float32:
                data = data.astype(np.float16)
            inputs[input_name] = np.ascontiguousarray(data)

            if input_name in dyn_info:
                inputs[f"resizer_crop_{input_name}"] = np.ascontiguousarray(
                    dyn_info[input_name].astype(np.int32)
                )

        return inputs

    def _build_xh2_inputs(self, sample):
        inputs = {}
        hmonnx_inputs = sample["hmonnx_inputs"]
        dyn_info = sample.get("meta", {}).get("dyn_info", {}) or {}

        for input_name in self.inputs_name:
            resizer_mode = self.resizer_modes.get(input_name, 0)
            data = hmonnx_inputs[input_name]
            if resizer_mode == 0:
                inputs[input_name] = np.ascontiguousarray(
                    self._cast_runtime_input(data)
                )
                continue

            fmt = self._get_yuv_format(input_name)
            yuv = data.astype(np.float16).flatten()
            valid_len = self._get_yuv_valid_len(yuv.size, fmt)
            inputs[input_name] = np.ascontiguousarray(yuv[:valid_len].reshape(1, -1))

            if resizer_mode in [1, 2]:
                if input_name not in dyn_info:
                    logger.fatal(
                        f"Missing dynamic resizer params for input: {input_name}"
                    )
                inputs[f"resizer_crop_{input_name}"] = np.ascontiguousarray(
                    dyn_info[input_name].astype(np.int32)
                )

        return inputs

    def _get_postprocess_inputs(self, sample):
        meta = sample.get("meta", {}) or {}
        if isinstance(meta.get("raw_inputs"), dict):
            return meta["raw_inputs"]
        if "image" in meta and len(self.inputs_name) == 1:
            return {self.inputs_name[0]: meta["image"]}
        return sample

    def _repeat_runtime_inputs(self, runtime_inputs):
        repeated = {}
        for name, data in runtime_inputs.items():
            if (
                name.startswith("resizer_crop_")
                and self.backend == "xh2"
                and self.roi_num > 1
            ):
                repeated[name] = np.repeat(data, repeats=self.roi_num, axis=0)
                continue

            batch = self.engine.get_input_batch_size(name)
            if data.shape[0] == batch:
                repeated[name] = data
                continue
            if batch % data.shape[0] != 0:
                logger.fatal(
                    f"Input '{name}' batch mismatch: data batch {data.shape[0]}, model batch {batch}"
                )
            repeated[name] = np.repeat(data, repeats=batch // data.shape[0], axis=0)
        return repeated

    def _get_yuv_format(self, input_name):
        resizer_cfg = self.inputs_cfg[input_name].get("resizer") or {}
        return resizer_cfg.get("toYUV_format", "YUV420SP")

    @staticmethod
    def _get_yuv_valid_len(size, fmt):
        if fmt == "YUV420SP":
            return size // 2
        if fmt == "YUV422SP":
            return size * 2 // 3
        if fmt in ["YUV444SP", "YUV400"]:
            return size
        logger.fatal(f"Unsupported YUV format: {fmt}")

    @staticmethod
    def _cast_runtime_input(data):
        if data.dtype == np.int64:
            return data.astype(np.int32)
        if data.dtype == np.float32:
            return data.astype(np.float16)
        return data

    @abc.abstractmethod
    def postprocess(self, outs: Dict[str, np.ndarray], in_datas: Dict[str, Any]) -> Any:
        """Postprocess model outputs."""
        pass

    def unload(self):
        """Unload the model and clean up resources."""
        pass

    @abc.abstractmethod
    def demo(self, dataloader):
        """Run model demonstration with a DataLoader."""
        pass

    @abc.abstractmethod
    def evaluate(self, dataloader, num=0):
        """Evaluate model performance with a DataLoader."""
        pass

    @property
    def ave_latency_ms(self):
        """float: Average latency in milliseconds for inference operations.

        Returns 0 if no inference operations have been performed.
        """
        if self.total == 0:
            return 0
        return (self.time_span / self.total) * 1000
