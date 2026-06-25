# Copyright 2025 HOUMO AI
#
# File: sam2_engine.py
# Description:
#   Sam2 Segmentation Engine implementation
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

import cv2
import numpy as np
from coco_segment_dataset import CocoSegmentDataset

try:
    import tcim_lite as tcim  # type: ignore[import-not-found]
except ImportError:
    tcim = None

try:
    import onnxruntime as ort  # type: ignore[import-not-found]
except ImportError:
    ort = None


HMM_MODEL_NOT_LOADED = "HMM model has not been loaded"


class BaseRuntimeModel:
    """单模型推理抽象类，infer 内部统一执行 set_input、run、get_output。"""

    def __init__(self, output_names):
        self.output_names = output_names

    def load(self, model_path):
        raise NotImplementedError

    def set_input(self, name, data):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def get_output(self, name):
        raise NotImplementedError

    def infer(self, inputs):
        for name, data in inputs.items():
            self.set_input(name, data)
        self.run()
        return {name: self.get_output(name) for name in self.output_names}


class HMMRuntimeModel(BaseRuntimeModel):
    """HMM 模型推理封装。"""

    def __init__(self, output_names, device_id=0):
        super().__init__(output_names)
        self.device_id = device_id
        self.model = None

    def load(self, model_path):
        if tcim is None:
            raise ImportError("Please install tcim_lite before using the HMM backend")
        self.model = tcim.runtime.load(
            model_path,
            tcim.runtime.Option(tcim.runtime.WeightManager(self.device_id)),
        )
        return self

    def set_input(self, name, data):
        if self.model is None:
            raise RuntimeError(HMM_MODEL_NOT_LOADED)
        self.model.set_input(name, data.astype(np.float16))

    def run(self):
        if self.model is None:
            raise RuntimeError(HMM_MODEL_NOT_LOADED)
        self.model.run()
        self.model.sync()

    def get_output(self, name):
        if self.model is None:
            raise RuntimeError(HMM_MODEL_NOT_LOADED)
        return self.model.get_output(name).numpy()


class ONNXRuntimeModel(BaseRuntimeModel):
    """ONNX Runtime 模型推理封装，与 HMMRuntimeModel 保持同样接口。"""

    def __init__(self, output_names, providers=None):
        super().__init__(output_names)
        self.providers = providers
        self.session = None
        self.inputs = {}
        self.outputs = {}
        self.input_dtypes = {}

    def load(self, model_path):
        if ort is None:
            raise ImportError("Please install onnxruntime before using the ONNX backend")
        providers = self.providers
        if providers is None:
            providers = ["CPUExecutionProvider"]
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_dtypes = {
            model_input.name: model_input.type for model_input in self.session.get_inputs()
        }
        return self

    def set_input(self, name, data):
        dtype_name = self.input_dtypes.get(name, "")
        if "float16" in dtype_name:
            self.inputs[name] = data.astype(np.float16)
        elif "float64" in dtype_name:
            self.inputs[name] = data.astype(np.float64)
        elif "int64" in dtype_name:
            self.inputs[name] = data.astype(np.int64)
        elif "int32" in dtype_name:
            self.inputs[name] = data.astype(np.int32)
        elif "uint8" in dtype_name:
            self.inputs[name] = data.astype(np.uint8)
        else:
            self.inputs[name] = data.astype(np.float32)

    def run(self):
        if self.session is None:
            raise RuntimeError("ONNX model has not been loaded")
        output_values = self.session.run(None, self.inputs)
        output_names = [output.name for output in self.session.get_outputs()]
        self.outputs = dict(zip(output_names, output_values))
        self.inputs = {}

    def get_output(self, name):
        return self.outputs[name]


class SAM2Engine:
    """SAM2 encoder/decoder 组合模型，支持 HMM 和 ONNX 两种后端。"""

    ENCODER_OUTPUTS = ["high0", "high1", "image_embed"]
    DECODER_OUTPUTS = ["masks", "iou_preds"]

    def __init__(self, backend="hmm", target_size=640, device_id=0):
        self.backend = backend
        self.target_size = target_size
        self.device_id = device_id
        self.encoder = None
        self.decoder = None

    def _create_runtime_model(self, output_names):
        if self.backend in ("hmm", "xh2"):
            return HMMRuntimeModel(output_names, device_id=self.device_id)
        if self.backend == "onnx":
            return ONNXRuntimeModel(output_names)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def load(self, encoder_path, decoder_path):
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"encoder model not found: {encoder_path}")
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(f"decoder model not found: {decoder_path}")

        print(f"[info] Backend: {self.backend}")
        print(f"[info] Loading encoder model: {encoder_path}")
        print(f"[info] Loading decoder model: {decoder_path}")
        self.encoder = self._create_runtime_model(self.ENCODER_OUTPUTS).load(encoder_path)
        self.decoder = self._create_runtime_model(self.DECODER_OUTPUTS).load(decoder_path)
        return self

    def preprocess(self, image):
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        im_resized = cv2.resize(image, (new_w, new_h))
        im_padded = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        im_padded[:new_h, :new_w] = im_resized

        im_rgb = cv2.cvtColor(im_padded, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        im_rgb = (im_rgb - mean) / std
        im_rgb = np.transpose(im_rgb, (2, 0, 1))[np.newaxis, ...]
        return im_rgb, scale, new_h, new_w, h, w

    def encode(self, image_tensor):
        if self.encoder is None:
            raise RuntimeError("SAM2 encoder has not been loaded")
        return self.encoder.infer({"image": image_tensor})

    def decode(self, image_features, point_coords, point_labels):
        if self.decoder is None:
            raise RuntimeError("SAM2 decoder has not been loaded")
        return self.decoder.infer({
            "image_embed": image_features["image_embed"],
            "high0": image_features["high0"],
            "high1": image_features["high1"],
            "point_coords": point_coords,
            "point_labels": point_labels,
        })

    def postprocess(self, decoder_outs, new_h, new_w, orig_h, orig_w):
        masks = decoder_outs["masks"]
        mask_logits = masks[0, 0].astype(np.float32)
        mask_logits = np.clip(mask_logits, -20.0, 20.0)
        prob = 1.0 / (1.0 + np.exp(-mask_logits))
        score = 1.0
        if "iou_preds" in decoder_outs:
            score = float(np.asarray(decoder_outs["iou_preds"]).reshape(-1)[0])
        return post_process_mask(prob, new_h, new_w, orig_h, orig_w), score

    def infer(self, image, point_coords, point_labels):
        image_tensor, _, new_h, new_w, orig_h, orig_w = self.preprocess(image)
        image_features = self.encode(image_tensor)
        decoder_outs = self.decode(image_features, point_coords, point_labels)
        return self.postprocess(decoder_outs, new_h, new_w, orig_h, orig_w)

    def eval(self, dataset_dir, num=0, max_ann_per_image=0, output_dir=None):
        """Evaluate SAM2 on COCO val2017 segmentation annotations."""
        dataset = CocoSegmentDataset(dataset_dir, num, max_ann_per_image, output_dir)
        return dataset.eval(self, num=num)


def post_process_mask(prob, new_h, new_w, orig_h, orig_w):
    """后处理 mask。"""
    thresh_high = 0.5
    thresh_low = 0.4
    open_kernel = 3
    close_kernel = 5
    min_area_ratio = 0.05

    mask_low = (prob > thresh_low).astype(np.uint8) * 255
    mask_high = (prob > thresh_high).astype(np.uint8) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    mask_open = cv2.morphologyEx(mask_high, cv2.MORPH_OPEN, kernel_open)

    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(mask_open, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        total_area = np.sum(areas)
        min_area = total_area * min_area_ratio
        mask_clean = np.zeros_like(mask_open)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask_clean[labels_map == i] = 255
    else:
        mask_clean = mask_open

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_dilated = cv2.dilate(mask_clean, kernel_dilate, iterations=1)
    mask_restored = mask_low & mask_dilated

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    mask_final = cv2.morphologyEx(mask_restored, cv2.MORPH_CLOSE, kernel_close)

    mask_cropped = mask_final[:new_h, :new_w]
    mask_full = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_full

