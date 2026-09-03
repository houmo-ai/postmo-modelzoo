# Copyright 2025 HOUMO AI
#
# File: loaders.py
# Description:
#   Base classes for model input data loaders.
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
import torch
import numpy as np
from typing import List

from ..utils import logger
from ..utils.preprocess import (
    convert_bgr_to_yuv,
    default_preprocess,
    resizer_preprocess,
)
from ..utils.utils import SUPPORT_IMAGE_FORMATS, gen_random_data, load_npz


class BaseDataLoader:
    """Base class for model input data loaders."""

    def __init__(
        self,
        data_dir=None,
        model_cfg=None,
        inputs_cfg=None,
        stage=None,
        num=0,
        dataset=None,
    ):
        self.data_dir = data_dir
        self.model_cfg = model_cfg or {}
        self.inputs_cfg = inputs_cfg or {}
        self.stage = stage
        self.num = num or 0
        self.dataset = dataset

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, _index):
        raise NotImplementedError


class RandomDataLoader(BaseDataLoader):
    """Generate one random sample for arbitrary model inputs."""

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        inputs = {}
        hmonnx_inputs = {}
        meta = {"source": "random", "dyn_info": {}}
        for input_name, input_cfg in self.inputs_cfg.items():
            data = gen_random_data(
                input_cfg["shape"], input_cfg.get("dtype", "float32")
            )
            hmonnx_data, dyn_info = random_resizer_input(input_cfg)
            inputs[input_name] = data
            hmonnx_inputs[input_name] = hmonnx_data
            if dyn_info is not None and _to_numpy(dyn_info).size > 0:
                meta["dyn_info"][input_name] = dyn_info
        return {"inputs": inputs, "hmonnx_inputs": hmonnx_inputs, "meta": meta}


class NpzDataLoader(BaseDataLoader):
    """Load preprocessed model inputs from .npz files."""

    def __init__(
        self,
        data_dir=None,
        model_cfg=None,
        inputs_cfg=None,
        stage=None,
        num=0,
        dataset=None,
    ):
        super().__init__(data_dir, model_cfg, inputs_cfg, stage, num, dataset)
        self.files = _find_files(data_dir, [".npz"])
        if self.num > 0:
            self.files = self.files[: self.num]
        if not self.files:
            logger.fatal(f"Not found npz data in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        npz_data = load_npz(path)
        inputs = {}
        hmonnx_inputs = {}
        meta = {"path": path, "dyn_info": {}}
        for input_name in self.inputs_cfg:
            if input_name not in npz_data:
                logger.fatal(f"Input '{input_name}' not found in NPZ file: {path}")
            input_cfg = self.inputs_cfg[input_name]
            data = npz_data[input_name]
            inputs[input_name] = data
            hmonnx_inputs[input_name] = data
            if _get_resizer_mode(input_cfg) != 0:
                inputs[input_name] = preprocess_image_input(data, input_cfg)
                hmonnx_data, dyn_info = preprocess_resizer_input(data, input_cfg)
                hmonnx_inputs[input_name] = hmonnx_data
                if dyn_info is not None and _to_numpy(dyn_info).size > 0:
                    meta["dyn_info"][input_name] = _to_numpy(dyn_info)
        return {"inputs": inputs, "hmonnx_inputs": hmonnx_inputs, "meta": meta}


class ImageDataLoader(BaseDataLoader):
    """Load single-input image data and apply configured preprocessing."""

    def __init__(
        self,
        data_dir=None,
        model_cfg=None,
        inputs_cfg=None,
        stage=None,
        num=0,
        dataset=None,
    ):
        super().__init__(data_dir, model_cfg, inputs_cfg, stage, num, dataset)
        if len(self.inputs_cfg) != 1:
            logger.fatal("ImageDataLoader only supports single input models")
        self.input_name = next(iter(self.inputs_cfg))
        self.input_cfg = self.inputs_cfg[self.input_name]
        if self.input_cfg.get("data_format") is None:
            logger.fatal("ImageDataLoader requires model input data_format")
        self.files = _find_files(data_dir, SUPPORT_IMAGE_FORMATS)
        if self.num > 0:
            self.files = self.files[: self.num]
        if not self.files:
            logger.fatal(f"Not found image data in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image = cv2.imread(path)
        if image is None:
            logger.fatal(f"Failed to load image: {path}")
        data = preprocess_image_input(image, self.input_cfg)
        hmonnx_data = data
        meta = {"path": path, "image": image, "dyn_info": {}}
        if _get_resizer_mode(self.input_cfg) != 0:
            hmonnx_data, dyn_info = preprocess_resizer_input(image, self.input_cfg)
            if dyn_info is not None and _to_numpy(dyn_info).size > 0:
                meta["dyn_info"][self.input_name] = _to_numpy(dyn_info)
        if self.stage == "quant":
            data = _repeat_to_model_batch(data, self.input_cfg, self.input_name)
            hmonnx_data = _repeat_to_model_batch(
                hmonnx_data, self.input_cfg, self.input_name
            )
        return {
            "inputs": {self.input_name: data},
            "hmonnx_inputs": {self.input_name: hmonnx_data},
            "meta": meta,
        }

    def _preprocess(self, image):
        return preprocess_image_input(image, self.input_cfg)


def _is_sequence_dataset(dataset):
    return hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")


def _is_paths_labels_pair(datas):
    """Detect ImageNet-style get_datas() return value: (paths, labels)."""
    return (
        isinstance(datas, tuple)
        and len(datas) == 2
        and isinstance(datas[0], (list, tuple))
        and isinstance(datas[1], (list, tuple))
        and len(datas[0]) == len(datas[1])
        and (not datas[0] or not isinstance(datas[0][0], (list, tuple, dict)))
    )


def _get_datas_items(datas, total):
    """Normalize get_datas() payload into a list of raw records."""
    if _is_paths_labels_pair(datas):
        paths, labels = datas[0][:total], datas[1][:total]
        return list(zip(paths, labels))
    if not isinstance(datas, (list, tuple)):
        logger.fatal("Dataset.get_datas() must return a list/tuple of records")
    return list(datas)[:total]


def dataset_length(dataset, num=0):
    """Return Dataset length after applying num slicing."""
    if dataset is None:
        return 0
    if _is_sequence_dataset(dataset):
        total = len(dataset)
    elif hasattr(dataset, "get_datas"):
        datas = dataset.get_datas(0)
        total = len(datas[0]) if _is_paths_labels_pair(datas) else len(datas)
    else:
        logger.fatal("Dataset must implement __len__/__getitem__ or get_datas")
    return min(num, total) if num and num > 0 else total


def dataset_records(dataset, num=0):
    """Return raw Dataset records with transition support for get_datas()."""
    total = dataset_length(dataset, num)
    if _is_sequence_dataset(dataset):
        return [
            _normalize_dataset_record(dataset[index], dataset, index)
            for index in range(total)
        ]

    if hasattr(dataset, "get_datas"):
        items = _get_datas_items(dataset.get_datas(total), total)
        return [
            _normalize_dataset_record(value, dataset, index)
            for index, value in enumerate(items)
        ]

    logger.fatal("Dataset must implement __len__/__getitem__ or get_datas")


def _normalize_dataset_record(record, dataset, index):
    if isinstance(record, dict):
        normalized = dict(record)
    elif isinstance(record, (list, tuple)):
        if not record:
            logger.fatal("Dataset record is empty")
        normalized = {"path": record[0]}
        if len(record) >= 2:
            # ImageNet-style (path, label); other loaders may override keys later.
            normalized["label"] = record[1]
        if len(record) >= 3:
            normalized["path2"] = record[2]
        if len(record) >= 4:
            normalized["flow_path"] = record[3]
        if len(record) >= 5:
            normalized["valid_path"] = record[4]
    else:
        normalized = {"path": record}

    image_ids = getattr(dataset, "image_ids", None)
    if (
        "image_id" not in normalized
        and image_ids is not None
        and index < len(image_ids)
    ):
        normalized["image_id"] = image_ids[index]

    labels = getattr(dataset, "labels", None)
    if "label" not in normalized and labels is not None and index < len(labels):
        normalized["label"] = labels[index]

    if "relative_path" not in normalized and hasattr(dataset, "get_relative_path"):
        normalized["relative_path"] = dataset.get_relative_path(index)

    return normalized


def validate_sample(sample, inputs_cfg):
    """Validate and normalize a DataLoader sample."""
    if not isinstance(sample, dict):
        logger.fatal("DataLoader sample must be a dict")
    inputs = sample.get("inputs")
    if not isinstance(inputs, dict):
        logger.fatal("DataLoader sample['inputs'] must be a dict")

    hmonnx_inputs = sample.get("hmonnx_inputs") or inputs
    if not isinstance(hmonnx_inputs, dict):
        logger.fatal("DataLoader sample['hmonnx_inputs'] must be a dict")

    meta = sample.get("meta", {})
    return {
        "inputs": _normalize_inputs(inputs, inputs_cfg, "inputs", check_shape=False),
        "hmonnx_inputs": _normalize_inputs(hmonnx_inputs, inputs_cfg, "hmonnx_inputs"),
        "meta": meta,
    }


def _normalize_inputs(inputs, inputs_cfg, field_name, check_shape=True):
    expected = list(inputs_cfg.keys())
    missing = [name for name in expected if name not in inputs]
    extra = [name for name in inputs if name not in inputs_cfg]
    if missing:
        logger.fatal(f"DataLoader {field_name} missing inputs: {missing}")
    if extra:
        logger.fatal(f"DataLoader {field_name} has unknown inputs: {extra}")

    normalized = {}
    for name in expected:
        value = inputs[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if not isinstance(value, np.ndarray):
            logger.fatal(
                f"DataLoader {field_name}['{name}'] must be np.ndarray or torch.Tensor"
            )
        expected_shape = inputs_cfg[name].get("shape")
        if check_shape and expected_shape and len(value.shape) != len(expected_shape):
            logger.fatal(
                f"DataLoader input '{name}' rank mismatch: got {list(value.shape)}, expected {expected_shape}"
            )
        normalized[name] = value

    return normalized


def _find_files(data_dir, exts: List[str]):
    if data_dir is None:
        return []
    if os.path.isfile(data_dir):
        return [data_dir] if _has_ext(data_dir, exts) else []
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if _has_ext(name, exts)
    )


def _has_ext(path, exts: List[str]):
    return os.path.splitext(path)[1] in exts


def _get_resizer_mode(input_cfg):
    if input_cfg.get("data_format") is None or "resizer" not in input_cfg:
        return 0
    resizer_cfg = input_cfg.get("resizer") or {}
    return resizer_cfg.get("resizer_mode", 3)


def _get_resizer_cfg(input_cfg):
    return input_cfg.get("resizer") or {}


def _preprocess_for_resizer(image, input_cfg, resizer_mode):
    _, _, height, width = input_cfg["shape"]
    resizer_cfg = _get_resizer_cfg(input_cfg)
    to_yuv_format = resizer_cfg.get("toYUV_format", "YUV420SP")
    resizer_input_size = resizer_cfg.get("resizer_input_size", [height, width])
    resizer_input_h, resizer_input_w = resizer_input_size
    resizer_crop = resizer_cfg.get(
        "resizer_crop", [0, 0, resizer_input_h, resizer_input_w]
    )
    data, dyn_info = resizer_preprocess(
        image,
        input_cfg["shape"],
        resizer_input_size=resizer_input_size,
        resizer_crop=resizer_crop,
        resizer_mode=resizer_mode,
        mean=input_cfg["mean"],
        std=input_cfg["std"],
        use_resize=resizer_mode in [0, 3],
        use_norm=resizer_mode == 0,
        use_rgb=input_cfg["data_format"] == "RGB" and resizer_mode == 0,
        resize_type=input_cfg["resize_type"],
        padding_mode=input_cfg.get("padding_mode"),
        padding_values=input_cfg.get("padding_values"),
        is_onnx=resizer_mode == 0,
        to_YUV=resizer_mode in [1, 2, 3],
        fmt=to_yuv_format,
    )
    return data, dyn_info


def preprocess_image_input(image, input_cfg):
    """Preprocess raw image into original ONNX input layout."""
    _, _, height, width = input_cfg["shape"]
    return default_preprocess(
        _ensure_hwc_uint8(image),
        size=(width, height),
        mean=input_cfg["mean"],
        std=input_cfg["std"],
        use_norm=True,
        use_rgb=input_cfg["data_format"] == "RGB",
        use_resize=True,
        resize_type=input_cfg["resize_type"],
        padding_mode=input_cfg.get("padding_mode"),
        padding_value=input_cfg.get("padding_values"),
        to_YUV=False,
    )


def preprocess_resizer_input(image, input_cfg):
    """Preprocess raw image into hmonnx/hmm resizer input and dynamic params."""
    resizer_mode = _get_resizer_mode(input_cfg)
    if resizer_mode == 0:
        return preprocess_image_input(image, input_cfg), None
    data, dyn_info = _preprocess_for_resizer(
        _ensure_hwc_uint8(image), input_cfg, resizer_mode
    )
    return _to_numpy(data), _to_numpy(dyn_info)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _repeat_to_model_batch(data, input_cfg, input_name):
    """Repeat a single calibration image to the model's native batch size."""
    data = _to_numpy(data)
    expected_batch = input_cfg["shape"][0]
    actual_batch = data.shape[0]
    if actual_batch == expected_batch:
        return data
    if actual_batch != 1:
        logger.fatal(
            f"Calibration input '{input_name}' batch mismatch: "
            f"got {actual_batch}, expected {expected_batch}"
        )
    return np.repeat(data, expected_batch, axis=0)


def _ensure_hwc_uint8(data):
    data = _to_numpy(data)
    if data.dtype != np.uint8:
        data = data.astype(np.uint8)
    if len(data.shape) == 4:
        data = data[0]
    return data


def random_resizer_input(input_cfg):
    """Generate random pre-resizer image data and optional dynamic resizer params."""
    resizer_mode = _get_resizer_mode(input_cfg)
    shape = input_cfg["shape"]
    if resizer_mode == 0:
        return gen_random_data(shape, input_cfg.get("dtype", "float32")), None

    batch, channels, height, width = shape
    resizer_cfg = _get_resizer_cfg(input_cfg)
    to_yuv_format = resizer_cfg.get("toYUV_format", "YUV420SP")
    resizer_input_size = resizer_cfg.get("resizer_input_size", [height, width])
    resizer_input_h, resizer_input_w = resizer_input_size
    random_bgr = torch.from_numpy(
        gen_random_data([1, channels, resizer_input_h, resizer_input_w], "uint8")
    )
    random_yuv = convert_bgr_to_yuv(random_bgr, to_yuv_format, to_NCHW=True).numpy()
    random_yuv = np.repeat(random_yuv, batch, axis=0)

    if resizer_mode == 1:
        dyn_info = np.array(
            [[0, 0, resizer_input_h, resizer_input_w, height, width, 0, 0, 0, 0]],
            dtype=np.int32,
        )
    elif resizer_mode == 2:
        dyn_info = np.array([[0, 0, resizer_input_h, resizer_input_w]], dtype=np.int32)
    else:
        dyn_info = None
    return random_yuv, dyn_info
