# Copyright 2025 HOUMO AI
#
# File: imagenet.py
# Description:
#   DataLoader for ImageNet dataset.
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

from ..utils import logger
from ..utils.utils import SUPPORT_IMAGE_FORMATS
from .loaders import (
    BaseDataLoader,
    dataset_records,
    preprocess_image_input,
    preprocess_resizer_input,
)


class ImageNetDataLoader(BaseDataLoader):
    """Load ImageNet validation data as model-ready DataLoader samples."""

    def __init__(
        self, data_dir=None, model_cfg=None, inputs_cfg=None, stage=None, num=0, dataset=None
    ):
        super().__init__(data_dir, model_cfg, inputs_cfg, stage, num, dataset)
        if len(self.inputs_cfg) != 1:
            logger.fatal("ImageNetDataLoader only supports single-input models")

        self.input_name = next(iter(self.inputs_cfg))
        self.input_cfg = self.inputs_cfg[self.input_name]
        if self.input_cfg.get("data_format") is None:
            logger.fatal("ImageNetDataLoader requires model input data_format")

        if self.dataset is not None:
            self.samples = dataset_records(self.dataset, self.num)
            self.dataset_name = getattr(
                self.dataset, "dataset_name", self.dataset.__class__.__name__
            )
        else:
            self.img_dir, self.val_file = self._resolve_imagenet_paths(data_dir)
            self.samples = self._load_samples()
            if self.num > 0:
                self.samples = self.samples[: self.num]
            self.dataset_name = "ILSVRC_2012Val"
        if not self.samples:
            logger.fatal(f"Not found ImageNet eval data in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if isinstance(sample, dict):
            path = sample.get("path")
            label = sample.get("label")
        else:
            path, label = sample
        if not path:
            logger.fatal(f"ImageNetDataLoader sample missing path at index {index}")
        if label is None:
            logger.fatal(f"ImageNetDataLoader sample missing label at index {index}")
        image = cv2.imread(path)
        if image is None:
            logger.fatal(f"Failed to load image: {path}")

        data = preprocess_image_input(image, self.input_cfg)
        hmonnx_data = data
        meta = {
            "path": path,
            "label": label,
            "dataset": self.dataset_name,
            "dyn_info": {},
        }

        if self._has_resizer():
            hmonnx_data, dyn_info = preprocess_resizer_input(image, self.input_cfg)
            if dyn_info is not None and np.asarray(dyn_info).size > 0:
                meta["dyn_info"][self.input_name] = np.asarray(dyn_info)

        return {
            "inputs": {self.input_name: data},
            "hmonnx_inputs": {self.input_name: hmonnx_data},
            "meta": meta,
        }

    @staticmethod
    def matches(data_dir):
        if data_dir is None or not os.path.isdir(data_dir):
            return False
        _, val_file = ImageNetDataLoader._resolve_imagenet_paths(data_dir, fatal=False)
        return val_file is not None

    @staticmethod
    def _resolve_imagenet_paths(data_dir, fatal=True):
        candidates = [
            ("ILSVRC2012_img_val", "val.txt"),
            ("ILSVRC2015_img_val", "ILSVRC2015_val.txt"),
        ]
        for img_dir_name, val_name in candidates:
            img_dir = os.path.join(data_dir, img_dir_name)
            val_file = os.path.join(data_dir, val_name)
            if os.path.isdir(img_dir) and os.path.exists(val_file):
                return img_dir, val_file
        if fatal:
            logger.fatal(
                "ImageNetDataLoader requires an ImageNet root containing "
                "ILSVRC2012_img_val/val.txt or ILSVRC2015_img_val/ILSVRC2015_val.txt"
            )
        return None, None

    def _load_samples(self):
        samples = []
        with open(self.val_file, "r") as f:
            for line in f:
                filename, label = line.strip().split()
                if os.path.splitext(filename)[1] not in SUPPORT_IMAGE_FORMATS:
                    continue
                path = os.path.join(self.img_dir, filename)
                if os.path.exists(path):
                    samples.append((path, int(label)))
        return samples

    def _has_resizer(self):
        return (
            self.input_cfg.get("data_format") is not None
            and "resizer" in self.input_cfg
        )
