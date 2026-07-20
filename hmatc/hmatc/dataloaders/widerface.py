# Copyright 2025 HOUMO AI
#
# File: widerface.py
# Description:
#   DataLoader for WiderFace face detection dataset.
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


class WiderFaceDataLoader(BaseDataLoader):
    """Load WiderFace validation images as model-ready samples."""

    def __init__(
        self, data_dir=None, model_cfg=None, inputs_cfg=None, stage=None, num=0, dataset=None
    ):
        super().__init__(data_dir, model_cfg, inputs_cfg, stage, num, dataset)
        if len(self.inputs_cfg) != 1:
            logger.fatal("WiderFaceDataLoader only supports single-input models")
        self.input_name = next(iter(self.inputs_cfg))
        self.input_cfg = self.inputs_cfg[self.input_name]
        if self.dataset is not None:
            self.samples = dataset_records(self.dataset, self.num)
            self.annotation_path = getattr(self.dataset, "annotation_path", None)
            self.dataset_name = getattr(
                self.dataset, "dataset_name", self.dataset.__class__.__name__
            )
        else:
            self.image_root = os.path.join(data_dir, "WIDER_val", "images")
            self.list_file = os.path.join(data_dir, "WIDER_val", "wider_val.txt")
            self.annotation_path = os.path.join(data_dir, "ground_truth", "val")
            for path in [self.image_root, self.list_file, self.annotation_path]:
                if not os.path.exists(path):
                    logger.fatal(f"WiderFace path not found -> {path}")
            self.samples = self._load_samples()
            if self.num > 0:
                self.samples = self.samples[: self.num]
            self.dataset_name = "widerface"
        if not self.samples:
            logger.fatal(f"Not found WiderFace data in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if isinstance(sample, dict):
            path = sample.get("path")
            rel_path = sample.get("relative_path", sample.get("path", ""))
        else:
            path, rel_path = sample
        if not path:
            logger.fatal(f"WiderFaceDataLoader sample missing path at index {index}")
        image = cv2.imread(path)
        if image is None:
            logger.fatal(f"Failed to load image: {path}")
        data = preprocess_image_input(image, self.input_cfg)
        hmonnx_data = data
        meta = {"path": path, "relative_path": rel_path, "image": image, "dyn_info": {}}
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
        return data_dir is not None and os.path.exists(
            os.path.join(data_dir, "WIDER_val", "wider_val.txt")
        )

    def _load_samples(self):
        samples = []
        with open(self.list_file, "r") as f:
            for line in f:
                rel_path = line.strip()
                path = self.image_root + rel_path
                if os.path.splitext(path)[
                    1
                ] in SUPPORT_IMAGE_FORMATS and os.path.exists(path):
                    samples.append((path, rel_path))
        return samples

    def _has_resizer(self):
        return (
            self.input_cfg.get("data_format") is not None
            and "resizer" in self.input_cfg
        )
