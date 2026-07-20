# Copyright 2025 HOUMO AI
#
# File: ccpd.py
# Description:
#   DataLoader for CCPD dataset.
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
import glob
import os
import cv2
import numpy as np

from ..utils import logger
from .loaders import (
    BaseDataLoader,
    dataset_records,
    preprocess_image_input,
    preprocess_resizer_input,
)


class CCPDImageDataLoader(BaseDataLoader):
    """Load CCPD-style OCR images as model-ready samples."""

    def __init__(
        self, data_dir=None, model_cfg=None, inputs_cfg=None, stage=None, num=0, dataset=None
    ):
        super().__init__(data_dir, model_cfg, inputs_cfg, stage, num, dataset)
        if len(self.inputs_cfg) != 1:
            logger.fatal("CCPDImageDataLoader only supports single-input models")
        self.input_name = next(iter(self.inputs_cfg))
        self.input_cfg = self.inputs_cfg[self.input_name]
        self.data_lines = []
        self.img_dir = data_dir
        if self.dataset is not None:
            self.samples = dataset_records(self.dataset, self.num)
            self.data_lines = list(getattr(self.dataset, "data_lines", []) or [])
            self.img_dir = getattr(self.dataset, "img_dir", data_dir)
            self.samples = self._attach_labels_from_data_lines(self.samples)
            self.dataset_name = getattr(
                self.dataset, "dataset_name", self.dataset.__class__.__name__
            )
        else:
            self.samples = self._load_samples(data_dir)
            if self.num > 0:
                self.samples = self.samples[: self.num]
            self.dataset_name = self._dataset_name(data_dir)
        if not self.samples:
            logger.fatal(f"Not found CCPD data in {data_dir}")

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
            logger.fatal(f"CCPDImageDataLoader sample missing path at index {index}")
        image = cv2.imread(path)
        if image is None:
            logger.fatal(f"Failed to load image: {path}")
        data = preprocess_image_input(image, self.input_cfg)
        hmonnx_data = data
        meta = {"path": path, "image": image, "label": label, "dyn_info": {}}
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
        if data_dir is None:
            return False
        return (
            os.path.exists(os.path.join(data_dir, "PPOCR", "val", "rec.txt"))
            or os.path.exists(os.path.join(data_dir, "PPOCR", "val", "det.txt"))
            or bool(glob.glob(os.path.join(data_dir, "*.jpg")))
        )

    def _attach_labels_from_data_lines(self, samples):
        """Fill missing labels from PPOCR-style data_lines (file\\tlabel)."""
        if not self.data_lines:
            return samples

        label_map = {}
        for line in self.data_lines:
            text = line.decode("utf-8") if isinstance(line, bytes) else str(line)
            parts = text.strip("\n").split("\t")
            if len(parts) < 2:
                continue
            file_name, label = parts[0], parts[1]
            label_map[file_name] = label
            label_map[os.path.basename(file_name)] = label

        enriched = []
        for sample in samples:
            if not isinstance(sample, dict):
                path, label = sample if isinstance(sample, (list, tuple)) else (sample, None)
                sample = {"path": path, "label": label}
            else:
                sample = dict(sample)
            if sample.get("label") is None:
                path = sample.get("path") or ""
                label = label_map.get(path) or label_map.get(os.path.basename(path))
                if label is None:
                    for key, value in label_map.items():
                        if path.endswith(key):
                            label = value
                            break
                sample["label"] = label
            enriched.append(sample)
        return enriched

    def _load_samples(self, data_dir):
        model_name = str(self.model_cfg.get("name", "")).lower()
        if "det" in model_name and os.path.exists(
            os.path.join(data_dir, "PPOCR", "val", "det.txt")
        ):
            return self._load_ppocr_det(data_dir)
        if os.path.exists(os.path.join(data_dir, "PPOCR", "val", "rec.txt")):
            return self._load_ppocr_rec(data_dir)
        if os.path.exists(os.path.join(data_dir, "PPOCR", "val", "det.txt")):
            return self._load_ppocr_det(data_dir)
        return [
            (path, None) for path in sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        ]

    def _load_ppocr_rec(self, data_dir):
        label_file = os.path.join(data_dir, "PPOCR", "val", "rec.txt")
        img_dir = os.path.join(data_dir, "PPOCR")
        self.img_dir = img_dir
        samples = []
        with open(label_file, "rb") as f:
            self.data_lines = f.readlines()
        for line in self.data_lines:
            file_name, label = line.decode("utf-8").strip("\n").split("\t")[:2]
            path = os.path.join(img_dir, file_name)
            if os.path.exists(path):
                samples.append((path, label))
        return samples

    def _load_ppocr_det(self, data_dir):
        label_file = os.path.join(data_dir, "PPOCR", "val", "det.txt")
        img_dir = os.path.join(data_dir, "ccpd_green")
        self.img_dir = img_dir
        samples = []
        with open(label_file, "rb") as f:
            self.data_lines = f.readlines()
        for line in self.data_lines:
            file_name, label = line.decode("utf-8").strip("\n").split("\t")[:2]
            path = os.path.join(img_dir, file_name)
            if os.path.exists(path):
                samples.append((path, label))
        return samples

    @staticmethod
    def _dataset_name(data_dir):
        if os.path.exists(os.path.join(data_dir, "PPOCR", "val", "rec.txt")):
            return "CCPD2020ValRec"
        if os.path.exists(os.path.join(data_dir, "PPOCR", "val", "det.txt")):
            return "CCPD2020Val"
        return "CCPD2019Sub"

    def _has_resizer(self):
        return (
            self.input_cfg.get("data_format") is not None
            and "resizer" in self.input_cfg
        )
