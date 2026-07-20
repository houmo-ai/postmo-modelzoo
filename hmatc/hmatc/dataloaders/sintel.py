# Copyright 2025 HOUMO AI
#
# File: sintel.py
# Description:
#   DataLoader for Sintel optical-flow dataset.
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
from .loaders import BaseDataLoader, dataset_records


class SintelDataLoader(BaseDataLoader):
    """Load Sintel optical-flow image pairs as model-ready samples."""

    def __init__(
        self, data_dir=None, model_cfg=None, inputs_cfg=None, stage=None, num=0, dataset=None
    ):
        super().__init__(data_dir, model_cfg, inputs_cfg, stage, num, dataset)
        if len(self.inputs_cfg) != 2:
            logger.fatal("SintelDataLoader requires two model inputs")
        self.input_names = list(self.inputs_cfg.keys())
        self.input_h, self.input_w = self.inputs_cfg[self.input_names[0]]["shape"][2:4]
        if self.dataset is not None:
            self.samples = self._samples_from_dataset(self.dataset, self.num)
            self.dataset_name = getattr(
                self.dataset, "dataset_name", self.dataset.__class__.__name__
            )
        else:
            self.dataset_name = "sintel_clean"
            self.samples = self._load_samples(data_dir, stage)
            if self.num > 0:
                self.samples = self.samples[: self.num]
        if not self.samples:
            logger.fatal(f"Not found Sintel data in {data_dir}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _samples_from_dataset(dataset, num):
        """Prefer full optical-flow pairs; fall back to generic dataset records."""
        if hasattr(dataset, "get_pairs"):
            pairs = dataset.get_pairs(num)
        elif getattr(dataset, "pairs", None) is not None:
            pairs = dataset.pairs if not num or num <= 0 else dataset.pairs[:num]
        else:
            return dataset_records(dataset, num)

        samples = []
        for pair in pairs:
            if isinstance(pair, dict):
                samples.append(pair)
                continue
            if len(pair) == 3:
                path1, path2, flow_path = pair
                samples.append((path1, path2, flow_path, None))
            else:
                samples.append(tuple(pair))
        return samples

    def __getitem__(self, index):
        sample = self.samples[index]
        if isinstance(sample, dict):
            path1 = sample.get("path")
            path2 = sample.get("path2")
            flow_path = sample.get("flow_path")
            valid_path = sample.get("valid_path")
        else:
            path1, path2, flow_path, valid_path = sample
        if not path1:
            logger.fatal(f"SintelDataLoader sample missing path at index {index}")
        if not path2:
            logger.fatal(f"SintelDataLoader sample missing path2 at index {index}")
        image1 = cv2.imread(path1)
        image2 = cv2.imread(path2)
        if image1 is None or image2 is None:
            logger.fatal(f"Failed to load image pair: {path1}, {path2}")

        data1 = self._preprocess(image1)
        data2 = self._preprocess(image2)
        inputs = {self.input_names[0]: data1, self.input_names[1]: data2}
        hmonnx_inputs = {
            self.input_names[0]: data1.astype(np.float16),
            self.input_names[1]: data2.astype(np.float16),
        }
        return {
            "inputs": inputs,
            "hmonnx_inputs": hmonnx_inputs,
            "meta": {
                "path": path1,
                "path2": path2,
                "flow_path": flow_path,
                "valid_path": valid_path,
                "raw_inputs": {
                    self.input_names[0]: image1,
                    self.input_names[1]: image2,
                },
                "dyn_info": {},
            },
        }

    @staticmethod
    def matches(data_dir):
        if data_dir is None or not os.path.isdir(data_dir):
            return False
        return os.path.isdir(os.path.join(data_dir, "clean")) and os.path.isdir(
            os.path.join(data_dir, "flow")
        )

    @staticmethod
    def load_flow(flow_path):
        return read_flo(flow_path)

    @staticmethod
    def load_valid(valid_path):
        if valid_path and os.path.exists(valid_path):
            return cv2.imread(valid_path, cv2.IMREAD_GRAYSCALE)
        return None

    def _load_samples(self, data_dir, stage):
        if self.matches(data_dir):
            return self._load_sintel_pairs(data_dir)
        return self._load_demo_pairs(data_dir) if stage == "demo" else []

    def _load_sintel_pairs(self, data_dir):
        pairs = []
        image_dir = os.path.join(data_dir, "clean")
        flow_dir = os.path.join(data_dir, "flow")
        scenes = sorted(
            d
            for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d))
        )
        for scene in scenes:
            scene_image_dir = os.path.join(image_dir, scene)
            scene_flow_dir = os.path.join(flow_dir, scene)
            if not os.path.exists(scene_flow_dir):
                continue
            frames = sorted(
                f for f in os.listdir(scene_image_dir) if f.endswith(".png")
            )
            for i in range(len(frames) - 1):
                flow_path = os.path.join(
                    scene_flow_dir, frames[i].replace(".png", ".flo")
                )
                if os.path.exists(flow_path):
                    pairs.append(
                        (
                            os.path.join(scene_image_dir, frames[i]),
                            os.path.join(scene_image_dir, frames[i + 1]),
                            flow_path,
                            None,
                        )
                    )
        return pairs

    @staticmethod
    def _load_demo_pairs(data_dir):
        files = sorted(
            os.path.join(data_dir, name)
            for name in os.listdir(data_dir)
            if os.path.splitext(name)[1] in [".jpg", ".jpeg", ".png", ".bmp"]
        )
        pairs = []
        for idx in range(0, len(files) - 1, 2):
            pairs.append((files[idx], files[idx + 1], None, None))
        return pairs

    def _preprocess(self, image):
        if image.shape[:2] != (self.input_h, self.input_w):
            image = cv2.resize(image, (self.input_w, self.input_h))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_chw = np.transpose(rgb_image.astype(np.float32), (2, 0, 1))
        return image_chw[np.newaxis, ...]


def read_flo(path):
    with open(path, "rb") as f:
        magic = np.frombuffer(f.read(4), dtype=np.float32)[0]
        if magic != 202021.25:
            logger.fatal(f"Invalid .flo file: {path}")
        width = np.frombuffer(f.read(4), dtype=np.int32)[0]
        height = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.frombuffer(f.read(2 * width * height * 4), dtype=np.float32)
        return data.reshape(2, height, width)
