# Copyright 2025 HOUMO AI
#
# File: coco.py
# Description:
#   DataLoader for COCO dataset.
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


class CocoDataLoader(BaseDataLoader):
    """Load COCO val images as model-ready DataLoader samples."""

    def __init__(
        self, data_dir=None, model_cfg=None, inputs_cfg=None, stage=None, num=0, dataset=None
    ):
        super().__init__(data_dir, model_cfg, inputs_cfg, stage, num, dataset)
        if len(self.inputs_cfg) != 1:
            logger.fatal("CocoDataLoader only supports single-input models")

        self.input_name = next(iter(self.inputs_cfg))
        self.input_cfg = self.inputs_cfg[self.input_name]
        if self.input_cfg.get("data_format") is None:
            logger.fatal("CocoDataLoader requires model input data_format")

        if self.dataset is not None:
            self.samples = dataset_records(self.dataset, self.num)
            self.annotations_file = getattr(self.dataset, "annotations_file", None)
            self.annotations_kpt = getattr(self.dataset, "annotations_kpt", None)
            self.image_ids = []
            for idx, sample in enumerate(self.samples):
                image_id = sample.get("image_id") if isinstance(sample, dict) else None
                if image_id is None:
                    logger.fatal(
                        f"CocoDataLoader sample missing image_id at index {idx}"
                    )
                self.image_ids.append(image_id)
            self.image_ids_dict = getattr(self.dataset, "image_ids_dict", {})
            self.dataset_name = getattr(
                self.dataset, "dataset_name", self.dataset.__class__.__name__
            )
        else:
            self.img_dir = os.path.join(data_dir, "val2017")
            self.annotations_file = os.path.join(
                data_dir, "annotations", "instances_val2017.json"
            )
            self.annotations_kpt = os.path.join(
                data_dir, "annotations", "person_keypoints_val2017.json"
            )
            if not os.path.isdir(self.img_dir):
                logger.fatal(f"COCO val image dir not found -> {self.img_dir}")
            if not os.path.exists(self.annotations_file):
                logger.fatal(
                    f"COCO annotations file not found -> {self.annotations_file}"
                )

            self.samples, self.image_ids, self.image_ids_dict = self._load_samples()
            if self.num > 0:
                self.samples = self.samples[: self.num]
                self.image_ids = [sample[1] for sample in self.samples]
            self.dataset_name = "coco_2017Val"

        if not self.samples:
            logger.fatal(f"Not found COCO eval data in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if isinstance(sample, dict):
            path = sample.get("path")
            image_id = sample.get("image_id")
        else:
            path, image_id = sample
        if not path:
            logger.fatal(f"CocoDataLoader sample missing path at index {index}")
        if image_id is None:
            logger.fatal(f"CocoDataLoader sample missing image_id at index {index}")
        image = cv2.imread(path)
        if image is None:
            logger.fatal(f"Failed to load image: {path}")

        data = preprocess_image_input(image, self.input_cfg)
        hmonnx_data = data
        meta = {
            "path": path,
            "image": image,
            "image_id": image_id,
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
        return os.path.isdir(os.path.join(data_dir, "val2017")) and os.path.exists(
            os.path.join(data_dir, "annotations", "instances_val2017.json")
        )

    def get_image_id(self, filename):
        return self.image_ids_dict[filename]

    def _load_samples(self):
        import json

        with open(self.annotations_file, "r") as f:
            annotations = json.load(f)

        samples = []
        image_ids = []
        image_ids_dict = {}
        for image in annotations["images"]:
            filename = image["file_name"]
            image_id = int(image["id"])
            path = os.path.join(self.img_dir, filename)
            if os.path.splitext(filename)[1] not in SUPPORT_IMAGE_FORMATS:
                continue
            if not os.path.exists(path):
                continue
            basename, _ = os.path.splitext(os.path.basename(path))
            image_ids_dict[basename] = image_id
            image_ids.append(image_id)
            samples.append((path, image_id))
        return samples, image_ids, image_ids_dict

    def _has_resizer(self):
        return (
            self.input_cfg.get("data_format") is not None
            and "resizer" in self.input_cfg
        )
