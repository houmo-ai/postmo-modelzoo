# Copyright 2025 HOUMO AI
#
# File: coco.py
# Description:
#   COCO dataset class
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
from ..base.base_dataset import BaseDataset
from ..utils import logger


class COCO2017Val(BaseDataset):
    """COCO 2017 validation dataset class for object detection and segmentation tasks.

    This class provides access to the COCO 2017 validation dataset, which contains
    images with object annotations in various formats. It handles loading image
    paths and associated metadata from the annotation files.
    """

    def __init__(self, root_path):
        """Initialize the COCO 2017 validation dataset.

        Args:
            root_path (str): Root directory of the COCO 2017 dataset containing
                           val2017 images and annotations subdirectories.
        """
        self.root_path = root_path
        if not os.path.exists(self.root_path):
            logger.error("root_path not exits -> {}".format(self.root_path))
            exit(-1)

        self.annotations_file = os.path.join(
            self.root_path, "annotations", "instances_val2017.json"
        )
        self.annotations_kpt = os.path.join(
            self.root_path, "annotations", "person_keypoints_val2017.json"
        )
        if not os.path.exists(self.annotations_file):
            logger.error(f"annotations_file not exist -> {self.annotations_file}")
            exit(-1)

        with open(self.annotations_file, "r") as f:
            annotations = json.load(f)
        images = annotations["images"]

        self.img_files = list()
        self.image_ids = list()
        self.image_ids_dict = dict()
        for image in images:
            filename = image["file_name"]
            image_id = int(image["id"])
            img_path = os.path.join(self.root_path, "val2017", filename)
            if not os.path.exists(img_path):
                # logger.warning(f"img_path not exist -> {img_path}")
                continue
            basename, _ = os.path.splitext(os.path.basename(img_path))
            self.image_ids_dict[basename] = image_id
            self.image_ids.append(image_id)
            self.img_files.append(img_path)

        self.total_num = len(self.img_files)

    def get_image_id(self, filename):
        """Get the image ID for a given filename.

        Args:
            filename (str): The basename of the image file (without extension).

        Returns:
            int: The corresponding image ID from the COCO dataset.
        """
        return self.image_ids_dict[filename]

    def get_next_batch(self):
        """Get the next batch of data.

        This method is not implemented in the current version.
        """
        pass

    def get_datas(self, num: int):
        """Get a subset of data with specified number of samples.

        Args:
            num (int): Number of data samples to retrieve. If 0, returns all data.
                     If greater than total number of samples, returns all data.

        Returns:
            list: List of image file paths.
        """
        if num == 0:
            num = self.total_num
        elif num > self.total_num:
            num = self.total_num
        img_paths = self.img_files[0:num]
        return img_paths

    @property
    def dataset_name(self):
        """str: Name of the dataset."""
        return "coco_2017Val"
