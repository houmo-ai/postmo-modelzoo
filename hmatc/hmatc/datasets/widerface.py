# Copyright 2025 HOUMO AI
#
# File: widerface.py
# Description:
#   WiderFace dataset
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
#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
from ..base.base_dataset import BaseDataset
from ..utils import logger


class WiderFace(BaseDataset):
    """
    WiderFace dataset class for face detection tasks.
    This class handles the loading and management of the WiderFace validation dataset.
    """

    def __init__(self, root_path):
        """
        Initialize the WiderFace dataset.

        Args:
            root_path (str): Root path of the WiderFace dataset directory
        """
        self._root_path = root_path
        if not os.path.exists(self._root_path):
            logger.fatal(f"root_path not exits -> {self._root_path}")

        self._list_file = os.path.join(self._root_path, "WIDER_val", "wider_val.txt")
        if not os.path.exists(self._list_file):
            logger.fatal(f"wider_val.txt not exits -> {self._list_file}")

        self._dataset_val_path = os.path.join(self._root_path, "WIDER_val", "images")
        if not os.path.exists(self._dataset_val_path):
            logger.fatal(f"image val not exits -> {self._dataset_val_path}")

        self._img_lists = list()
        self._img_relative_path = list()
        with open(self._list_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                subpath = line.strip()
                img_path = self._dataset_val_path + subpath
                if os.path.exists(img_path):
                    self._img_lists.append(img_path)
                    self._img_relative_path.append(subpath)
        self._total_num = len(self._img_lists)

        self._annotation_path = os.path.join(self._root_path, "ground_truth", "val")
        if not os.path.exists(self._annotation_path):
            logger.fatal(f"annotation_path not exits -> {self._annotation_path}")

    def get_next_batch(self):
        """
        Get the next batch of data.
        This method is currently not implemented.
        """
        pass

    def get_datas(self, num: int):
        """
        Get a specified number of image paths from the dataset.

        Args:
            num (int): Number of images to retrieve. If 0, returns all images.

        Returns:
            list: List of image file paths
        """
        if num == 0:
            num = self._total_num
        elif num > self._total_num:
            num = self._total_num

        img_paths = self._img_lists[0:num]
        return img_paths

    def get_relative_path(self, idx):
        """
        Get the relative path of an image at the specified index.

        Args:
            idx (int): Index of the image

        Returns:
            str: Relative path of the image
        """
        return self._img_relative_path[idx]

    @property
    def annotation_path(self):
        """
        Get the path to the annotation directory.

        Returns:
            str: Path to the ground truth annotation directory
        """
        return self._annotation_path

    @property
    def dataset_name(self):
        """
        Get the name of the dataset.

        Returns:
            str: Name of the dataset ("widerface")
        """
        return "widerface"
