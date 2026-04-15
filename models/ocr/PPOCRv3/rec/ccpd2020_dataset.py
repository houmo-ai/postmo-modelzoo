#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: dataset.py
# Description:
#   CCPD2020 dataset for ppocrv3 recognition model to evaluate.
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
import glob
from hmatc.utils import logger
from hmatc.base.base_dataset import BaseDataset


class CCPD2020DataSet(BaseDataset):
    """CCPD2020 dataset for OCR recognition evaluation."""

    def __init__(self, root_path):
        """Initialize dataset with root path.

        Args:
            root_path (str): Root directory containing the dataset.
        """
        self.root_path = root_path
        if not os.path.exists(self.root_path):
            logger.fatal(f"root_path not exists -> {self.root_path}")

        # Dataset paths for recognition
        self.labels_file = os.path.join(self.root_path, "PPOCR/val/rec.txt")
        self.img_dir = os.path.join(self.root_path, "PPOCR")
        self.img_files = glob.glob(os.path.join(self.img_dir, "val/crop_imgs/*.jpg"))
        self.total_num = len(self.img_files)
        self.data_lines = self._get_image_info_list([self.labels_file])

    def _get_image_info_list(self, file_list):
        """Read image info from label files.

        Args:
            file_list (list): List of label file paths.

        Returns:
            list: List of data lines from label files.
        """
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for file in file_list:
            with open(file, "rb") as f:
                lines = f.readlines()
                data_lines.extend(lines)
        return data_lines

    def get_datas(self, num: int):
        """Get subset of image paths.

        Args:
            num (int): Number of samples to retrieve. 0 means all.

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
        """Return dataset name."""
        return "CCPD2020ValRec"

    def get_next_batch(self):
        """Get next batch of data. Not implemented for this dataset."""
        pass
