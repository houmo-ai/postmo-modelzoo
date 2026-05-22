#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: dataset.py
# Description:
#   CCPD2019 sub dataset for lprnet model to evaluate.
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


class CCPD2019SubDataSet(BaseDataset):
    def __init__(self, root_path):
        self.root_path = root_path
        if not os.path.exists(self.root_path):
            logger.error("root_path not exits -> {}".format(self.root_path))
            exit(-1)
        self.img_files = glob.glob(os.path.join(self.root_path, "*.jpg"))
        self.total_num = len(self.img_files)

    def get_datas(self, num: int):
        if num == 0:
            num = self.total_num
        elif num > self.total_num:
            num = self.total_num
        img_paths = self.img_files[0:num]
        return img_paths

    @property
    def dataset_name(self):
        return "CCPD2019Sub"

    def get_next_batch(self):
        pass
