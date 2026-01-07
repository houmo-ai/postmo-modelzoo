# Copyright 2025 HOUMO AI
#
# File: dataset.py
# Description:
#   Dataset class for YOLOv8 segmentation model.
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
from hmatc.datasets.coco import COCO2017Val


class Dataset(COCO2017Val):
    """
    Dataset class for YOLOv8 segmentation model.

    This class provides the dataset implementation for the YOLOv8 segmentation model,
    using the COCO2017 validation dataset as base. It handles data loading and
    preprocessing required for the YOLOv8 segmentation model evaluation.

    Args:
        **kwargs: Arguments passed to the parent COCO2017Val class
    """

    def __init__(self, **kwargs):
        """
        Initialize the YOLOv8 segmentation dataset.

        Args:
            **kwargs: Arguments passed to the parent COCO2017Val class
        """
        super().__init__(**kwargs)
