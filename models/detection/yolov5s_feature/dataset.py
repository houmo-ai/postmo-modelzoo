# Copyright 2025 HOUMO AI
#
# File: dataset.py
# Description:
#   COCO2017 Val dataset for object detection
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
    Dataset class for YOLOv5s object detection model.

    This class inherits from COCO2017Val and is used for object detection
    tasks with the YOLOv5s model. It provides the necessary data
    handling and preprocessing for COCO format datasets.
    """

    def __init__(self, **kwargs):
        """
        Initialize Dataset instance.

        Args:
            **kwargs: Keyword arguments passed to the parent COCO2017Val class
        """
        super().__init__(**kwargs)
