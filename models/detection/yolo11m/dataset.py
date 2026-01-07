# Copyright 2025 HOUMO AI
#
# File: dataset.py
# Description:
#   Dataset class for YOLO11m model using COCO2017 validation set.
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
    Dataset class for YOLO11m model.

    This class inherits from COCO2017Val and is used for data processing
    in the YOLOv8m pose estimation model. It uses the COCO2017 validation
    set as the base dataset format.
    """

    def __init__(self, **kwargs):
        """
        Initialize Dataset instance.

        Args:
            **kwargs: Keyword arguments passed to the parent class COCO2017Val
        """
        super().__init__(**kwargs)
