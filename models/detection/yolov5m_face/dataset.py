# Copyright 2025 HOUMO AI
#
# File: dataset.py
# Description:
#   WiderFace dataset for face detection
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
from hmatc.datasets.widerface import WiderFace


class Dataset(WiderFace):
    """
    Dataset class for YOLOv5m face detection model.

    This class inherits from WiderFace and is used for face detection
    tasks with the YOLOv5m model. It provides the necessary data handling.
    """

    def __init__(self, **kwargs):
        """
        Initialize Dataset instance.

        Args:
            **kwargs: Keyword arguments passed to the parent WiderFace class
        """
        super().__init__(**kwargs)
