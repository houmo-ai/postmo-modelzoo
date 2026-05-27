# Copyright 2026 HOUMO AI
#
# File: dataset.py
# Description:
#   Dataset class for DINOv3 base model.
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
from hmatc.datasets.imagenet import ILSVRC2012


class Dataset(ILSVRC2012):
    """
    Dataset class for DINOv3 base model.
    Inherits from ILSVRC2012 dataset class to provide ImageNet dataset functionality
    specifically tailored for DINOv3 base model.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Dataset instance.

        Args:
            **kwargs: Additional keyword arguments passed to parent ILSVRC2012 constructor
        """
        super().__init__(**kwargs)
