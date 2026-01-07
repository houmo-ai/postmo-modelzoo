# Copyright 2025 HOUMO AI
#
# File: dataset.py
# Description:
#  This file contains the dataset class for the MobileNetV2 model.
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
    Dataset class for MobileNetV2 model.

    This class provides the dataset implementation for the MobileNetV2 model,
    using the ILSVRC2012 (ImageNet) validation dataset as base. It handles data loading
    and preprocessing required for the MobileNetV2 model evaluation.

    Args:
        **kwargs: Arguments passed to the parent ILSVRC2012 class
    """

    def __init__(self, **kwargs):
        """
        Initialize the MobileNetV2 dataset.

        Args:
            **kwargs: Arguments passed to the parent ILSVRC2012 class
        """
        super().__init__(**kwargs)
