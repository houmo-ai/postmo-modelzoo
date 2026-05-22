# Copyright 2025 HOUMO AI
#
# File: base_dataset.py
# Description:
#   Base dataset class
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
import abc


class BaseDataset(object, metaclass=abc.ABCMeta):
    """Base dataset class that provides image paths and labels.

    This abstract class defines the interface for dataset implementations,
    providing methods for retrieving batches of data and accessing dataset properties.
    """

    def __init__(self, **kwargs):
        """Initialize the dataset with dataset directory.

        Args:
            **kwargs: Arbitrary keyword arguments for dataset configuration,
                     typically including dataset directory path and other settings.
        """
        pass

    @abc.abstractmethod
    def get_next_batch(self):
        """Get the next batch of data.

        Returns:
            A batch of data containing image paths and corresponding labels.
        """
        pass

    @abc.abstractmethod
    def get_datas(self, num: int):
        """Get a subset of data with specified number of samples.

        Args:
            num (int): Number of data samples to retrieve.

        Returns:
            A subset of dataset containing specified number of samples.
        """
        pass

    @property
    @abc.abstractmethod
    def dataset_name(self):
        """str: Name of the dataset implementation."""
        return "BaseDataset"
