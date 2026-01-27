# Copyright 2025 HOUMO AI
#
# File: base_infer.py
# Description:
#   Base infer class
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
import abc


class BaseInfer(object, metaclass=abc.ABCMeta):
    """Base inference class for model inference operations.

    This abstract class defines the interface for model inference implementations,
    providing methods for loading models, running inference, and managing inference
    statistics such as timing and performance metrics.
    """

    def __init__(self, **kwargs):
        """Initialize the inference instance.

        Args:
            **kwargs: Arbitrary keyword arguments for inference configuration.
        """
        self.time_span = 0  # Total time span for inference operations
        self.total = 0  # Total number of inference operations performed
        self.engine = None  # Inference engine instance
        self.backend = "onnx"  # Backend type: onnx/hmquant/xh1/xh2
        self.device = "cpu"  # Device for inference execution
        self.inputs_batch = dict()

    @abc.abstractmethod
    def load(self, model_path, device_id=0):
        """Load the model for inference.

        Args:
            model_path (str): Path to the model file to be loaded.
            device_id (int): Device ID to load the model on (default: 0).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, in_datas: dict, to_file=False):
        """Execute inference on input data.

        Args:
            in_datas (dict): Dictionary containing input data for inference.
            to_file (bool): Whether to save results to file (default: False).

        Returns:
            Inference results based on the specific implementation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def unload(self):
        """Unload the model and clean up resources.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def ave_latency_ms(self):
        """float: Average latency in milliseconds for inference operations.

        Returns 0 if no inference operations have been performed.
        """
        if self.total == 0:
            return 0
        return self.time_span / self.total

    def get_input_batch_size(self, name):
        return self.inputs_batch.get(name, 0)
