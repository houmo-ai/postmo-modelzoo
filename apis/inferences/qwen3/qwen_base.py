#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025 HOUMO AI
#
# File: qwen_base.py
# Description:
#   Base class for Qwen3 models - Defines common interface and initialization
#   logic for both prefill and decode models, including loading models and tokenizers,
#   managing KV caches, and handling embeddings.
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

from abc import ABC, abstractmethod
import os
import re
import numpy as np
import torch
from transformers import AutoTokenizer
from loguru import logger
from typing import List

import tcim_lite as tcim
from hmatc.python.get_hm_devices import get_hm_devices


class HmQwenBase(ABC):
    """
    Abstract base class for Qwen3 model implementations.
    Provides common functionality for loading models, managing KV caches,
    and handling embeddings for both prefill and decode operations.
    """

    def __init__(
        self, prefill_path, decode_path, embedding_path, tokenizer_dir, ndevice
    ):
        """
        Initialize the base Qwen model with specified paths and configurations.

        Args:
            prefill_path: Path to the prefill model file
            decode_path: Path to the decode model file
            embedding_path: Path to the embedding weights file
            tokenizer_dir: Directory containing tokenizer files
            ndevice: Number of devices to use for inference
        """
        # Validate that all required model files exist
        if (
            not self._check_file_path(prefill_path)
            or not self._check_file_path(decode_path)
            or not self._check_file_path(embedding_path)
        ):
            raise ValueError("Invalid model file path.")
        # Validate device number
        if ndevice <= 0 or ndevice > 2:
            raise ValueError(f"Unsupported device number {ndevice}.")

        self.ndevice = ndevice
        device_list = get_hm_devices(self.ndevice)
        # Initialize device manager and weight manager for TCIM runtime
        dev_manager = tcim.runtime.DevManager(device_list, "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)

        # Load the prefill model
        prefill_option = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=prefill_option)
        logger.info("prefill model loaded")

        # Determine the number of transformer blocks in the model
        self.nblocks = self.get_nblocks()
        # Load the decode model with dummy tensors for KV cache inputs
        decode_option = tcim.runtime.Option(weight_manager)
        dummy_tensor_names = [
            f"model_layers_{i}_self_attn_kcache_input" for i in range(self.nblocks)
        ]
        dummy_tensor_names += [
            f"model_layers_{i}_self_attn_vcache_input" for i in range(self.nblocks)
        ]
        decode_option.set_dummy_tensors(dummy_tensor_names)
        self.decode = tcim.runtime.load(decode_path, option=decode_option)
        logger.info("decode model loaded")

        # Extract important dimensions from model inputs
        self.prefill_length = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        self.embedding_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]
        self.context_max_length = self.decode.get_input_info(
            self.decode.get_input_name(3)
        ).shape[2]
        self.batch = self.decode.get_input_info(self.decode.get_input_name(0)).shape[0]

        # Initialize KV caches from prefill to decode model
        for i in range(3, 2 * self.nblocks + 3):
            cache = self.prefill.get_dev_input(self.prefill.get_input_name(i))
            self.decode.set_input(self.decode.get_input_name(i), cache)
        # Set initial current length input for decode model
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode.get_input_name(2)
        self.decode.set_input(decode_current_length_name, current_length_input_1)

        # Initialize tokenizer from pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )
        # Load and prepare embedding weights
        embedding_weight = torch.load(embedding_path, map_location="cpu")
        embedding_weight = embedding_weight["weight"]  # Only use in XH2 backend
        self.embedding_weight = embedding_weight.reshape(-1, self.embedding_len).float()

        self.context_length = 0
        # Store input/output names for easier access
        self.prefill_input_name = self.prefill.get_input_name(0)
        self.prefill_valid_length_name = self.prefill.get_input_name(1)
        self.prefill_current_length_name = self.prefill.get_input_name(2)
        self.decode_input_name = self.decode.get_input_name(0)
        self.decode_valid_length_name = self.decode.get_input_name(1)

    @staticmethod
    def _check_file_path(file_path) -> bool:
        """
        Check if the specified file path exists.

        Args:
            file_path: Path to check

        Returns:
            Boolean indicating whether the file exists
        """
        if os.path.exists(file_path):
            return True
        return False

    def get_nblocks(self) -> int:
        """
        Determine the number of transformer blocks in the model by analyzing input names.

        Returns:
            Number of transformer blocks in the model
        """
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        # Pattern to match KV cache input names which indicate transformer layers
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def _get_module(self, module_type: str):
        """
        Get the appropriate model module based on type.

        Args:
            module_type: Type of module ('prefill' or 'decode')

        Returns:
            The requested model module
        """
        if module_type == "prefill":
            module = self.prefill
        elif module_type == "decode":
            module = self.decode
        else:
            raise ValueError(f"Invalid module type: {module_type}")

        return module

    def get_input_num(self, module_type) -> int:
        """
        Get the number of inputs for the specified module.

        Args:
            module_type: Type of module ('prefill' or 'decode')

        Returns:
            Number of inputs for the module
        """
        return self._get_module(module_type).get_num_inputs()

    def get_output_num(self, module_type) -> int:
        """
        Get the number of outputs for the specified module.

        Args:
            module_type: Type of module ('prefill' or 'decode')

        Returns:
            Number of outputs for the module
        """
        return self._get_module(module_type).get_num_outputs()

    def get_input_names(self, module_type) -> List[str]:
        """
        Get all input names for the specified module.

        Args:
            module_type: Type of module ('prefill' or 'decode')

        Returns:
            List of input names for the module
        """
        module = self._get_module(module_type)

        input_names = []
        for i in range(self.get_input_num(module_type)):
            input_names.append(module.get_input_name(i))
        return input_names

    def get_output_names(self, module_type) -> List[str]:
        """
        Get all output names for the specified module.

        Args:
            module_type: Type of module ('prefill' or 'decode')

        Returns:
            List of output names for the module
        """
        module = self._get_module(module_type)

        output_names = []
        for i in range(self.get_output_num(module_type)):
            output_names.append(module.get_output_name(i))
        return output_names

    @abstractmethod
    def chat(self, question: str, **kwargs):
        """
        Abstract method for performing chat interactions.
        Must be implemented by subclasses.
        """
        pass
