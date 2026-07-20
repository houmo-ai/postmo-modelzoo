# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   Vision Transformer (ViT-B-16) model implementation for image classification tasks.
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
import numpy as np
from typing import Dict, Any
from hmatc.utils.postprocess import softmax
from hmatc.base.task_models import ClassificationModel


class ViT(ClassificationModel):
    """
    Vision Transformer (ViT-B-16) model implementation for image classification tasks.
    Inherits from BaseModel and provides postprocessing, demo and evaluation functionality.
    """

    def __init__(self, **kwargs):
        """
        Initialize ViT model instance.

        Args:
            **kwargs: Additional keyword arguments passed to BaseModel constructor
        """
        super(ViT, self).__init__(**kwargs)
        self.input_name = self.inputs_name[0]

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Postprocess the model outputs to get classification results.

        Applies softmax activation to the model outputs and finds the class with
        the highest probability for each sample in the batch.

        Args:
            outs: Dictionary containing model output tensors
            in_datas: Dictionary containing input data (not used in this implementation)

        Returns:
            List of tuples containing (class_index, confidence_score) for each sample in batch
        """
        output_name = list(outs.keys())[0]
        out = softmax(outs[output_name], axis=1, keepdims=True)
        max_idxes = np.argmax(out, axis=1, keepdims=True)
        batch = max_idxes.shape[0]
        res = list()
        for i in range(batch):
            max_idx = max_idxes[i][0]
            max_score = out[i][max_idx]
            res.append((max_idx, max_score))  # (cls_idx, score)
        return res


