# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YOLOv8 Classification model implementation.
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


class YoloV8Cls(ClassificationModel):
    """
    YOLOv8 Classification model implementation.

    This class implements the YOLOv8 model for image classification tasks,
    inheriting from the BaseModel class. It provides functionality for
    postprocessing, model demonstration, and evaluation.
    """

    def __init__(self, **kwargs):
        """
        Initializes the YoloV8Cls model.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the parent BaseModel class.
        """
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Postprocesses the model outputs to get classification results.

        Applies softmax to the output and returns the class index with the highest score
        along with its confidence score for each sample in the batch.

        Args:
            outs: Dictionary containing model outputs
            in_datas: Dictionary containing input data

        Returns:
            List of tuples containing (class_index, score) for each sample in the batch
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


