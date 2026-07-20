# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   This file contains the EfficientNet model implementation,
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


class EfficientNet(ClassificationModel):
    """
    EfficientNet model implementation for image classification tasks.

    This class implements the EfficientNet model with preprocessing, postprocessing,
    evaluation and demo capabilities. It inherits from BaseModel and provides
    specific implementation for image classification using the EfficientNet architecture.

    Args:
        **kwargs: Arguments passed to the parent BaseModel class including model configuration
    """

    def __init__(self, **kwargs):
        """
        Initialize the EfficientNet model.

        Sets up the model with input configuration and other model-specific parameters.

        Args:
            **kwargs: Arguments passed to the parent BaseModel class
        """
        super(EfficientNet, self).__init__(**kwargs)
        self.input_name = self.inputs_name[0]

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Postprocess the model outputs to generate final classification results.

        Applies softmax to convert logits to probabilities and finds the class
        with the highest probability for each input in the batch.

        Args:
            outs: Model output dictionary containing raw predictions (logits)
            in_datas: Input data dictionary containing the original images

        Returns:
            list: List of tuples containing (class_index, probability) for each input in batch
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


