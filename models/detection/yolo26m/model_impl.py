# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YOLO26 model implementation.
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
import torch
import numpy as np
from typing import Dict, Any
from hmatc.base.task_models import CocoDetectionModel
from hmatc.utils.postprocess import scale_coords


class Yolo26(CocoDetectionModel):
    """
    YOLO26 object detection model implementation.

    This class implements the YOLO26 model for object detection tasks,
    inheriting from BaseModel. It provides functionality for post-processing
    model outputs, running inference on images, and evaluating model performance
    using COCO metrics.
    """

    def __init__(self, **kwargs):
        """
        Initialize Yolo26 model instance.

        Args:
            **kwargs: Keyword arguments passed to the parent BaseModel class
        """
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)  # HW
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_det = 300
        self.to_coco91 = True

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Post-process the model outputs to extract detections.

        Applies topk filtering and scales the detection coordinates
        from the model input size to the original image size.

        Args:
            outs: Model outputs as dictionary of numpy arrays
            in_datas: Input data as dictionary of numpy arrays

        Returns:
            Processed detections as numpy array with format [x1, y1, x2, y2, confidence, class_idx]

        Raises:
            ValueError: If the model has more than one output
        """
        if len(outs) != 1:
            raise ValueError("Yolo26 model only has one output")
        out = torch.from_numpy(list(outs.values())[0])  # [bs, 84, 8400]
        pred = out[:1, ...]  # [1, 300, 6]
        mask = pred[..., 4] > self.conf_threshold
        outputs = [p[mask[idx]] for idx, p in enumerate(pred)]
        cv_image = list(in_datas.values())[0]
        output = outputs[0]
        output[:, :4] = scale_coords(
            self.input_size, output[:, :4], cv_image.shape
        ).round()
        output = output.detach().cpu().numpy()
        return output


