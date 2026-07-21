# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YOLOv9m model implementation for object detection.
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
from hmatc.utils.postprocess import non_max_suppression, scale_coords


class YoloV9(CocoDetectionModel):
    """
    YOLOv9 Detection Model implementation.

    This class implements the YOLOv9 object detection model with preprocessing,
    postprocessing, evaluation and demo capabilities. It inherits from BaseModel
    and provides specific implementation for YOLOv9 including NMS postprocessing,
    detection scaling, and COCO evaluation.

    Args:
        **kwargs: Arguments passed to the parent BaseModel class including model configuration
    """

    def __init__(self, **kwargs):
        """
        Initialize the YOLOv9 model.

        Sets up the model with input configuration, default thresholds for postprocessing,
        and COCO format settings.

        Args:
            **kwargs: Arguments passed to the parent BaseModel class
        """
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)  # HW
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.to_coco91 = True

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Postprocess the model outputs to generate final detections.

        Applies non-maximum suppression to filter detections based on confidence
        and IoU thresholds, and scales coordinates back to original image dimensions.

        Args:
            outs: Model output dictionary containing raw predictions
            in_datas: Input data dictionary containing the original images

        Returns:
            numpy.ndarray: Processed detections with format [x1, y1, x2, y2, confidence, class]

        Raises:
            ValueError: If the model doesn't have exactly one output
        """
        if len(outs) != 1:
            raise ValueError("YoloV9 model only has one output")
        pred = list(outs.values())[0]  # [bs, 84, 8400]
        pred = torch.from_numpy(pred)
        pred = pred[:1, ...]  # [1, 84, 8400]
        cv_image = list(in_datas.values())[0]
        outputs = non_max_suppression(pred, self.conf_threshold, self.iou_threshold)
        output = outputs[0]
        output[:, :4] = scale_coords(
            self.input_size, output[:, :4], cv_image.shape
        ).round()
        output = output.detach().cpu().numpy()
        return output


