# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YOLOv7 model implementation for object detection
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


class YoloV7(CocoDetectionModel):
    """
    YOLOv7 model implementation for object detection tasks.

    This class implements the YOLOv7 model with preprocessing, postprocessing,
    inference, evaluation, and visualization capabilities. It inherits from BaseModel
    and provides specific implementations for object detection using the YOLOv7 architecture.
    """

    def __init__(self, **kwargs):
        """
        Initialize the YOLOv7 model.

        Sets up input configurations, default thresholds for postprocessing,
        and other model-specific parameters.
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
        """
        pred = list(outs.values())[0]  # [bs, 25200, 85]
        pred = torch.from_numpy(pred)
        pred = pred[:1, ...]  # [1, 25200, 85]
        cv_image = list(in_datas.values())[0]
        outputs = non_max_suppression(
            pred,
            self.conf_threshold,
            self.iou_threshold,
            exist_obj_conf=True,
        )
        output = outputs[0]
        output[:, :4] = scale_coords(
            self.input_size, output[:, :4], cv_image.shape
        ).round()
        output = output.detach().cpu().numpy()
        return output


