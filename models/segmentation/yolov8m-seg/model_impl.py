# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YOLOv8 Segmentation Model implementation
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
import cv2
import torch
import numpy as np
from typing import Dict, Any, List
from hmatc.base.task_models import CocoSegmentationModel
from hmatc.utils.postprocess import (
    non_max_suppression,
    scale_coords,
    process_mask,
    scale_coords_mask,
)
from hmatc.utils.metrics import detections_mask2json, merge_json, coco_eval


class YoloV8Seg(CocoSegmentationModel):
    """
    YOLOv8 Segmentation Model implementation.

    This class implements the YOLOv8 instance segmentation model with preprocessing,
    postprocessing, evaluation and demo capabilities. It inherits from BaseModel
    and provides specific implementation for instance segmentation with mask generation.

    Args:
        **kwargs: Arguments passed to the parent BaseModel class including model configuration
    """

    def __init__(self, **kwargs):
        """
        Initialize the YOLOv8 Segmentation model.

        Sets up the model with input configuration, default thresholds for postprocessing,
        and other model-specific parameters for segmentation.

        Args:
            **kwargs: Arguments passed to the parent BaseModel class
        """
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)  # HW
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Postprocess the model outputs to generate final segmentation results.

        Applies non-maximum suppression and mask processing to generate
        final detection and segmentation results.

        Args:
            outs: Model output dictionary containing raw predictions
            in_datas: Input data dictionary containing the original images

        Returns:
            tuple: A tuple containing:
                - detections: numpy array of detections with format [x1, y1, x2, y2, confidence, class]
                - masks: list of segmentation masks
                - contours: list of contours for each mask
        """
        outs = list(outs.values())
        cv_image = list(in_datas.values())[0]
        det_out = torch.from_numpy(outs[0])  # bs, 116, 8400
        seg_out = torch.from_numpy(outs[1])  # bs, 32, 160, 160
        det_out = det_out[:1, ...]
        seg_out = seg_out[:1, ...]
        detections = non_max_suppression(
            det_out,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            nm=32,
        )
        detections = detections[0]
        _contours = list()
        _masks = list()
        if detections.shape[0] > 0:
            masks = process_mask(
                seg_out[0],
                detections[:, 6:],
                detections[:, :4],
                self.input_size,
                upsample=True,
            )  # HWC
            detections[:, :4] = scale_coords(
                self.input_size, detections[:, :4], cv_image.shape
            ).round()
            masks = masks.numpy()
            h, w, _ = cv_image.shape
            for _, mask in enumerate(masks):
                contours, _ = cv2.findContours(
                    mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                if isinstance(contours, tuple):
                    contours = list(contours)
                contours = scale_coords_mask(self.input_size, contours, cv_image.shape)
                tmp_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(tmp_mask, contours, 255)
                _masks.append(tmp_mask)
                _contours.append(contours)
        return detections, _masks, _contours


