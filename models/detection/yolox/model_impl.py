# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YOLOX Detection Model.
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


def nms(boxes, scores, nms_thr):
    """
    Single class NMS implemented in Numpy.

    Performs non-maximum suppression to filter overlapping bounding boxes
    for a single object class.

    Args:
        boxes: Array of bounding boxes in format [x1, y1, x2, y2]
        scores: Array of confidence scores for each box
        nms_thr: Threshold for IoU to determine if boxes overlap

    Returns:
        List of indices of boxes to keep after NMS
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """
    Multiclass NMS implemented in Numpy. Class-agnostic version.

    Performs non-maximum suppression across multiple classes without
    considering class labels during suppression.

    Args:
        boxes: Array of bounding boxes in format [x1, y1, x2, y2]
        scores: Array of class scores for each box
        nms_thr: Threshold for IoU to determine if boxes overlap
        score_thr: Threshold for minimum confidence score

    Returns:
        Array of detections with format [x1, y1, x2, y2, confidence, class]
    """
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        dets = []
        return dets
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [
                valid_boxes[keep],
                valid_scores[keep, None],
                valid_cls_inds[keep, None],
            ],
            axis=1,
        )
    return dets


class YoloX(CocoDetectionModel):
    """
    YOLOX Detection Model implementation.

    This class implements the YOLOX object detection model with preprocessing,
    postprocessing, evaluation and demo capabilities. It inherits from BaseModel
    and provides specific implementation for YOLOX including NMS postprocessing,
    detection scaling, and COCO evaluation.

    Args:
        **kwargs: Arguments passed to the parent BaseModel class including model configuration
    """

    def __init__(self, **kwargs):
        """
        Initialize the YOLOX model.

        Sets up the model with input configuration, default thresholds for postprocessing,
        and other model-specific parameters.

        Args:
            **kwargs: Arguments passed to the parent BaseModel class
        """
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)  # HW
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.feats = [(80, 80), (40, 40), (20, 20)]  # hw
        self.strides = [8.0, 16.0, 32.0]
        self.to_coco91 = True

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Postprocess the model outputs to generate final detections.

        Applies grid-based coordinate transformation, exponential scaling,
        and multiclass NMS to generate final detection results.

        Args:
            outs: Model output dictionary containing raw predictions
            in_datas: Input data dictionary containing the original images

        Returns:
            numpy.ndarray: Processed detections with format [x1, y1, x2, y2, confidence, class]
        """
        key = list(outs.keys())[0]
        cv_image = list(in_datas.values())[0]
        outputs = outs[key]
        grids = []
        expanded_strides = []
        strides = self.strides
        img_size = self.input_size
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        boxes = outputs[:, :, :4]
        scores = outputs[:, :, 4:5] * outputs[:, :, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2] / 2.0
        boxes_xyxy[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3] / 2.0
        boxes_xyxy[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2] / 2.0
        boxes_xyxy[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3] / 2.0

        outputs = multiclass_nms_class_agnostic(
            boxes_xyxy[0, :, :],
            scores[0, :, :],
            self.iou_threshold,
            self.conf_threshold,
        )
        if len(outputs) != 0:
            outputs[:, :4] = scale_coords(
                self.input_size,
                outputs[:, :4],
                cv_image.shape,
                need_pad=False,
            ).round()
        return outputs


