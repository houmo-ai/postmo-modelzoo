# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YOLO12 model implementation.
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


class Yolo12(CocoDetectionModel):
    """
    YOLOv12 object detection model implementation.

    This class implements the YOLOv12 model for object detection tasks,
    inheriting from BaseModel. It provides functionality for post-processing
    model outputs, running inference on images, and evaluating model performance
    using COCO metrics. The model uses anchor-based detection with predefined
    feature map sizes and strides.
    """

    def __init__(self, **kwargs):
        """
        Initialize Yolo12 model instance.

        Sets up model configuration including input dimensions, confidence thresholds,
        feature map sizes, and anchor points for detection.

        Args:
            **kwargs: Keyword arguments passed to the parent BaseModel class
        """
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)  # HW
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.feats = [(80, 80), (40, 40), (20, 20)]  # hw
        self.strides = [8.0, 16.0, 32.0]
        self.anchor_points, self.stride_tensor = self.make_anchors(
            self.feats, self.strides
        )
        self.to_coco91 = True

    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        """
        Generate anchor points and stride tensors for detection.

        Creates anchor points for each feature map based on the specified
        feature map sizes and strides.

        Args:
            feats: List of tuples containing feature map dimensions (height, width)
            strides: List of stride values for each feature map
            grid_cell_offset: Offset value for grid cell centers (default 0.5)

        Returns:
            Tuple of (anchor_points, stride_tensor) as PyTorch tensors
        """
        anchor_points, stride_tensor = [], []
        for i, stride in enumerate(strides):
            h, w = feats[i]
            sx = torch.arange(end=w, dtype=torch.float32) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, dtype=torch.float32) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float))
        return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)

    def decode(self, outs: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Decode model outputs to bounding box coordinates.

        Transforms the raw model outputs to proper bounding box coordinates
        by applying anchor-based transformations.

        Args:
            outs: Model outputs as dictionary of numpy arrays

        Returns:
            Decoded predictions as PyTorch tensor with shape [batch_size, 84, 8400]
        """
        pred = list()
        for _, name in enumerate(outs):
            out = torch.from_numpy(outs[name])
            bs, box_or_cls, num_anchors = out.shape
            if box_or_cls != 4:
                pred.append(out)
                continue
            # bs, 4, 8400
            bbox_data = out.permute(0, 2, 1)  # bs, 8400, 4
            bbox_data[:, :, 0:2] = self.anchor_points - bbox_data[:, :, 0:2]
            bbox_data[:, :, 2:4] = self.anchor_points + bbox_data[:, :, 2:4]
            bbox_data_xy = (bbox_data[:, :, 0:2] + bbox_data[:, :, 2:4]) * 0.5
            bbox_data_wh = bbox_data[:, :, 2:4] - bbox_data[:, :, 0:2]
            bbox_data[:, :, 0:2] = bbox_data_xy
            bbox_data[:, :, 2:4] = bbox_data_wh
            bbox_data *= self.stride_tensor
            bbox_data = bbox_data.permute(0, 2, 1).contiguous()  # 1, 4, 8400
            pred.insert(0, bbox_data)
        return torch.cat(pred, dim=1)  # [bs, 84, 8400]

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Post-process the model outputs to extract detections.

        Applies non-maximum suppression and scales the detection coordinates
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
            raise ValueError("Yolo12 model only has one output")
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


