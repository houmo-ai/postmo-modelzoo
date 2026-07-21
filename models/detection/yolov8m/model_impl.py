# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YOLOv8 model implementation for object detection.
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
from hmatc.utils import logger
from hmatc.base.task_models import CocoDetectionModel
from hmatc.utils.postprocess import non_max_suppression, scale_coords


class YoloV8(CocoDetectionModel):
    """
    YOLOv8 model implementation for object detection tasks.

    This class implements the YOLOv8 model with preprocessing, postprocessing,
    inference, evaluation, and visualization capabilities.
    """

    def __init__(self, **kwargs):
        """
        Initialize the YOLOv8 model.

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
        self.anchor_points, self.stride_tensor = self.make_anchors(
            self.feats, self.strides
        )
        self.to_coco91 = True

    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        """
        Generate anchors for YOLOv8 model based on feature map sizes and strides.

        Args:
            feats: List of feature map sizes as (height, width) tuples
            strides: List of stride values for each feature level
            grid_cell_offset: Offset value for grid cell center

        Returns:
            Tuple of anchor points and stride tensor
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
        Decode model outputs to bounding boxes.

        Args:
            outs: Model output dictionary containing raw predictions

        Returns:
            torch.Tensor: Decoded predictions
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
        Postprocess the model outputs to generate final detections.

        Args:
            outs: Model output dictionary containing raw predictions
            in_datas: Input data dictionary containing the original images

        Returns:
            numpy.ndarray: Processed detections with format [x1, y1, x2, y2, confidence, class]
        """
        if len(outs) == 2:
            # clip
            pred = self.decode(outs)  # [bs, 80, 8400], [bs, 4, 8400]
        elif len(outs) == 1:
            pred = list(outs.values())[0]  # [bs, 84, 8400]
            pred = torch.from_numpy(pred)
        else:
            logger.error(f"Output length error: {len(outs)}")
            exit(-1)
        pred = pred[:1, ...]  # [1, 84, 8400]
        cv_image = list(in_datas.values())[0]
        outputs = non_max_suppression(pred, self.conf_threshold, self.iou_threshold)
        output = outputs[0]
        output[:, :4] = scale_coords(
            self.input_size, output[:, :4], cv_image.shape
        ).round()
        output = output.detach().cpu().numpy()
        return output


