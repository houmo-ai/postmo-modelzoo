# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YOLOv5 object detection model implementation.
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


class YoloV5(CocoDetectionModel):
    """
    YOLOv5 object detection model implementation that feature as inputs.

    This class implements the YOLOv5 model for object detection tasks,
    inheriting from BaseModel. It provides functionality for post-processing
    model outputs using anchor-based detection, running inference on images,
    and evaluating model performance using COCO metrics.
    """

    def __init__(self, **kwargs):
        """
        Initialize YoloV5 model instance.

        Sets up model configuration including input dimensions, confidence thresholds,
        strides, and anchor boxes for multi-scale detection.

        Args:
            **kwargs: Keyword arguments passed to the parent BaseModel class
        """
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)  # HW
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.strides = [8.0, 16.0, 32.0]
        self.anchors = (
            torch.Tensor(
                [
                    10,
                    13,
                    16,
                    30,
                    33,
                    23,
                    30,
                    61,
                    62,
                    45,
                    59,
                    119,
                    116,
                    90,
                    156,
                    198,
                    373,
                    326,
                ]
            )
            .float()
            .view(3, 3, 2)
        )
        self.to_coco91 = True

    def decode(self, outs: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Decode model outputs using anchor-based transformation.

        Converts the raw model outputs to bounding box coordinates by applying
        YOLOv5's anchor-based transformation with grid shifts and scaling.

        Args:
            outs: Model outputs as dictionary of numpy arrays

        Returns:
            Decoded predictions as PyTorch tensor with shape [batch_size, num_anchors, num_classes+5]
        """
        pred = list()
        for idx, name in enumerate(outs):
            out = torch.from_numpy(outs[name])
            assert len(out.shape) == 5, f"output shape error: {out.shape}"
            _, na, h, w, no = out.shape
            sx = torch.arange(end=w, dtype=torch.float)  # shift x
            sy = torch.arange(end=h, dtype=torch.float)  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            grid = torch.stack((sx, sy), dim=2).expand(1, na, h, w, 2) - 0.5
            anchor_grid = self.anchors[idx].view(1, na, 1, 1, 2).expand(1, na, h, w, 2)
            # xy, wh, conf = out.sigmoid().split((2, 2, no - 4), dim=4)
            xy, wh, conf = out.split((2, 2, no - 4), dim=4)
            xy = (xy * 2 + grid) * self.strides[idx]
            wh = (wh * 2) ** 2 * anchor_grid
            y = torch.cat([xy, wh, conf], dim=4)
            pred.append(y.view(1, -1, no))
        return torch.cat(pred, dim=1)

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Post-process the model outputs to extract detections.

        Handles both single and triple output formats, either decodes the model outputs
        using anchor-based transformation if there are 3 outputs, or uses the single
        output directly. Then applies non-maximum suppression and scales the detection
        coordinates from the model input size to the original image size.

        Args:
            outs: Model outputs as dictionary of numpy arrays
            in_datas: Input data as dictionary of numpy arrays

        Returns:
            Processed detections as numpy array with format [x1, y1, x2, y2, confidence, class_idx]

        Raises:
            SystemExit: If the number of outputs is not 1 or 3
        """
        if len(outs) == 3:
            pred = self.decode(outs)  # [bs, 25200, 85]
        elif len(outs) == 1:
            pred = list(outs.values())[0]  # [bs, 25200, 85]
            pred = torch.from_numpy(pred)
        else:
            logger.error(f"Output length error: {len(outs)}")
            exit(-1)
        # Only take batch 0, multi-batch data is copied, no need to waste time processing it
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


