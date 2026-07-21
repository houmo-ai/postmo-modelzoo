# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YOLOv10 model implementation.
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
from typing import Dict, Any, List
from hmatc.base.task_models import CocoDetectionModel
from hmatc.utils.postprocess import scale_coords


class YoloV10(CocoDetectionModel):
    """
    YOLOv10 Detection Model implementation.

    This class implements the YOLOv10 object detection model with preprocessing,
    postprocessing, evaluation and demo capabilities. It inherits from BaseModel
    and provides specific implementation for YOLOv10 model.

    Args:
        **kwargs: Arguments passed to the parent BaseModel class including model configuration
    """

    def __init__(self, **kwargs):
        """
        Initialize the YOLOv10 model.

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
        self.max_det = 300
        self.strides = [8, 16, 32]
        self.proj = torch.arange(16, dtype=torch.float32).view(16, 1)
        self.anchor_points, self.stride_tensor = self.make_anchors(H, W, self.strides)
        self.to_coco91 = True

    @staticmethod
    def make_anchors(H, W, strides, grid_cell_offset=0.5):
        """
        Generate anchors from features.

        Creates anchor points and stride tensors for the model based on input dimensions
        and specified strides.

        Args:
            H: Input height
            W: Input width
            strides: List of stride values for different feature levels
            grid_cell_offset: Offset value for grid cell center positioning

        Returns:
            Tuple of anchor points and stride tensor
        """
        anchor_points, stride_tensor = [], []
        for i, stride in enumerate(strides):
            h, w = int(H / stride), int(W / stride)
            sx = torch.arange(end=w, dtype=torch.float32) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, dtype=torch.float32) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(
                torch.full((h * w, 1), fill_value=stride, dtype=torch.float32)
            )
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def _decode(self, outputs: List[np.ndarray]):
        """
        Decode the model outputs to bounding boxes and class probabilities.

        Processes the raw model outputs to generate proper bounding box coordinates
        and class probabilities.

        Args:
            outputs: List of raw model outputs as numpy arrays

        Returns:
            torch.Tensor: Decoded predictions with shape [batch_size, 84, 8400]
        """
        bs = outputs[0].shape[0]
        nc = outputs[0].shape[1]
        cls_data = np.concatenate(
            [
                outputs[4].reshape(bs, nc, -1),
                outputs[2].reshape(bs, nc, -1),
                outputs[0].reshape(bs, nc, -1),
            ],
            axis=2,
        )  # bs, 80, 8400
        box_data = np.concatenate(
            [
                outputs[5].reshape(bs, 64, -1),
                outputs[3].reshape(bs, 64, -1),
                outputs[1].reshape(bs, 64, -1),
            ],
            axis=2,
        )  # bs, 64, 8400
        cls_data = torch.from_numpy(cls_data)
        box_data = torch.from_numpy(box_data)
        box_data = (
            box_data.view(bs, 4, 16, 8400)
            .permute(0, 3, 1, 2)
            .contiguous()
            .softmax(dim=3)
            .view(-1, 16)
            .matmul(self.proj)
            .view(bs, 8400, 4)
        )
        box_data[:, :, 0:2] = self.anchor_points - box_data[:, :, 0:2]
        box_data[:, :, 2:4] = self.anchor_points + box_data[:, :, 2:4]
        box_data *= self.stride_tensor
        box_data = box_data.permute(0, 2, 1).contiguous()  # bs, 4, 8400
        cls_data = torch.sigmoid(cls_data)  # bs, 80, 8400
        return torch.cat([box_data, cls_data], dim=1)

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Postprocess the model outputs to generate final detections.

        Applies decoding, filtering, and scaling to generate final detection results.

        Args:
            outs: Model output dictionary containing raw predictions
            in_datas: Input data dictionary containing the original images

        Returns:
            numpy.ndarray: Processed detections with format [x1, y1, x2, y2, confidence, class]

        Raises:
            ValueError: If the model output doesn't have 1 or 6 elements
        """
        outs = list(outs.values())
        if len(outs) == 6:
            out = self._decode(outs)
        elif len(outs) == 1:
            out = torch.from_numpy(outs[0])  # [bs, 84, 8400]
        else:
            raise ValueError("YoloV10 model only has 1 or 6 output")
        pred = out[:1, ...]  # [1, 84, 8400]
        nc = pred.shape[1] - 4
        pred = pred.permute(0, 2, 1)
        batch_size, anchors, _ = pred.shape
        boxes, scores = pred.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(self.max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(self.max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        outputs = torch.cat(
            [boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()],
            dim=-1,
        )
        mask = outputs[..., 4] > self.conf_threshold
        outputs = [p[mask[idx]] for idx, p in enumerate(outputs)]
        cv_image = list(in_datas.values())[0]
        output = outputs[0]
        output[:, :4] = scale_coords(
            self.input_size, output[:, :4], cv_image.shape
        ).round()
        output = output.detach().cpu().numpy()
        return output


