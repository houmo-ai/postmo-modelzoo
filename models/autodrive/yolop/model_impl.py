# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YoloP model implementation for autonomous driving tasks.
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
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from hmatc.utils import logger
from hmatc.base.base_model import BaseModel
from hmatc.utils.preprocess import calc_padding_size
from hmatc.utils.postprocess import non_max_suppression, scale_coords


class YoloP(BaseModel):
    """
    YoloP model implementation for autonomous driving tasks.

    This class implements the YoloP model which performs three tasks simultaneously:
    object detection, drivable area segmentation, and lane line segmentation.
    It inherits from BaseModel and provides specific implementation for multi-task
    autonomous driving applications.

    Args:
        **kwargs: Arguments passed to the parent BaseModel class including model configuration
    """

    def __init__(self, **kwargs):
        """
        Initialize the YoloP model.

        Sets up the model with input configuration, default thresholds for postprocessing,
        and other model-specific parameters for autonomous driving tasks.

        Args:
            **kwargs: Arguments passed to the parent BaseModel class
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
                [3, 9, 5, 11, 4, 20, 7, 18, 6, 39, 12, 31, 19, 50, 38, 81, 68, 157]
            )
            .float()
            .view(3, 3, 2)
        )
        self.to_coco91 = True

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Postprocess the model outputs to generate final results for all three tasks.

        Performs postprocessing for object detection, drivable area segmentation,
        and lane line segmentation. Applies non-maximum suppression for detection
        and processes segmentation masks.

        Args:
            outs: Model output dictionary containing raw predictions for all tasks
            in_datas: Input data dictionary containing the original images

        Returns:
            tuple: A tuple containing:
                - det_out: numpy array of detection results [x1, y1, x2, y2, confidence, class]
                - mask: numpy array of combined segmentation mask with drivable area and lane line
        """
        da_seg_out = outs["drive_area_seg"]  # bs, 2, 640, 640
        ll_seg_out = outs["lane_line_seg"]  # bs, 2, 640, 640
        bbox_outs = list()
        for key in outs:
            if key not in ["drive_area_seg", "lane_line_seg"]:
                bbox_outs.append(outs[key])
        # det
        z = list()
        for idx, bbox_out in enumerate(bbox_outs):
            bs, na, ny, nx, nc = bbox_out.shape
            p = torch.from_numpy(bbox_out)
            sx = torch.arange(end=nx, dtype=torch.float32)  # shift x
            sy = torch.arange(end=ny, dtype=torch.float32)  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            grid = torch.stack((sx, sy), dim=2).expand(1, 3, ny, nx, 2) - 0.5
            anchor_grid = self.anchors[idx].view(1, 3, 1, 1, 2).expand(1, 3, ny, nx, 2)
            # p = (p.view(bs, 3, c // 3, ny, nx).permute(0, 1, 3, 4, 2).contiguous())
            p[..., 0:2] = (p[..., 0:2] * 2.0 + grid) * self.strides[idx]  # xy
            p[..., 2:4] = (p[..., 2:4] * 2) ** 2 * anchor_grid
            z.append(p.view(bs, -1, nc))  # bs, -1, 6
        det_out = torch.cat(z, dim=1)  # bs, -1, 6
        det_out = det_out[:1, ...]  # bs, -1, 6
        cv_image = list(in_datas.values())[0]
        outputs = non_max_suppression(
            det_out,
            self.conf_threshold,
            self.iou_threshold,
            exist_obj_conf=True,
        )
        output = outputs[0]
        output[:, :4] = scale_coords(
            self.input_size, output[:, :4], cv_image.shape
        ).round()
        det_out = output.detach().cpu().numpy()

        # da_seg
        H, W, _ = cv_image.shape
        target_size = (self.input_size[1], self.input_size[0])  # (W, H)
        padding_size, size, scale = calc_padding_size(
            (H, W), target_size, padding_mode=1
        )
        top, left, bottom, right = padding_size
        nh, nw = size
        da_seg_mask = torch.from_numpy(
            da_seg_out[:, :, top : top + nh, left : left + nw]
        )  # 2, nh, nw
        ll_seg_mask = torch.from_numpy(
            ll_seg_out[:, :, top : top + nh, left : left + nw]
        )  # 2, nh, nw
        da_seg_mask = (
            torch.nn.functional.interpolate(
                da_seg_mask, size=(H, W), mode="bilinear", align_corners=False
            )
            .detach()
            .cpu()
            .numpy()
        )
        ll_seg_mask = (
            torch.nn.functional.interpolate(
                ll_seg_mask, size=(H, W), mode="bilinear", align_corners=False
            )
            .detach()
            .cpu()
            .numpy()
        )
        da_seg_mask = np.argmax(da_seg_mask, axis=1)[0]  # (0|1)
        ll_seg_mask = np.argmax(ll_seg_mask, axis=1)[0]  # (0|1)
        mask = np.zeros((H, W, 3), dtype=np.uint8)
        mask[da_seg_mask == 1] = [0, 255, 0]  # 1: drive area
        mask[ll_seg_mask == 1] = [255, 0, 0]  # 1: lane line
        return det_out, mask

    def demo(self, filepaths: list):
        """
        Run inference on input images and save visualized results.

        Performs multi-task inference on the input images, draws bounding boxes
        for detected objects and overlays segmentation masks for drivable areas
        and lane lines, then saves the results.

        Args:
            filepaths: List of paths to input images for inference
        """
        in_datas = dict()
        save_dir = f"vis_{self.backend}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx, filepath in enumerate(filepaths):
            basename, _ = os.path.splitext(os.path.basename(filepath))
            save_path = os.path.join(save_dir, f"{basename}.jpg")
            cv_image = cv2.imread(filepath)
            if cv_image is None:
                logger.warning(f"{filepath} not exists or decode failed")
                continue
            in_datas[self.input_name] = cv_image
            logger.info(f"Image[{idx}] {filepath}")
            det_outs, mask = self.run(in_datas)
            for idx, detection in enumerate(det_outs):
                x1, y1, x2, y2, score, cls_idx = detection
                x1 = int(x1) if x1 > 0 else 0
                y1 = int(y1) if y1 > 0 else 0
                x2 = int(x2) if x2 < cv_image.shape[1] else cv_image.shape[1]
                y2 = int(y2) if y2 < cv_image.shape[0] else cv_image.shape[0]
                cls_idx = int(cls_idx)
                logger.info(
                    f"Detection[{idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, score: {score:.3f}, cls: {cls_idx:2}"
                )
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv_image = np.where(
                mask == 255, cv_image * 0.5 + mask * 0.5, cv_image
            ).astype(np.uint8)
            cv2.imwrite(save_path, cv_image)
            logger.info(f"Save result to {save_path}")

    def evaluate(self, dataset, num=0):
        """
        Evaluate the model performance on a given dataset.

        Currently not implemented for YoloP model as it performs multiple tasks
        that require different evaluation metrics.

        Args:
            dataset: Dataset object containing evaluation data
            num: Number of samples to evaluate (0 means all samples)

        Raises:
            NotImplementedError: Evaluation is not implemented for YoloP model
        """
        raise NotImplementedError("Evaluation is not implemented for YoloP model.")
