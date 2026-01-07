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
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List
from hmatc.utils import logger
from hmatc.base.base_model import BaseModel, COLORS
from hmatc.utils.postprocess import (
    non_max_suppression,
    scale_coords,
    process_mask,
    scale_coords_mask,
)
from hmatc.utils.metrics import detections_mask2json, merge_json, coco_eval


class YoloV8Seg(BaseModel):
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

    def demo(self, filepaths: list):
        """
        Run inference on input images and save visualized segmentation results.

        Performs segmentation on the input images, draws bounding boxes,
        segmentation masks and contours, and saves the results with detections visualized.

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
            detections, masks, contours = self.run(in_datas)
            for idx, detection in enumerate(detections):
                (x1, y1, x2, y2, score, cls_idx) = detection[0:6]
                color = np.array(COLORS[int(cls_idx) % len(COLORS)])
                # 画mask
                mask = masks[idx]
                new_masks = np.array([mask, mask, mask]).transpose((1, 2, 0))
                cv_image = np.where(
                    new_masks == 255, cv_image * 0.5 + color * 0.5, cv_image
                )
                # 画轮廓
                contour = contours[idx]
                cv2.drawContours(cv_image, contour, -1, color.tolist(), 2)
                # 画框
                x1 = int(x1) if x1 > 0 else 0
                y1 = int(y1) if y1 > 0 else 0
                x2 = int(x2) if x2 < cv_image.shape[1] else cv_image.shape[1]
                y2 = int(y2) if y2 < cv_image.shape[0] else cv_image.shape[0]
                logger.info(
                    f"Detection[{idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, score: {score:.3f}, cls: {int(cls_idx)}"
                )
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color.tolist(), 2)

            cv2.imwrite(save_path, cv_image)
            logger.info(f"Save result to {save_path}")

    def evaluate(self, dataset, num=0):
        """
        Evaluate the model performance on a given dataset.

        Runs inference on the dataset images, performs postprocessing,
        converts detections to COCO format for segmentation, and calculates mAP metrics.

        Args:
            dataset: Dataset object containing evaluation data
            num: Number of samples to evaluate (0 means all samples)

        Returns:
            dict: Dictionary containing evaluation metrics including mAP50-95, mAP50,
                  input size, dataset name, number of samples, and latency
        """
        self.iou_threshold = 0.65
        self.conf_threshold = 0.01
        img_paths = dataset.get_datas(num)
        save_results = f"results_{self.backend}"
        if not os.path.exists(save_results):
            os.makedirs(save_results)
        in_datas = dict()
        for idx, img_path in enumerate(tqdm(img_paths)):
            basename, _ = os.path.splitext(os.path.basename(img_path))
            image_id = dataset.get_image_id(basename)
            out_path = os.path.join(save_results, f"{image_id}.json")
            if os.path.exists(out_path):
                continue
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning(f"{img_path} not exists or decode failed")
                continue
            in_datas[self.input_name] = cv_image
            logger.debug(f"Image[{idx}] {img_path}")
            detections, masks, contours = self.run(in_datas)
            detections_mask2json(detections, contours, out_path)
        pred_json = f"pred_{self.backend}.json"
        merge_json(save_results, pred_json)
        map50_95, map50 = coco_eval(
            pred_json,
            dataset.annotations_file,
            dataset.image_ids,
            iou_type="segm",
        )
        return {
            "input_size": self.inputs_cfg[self.input_name]["shape"],
            "dataset": dataset.dataset_name,
            "num": len(img_paths),
            "map50_95": f"{map50_95:.6f}",
            "map50": f"{map50:.6f}",
            "latency": f"{self.ave_latency_ms:.6f}",
        }
