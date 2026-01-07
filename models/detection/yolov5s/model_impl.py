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
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from hmatc.utils import logger
from hmatc.base.base_model import BaseModel
from hmatc.utils.postprocess import non_max_suppression, scale_coords
from hmatc.utils.metrics import detections2txt, detection_txt2json, coco_eval


class YoloV5(BaseModel):
    """
    YOLOv5 object detection model implementation.

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

        Either decodes the model outputs using anchor-based transformation if there are 3 outputs,
        otherwise uses the single output directly. Then applies non-maximum suppression and
        scales the detection coordinates from the model input size to the original image size.

        Args:
            outs: Model outputs as dictionary of numpy arrays
            in_datas: Input data as dictionary of numpy arrays

        Returns:
            Processed detections as numpy array with format [x1, y1, x2, y2, confidence, class_idx]
        """
        if len(outs) == 3:
            pred = self.decode(outs)  # [bs, 25200, 85]
        else:
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

    def demo(self, filepaths: list):
        """
        Run inference on input images and visualize results.

        Performs object detection on the input images, draws bounding boxes
        on detected objects, and saves the annotated images to the output directory.

        Args:
            filepaths: List of image file paths to process
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
            outs = self.run(in_datas)
            for idx, detection in enumerate(outs):
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
            cv2.imwrite(save_path, cv_image)
            logger.info(f"Save result to {save_path}")

    def evaluate(self, dataset, num=0):
        """
        Evaluate model performance on a dataset.

        Runs inference on all images in the dataset and calculates COCO metrics
        including mAP@0.5:0.95 and mAP@0.5.

        Args:
            dataset: Dataset object containing images and annotations
            num: Number of samples to evaluate (0 means all samples)

        Returns:
            Dictionary containing evaluation metrics:
            - input_size: Input shape of the model
            - dataset: Name of the dataset
            - num: Number of evaluated samples
            - map50_95: Mean Average Precision at IoU 0.5:0.05:0.95
            - map50: Mean Average Precision at IoU 0.5
            - latency: Average inference latency
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
            out_path = os.path.join(save_results, f"{image_id}.txt")
            if os.path.exists(out_path):
                continue
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning(f"{img_path} not exists or decode failed")
                continue
            in_datas[self.input_name] = cv_image
            logger.debug(f"Image[{idx}] {img_path}")
            detections = self.run(in_datas)
            detections2txt(detections, out_path)
        pred_json = f"pred_{self.backend}.json"
        detection_txt2json(save_results, pred_json, to_coco91=self.to_coco91)
        map50_95, map50 = coco_eval(
            pred_json, dataset.annotations_file, dataset.image_ids
        )
        return {
            "input_size": self.inputs_cfg[self.input_name]["shape"],
            "dataset": dataset.dataset_name,
            "num": len(img_paths),
            "map50_95": f"{map50_95:.6f}",
            "map50": f"{map50:.6f}",
            "latency": f"{self.ave_latency_ms:.6f}",
        }
