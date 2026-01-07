# Copyright 2025 HOUMO AI
#
# File: model_impl.py
# Description:
#   YOLOv5M face detection model implementation with 5 facial landmarks.
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
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)

import cv2
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from hmatc.utils import logger
from hmatc.base.base_model import BaseModel
from hmatc.utils.postprocess import (
    xywh2xyxy,
    xyxy2xywh,
    scale_coords,
    scale_coords_landmarks,
)
from hmatc.utils.metrics import detections_face2txt
from evaluation import evaluation


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
):
    """
    Perform non-maximum suppression on face detection predictions.

    Filters detections based on confidence threshold and applies NMS to
    remove duplicate detections based on IoU threshold.

    Args:
        prediction: Model predictions tensor with shape [batch_size, num_anchors, 16]
                   where 16 = 4 bbox coords + 1 obj conf + 10 landmarks + 1 class
        conf_thres: Confidence threshold for filtering detections
        iou_thres: IoU threshold for non-maximum suppression

    Returns:
        List of filtered detections for each image in the batch
    """
    xc = prediction[..., 4] > conf_thres  # candidates
    max_wh = 4096
    output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 15:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 15:].max(1, keepdim=True)
        x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]
        n = x.shape[0]
        if not n:
            continue
        c = x[:, 15:16] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[i]

    return output


class YoloV5MFace(BaseModel):
    """
    YOLOv5M face detection model implementation.

    This class implements the YOLOv5M model for face detection tasks,
    including 5 facial landmarks detection. It provides functionality
    for post-processing model outputs, running inference on images,
    and evaluating model performance.
    """

    def __init__(self, **kwargs):
        """
        Initialize YoloV5MFace model instance.

        Args:
            **kwargs: Keyword arguments passed to the parent BaseModel class
        """
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)  # HW
        self.conf_threshold = 0.2
        self.iou_threshold = 0.5

    def postprocess(
        self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]
    ) -> Any:
        """
        Post-process the model outputs to extract face detections and landmarks.

        Applies non-maximum suppression and scales the detection coordinates
        and landmarks from the model input size to the original image size.

        Args:
            outs: Model outputs as dictionary of numpy arrays with shape [batch_size, 25200, 16]
                  where 16 = 4 bbox coords + 1 obj conf + 10 landmarks + 1 class
            in_datas: Input data as dictionary of numpy arrays

        Returns:
            Tuple of (boxes, landmarks_list) where:
            - boxes: List of face bounding boxes [x, y, width, height, confidence]
            - landmarks_list: List of facial landmarks for each detection
        """
        # 4 (c_x, c_y, w, h) + 1 (obj_conf) + 10 (x1, y1, x2, y2...) + 1 (class_num) = 16
        pred = list(outs.values())[0]  # [bs, 25200, 16]
        pred = torch.from_numpy(pred)
        cv_image = list(in_datas.values())[0]
        output = non_max_suppression(
            pred,
            self.conf_threshold,
            self.iou_threshold,
        )[0]
        gn = torch.tensor(cv_image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        gn_lks = torch.tensor(cv_image.shape)[
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        ]  # normalization gain landmarks
        h, w, c = cv_image.shape

        output[:, :4] = scale_coords(
            self.input_size, output[:, :4], cv_image.shape
        ).round()
        output[:, 5:15] = scale_coords_landmarks(
            self.input_size, output[:, 5:15], cv_image.shape
        ).round()
        boxes = []
        landmarks_list = []
        for j in range(output.size()[0]):
            xywh = (xyxy2xywh(output[j, :4].view(1, 4)) / gn).view(-1)
            xywh = xywh.data.cpu().numpy()
            conf = output[j, 4].cpu().numpy()
            landmarks = (output[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)

            boxes.append([x1, y1, (x2 - x1), (y2 - y1), conf])
            landmarks_list.append(landmarks)

        return boxes, landmarks_list

    def demo(self, filepaths: list):
        """
        Run face detection inference on input images and visualize results.

        Performs face detection on the input images, draws bounding boxes
        and facial landmarks on detected faces, and saves the annotated
        images to the output directory.

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
            outs, landmark_list = self.run(in_datas)
            h, w, c = cv_image.shape
            tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
            for idx, detection in enumerate(outs):
                x1 = int(detection[0])
                y1 = int(detection[1])
                x2 = x1 + int(detection[2])
                y2 = y1 + int(detection[3])
                conf = detection[4]
                landmarks = landmark_list[idx]
                logger.info(
                    f"Detection[{idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, conf: {conf:.3f}"
                )
                cv2.rectangle(
                    cv_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    thickness=tl,
                    lineType=cv2.LINE_AA,
                )

                clors = [
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (255, 255, 0),
                    (0, 255, 255),
                ]
                for i in range(5):
                    point_x = int(landmarks[2 * i] * w)
                    point_y = int(landmarks[2 * i + 1] * h)
                    cv2.circle(cv_image, (point_x, point_y), tl + 1, clors[i], -1)
                tf = max(tl - 1, 1)  # font thickness
                label = str(conf)[:5]
                cv2.putText(
                    cv_image,
                    label,
                    (x1, y1 - 2),
                    0,
                    tl / 3,
                    [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )
            cv2.imwrite(save_path, cv_image)
            logger.info(f"Save result to {save_path}")

    def evaluate(self, dataset, num=0):
        """
        Evaluate model performance on a face detection dataset.

        Runs inference on all images in the dataset and calculates
        face detection evaluation metrics.

        Args:
            dataset: Dataset object containing images and annotations
            num: Number of samples to evaluate (0 means all samples)

        Returns:
            Dictionary containing evaluation metrics:
            - input_size: Input shape of the model
            - dataset: Name of the dataset
            - num: Number of evaluated samples
            - ap_easy: Average precision for easy faces
            - ap_medium: Average precision for medium faces
            - ap_hard: Average precision for hard faces
        """
        self.iou_threshold = 0.5
        self.conf_threshold = 0.02
        img_paths = dataset.get_datas(num)
        save_results = f"results_{self.backend}"
        if not os.path.exists(save_results):
            os.makedirs(save_results)
        in_datas = dict()
        for idx, img_path in enumerate(tqdm(img_paths)):
            image_name = os.path.basename(img_path)
            txt_name = os.path.splitext(image_name)[0] + ".txt"
            folder_name = img_path.rsplit("/", 2)[-2]
            out_path = os.path.join(save_results, folder_name, txt_name)
            dirname = os.path.dirname(out_path)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning(f"{img_path} not exists or decode failed")
                continue
            in_datas[self.input_name] = cv_image
            logger.debug(f"Image[{idx}] {img_path}")
            detections, _ = self.run(in_datas)
            detections_face2txt(detections, out_path)

        aps = evaluation(save_results, dataset.annotation_path)
        return {
            "input_size": self.inputs_cfg[self.input_name]["shape"],
            "dataset": dataset.dataset_name,
            "num": len(img_paths),
            "ap_easy": f"{aps[0]:.6f}",
            "ap_medium": f"{aps[1]:.6f}",
            "ap_hard": f"{aps[2]:.6f}",
        }
