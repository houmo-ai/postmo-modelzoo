# Copyright 2025 HOUMO AI
#
# File: task_models.py
# Description:
#   Base classes for different types of models.
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
import numpy as np
from tqdm import tqdm

from .base_model import BaseModel, COLORS
from ..utils import logger
from ..datasets.imagenet import ILSVRC2012_LABELS
from ..utils.metrics import (
    coco_eval,
    detection_txt2json,
    detections2txt,
)


class ClassificationModel(BaseModel):
    """Base class for ImageNet-style classification model examples."""

    def demo(self, dataloader):
        for idx in range(len(dataloader)):
            sample = dataloader[idx]
            meta = sample.get("meta", {})
            path = meta.get("path", "")
            logger.info(f"[{idx}] {path}")

            outs = self.run(sample)
            cls_idx, score = outs[0]
            cls_idx = str(cls_idx)
            cls_name = ILSVRC2012_LABELS[cls_idx][0]
            logger.info(f"score: {score:.3f}, cls_idx: {cls_idx}, cls_name: {cls_name}")

    def evaluate(self, dataloader, num=0):
        total = len(dataloader) if num == 0 else min(num, len(dataloader))
        if total == 0:
            logger.fatal("No eval data found")

        top1_acc = 0
        for idx in tqdm(range(total)):
            sample = dataloader[idx]
            meta = sample.get("meta", {})
            label = meta.get("label")
            if label is None:
                logger.fatal("Classification eval requires sample['meta']['label']")

            logger.debug(f"[{idx}] {meta.get('path', '')}")
            outs = self.run(sample)
            cls_idx = str(outs[0][0])
            if cls_idx == str(label):
                top1_acc += 1

        return {
            "input_size": self.inputs_cfg[self.input_name]["shape"],
            "dataset": getattr(
                dataloader, "dataset_name", dataloader.__class__.__name__
            ),
            "num": total,
            "top1_acc": f"{top1_acc / total:.6f}",
            "latency_ms": f"{self.ave_latency_ms:.6f}",
        }


class CocoDetectionModel(BaseModel):
    """Base class for COCO-style detection model examples."""

    def demo(self, dataloader):
        save_dir = f"vis_{self.backend}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(dataloader)):
            sample = dataloader[idx]
            meta = sample.get("meta", {})
            filepath = meta.get("path", "")
            cv_image = meta.get("image")
            if cv_image is None:
                logger.fatal("Detection demo requires sample['meta']['image']")

            basename, _ = os.path.splitext(os.path.basename(filepath))
            save_path = os.path.join(save_dir, f"{basename}.jpg")
            logger.info(f"Image[{idx}] {filepath}")

            outs = self.run(sample)
            self.draw_detections(cv_image, outs)
            cv2.imwrite(save_path, cv_image)
            logger.info(f"Save result to {save_path}")

    def draw_detections(self, cv_image, detections):
        for det_idx, detection in enumerate(detections):
            x1, y1, x2, y2, score, cls_idx = detection[:6]
            x1 = int(x1) if x1 > 0 else 0
            y1 = int(y1) if y1 > 0 else 0
            x2 = int(x2) if x2 < cv_image.shape[1] else cv_image.shape[1]
            y2 = int(y2) if y2 < cv_image.shape[0] else cv_image.shape[0]
            cls_idx = int(cls_idx)
            logger.info(
                f"Detection[{det_idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, score: {score:.3f}, cls: {cls_idx:2}"
            )
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    def evaluate(self, dataloader, num=0):
        self.iou_threshold = 0.65
        self.conf_threshold = 0.01
        total = len(dataloader) if num == 0 else min(num, len(dataloader))
        if total == 0:
            logger.fatal("No eval data found")

        save_results = f"results_{self.backend}"
        if not os.path.exists(save_results):
            os.makedirs(save_results)

        for idx in tqdm(range(total)):
            sample = dataloader[idx]
            meta = sample.get("meta", {})
            image_id = meta.get("image_id")
            if image_id is None:
                logger.fatal("COCO eval requires sample['meta']['image_id']")

            out_path = os.path.join(save_results, f"{image_id}.txt")
            if os.path.exists(out_path):
                continue
            logger.debug(f"Image[{idx}] {meta.get('path', '')}")
            detections = self.run(sample)
            detections2txt(detections, out_path)

        pred_json = f"pred_{self.backend}.json"
        to_coco91 = getattr(self, "to_coco91", False)
        detection_txt2json(save_results, pred_json, to_coco91=to_coco91)
        map50_95, map50 = coco_eval(
            pred_json, dataloader.annotations_file, dataloader.image_ids
        )
        return self.metric_result(dataloader, total, map50_95, map50)

    def metric_result(self, dataloader, total, map50_95, map50):
        return {
            "input_size": self.inputs_cfg[self.input_name]["shape"],
            "dataset": getattr(
                dataloader, "dataset_name", dataloader.__class__.__name__
            ),
            "num": total,
            "map50_95": f"{map50_95:.6f}",
            "map50": f"{map50:.6f}",
            "latency": f"{self.ave_latency_ms:.6f}",
        }


class CocoSegmentationModel(CocoDetectionModel):
    """Base class for COCO instance segmentation examples."""

    def demo(self, dataloader):
        save_dir = f"vis_{self.backend}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(dataloader)):
            sample = dataloader[idx]
            meta = sample.get("meta", {})
            filepath = meta.get("path", "")
            cv_image = meta.get("image")
            if cv_image is None:
                logger.fatal("Segmentation demo requires sample['meta']['image']")

            basename, _ = os.path.splitext(os.path.basename(filepath))
            save_path = os.path.join(save_dir, f"{basename}.jpg")
            logger.info(f"Image[{idx}] {filepath}")

            detections, masks, contours = self.run(sample)
            for det_idx, detection in enumerate(detections):
                x1, y1, x2, y2, score, cls_idx = detection[0:6]
                color = np.array(COLORS[int(cls_idx) % len(COLORS)])
                mask = masks[det_idx]
                new_masks = np.array([mask, mask, mask]).transpose((1, 2, 0))
                cv_image = np.where(
                    new_masks == 255, cv_image * 0.5 + color * 0.5, cv_image
                )
                contour = contours[det_idx]
                cv2.drawContours(cv_image, contour, -1, color.tolist(), 2)
                x1 = int(x1) if x1 > 0 else 0
                y1 = int(y1) if y1 > 0 else 0
                x2 = int(x2) if x2 < cv_image.shape[1] else cv_image.shape[1]
                y2 = int(y2) if y2 < cv_image.shape[0] else cv_image.shape[0]
                logger.info(
                    f"Detection[{det_idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, score: {score:.3f}, cls: {int(cls_idx)}"
                )
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color.tolist(), 2)

            cv2.imwrite(save_path, cv_image)
            logger.info(f"Save result to {save_path}")

    def evaluate(self, dataloader, num=0):
        from ..utils.metrics import detections_mask2json, merge_json

        self.iou_threshold = 0.65
        self.conf_threshold = 0.01
        total = len(dataloader) if num == 0 else min(num, len(dataloader))
        if total == 0:
            logger.fatal("No eval data found")

        save_results = f"results_{self.backend}"
        if not os.path.exists(save_results):
            os.makedirs(save_results)

        for idx in tqdm(range(total)):
            sample = dataloader[idx]
            meta = sample.get("meta", {})
            image_id = meta.get("image_id")
            if image_id is None:
                logger.fatal("COCO seg eval requires sample['meta']['image_id']")
            out_path = os.path.join(save_results, f"{image_id}.json")
            if os.path.exists(out_path):
                continue
            logger.debug(f"Image[{idx}] {meta.get('path', '')}")
            detections, _, contours = self.run(sample)
            detections_mask2json(detections, contours, out_path)

        pred_json = f"pred_{self.backend}.json"
        merge_json(save_results, pred_json)
        map50_95, map50 = coco_eval(
            pred_json,
            dataloader.annotations_file,
            dataloader.image_ids,
            iou_type="segm",
        )
        return self.metric_result(dataloader, total, map50_95, map50)


class CocoPoseModel(CocoDetectionModel):
    """Base class for COCO keypoint examples."""

    def demo(self, dataloader):
        from ..utils.postprocess import plot_skeleton_kpts

        save_dir = f"vis_{self.backend}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(dataloader)):
            sample = dataloader[idx]
            meta = sample.get("meta", {})
            filepath = meta.get("path", "")
            cv_image = meta.get("image")
            if cv_image is None:
                logger.fatal("Pose demo requires sample['meta']['image']")

            basename, _ = os.path.splitext(os.path.basename(filepath))
            save_path = os.path.join(save_dir, f"{basename}.jpg")
            logger.info(f"Image[{idx}] {filepath}")

            outs = self.run(sample)
            for det_idx, detection in enumerate(outs):
                plot_skeleton_kpts(cv_image, detection[6:].T, 3)
                x1, y1, x2, y2, score, cls_idx = detection[0:6]
                x1 = int(x1) if x1 > 0 else 0
                y1 = int(y1) if y1 > 0 else 0
                x2 = int(x2) if x2 < cv_image.shape[1] else cv_image.shape[1]
                y2 = int(y2) if y2 < cv_image.shape[0] else cv_image.shape[0]
                logger.info(
                    f"Detection[{det_idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, score: {score:.3f}, cls: {int(cls_idx)}"
                )
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(save_path, cv_image)
            logger.info(f"Save result to {save_path}")

    def evaluate(self, dataloader, num=0):
        from ..utils.metrics import detections_kpt2json, merge_json

        self.iou_threshold = 0.65
        self.conf_threshold = 0.01
        total = len(dataloader) if num == 0 else min(num, len(dataloader))
        if total == 0:
            logger.fatal("No eval data found")

        save_results = f"results_{self.backend}"
        if not os.path.exists(save_results):
            os.makedirs(save_results)

        for idx in tqdm(range(total)):
            sample = dataloader[idx]
            meta = sample.get("meta", {})
            image_id = meta.get("image_id")
            if image_id is None:
                logger.fatal("COCO pose eval requires sample['meta']['image_id']")
            out_path = os.path.join(save_results, f"{image_id}.json")
            if os.path.exists(out_path):
                continue
            logger.debug(f"Image[{idx}] {meta.get('path', '')}")
            detections = self.run(sample)
            detections_kpt2json(detections, out_path)

        pred_json = f"pred_{self.backend}.json"
        merge_json(save_results, pred_json)
        map50_95, map50 = coco_eval(
            pred_json,
            dataloader.annotations_kpt,
            dataloader.image_ids,
            iou_type="keypoints",
        )
        return self.metric_result(dataloader, total, map50_95, map50)


class WiderFaceModel(BaseModel):
    """Base class for WiderFace face detection examples."""

    def demo(self, dataloader):
        save_dir = f"vis_{self.backend}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(dataloader)):
            sample = dataloader[idx]
            meta = sample.get("meta", {})
            filepath = meta.get("path", "")
            cv_image = meta.get("image")
            if cv_image is None:
                logger.fatal("WiderFace demo requires sample['meta']['image']")

            basename, _ = os.path.splitext(os.path.basename(filepath))
            save_path = os.path.join(save_dir, f"{basename}.jpg")
            logger.info(f"Image[{idx}] {filepath}")
            boxes, landmarks_list = self.run(sample)
            h, w, _ = cv_image.shape
            tl = 1
            for det_idx, detection in enumerate(boxes):
                x1 = int(detection[0])
                y1 = int(detection[1])
                x2 = x1 + int(detection[2])
                y2 = y1 + int(detection[3])
                conf = detection[4]
                landmarks = landmarks_list[det_idx]
                logger.info(
                    f"Detection[{det_idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, conf: {conf:.3f}"
                )
                cv2.rectangle(
                    cv_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    thickness=tl,
                    lineType=cv2.LINE_AA,
                )
                colors = [
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (255, 255, 0),
                    (0, 255, 255),
                ]
                for i in range(5):
                    point_x = int(landmarks[2 * i] * w)
                    point_y = int(landmarks[2 * i + 1] * h)
                    cv2.circle(cv_image, (point_x, point_y), tl + 1, colors[i], -1)
                cv2.putText(
                    cv_image,
                    str(conf)[:5],
                    (x1, y1 - 2),
                    0,
                    tl / 3,
                    [225, 255, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            cv2.imwrite(save_path, cv_image)
            logger.info(f"Save result to {save_path}")

    def evaluate(self, dataloader, num=0):
        from ..utils.metrics import detections_face2txt
        from evaluation import evaluation

        self.iou_threshold = 0.5
        self.conf_threshold = 0.02
        total = len(dataloader) if num == 0 else min(num, len(dataloader))
        save_results = f"results_{self.backend}"
        if not os.path.exists(save_results):
            os.makedirs(save_results)

        for idx in tqdm(range(total)):
            sample = dataloader[idx]
            meta = sample.get("meta", {})
            img_path = meta.get("path", "")
            image_name = os.path.basename(img_path)
            txt_name = os.path.splitext(image_name)[0] + ".txt"
            folder_name = img_path.rsplit("/", 2)[-2]
            out_path = os.path.join(save_results, folder_name, txt_name)
            dirname = os.path.dirname(out_path)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            logger.debug(f"Image[{idx}] {img_path}")
            detections, _ = self.run(sample)
            detections_face2txt(detections, out_path)

        aps = evaluation(save_results, dataloader.annotation_path)
        return {
            "input_size": self.inputs_cfg[self.input_name]["shape"],
            "dataset": getattr(
                dataloader, "dataset_name", dataloader.__class__.__name__
            ),
            "num": total,
            "ap_easy": f"{aps[0]:.6f}",
            "ap_medium": f"{aps[1]:.6f}",
            "ap_hard": f"{aps[2]:.6f}",
        }


class YoloPDemoModel(BaseModel):
    """Base class for YoloP-style demo-only examples."""

    def demo(self, dataloader):
        save_dir = f"vis_{self.backend}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx in range(len(dataloader)):
            sample = dataloader[idx]
            meta = sample.get("meta", {})
            filepath = meta.get("path", "")
            cv_image = meta.get("image")
            if cv_image is None:
                logger.fatal("YoloP demo requires sample['meta']['image']")
            basename, _ = os.path.splitext(os.path.basename(filepath))
            save_path = os.path.join(save_dir, f"{basename}.jpg")
            logger.info(f"Image[{idx}] {filepath}")
            det_outs, mask = self.run(sample)
            for det_idx, detection in enumerate(det_outs):
                x1, y1, x2, y2, score, cls_idx = detection
                x1 = int(x1) if x1 > 0 else 0
                y1 = int(y1) if y1 > 0 else 0
                x2 = int(x2) if x2 < cv_image.shape[1] else cv_image.shape[1]
                y2 = int(y2) if y2 < cv_image.shape[0] else cv_image.shape[0]
                logger.info(
                    f"Detection[{det_idx:2}] x1: {x1:4}, y1: {y1:4}, x2: {x2:4}, y2: {y2:4}, score: {score:.3f}, cls: {int(cls_idx):2}"
                )
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv_image = np.where(
                mask == 255, cv_image * 0.5 + mask * 0.5, cv_image
            ).astype(np.uint8)
            cv2.imwrite(save_path, cv_image)
            logger.info(f"Save result to {save_path}")

    def evaluate(self, dataloader, num=0):
        raise NotImplementedError("Evaluation is not implemented for YoloP model.")
