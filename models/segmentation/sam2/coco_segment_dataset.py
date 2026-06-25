# Copyright 2025 HOUMO AI
#
# File: coco_segment_dataset.py
# Description:
#   COCO segmentation dataset and evaluation helper for SAM2.
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

import json
import os
import time
import traceback

import cv2
import numpy as np
from tqdm import tqdm


def coco80_to_coco91_class():
    """Convert COCO 80-class index to COCO 91-class category id."""
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
    ]


def detections_mask2json(detections, contours_lists, filepath):
    """Write detection and contour results to one COCO segmentation JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        if not contours_lists:
            return
        image_id = int(os.path.splitext(os.path.basename(filepath))[0])
        pred_lists = []
        category_ids = coco80_to_coco91_class()
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = [int(value) for value in det[0:4]]
            conf = float(det[4])
            cls = int(det[5])
            contours = contours_lists[idx]
            new_contours = []
            area = 0
            for contour in contours:
                if contour.shape[0] <= 2:
                    continue
                area += cv2.contourArea(contour)
                new_contour = contour.flatten().tolist()
                if len(new_contour) == 4:
                    new_contour.append(new_contour[-1])
                new_contours.append(new_contour)
            if not new_contours:
                continue
            pred_lists.append({
                "image_id": image_id,
                "category_id": category_ids[cls],
                "bbox": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                "score": conf,
                "segmentation": new_contours,
                "area": area,
                "iscrowd": 0,
            })
        f.write(json.dumps(pred_lists))


def merge_json(save_results, pred_json):
    """Merge per-image COCO segmentation JSON files."""
    results = []
    for filename in os.listdir(save_results):
        if os.path.splitext(filename)[1] != ".json":
            continue
        with open(os.path.join(save_results, filename), "r", encoding="utf-8") as f:
            line = f.read().strip()
            if not line:
                continue
            detections = json.loads(line)
            if detections:
                results.extend(detections)
    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(results, f)


def coco_eval(pred_json, anno_json, image_ids, iou_type="segm"):
    """Run pycocotools COCO evaluation."""
    print(f"[info] Evaluating pycocotools mAP... saving {pred_json}...")
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(anno_json)
        pred = coco_gt.loadRes(pred_json)
        evaluator = COCOeval(coco_gt, pred, iou_type)
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return evaluator.stats[:2]
    except Exception as e:
        print(f"[error] pycocotools unable to run: {e}\n{traceback.format_exc()}")
        raise


class CocoSegmentDataset:
    """COCO val2017 segmentation dataset and evaluation helper for SAM2."""

    dataset_name = "coco_val2017"

    def __init__(self, dataset_dir, num=0, max_ann_per_image=0, output_dir=None):
        from pycocotools.coco import COCO

        self.dataset_dir = dataset_dir
        self.max_ann_per_image = max_ann_per_image
        self.output_dir = output_dir
        self.image_dir = os.path.join(dataset_dir, "val2017")
        self.annotations_file = os.path.join(
            dataset_dir, "annotations", "instances_val2017.json"
        )
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"COCO val image directory not found: {self.image_dir}")
        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(f"COCO annotation file not found: {self.annotations_file}")

        self.coco = COCO(self.annotations_file)
        self.image_ids = self._existing_image_ids(num)
        self.category_ids = self._build_category_ids()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = self._read_image(image_path)
        return {
            "path": image_path,
            "image": image,
            "image_id": image_id,
            "anns": self._load_annotations(image_id),
        }

    def eval(self, engine, num=0):
        """Evaluate segmentation mAP with a SAM2Engine instance."""
        total = len(self) if num == 0 else min(num, len(self))
        if total == 0:
            raise RuntimeError("No eval data found")

        save_results = self._save_results_dir(engine.backend)
        if not os.path.exists(save_results):
            os.makedirs(save_results)

        time_span = 0
        evaluated = 0
        for idx in tqdm(range(total)):
            sample = self[idx]
            image_id = sample.get("image_id")
            out_path = os.path.join(save_results, f"{image_id}.json")
            if os.path.exists(out_path):
                continue
            print(f"[debug] Image[{idx}] {sample.get('path', '')}")
            detections, _, contours, latency = self.run(engine, sample)
            time_span += latency
            evaluated += 1
            detections_mask2json(detections, contours, out_path)

        pred_json = self._pred_json_path(engine.backend)
        merge_json(save_results, pred_json)
        eval_image_ids = self.image_ids[:total]
        map50_95, map50 = coco_eval(
            pred_json,
            self.annotations_file,
            eval_image_ids,
            iou_type="segm",
        )
        return self._metric_result(engine, total, time_span, evaluated, map50_95, map50)

    def run(self, engine, sample):
        """Run SAM2 prompt segmentation for one COCO image sample."""
        image = sample.get("image")
        image_tensor, scale, new_h, new_w, h, w = engine.preprocess(image)
        image_features = engine.encode(image_tensor)
        detections = []
        masks = []
        contours = []

        start_time = time.time()
        for ann in sample.get("anns", []):
            result = self._predict_annotation(
                engine, image_features, ann, scale, new_h, new_w, h, w
            )
            if result is None:
                continue
            detection, mask, contour = result
            detections.append(detection)
            masks.append(mask)
            contours.append(contour)
        latency = time.time() - start_time
        return np.asarray(detections, dtype=np.float32), masks, contours, latency

    def _existing_image_ids(self, num):
        image_ids = []
        for image_id in self.coco.getImgIds():
            image_info = self.coco.loadImgs([image_id])[0]
            image_path = os.path.join(self.image_dir, image_info["file_name"])
            if os.path.exists(image_path):
                image_ids.append(image_id)
        if num > 0:
            return image_ids[:num]
        return image_ids

    def _load_annotations(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        anns = [ann for ann in anns if ann.get("area", 0) > 0]
        if self.max_ann_per_image > 0:
            return anns[: self.max_ann_per_image]
        return anns

    def _save_results_dir(self, backend):
        if self.output_dir is None:
            return f"results_{backend}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        return os.path.join(self.output_dir, f"results_{backend}")

    def _pred_json_path(self, backend):
        if self.output_dir is None:
            return f"pred_{backend}.json"
        return os.path.join(self.output_dir, f"pred_{backend}.json")

    @staticmethod
    def _read_image(image_path):
        image_data = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        return image

    @staticmethod
    def _build_category_ids():
        return coco80_to_coco91_class()

    def _predict_annotation(self, engine, image_features, ann, scale, new_h, new_w, h, w):
        coords, labels = self._build_prompt_from_bbox(ann["bbox"], scale)
        if coords is None:
            return None

        decoder_outs = engine.decode(image_features, coords, labels)
        mask, score = engine.postprocess(decoder_outs, new_h, new_w, h, w)
        contour = self._mask_to_contour(mask)
        if not contour:
            return None

        x, y, box_w, box_h = ann["bbox"]
        cls_idx = self._category_to_coco80_index(ann["category_id"])
        detection = [x, y, x + box_w, y + box_h, score, cls_idx]
        return detection, mask, contour

    @staticmethod
    def _build_prompt_from_bbox(bbox, scale):
        x, y, box_w, box_h = bbox
        if box_w <= 1 or box_h <= 1:
            return None, None

        x1 = x * scale
        y1 = y * scale
        x2 = (x + box_w) * scale
        y2 = (y + box_h) * scale
        cx = (x + box_w * 0.5) * scale
        cy = (y + box_h * 0.5) * scale
        coords = np.array([[[cx, cy], [x1, y1], [x2, y2]]], dtype=np.float32)
        labels = np.array([[1, 2, 3]], dtype=np.float32)
        return coords, labels

    @staticmethod
    def _mask_to_contour(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if isinstance(contours, tuple):
            contours = list(contours)
        return [contour for contour in contours if contour.shape[0] > 2]

    def _category_to_coco80_index(self, category_id):
        if category_id in self.category_ids:
            return self.category_ids.index(category_id)
        return 0

    def _metric_result(self, engine, total, time_span, evaluated, map50_95, map50):
        ave_latency_ms = 0
        if evaluated > 0:
            ave_latency_ms = time_span / evaluated * 1000
        return {
            "backend": engine.backend,
            "input_size": [1, 3, engine.target_size, engine.target_size],
            "dataset": self.dataset_name,
            "num": total,
            "map50_95": f"{map50_95:.6f}",
            "map50": f"{map50:.6f}",
            "latency": f"{ave_latency_ms:.6f}",
        }
