#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: demo.py
# Description:
#   PaddleOCR V5 detection + recognition HMM inference demo on HOUMO AI device.
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

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import pyclipper
    from shapely.geometry import Polygon
except ImportError:
    pyclipper = None
    Polygon = None

import tcim_lite as tcim


SCRIPT_DIR = Path(__file__).resolve().parent
IMODELZOO_UTILS_PATH = SCRIPT_DIR.parents[2] / "utils" / "python"
if str(IMODELZOO_UTILS_PATH) not in sys.path:
    sys.path.insert(0, str(IMODELZOO_UTILS_PATH))
from houmo_engine.perf import PerfTracker

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


class CTCLabelDecode:
    def __init__(self, character_dict_path: str, use_space_char: bool = True):
        dict_character = []
        with open(character_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                # Keep literal space entries. PPOCRv5 dict starts with a space
                # character; dropping it shifts every following class id and
                # makes CTC decoding produce wrong text.
                dict_character.append(line.rstrip("\n\r"))
        if use_space_char:
            dict_character.append(" ")
        self.character = ["blank"] + dict_character


    def __call__(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        return self.decode(preds_idx, preds_prob, is_remove_duplicate=True)

    def decode(
        self,
        text_index: np.ndarray,
        text_prob: np.ndarray | None = None,
        is_remove_duplicate: bool = False,
    ) -> List[Tuple[str, float]]:
        result_list = []
        ignored_tokens = {0}
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token
            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
                if 0 <= text_id < len(self.character)
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1.0] * len(char_list)
            if len(conf_list) == 0:
                conf_list = [0.0]
            result_list.append(("".join(char_list), float(np.mean(conf_list))))
        return result_list


def _as_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _module_input_shape(module) -> List[int]:
    name = module.get_input_name(0)
    return [int(dim) for dim in module.get_input_info(name).shape]


def _module_output(module):
    return _as_numpy(module.get_output(module.get_output_name(0)))


def _resize_det_image(image: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    resized = resized.astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32").reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype="float32").reshape(1, 1, 3)
    resized = (resized - mean) / std
    return resized.transpose(2, 0, 1)[np.newaxis, :].astype("float16")


def _resize_rec_image(image: np.ndarray, input_h: int, input_w: int, use_rgb: bool = False) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((3, input_h, input_w), dtype="float16")
    if use_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Match PaddleOCR recognition preprocessing:
    #   1. fixed height + aspect-ratio width
    #   2. normalize by mean/std 0.5
    #   3. right padding with normalized value 0
    ratio = w / float(h)
    resized_w = int(np.ceil(input_h * ratio))
    resized_w = min(input_w, max(1, resized_w))

    resized = cv2.resize(image, (resized_w, input_h), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype("float32") / 255.0
    resized = (resized - 0.5) / 0.5
    resized = resized.transpose(2, 0, 1)

    padded = np.zeros((3, input_h, input_w), dtype="float32")
    padded[:, :, :resized_w] = resized
    return padded.astype("float32")


def _get_mini_boxes(contour: np.ndarray):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2
    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def _box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    h, w = bitmap.shape[:2]
    box = box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
    crop = bitmap[ymin : ymax + 1, xmin : xmax + 1].astype("float32", copy=False)
    return cv2.mean(crop, mask)[0]


def _unclip(box: np.ndarray, unclip_ratio: float) -> np.ndarray:
    if pyclipper is None or Polygon is None:
        return box
    poly = Polygon(box)
    if poly.length == 0:
        return box
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if not expanded:
        return box
    return np.array(expanded)


def _boxes_from_bitmap(
    pred: np.ndarray,
    bitmap: np.ndarray,
    dest_width: int,
    dest_height: int,
    box_thresh: float,
    unclip_ratio: float,
    min_size: int,
    max_candidates: int,
) -> List[np.ndarray]:
    height, width = bitmap.shape
    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[0] if len(outs) == 2 else outs[1]
    boxes = []
    for contour in contours[:max_candidates]:
        points, sside = _get_mini_boxes(contour)
        if sside < min_size:
            continue
        points = np.array(points)
        score = _box_score_fast(pred, points.reshape(-1, 2))
        if score < box_thresh:
            continue

        box = _unclip(points, unclip_ratio)
        if box.ndim == 3 and len(box) > 1:
            continue
        box = np.array(box).reshape(-1, 1, 2)
        box, sside = _get_mini_boxes(box)
        if sside < min_size + 2:
            continue
        box = np.array(box)
        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.astype("int32"))
    return boxes


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(np.array(tmp), axis=1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]
    return rect


def _clip_det_res(points: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points


def _filter_tag_det_res(dt_boxes: Sequence[np.ndarray], image_shape: Tuple[int, int]) -> List[np.ndarray]:
    img_height, img_width = image_shape[:2]
    dt_boxes_new = []
    for box in dt_boxes:
        box = _order_points_clockwise(np.array(box, dtype="float32"))
        box = _clip_det_res(box, img_height, img_width)
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        dt_boxes_new.append(box.astype("int32"))
    return dt_boxes_new


def _sort_boxes(boxes: Sequence[np.ndarray]) -> List[np.ndarray]:
    num_boxes = len(boxes)
    sorted_boxes = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
    boxes = list(sorted_boxes)
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10 and boxes[j + 1][0][0] < boxes[j][0][0]:
                boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
            else:
                break
    return boxes


def _group_adjacent_boxes(boxes: Sequence[np.ndarray], image_shape: Tuple[int, int]) -> List[List[np.ndarray]]:
    if len(boxes) < 2:
        return [[box] for box in boxes]
    image_h, image_w = image_shape
    groups: List[List[np.ndarray]] = []
    used = [False] * len(boxes)
    for i, box in enumerate(boxes):
        if used[i]:
            continue
        group = [box]
        used[i] = True
        rect_i = cv2.minAreaRect(box.astype("float32"))
        angle_i = rect_i[2]
        center_i = np.array(rect_i[0], dtype="float32")
        width_i = max(rect_i[1])
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            rect_j = cv2.minAreaRect(boxes[j].astype("float32"))
            center_j = np.array(rect_j[0], dtype="float32")
            angle_diff = abs(angle_i - rect_j[2])
            angle_diff = min(angle_diff, 180.0 - angle_diff)
            center_dist = np.linalg.norm(center_i - center_j)
            if angle_diff > 15.0 or center_dist > width_i * 1.5:
                continue
            group.append(boxes[j])
            used[j] = True
        groups.append(group)
    return groups


def _merge_box_group(group: Sequence[np.ndarray], image_shape: Tuple[int, int]) -> np.ndarray:
    if len(group) == 1:
        return np.array(group[0]).astype("int32")
    image_h, image_w = image_shape
    points = np.concatenate(group, axis=0).astype("float32")
    merged_box, _ = _get_mini_boxes(points.reshape(-1, 1, 2))
    merged_box = np.array(merged_box)
    height_i = max(min(cv2.minAreaRect(group[0].astype("float32"))[1]), 1.0)
    pad = max(2.0, height_i * 0.15)
    rect = cv2.minAreaRect(merged_box.astype("float32"))
    rect = (rect[0], (rect[1][0] + pad, rect[1][1] + pad), rect[2])
    merged_box = cv2.boxPoints(rect)
    merged_box[:, 0] = np.clip(np.round(merged_box[:, 0]), 0, image_w)
    merged_box[:, 1] = np.clip(np.round(merged_box[:, 1]), 0, image_h)
    return np.array(_get_mini_boxes(merged_box.reshape(-1, 1, 2))[0]).astype("int32")


def _merge_adjacent_boxes(boxes: Sequence[np.ndarray], image_shape: Tuple[int, int]) -> List[np.ndarray]:
    return _sort_boxes([_merge_box_group(group, image_shape) for group in _group_adjacent_boxes(boxes, image_shape)])



def _expand_text_box(points: np.ndarray, image_shape: Tuple[int, int], pad_ratio: float) -> np.ndarray:
    if pad_ratio <= 0:
        return points.astype("float32")
    img_h, img_w = image_shape[:2]
    points = points.astype("float32")
    center = points.mean(axis=0, keepdims=True)
    expanded = center + (points - center) * (1.0 + pad_ratio * 2.0)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, img_w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, img_h - 1)
    return expanded.astype("float32")


def _crop_text_region(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    # Keep this function byte-for-byte close to PaddleOCR's
    # tools.infer.utility.get_rotate_crop_image(). Recognition accuracy is very
    # sensitive to interpolation, border handling, point order, and tall-crop
    # rotation.
    points = points.astype("float32")
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    img_crop_width = max(img_crop_width, 1)
    img_crop_height = max(img_crop_height, 1)
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    matrix = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        image,
        matrix,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / max(dst_img_width, 1) >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


class PaddleOCRV5Demo:
    def __init__(
        self,
        det_path: str,
        rec_path: str,
        character_dict_path: str,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.5,
        max_candidates: int = 1000,
        drop_score: float = 0.5,
        crop_pad_ratio: float = 0.25,
        rec_rgb: bool = False,
        debug_dir: str | None = None,
        perf: bool = False,
    ):
        self.perf = PerfTracker.create(perf)
        with self.perf.scope("ocr.init"):
            logger.info("Loading det model: {}", det_path)
            self.det = tcim.runtime.load(det_path)
            logger.info("Loading rec model: {}", rec_path)
            self.rec = tcim.runtime.load(rec_path)
            self.det_input_name = self.det.get_input_name(0)
            self.rec_input_name = self.rec.get_input_name(0)
            self.det_shape = _module_input_shape(self.det)
            self.rec_shape = _module_input_shape(self.rec)
            self.ctc_decode = CTCLabelDecode(character_dict_path, use_space_char=True)
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.min_size = 8
        self.max_candidates = max_candidates
        self.drop_score = drop_score
        self.crop_pad_ratio = crop_pad_ratio
        self.rec_rgb = rec_rgb
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info("det input shape: {}", self.det_shape)
        logger.info("rec input shape: {}", self.rec_shape)

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        _, _, input_h, input_w = self.det_shape
        with self.perf.scope("ocr.det.preprocess"):
            det_input = _resize_det_image(image, input_h, input_w)
        with self.perf.scope("ocr.det.set_input"):
            self.det.set_input(self.det_input_name, det_input)
        with self.perf.scope("ocr.det.infer"):
            self.det.run()
            self.det.sync()
        with self.perf.scope("ocr.det.get_output"):
            pred = _module_output(self.det).astype("float32", copy=False)
        with self.perf.scope("ocr.det.postprocess"):
            pred = pred[:, 0, :, :]
            mask = pred[0] > self.thresh
            src_h, src_w = image.shape[:2]
            boxes = _boxes_from_bitmap(
                pred[0],
                mask,
                src_w,
                src_h,
                box_thresh=self.box_thresh,
                unclip_ratio=self.unclip_ratio,
                min_size=self.min_size,
                max_candidates=self.max_candidates,
            )
            boxes = _filter_tag_det_res(boxes, image.shape)
            boxes = _sort_boxes(boxes)
        return boxes

    def _recognize_crops(self, crops: Sequence[np.ndarray], debug_names: Sequence[str] | None = None) -> List[Tuple[str, float]]:
        if not crops:
            return []
        rec_batch, _, rec_h, rec_w = self.rec_shape
        results = []
        for start in range(0, len(crops), rec_batch):
            batch_crops = crops[start : start + rec_batch]
            rec_input = _recognize_crop_batch(
                self,
                batch_crops,
                start,
                rec_batch,
                rec_h,
                rec_w,
                debug_names,
            )
            decoded = _run_recognition_batch(self, rec_input)
            results.extend(decoded[: len(batch_crops)])
        return results

    def recognize(self, image: np.ndarray, boxes: Sequence[np.ndarray]) -> List[Tuple[str, float]]:
        with self.perf.scope("ocr.rec.crop"):
            crops = [_crop_text_region(image, _expand_text_box(box, image.shape, self.crop_pad_ratio)) for box in boxes]
            debug_names = [f"{idx:03d}" for idx in range(len(crops))]
        return self._recognize_crops(crops, debug_names)

    def recognize_groups(self, image: np.ndarray, groups: Sequence[Sequence[np.ndarray]]) -> List[Tuple[str, float]]:
        crops = []
        debug_names = []
        group_spans = []
        for group_idx, group in enumerate(groups):
            sorted_group = sorted(group, key=lambda box: float(box[:, 0].mean()))
            start = len(crops)
            for member_idx, box in enumerate(sorted_group):
                with self.perf.scope("ocr.rec.crop"):
                    crops.append(_crop_text_region(image, _expand_text_box(box, image.shape, self.crop_pad_ratio)))
                debug_names.append(f"g{group_idx:03d}_{member_idx:03d}")
            group_spans.append((start, len(crops)))

        crop_results = self._recognize_crops(crops, debug_names)
        results = []
        for start, end in group_spans:
            parts = crop_results[start:end]
            text = "".join(part[0] for part in parts)
            score = float(np.mean([part[1] for part in parts])) if parts else 0.0
            results.append((text, score))
        return results

    def run(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        # PaddleOCR preprocessing pipelines conventionally consume OpenCV BGR
        # images. Keep BGR here; converting to RGB shifts channels and can make
        # the recognition logits decode to visually similar but wrong text.
        with self.perf.scope("ocr.e2e"):
            start = time.time()
            det_start = time.time()
            with self.perf.scope("ocr.det"):
                raw_boxes = self.detect(image)
            boxes = raw_boxes
            det_time = time.time() - det_start
            rec_start = time.time()
            with self.perf.scope("ocr.rec"):
                raw_rec_results = self.recognize(image, boxes)
            for idx, (box, rec_result) in enumerate(zip(boxes, raw_rec_results)):
                logger.info("raw_rec[{}] score={:.4f} box={} text={}", idx, rec_result[1], box.tolist(), rec_result[0])
            filtered_boxes = []
            filtered_rec_results = []
            for box, rec_result in zip(boxes, raw_rec_results):
                if rec_result[1] >= self.drop_score:
                    filtered_boxes.append(box)
                    filtered_rec_results.append(rec_result)
            boxes = filtered_boxes
            rec_results = filtered_rec_results
            rec_time = time.time() - rec_start
            total_time = time.time() - start
        if self.debug_dir:
            raw_debug_image = image.copy()
            for idx, box in enumerate(raw_boxes):
                cv2.polylines(raw_debug_image, [box.astype("int32").reshape(-1, 1, 2)], True, (255, 0, 0), 2)
                x = int(box[:, 0].min())
                y = int(box[:, 1].min())
                cv2.putText(raw_debug_image, str(idx), (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(self.debug_dir / "det_boxes_raw.jpg"), raw_debug_image)

            debug_image = image.copy()
            for idx, box in enumerate(boxes):
                cv2.polylines(debug_image, [box.astype("int32").reshape(-1, 1, 2)], True, (0, 255, 255), 2)
                x = int(box[:, 0].min())
                y = int(box[:, 1].min())
                cv2.putText(debug_image, str(idx), (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(self.debug_dir / "det_boxes.jpg"), debug_image)
        return image, boxes, rec_results, {"det": det_time, "rec": rec_time, "total": total_time}


def _load_cjk_font(font_path: str, font_size: int):
    """Load a CJK-capable font. For .ttc collections, prefer Simplified Chinese face."""
    if ImageFont is None:
        return None

    suffix = Path(font_path).suffix.lower()
    # Noto/Source Han CJK collections commonly put SC at index 2.
    indices = [2, 0, 1, 3, 4] if suffix == ".ttc" else [0]
    last_error = None
    for index in indices:
        try:
            return ImageFont.truetype(font_path, font_size, index=index)
        except Exception as exc:  # noqa: BLE001 - try next face/path
            last_error = exc
            continue
    if last_error is not None:
        logger.debug("Failed to load font {}: {}", font_path, last_error)
    return None


_VIS_FONT_FALLBACKS = [
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def _resolve_vis_font(font_path: str | None, font_size: int):
    font_paths = []
    if font_path:
        font_paths.append(font_path)
    font_paths.extend(_VIS_FONT_FALLBACKS)

    base_font = None
    selected_font_path = None
    for candidate_font_path in font_paths:
        if not Path(candidate_font_path).is_file():
            continue
        base_font = _load_cjk_font(candidate_font_path, max(12, int(font_size)))
        if base_font is not None:
            selected_font_path = candidate_font_path
            break
    if base_font is None:
        logger.warning("No CJK font found; Chinese text may render incorrectly")
        base_font = ImageFont.load_default()
    else:
        logger.info("Visualization font: {}", selected_font_path)
    return base_font, selected_font_path


def _text_bbox(font, text: str) -> tuple[int, int]:
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(text)
        return max(1, right - left), max(1, bottom - top)
    width, height = font.getsize(text)
    return max(1, width), max(1, height)


def _draw_vertical_text(draw: ImageDraw.ImageDraw, origin_x: int, origin_y: int, text: str, font, fill: tuple[int, int, int]) -> None:
    if not text:
        return
    char_sizes = [_text_bbox(font, ch) for ch in text]
    char_width = max(width for width, _ in char_sizes)
    spacing = max(1, int(round(_text_bbox(font, "0")[1] * 0.12)))
    y = origin_y
    for ch, (width, height) in zip(text, char_sizes):
        x = origin_x + max(0, (char_width - width) // 2)
        draw.text((x, y), ch, fill=fill, font=font)
        y += height + spacing


def _draw_single_result(
    draw: ImageDraw.ImageDraw,
    box: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    w: int,
    base_font,
    selected_font_path: str | None,
    font_cache: dict[int, object],
    font_size: int,
) -> None:
    x1 = int(box[:, 0].min())
    y1 = int(box[:, 1].min())
    x2 = int(box[:, 0].max())
    y2 = int(box[:, 1].max())
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    # Keep text readable on the right panel; do not shrink to tiny box height.
    item_font_size = max(16, min(int(font_size), max(16, int(box_h * 1.8))))
    font = font_cache.get(item_font_size)
    if font is None and selected_font_path is not None:
        font = _load_cjk_font(selected_font_path, item_font_size) or base_font
        font_cache[item_font_size] = font
    elif font is None:
        font = base_font

    rgb_color = (color[2], color[1], color[0])
    right_x1 = w + x1
    right_y1 = y1
    right_x2 = w + x2
    right_y2 = y2
    draw.rectangle([right_x1, right_y1, right_x2, right_y2], outline=rgb_color, width=1)

    text_w, text_h = _text_bbox(font, text if text else "0")
    is_vertical_box = box_h >= box_w * 1.2
    if is_vertical_box:
        vertical_spacing = max(1, int(round(text_h * 0.15)))
        total_text_h = sum(_text_bbox(font, ch)[1] for ch in text) + vertical_spacing * max(0, len(text) - 1)
        start_x = right_x1 + max(0, (box_w - text_w) // 2)
        start_y = right_y1 + max(0, (box_h - total_text_h) // 2)
        _draw_vertical_text(draw, start_x, start_y, text, font, (0, 0, 0))
    else:
        draw.text((right_x1, max(0, right_y1 - 2)), text, fill=(0, 0, 0), font=font)


def _draw_results(
    image: np.ndarray,
    boxes: Sequence[np.ndarray],
    rec_results: Sequence[Tuple[str, float]],
    font_path: str | None = None,
    font_size: int = 24,
) -> np.ndarray:
    h, w = image.shape[:2]
    canvas = np.ones((h, w * 2, 3), dtype=np.uint8) * 255
    canvas[:, :w] = image.copy()

    colors = [
        (255, 128, 128),
        (128, 255, 128),
        (128, 128, 255),
        (255, 255, 128),
        (255, 128, 255),
        (128, 255, 255),
    ]
    overlay = canvas[:, :w].copy()
    for idx, box in enumerate(boxes):
        color = colors[idx % len(colors)]
        cv2.fillPoly(overlay, [box.astype("int32").reshape(-1, 1, 2)], color)
        cv2.polylines(canvas[:, :w], [box.astype("int32").reshape(-1, 1, 2)], True, color, 2)
    canvas[:, :w] = cv2.addWeighted(overlay, 0.25, canvas[:, :w], 0.75, 0)

    if Image is None or ImageDraw is None:
        logger.warning("Pillow is not available; skip OCR text visualization")
        return canvas

    base_font, selected_font_path = _resolve_vis_font(font_path, font_size)
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_cache = {max(12, int(font_size)): base_font}
    for idx, (box, rec) in enumerate(zip(boxes, rec_results)):
        text, _ = rec
        color = colors[idx % len(colors)]
        _draw_single_result(draw, box, text, color, w, base_font, selected_font_path, font_cache, font_size)

    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def _recognize_crop_batch(
    demo: PaddleOCRV5Demo,
    batch_crops: Sequence[np.ndarray],
    start: int,
    rec_batch: int,
    rec_h: int,
    rec_w: int,
    debug_names: Sequence[str] | None,
):
    rec_inputs = []
    with demo.perf.scope("ocr.rec.preprocess"):
        for local_idx, crop in enumerate(batch_crops):
            rec_preprocessed = _resize_rec_image(crop, rec_h, rec_w, use_rgb=demo.rec_rgb)
            if demo.debug_dir:
                crop_index = start + local_idx
                debug_name = debug_names[crop_index] if debug_names else f"{crop_index:03d}"
                cv2.imwrite(str(demo.debug_dir / f"crop_{debug_name}.jpg"), crop)
                debug_rec = rec_preprocessed.transpose(1, 2, 0).astype("float32")
                debug_rec = ((debug_rec * 127.5) + 127.5).clip(0, 255).astype("uint8")
                if demo.rec_rgb:
                    debug_rec = cv2.cvtColor(debug_rec, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(demo.debug_dir / f"rec_input_{debug_name}.jpg"), debug_rec)
            rec_inputs.append(rec_preprocessed)
        while len(rec_inputs) < rec_batch:
            rec_inputs.append(np.zeros((3, rec_h, rec_w), dtype="float32"))
        rec_input = np.stack(rec_inputs, axis=0).astype("float32")
    return rec_input


def _run_recognition_batch(
    demo: PaddleOCRV5Demo,
    rec_input: np.ndarray,
):
    with demo.perf.scope("ocr.rec.set_input"):
        demo.rec.set_input(demo.rec_input_name, rec_input)
    with demo.perf.scope("ocr.rec.infer"):
        demo.rec.run()
        demo.rec.sync()
    with demo.perf.scope("ocr.rec.get_output"):
        pred = _module_output(demo.rec)
    logger.info("rec output shape: {}", pred.shape)
    with demo.perf.scope("ocr.rec.decode"):
        return demo.ctc_decode(pred)


def _flush_rec_batch(
    demo: PaddleOCRV5Demo,
    batch_paths: List[str],
    batch_labels: List[str],
    batch_crops: List[np.ndarray],
    total: int,
    exact_correct: int,
    loose_correct: int,
    score_sum: float,
):
    if not batch_crops:
        return total, exact_correct, loose_correct, score_sum
    rec_batch, _, rec_h, rec_w = demo.rec_shape
    rec_input = _recognize_crop_batch(demo, batch_crops, 0, rec_batch, rec_h, rec_w, None)
    preds = _run_recognition_batch(demo, rec_input)
    for path, label, (pred_text, pred_score) in zip(batch_paths, batch_labels, preds):
        total += 1
        score_sum += float(pred_score)
        if label and pred_text == label:
            exact_correct += 1
        if label and _normalize_rec_text(pred_text) == _normalize_rec_text(label):
            loose_correct += 1
        if label:
            logger.info("rec[{}] score={:.4f} pred={} gt={} path={}", total - 1, pred_score, pred_text, label, path)
        else:
            logger.info("rec[{}] score={:.4f} pred={} path={}", total - 1, pred_score, pred_text, path)
    batch_paths.clear()
    batch_labels.clear()
    batch_crops.clear()
    return total, exact_correct, loose_correct, score_sum


def _run_rec_only(demo: PaddleOCRV5Demo, samples: Sequence[Tuple[str, str]]) -> None:
    if not samples:
        logger.warning("No rec samples found")
        return

    batch_size = demo.rec_shape[0]
    total = 0
    exact_correct = 0
    loose_correct = 0
    score_sum = 0.0
    batch_paths: List[str] = []
    batch_labels: List[str] = []
    batch_crops: List[np.ndarray] = []

    for path, label in samples:
        image = cv2.imread(path)
        if image is None:
            logger.warning("Skip unreadable image: {}", path)
            continue
        batch_paths.append(path)
        batch_labels.append(label)
        batch_crops.append(image)
        if len(batch_crops) == batch_size:
            total, exact_correct, loose_correct, score_sum = _flush_rec_batch(
                demo,
                batch_paths,
                batch_labels,
                batch_crops,
                total,
                exact_correct,
                loose_correct,
                score_sum,
            )
    total, exact_correct, loose_correct, score_sum = _flush_rec_batch(
        demo,
        batch_paths,
        batch_labels,
        batch_crops,
        total,
        exact_correct,
        loose_correct,
        score_sum,
    )

    if total == 0:
        logger.warning("No valid rec images were loaded")
        return

    labeled_total = sum(1 for _, label in samples if label)
    if labeled_total > 0:
        logger.info(
            "Rec-only summary: samples={} exact_accuracy={:.4f} loose_accuracy={:.4f} avg_score={:.4f}",
            total,
            exact_correct / total,
            loose_correct / total,
            score_sum / total,
        )
    else:
        logger.info("Rec-only summary: samples={} avg_score={:.4f}", total, score_sum / total)


def _default_dict_path() -> str:
    return str(SCRIPT_DIR / "ppocrv5_dict.txt")


def _load_rec_samples(rec_list_path: str, rec_root: str | None = None, limit: int = 0) -> List[Tuple[str, str]]:
    rec_list = Path(rec_list_path)
    root = Path(rec_root) if rec_root else rec_list.resolve().parent.parent
    samples: List[Tuple[str, str]] = []
    with rec_list.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                logger.warning("Skip malformed rec list line: {}", raw_line.rstrip("\n"))
                continue
            rel_path, label = parts
            samples.append((str((root / rel_path).resolve()), label))
            if limit > 0 and len(samples) >= limit:
                break
    return samples


def _normalize_rec_text(text: str) -> str:
    if not text:
        return ""
    remove_chars = {"-", "·", "•", ":", "：", " ", "　"}
    return "".join(ch for ch in text.upper() if ch not in remove_chars)


def _log_rec_prediction(path: str, label: str, pred_text: str, pred_score: float, sample_index: int) -> None:
    if label:
        logger.info("rec[{}] score={:.4f} pred={} gt={} path={}", sample_index, pred_score, pred_text, label, path)
    else:
        logger.info("rec[{}] score={:.4f} pred={} path={}", sample_index, pred_score, pred_text, path)


def _flush_rec_batch(
    demo: PaddleOCRV5Demo,
    batch_paths: List[str],
    batch_labels: List[str],
    batch_crops: List[np.ndarray],
    metrics: dict,
) -> None:
    if not batch_crops:
        return

    preds = demo._recognize_crops(batch_crops)
    for path, label, (pred_text, pred_score) in zip(batch_paths, batch_labels, preds):
        metrics["total"] += 1
        metrics["score_sum"] += float(pred_score)
        if label and pred_text == label:
            metrics["exact_correct"] += 1
        if label and _normalize_rec_text(pred_text) == _normalize_rec_text(label):
            metrics["loose_correct"] += 1
        _log_rec_prediction(path, label, pred_text, pred_score, metrics["total"] - 1)

    batch_paths.clear()
    batch_labels.clear()
    batch_crops.clear()


def _run_rec_only(demo: PaddleOCRV5Demo, samples: Sequence[Tuple[str, str]]) -> None:
    if not samples:
        logger.warning("No rec samples found")
        return

    batch_size = demo.rec_shape[0]
    metrics = {
        "total": 0,
        "exact_correct": 0,
        "loose_correct": 0,
        "score_sum": 0.0,
    }
    batch_paths: List[str] = []
    batch_labels: List[str] = []
    batch_crops: List[np.ndarray] = []

    for path, label in samples:
        image = cv2.imread(path)
        if image is None:
            logger.warning("Skip unreadable image: {}", path)
            continue
        batch_paths.append(path)
        batch_labels.append(label)
        batch_crops.append(image)
        if len(batch_crops) == batch_size:
            _flush_rec_batch(demo, batch_paths, batch_labels, batch_crops, metrics)

    _flush_rec_batch(demo, batch_paths, batch_labels, batch_crops, metrics)

    if metrics["total"] == 0:
        logger.warning("No valid rec images were loaded")
        return

    labeled_total = sum(1 for _, label in samples if label)
    if labeled_total > 0:
        logger.info(
            "Rec-only summary: samples={} exact_accuracy={:.4f} loose_accuracy={:.4f} avg_score={:.4f}",
            metrics["total"],
            metrics["exact_correct"] / metrics["total"],
            metrics["loose_correct"] / metrics["total"],
            metrics["score_sum"] / metrics["total"],
        )
    else:
        logger.info("Rec-only summary: samples={} avg_score={:.4f}", metrics["total"], metrics["score_sum"] / metrics["total"])


def get_args():
    parser = argparse.ArgumentParser(description="Run PaddleOCR V5 det+rec HMM demo or rec-only benchmark")
    parser.add_argument("--image", default=None, help="input image path for det+rec mode")
    parser.add_argument("--rec-image", default=None, help="single cropped recognition image path")
    parser.add_argument("--rec-list", default=None, help="rec.txt path with relative crop image paths and labels")
    parser.add_argument("--rec-root", default=None, help="root directory for paths in rec.txt; defaults to the rec list parent")
    parser.add_argument("--rec-limit", type=int, default=0, help="limit number of rec samples when using --rec-list")
    parser.add_argument(
        "--det-path",
        default=str(SCRIPT_DIR / "output" / HOUMO_TARGET / "paddleocr_v5_det_xh2_b1_1core_O2.hmm"),
        help="compiled detection .hmm path",
    )
    parser.add_argument(
        "--rec-path",
        default=str(SCRIPT_DIR / "output" / HOUMO_TARGET / "paddleocr_v5_rec_xh2_b6_1core_O2.hmm"),
        help="compiled recognition .hmm path",
    )
    parser.add_argument("--character-dict", default=_default_dict_path())
    parser.add_argument("--output", default=str(SCRIPT_DIR / "ocr_result.jpg"), help="visualized result image")
    parser.add_argument("--thresh", type=float, default=0.3)
    parser.add_argument("--box-thresh", type=float, default=0.6)
    parser.add_argument("--unclip-ratio", type=float, default=1.5)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--drop-score", type=float, default=0.5)
    parser.add_argument("--crop-pad-ratio", type=float, default=0.0, help="expand each detected text box before perspective crop")
    parser.add_argument("--rec-rgb", action="store_true", help="convert cropped rec images from BGR to RGB before recognition")
    parser.add_argument(
        "--vis-font-path",
        default="/data/qianqian.zhao/NotoSansCJK-Regular.ttc",
        help="font path used to draw non-ASCII OCR text; for .ttc prefer a CJK SC face",
    )
    parser.add_argument("--vis-font-size", type=int, default=24, help="font size used to draw OCR text")
    parser.add_argument("--vis-stroke-width", type=int, default=2, help="outline width used to draw OCR text")
    parser.add_argument("--debug-dir", default=None, help="save detection crops and recognition inputs for debugging")
    parser.add_argument("--perf", action="store_true", help="enable performance summary for det+rec demo")
    return parser.parse_args()


def main():
    args = get_args()
    rec_mode = bool(args.rec_image or args.rec_list)
    if rec_mode and args.image:
        logger.warning("--image is ignored in rec-only mode")
    if args.rec_image and args.rec_list:
        raise ValueError("Use only one of --rec-image or --rec-list")

    demo = PaddleOCRV5Demo(
        det_path=args.det_path,
        rec_path=args.rec_path,
        character_dict_path=args.character_dict,
        thresh=args.thresh,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
        max_candidates=args.max_candidates,
        drop_score=args.drop_score,
        crop_pad_ratio=args.crop_pad_ratio,
        rec_rgb=args.rec_rgb,
        debug_dir=args.debug_dir,
        perf=args.perf,
    )

    if args.rec_image:
        _run_rec_only(demo, [(args.rec_image, "")])
        return

    if args.rec_list:
        samples = _load_rec_samples(args.rec_list, rec_root=args.rec_root, limit=args.rec_limit)
        _run_rec_only(demo, samples)
        return

    if not args.image:
        raise ValueError("Either --image, --rec-image, or --rec-list must be provided")

    image, boxes, rec_results, timings = demo.run(args.image)
    for idx, (box, rec) in enumerate(zip(boxes, rec_results)):
        text, score = rec
        logger.info("[{}] score={:.4f} box={} text={}", idx, score, box.tolist(), text)
    logger.info(
        "PaddleOCR V5 done: boxes={}, det={:.3f}s rec={:.3f}s total={:.3f}s",
        len(boxes),
        timings["det"],
        timings["rec"],
        timings["total"],
    )
    with demo.perf.scope("ocr.vis"):
        output = _draw_results(
            image,
            boxes,
            rec_results,
            font_path=args.vis_font_path,
            font_size=args.vis_font_size,
        )
        cv2.imwrite(args.output, output)
    logger.info("Visualization saved to {}", args.output)
    demo.perf.print_summary()


if __name__ == "__main__":
    main()
