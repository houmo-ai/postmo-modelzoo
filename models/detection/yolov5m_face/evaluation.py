# Copyright 2025 HOUMO AI
#
# File: evaluation.py
# Description:
#   Evaluation code for YOLOv5m face detection model on WiderFace dataset.
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
import pickle

import numpy as np
import tqdm
from scipy.io import loadmat


def _load_widerface_mat_files(gt_dir):
    return {
        "base": loadmat(os.path.join(gt_dir, "wider_face_val.mat")),
        "easy": loadmat(os.path.join(gt_dir, "wider_easy_val.mat")),
        "medium": loadmat(os.path.join(gt_dir, "wider_medium_val.mat")),
        "hard": loadmat(os.path.join(gt_dir, "wider_hard_val.mat")),
    }


def get_gt_boxes(gt_dir):
    """Load WiderFace validation annotations from mat files."""

    mat_files = _load_widerface_mat_files(gt_dir)
    base_annotations = mat_files["base"]

    face_boxes_by_event = base_annotations["face_bbx_list"]
    event_names = base_annotations["event_list"]
    image_names = base_annotations["file_list"]

    hard_gt = mat_files["hard"]["gt_list"]
    medium_gt = mat_files["medium"]["gt_list"]
    easy_gt = mat_files["easy"]["gt_list"]

    print(
        "event_list num:",
        len(event_names),
        ", easy num:",
        len(easy_gt),
        ", medium num:",
        len(medium_gt),
        ", hard num:",
        len(hard_gt),
    )

    return (
        face_boxes_by_event,
        event_names,
        image_names,
        hard_gt,
        medium_gt,
        easy_gt,
    )


def get_gt_boxes_from_txt(gt_path, cache_dir):
    cache_file = os.path.join(cache_dir, "gt_cache.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as file_obj:
            return pickle.load(file_obj)

    with open(gt_path, "r") as file_obj:
        raw_lines = [line.rstrip("\r\n") for line in file_obj]

    print(len(raw_lines))
    boxes_by_image = {}
    current_name = None
    current_boxes = []
    parse_state = 0

    for line in raw_lines:
        if "--" in line:
            if parse_state == 2 and current_name is not None:
                boxes_by_image[current_name] = np.asarray(
                    current_boxes, dtype=np.float32
                )
            current_name = line
            current_boxes = []
            parse_state = 1
            continue

        if parse_state == 1:
            parse_state = 2
            continue

        if parse_state == 2:
            current_boxes.append([float(value) for value in line.split(" ")[:4]])

    if parse_state == 2 and current_name is not None:
        boxes_by_image[current_name] = np.asarray(current_boxes, dtype=np.float32)

    with open(cache_file, "wb") as file_obj:
        pickle.dump(boxes_by_image, file_obj)

    return boxes_by_image


def _parse_prediction_lines(lines):
    detections = []
    for line in lines:
        fields = line.rstrip("\r\n").split(" ")
        if not fields or fields[0] == "":
            continue
        detections.append([float(fields[index]) for index in range(5)])
    return np.asarray(detections)


def read_pred_file(filepath):
    with open(filepath, "r") as file_obj:
        lines = file_obj.readlines()

    image_path = lines[0].rstrip("\n\r")
    boxes = _parse_prediction_lines(lines[2:])
    return image_path.split("/")[-1], boxes


def get_preds(pred_dir):
    event_names = os.listdir(pred_dir)
    predictions = {}
    progress = tqdm.tqdm(event_names)

    for event_name in progress:
        progress.set_description("Reading Predictions ")
        event_dir = os.path.join(pred_dir, event_name)
        image_files = os.listdir(event_dir)
        event_predictions = {}
        for image_file in image_files:
            image_name, image_boxes = read_pred_file(
                os.path.join(event_dir, image_file)
            )
            event_predictions[image_name.rstrip(".jpg")] = image_boxes
        predictions[event_name] = event_predictions

    return predictions


def norm_score(pred):
    """Normalize confidence scores into [0, 1]."""

    global_max = 0
    global_min = 1

    for event_predictions in pred.values():
        for image_predictions in event_predictions.values():
            if len(image_predictions) == 0:
                continue
            global_min = min(global_min, np.min(image_predictions[:, -1]))
            global_max = max(global_max, np.max(image_predictions[:, -1]))

    score_range = global_max - global_min
    if score_range == 0:
        return

    for event_predictions in pred.values():
        for image_predictions in event_predictions.values():
            if len(image_predictions) == 0:
                continue
            image_predictions[:, -1] = (
                image_predictions[:, -1] - global_min
            ) / score_range


def bbox_overlaps(boxes, query_boxes):
    """Compute IoU matrix between predicted boxes and target boxes."""

    num_boxes = boxes.shape[0]
    num_queries = query_boxes.shape[0]
    iou_matrix = np.zeros((num_boxes, num_queries), dtype=np.float64)

    for query_index in range(num_queries):
        query_area = (query_boxes[query_index, 2] - query_boxes[query_index, 0] + 1) * (
            query_boxes[query_index, 3] - query_boxes[query_index, 1] + 1
        )
        for box_index in range(num_boxes):
            inter_width = (
                min(boxes[box_index, 2], query_boxes[query_index, 2])
                - max(boxes[box_index, 0], query_boxes[query_index, 0])
                + 1
            )
            if inter_width <= 0:
                continue

            inter_height = (
                min(boxes[box_index, 3], query_boxes[query_index, 3])
                - max(boxes[box_index, 1], query_boxes[query_index, 1])
                + 1
            )
            if inter_height <= 0:
                continue

            union_area = float(
                (boxes[box_index, 2] - boxes[box_index, 0] + 1)
                * (boxes[box_index, 3] - boxes[box_index, 1] + 1)
                + query_area
                - inter_width * inter_height
            )
            iou_matrix[box_index, query_index] = inter_width * inter_height / union_area

    return iou_matrix


def _convert_xywh_to_xyxy(box_array):
    converted = box_array.copy()
    converted[:, 2] = converted[:, 2] + converted[:, 0]
    converted[:, 3] = converted[:, 3] + converted[:, 1]
    return converted


def image_eval(pred, gt, ignore, iou_thresh):
    """Evaluate predictions for a single image."""

    pred_boxes = _convert_xywh_to_xyxy(pred.copy())
    gt_boxes = _convert_xywh_to_xyxy(gt.copy())

    cumulative_recall = np.zeros(pred_boxes.shape[0])
    gt_match_state = np.zeros(gt_boxes.shape[0])
    proposal_state = np.ones(pred_boxes.shape[0])

    overlaps = bbox_overlaps(pred_boxes[:, :4], gt_boxes)

    for pred_index in range(pred_boxes.shape[0]):
        row_overlaps = overlaps[pred_index]
        best_iou = row_overlaps.max()
        best_gt_index = row_overlaps.argmax()

        if best_iou >= iou_thresh:
            if ignore[best_gt_index] == 0:
                gt_match_state[best_gt_index] = -1
                proposal_state[pred_index] = -1
            elif gt_match_state[best_gt_index] == 0:
                gt_match_state[best_gt_index] = 1

        cumulative_recall[pred_index] = len(np.where(gt_match_state == 1)[0])

    return cumulative_recall, proposal_state


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2), dtype=float)
    for thresh_index in range(thresh_num):
        score_thresh = 1 - (thresh_index + 1) / thresh_num
        passed_indices = np.where(pred_info[:, 4] >= score_thresh)[0]
        if len(passed_indices) == 0:
            continue

        last_index = passed_indices[-1]
        valid_proposals = np.where(proposal_list[: last_index + 1] == 1)[0]
        pr_info[thresh_index, 0] = len(valid_proposals)
        pr_info[thresh_index, 1] = pred_recall[last_index]

    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    aggregated_curve = np.zeros((thresh_num, 2), dtype=float)
    for thresh_index in range(thresh_num):
        if pr_curve[thresh_index, 0] != 0:
            aggregated_curve[thresh_index, 0] = (
                pr_curve[thresh_index, 1] / pr_curve[thresh_index, 0]
            )
        if count_face != 0:
            aggregated_curve[thresh_index, 1] = pr_curve[thresh_index, 1] / count_face
    return aggregated_curve


def voc_ap(rec, prec):
    recall_points = np.concatenate(([0.0], rec, [1.0]))
    precision_points = np.concatenate(([0.0], prec, [0.0]))

    for index in range(precision_points.size - 1, 0, -1):
        precision_points[index - 1] = np.maximum(
            precision_points[index - 1], precision_points[index]
        )

    changed_indices = np.where(recall_points[1:] != recall_points[:-1])[0]
    return np.sum(
        (recall_points[changed_indices + 1] - recall_points[changed_indices])
        * precision_points[changed_indices + 1]
    )


def _iter_setting_data(facebox_list, file_list, gt_list, event_list):
    for event_index, event_entry in enumerate(event_list):
        yield {
            "event_name": str(event_entry[0][0]),
            "image_list": file_list[event_index][0],
            "keep_list": gt_list[event_index][0],
            "gt_box_list": facebox_list[event_index][0],
        }


def _evaluate_one_setting(
    setting_name,
    pred,
    facebox_list,
    file_list,
    gt_list,
    event_list,
    iou_thresh,
    thresh_num,
):
    total_faces = 0
    pr_curve = np.zeros((thresh_num, 2), dtype=float)
    progress = tqdm.tqdm(
        _iter_setting_data(facebox_list, file_list, gt_list, event_list),
        total=len(event_list),
    )

    for event_data in progress:
        progress.set_description(f"Processing {setting_name}")
        event_name = event_data["event_name"]
        if event_name not in pred:
            continue

        event_predictions = pred[event_name]
        image_list = event_data["image_list"]
        keep_list = event_data["keep_list"]
        gt_box_list = event_data["gt_box_list"]

        for image_index, image_entry in enumerate(image_list):
            image_name = str(image_entry[0][0])
            if image_name not in event_predictions:
                continue

            pred_info = event_predictions[image_name]
            gt_boxes = gt_box_list[image_index][0].astype(float)
            keep_index = keep_list[image_index][0]
            total_faces += len(keep_index)

            if len(gt_boxes) == 0 or len(pred_info) == 0:
                continue

            ignore = np.zeros(gt_boxes.shape[0])
            if len(keep_index) != 0:
                ignore[keep_index - 1] = 1

            pred_recall, proposal_list = image_eval(
                pred_info, gt_boxes, ignore, iou_thresh
            )
            pr_curve += img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

    normalized_curve = dataset_pr_info(thresh_num, pr_curve, total_faces)
    precision = normalized_curve[:, 0]
    recall = normalized_curve[:, 1]
    return voc_ap(recall, precision)


def evaluation(pred, gt_path, iou_thresh=0.5):
    pred = get_preds(pred)
    norm_score(pred)

    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = (
        get_gt_boxes(gt_path)
    )

    thresh_num = 1000
    setting_pairs = [
        ("easy", easy_gt_list),
        ("medium", medium_gt_list),
        ("hard", hard_gt_list),
    ]

    aps = []
    for setting_name, gt_list in setting_pairs:
        aps.append(
            _evaluate_one_setting(
                setting_name,
                pred,
                facebox_list,
                file_list,
                gt_list,
                event_list,
                iou_thresh,
                thresh_num,
            )
        )

    print("==================== Results ====================")
    print("Easy   Val AP: {}".format(aps[0]))
    print("Medium Val AP: {}".format(aps[1]))
    print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")

    return aps
