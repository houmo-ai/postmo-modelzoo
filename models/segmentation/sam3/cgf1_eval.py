# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Modifications Copyright (c) 2026 HOUMO AI
# Modified by HOUMO AI on 2026-09-04.
# Changes: compact standalone implementation for the iModelzoo SAM3 example.
# Original source: https://github.com/facebookresearch/sam3,
# sam3/eval/cgf1_eval.py.
#
# File: cgf1_eval.py
# Description:
#   Local CGF1 evaluator for SAM3 SA-Co Gold model example evaluation.
#
# This derivative of the SAM3 evaluation code is subject to the SAM License:
# https://github.com/facebookresearch/sam3/blob/main/LICENSE
"""Compact local implementation of the SA-Co Gold CGF1 evaluation contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


class CGF1Evaluator:
	"""Evaluate open-vocabulary instance predictions against one or more GT files."""

	def __init__(self, gt_path, verbose: bool = False, iou_type: str = "segm"):
		self.gt_paths = [gt_path] if isinstance(gt_path, (str, Path)) else list(gt_path)
		if iou_type not in ("segm", "bbox"):
			raise ValueError(f"Unsupported iou_type: {iou_type}")
		self.iou_type = iou_type
		self.verbose = verbose
		self.ground_truths = [json.loads(Path(path).read_text()) for path in self.gt_paths]

	@staticmethod
	def _iou(detections, ground_truths, iou_type: str) -> np.ndarray:
		if not detections or not ground_truths:
			return np.zeros((len(detections), len(ground_truths)), dtype=np.float32)
		if iou_type == "segm":
			return np.asarray(
				mask_utils.iou(
					[d["segmentation"] for d in detections],
					[g["segmentation"] for g in ground_truths],
					[bool(g.get("iscrowd", 0)) for g in ground_truths],
				),
				dtype=np.float32,
			)
		return np.asarray(
				mask_utils.iou(
					[d["bbox"] for d in detections],
					[g["bbox"] for g in ground_truths],
					[bool(g.get("iscrowd", 0)) for g in ground_truths],
				),
				dtype=np.float32,
			)

	def evaluate(self, prediction_path: str) -> dict[str, float]:
		predictions = json.loads(Path(prediction_path).read_text())
		pred_by_id: dict[int, list[dict]] = {}
		for prediction in predictions:
			pred_by_id.setdefault(int(prediction["image_id"]), []).append(prediction)
		rows_by_id: dict[int, list[dict]] = {}
		for ground_truth in self.ground_truths:
			for row in ground_truth.get("images", []):
				if row.get("is_instance_exhaustive", False):
					rows_by_id.setdefault(int(row["id"]), []).append(ground_truth)

		thresholds = np.arange(0.5, 1.0, 0.05)
		values = []
		for image_id, ground_truths in rows_by_id.items():
			preds = [
				prediction for prediction in pred_by_id.get(image_id, [])
				if float(prediction["score"]) >= 0.5
			]
			best = None
			best_score = -1.0
			for ground_truth in ground_truths:
				gts = [
					a for a in ground_truth.get("annotations", [])
					if int(a["image_id"]) == image_id and not a.get("ignore", False)
				]
				ious = self._iou(preds, gts, self.iou_type)
				if ious.size:
					matched_pred, matched_gt = linear_sum_assignment(-ious)
					matched = ious[matched_pred, matched_gt]
				else:
					matched = np.empty(0, dtype=np.float32)
				true_positives = np.asarray(
					[(matched >= threshold).sum() for threshold in thresholds],
					dtype=np.float64,
				)
				false_positives = len(preds) - true_positives
				false_negatives = len(gts) - true_positives
				precision = true_positives / (true_positives + false_positives + 1e-4)
				recall = true_positives / (true_positives + false_negatives + 1e-4)
				local_f1 = 2 * precision * recall / (precision + recall + 1e-4)
				candidate_score = 1.0 if not preds and not gts else float(local_f1.mean())
				if candidate_score > best_score:
					best_score = candidate_score
					best = (len(preds), len(gts), true_positives)
			if best is not None:
				values.append(best)

		tp = np.zeros(len(thresholds), dtype=np.float64)
		fp = np.zeros(len(thresholds), dtype=np.float64)
		fn = np.zeros(len(thresholds), dtype=np.float64)
		il_tp = il_fp = il_tn = il_fn = 0
		positive_micro_fp = np.zeros(len(thresholds), dtype=np.float64)
		for num_pred, num_gt, true_positives in values:
			tp += true_positives
			fp += num_pred - true_positives
			fn += num_gt - true_positives
			if num_gt > 0 and num_pred > 0:
				positive_micro_fp += num_pred - true_positives
			if num_gt and num_pred:
				il_tp += 1
			elif num_pred:
				il_fp += 1
			else:
				il_tn += int(num_gt == 0)
			il_fn += int(num_gt > 0 and num_pred == 0)
		positive_micro_precision = tp / (tp + positive_micro_fp + 1e-4)
		recall = tp / (tp + fn + 1e-4)
		pmf1 = 2 * positive_micro_precision * recall / (
			positive_micro_precision + recall + 1e-4
		)
		il_mcc = (il_tp * il_tn - il_fp * il_fn) / np.sqrt(
			(il_tp + il_fp) * (il_tp + il_fn) * (il_tn + il_fp) * (il_tn + il_fn) + 1e-6
		)
		return {
			f"cgF1_eval_{self.iou_type}_cgF1": float(np.mean(pmf1) * il_mcc),
			f"cgF1_eval_{self.iou_type}_IL_MCC": float(il_mcc),
			f"cgF1_eval_{self.iou_type}_positive_micro_F1": float(np.mean(pmf1)),
		}
