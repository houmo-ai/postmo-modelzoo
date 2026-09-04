# Copyright (c) 2026 HOUMO AI
#
# File: eval_util.py
# Description:
#   SAM3 SA-Co Gold dataset discovery, inference, and evaluation utilities.
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
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


GOLD_GTS = {
	"metaclip_nps": [
		"gold_metaclip_merged_a_release_test.json",
		"gold_metaclip_merged_b_release_test.json",
		"gold_metaclip_merged_c_release_test.json",
	],
	"sa1b_nps": [
		"gold_sa1b_merged_a_release_test.json",
		"gold_sa1b_merged_b_release_test.json",
		"gold_sa1b_merged_c_release_test.json",
	],
	"crowded": [
		"gold_crowded_merged_a_release_test.json",
		"gold_crowded_merged_b_release_test.json",
		"gold_crowded_merged_c_release_test.json",
	],
	"fg_food": [
		"gold_fg_food_merged_a_release_test.json",
		"gold_fg_food_merged_b_release_test.json",
		"gold_fg_food_merged_c_release_test.json",
	],
	"fg_sports_equipment": [
		"gold_fg_sports_equipment_merged_a_release_test.json",
		"gold_fg_sports_equipment_merged_b_release_test.json",
		"gold_fg_sports_equipment_merged_c_release_test.json",
	],
	"attributes": [
		"gold_attributes_merged_a_release_test.json",
		"gold_attributes_merged_b_release_test.json",
		"gold_attributes_merged_c_release_test.json",
	],
	"wiki_common": [
		"gold_wiki_common_merged_a_release_test.json",
		"gold_wiki_common_merged_b_release_test.json",
		"gold_wiki_common_merged_c_release_test.json",
	],
}


def _dataset_root(path: Path) -> Path | None:
	path = path.expanduser().resolve()
	if (path / "gt-annotations").is_dir():
		return path
	if (path / "saco_gold" / "gt-annotations").is_dir():
		return path / "saco_gold"
	return None


def _candidate_roots(script_dir: Path) -> list[Path]:
	candidates = [script_dir, script_dir / "data" / "dataset"]
	for variable in ("HOUMO_DATASETS_PATH", "HOUMO_EXAMPLES_PATH"):
		value = os.environ.get(variable)
		if value:
			base = Path(value).expanduser()
			candidates.extend([base, base / "data" / "dataset", base / "data" / "datasets"])
	return candidates


def locate_dataset(dataset_path: str | Path | None, script_dir: Path) -> Path:
	"""Locate or extract the saco_gold dataset using repository environment paths."""
	candidates = [Path(dataset_path)] if dataset_path else _candidate_roots(script_dir)
	for candidate in candidates:
		root = _dataset_root(candidate)
		if root:
			return root

	zip_paths: list[Path] = []
	for candidate in candidates:
		candidate = candidate.expanduser().resolve()
		if candidate.is_file() and candidate.name == "saco_gold.zip":
			zip_paths.append(candidate)
		elif candidate.is_dir():
			zip_paths.extend(candidate.rglob("saco_gold.zip"))
	if not zip_paths:
		raise FileNotFoundError(
			"Cannot find saco_gold dataset or saco_gold.zip. "
			"Please run get_model.py --type raw or specify --dataset_path."
		)

	archive = zip_paths[0]
	extract_parent = archive.parent
	shutil.unpack_archive(str(archive), str(extract_parent))
	root = _dataset_root(extract_parent)
	if root is None:
		raise RuntimeError(f"Failed to locate extracted saco_gold dataset: {extract_parent}")
	return root


def _image_path(image_root: Path, file_name: str) -> Path:
	path = image_root / file_name
	if path.is_file():
		return path
	matches = list(image_root.rglob(Path(file_name).name))
	if len(matches) == 1:
		return matches[0]
	raise FileNotFoundError(f"Image not found: {path}")


def _load_rows(annotation_paths: list[Path], limit: int) -> list[dict]:
	rows = json.loads(annotation_paths[0].read_text())["images"]
	return rows if limit == 0 else rows[:limit]


def _encode_mask(mask: np.ndarray) -> dict:
	encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
	encoded["counts"] = encoded["counts"].decode("ascii")
	return encoded


def _filtered_gt_paths(
	gt_paths: list[Path], rows: list[dict], output_dir: Path
) -> list[Path]:
	selected_ids = {int(row["id"]) for row in rows}
	output_dir.mkdir(parents=True, exist_ok=True)
	filtered_paths = []
	for path in gt_paths:
		data = json.loads(path.read_text())
		data["images"] = [x for x in data["images"] if int(x["id"]) in selected_ids]
		data["annotations"] = [
			x for x in data["annotations"] if int(x["image_id"]) in selected_ids
		]
		output_path = output_dir / path.name
		output_path.write_text(json.dumps(data))
		filtered_paths.append(output_path)
	return filtered_paths


def _evaluate(prediction_path: Path, gt_paths: list[Path]) -> dict:
	evaluator_cls = _load_cgf1_evaluator()

	result = {}
	for iou_type in ("segm", "bbox"):
		summary = evaluator_cls(
			[str(path) for path in gt_paths], verbose=True, iou_type=iou_type
		).evaluate(str(prediction_path))
		result[iou_type] = {
			"cgf1": summary[f"cgF1_eval_{iou_type}_cgF1"] * 100,
			"il_mcc": summary[f"cgF1_eval_{iou_type}_IL_MCC"],
			"pmf1": summary[f"cgF1_eval_{iou_type}_positive_micro_F1"] * 100,
		}
	return result


def _load_cgf1_evaluator():
	"""Load the evaluator migrated into this model example."""
	from cgf1_eval import CGF1Evaluator

	return CGF1Evaluator


def evaluate_engine(
	engine,
	dataset_root: Path,
	subset: str,
	limit: int,
	output_dir: Path,
) -> dict:
	"""Run one engine on SA-Co Gold and return CGF1 metrics."""
	if subset not in GOLD_GTS:
		raise ValueError(f"Unsupported evaluation subset: {subset}")
	gt_dir = dataset_root / "gt-annotations"
	gt_paths = [gt_dir / name for name in GOLD_GTS[subset]]
	missing = [str(path) for path in gt_paths if not path.is_file()]
	if missing:
		raise FileNotFoundError(f"Missing evaluation annotations: {missing}")
	image_root = dataset_root / ("sa1b-images" if subset == "sa1b_nps" else "metaclip-images")
	rows = _load_rows(gt_paths, limit)
	if not image_root.is_dir():
		raise FileNotFoundError(f"Missing evaluation image directory: {image_root}")

	predictions = []
	for row in tqdm(rows, desc=f"{engine.backend} SAM3 evaluation"):
		path = _image_path(image_root, row["file_name"])
		with Image.open(path) as image_file:
			image = np.asarray(image_file.convert("RGB"))[:, :, ::-1].copy()
		results = engine.infer(image, row["text_input"], ensure_one=False)
		for result in results:
			box = np.asarray(result["box"], dtype=np.float32)
			mask = np.asarray(result["mask"] > 0.5, dtype=np.uint8)
			predictions.append(
				{
					"image_id": int(row["id"]),
					"category_id": 1,
					"bbox": [
						float(box[0] / image.shape[1]),
						float(box[1] / image.shape[0]),
						float(max(0.0, box[2] - box[0]) / image.shape[1]),
						float(max(0.0, box[3] - box[1]) / image.shape[0]),
					],
					"segmentation": _encode_mask(mask),
					"score": float(result["score"]),
					"area": float(mask.sum()),
				}
			)

	output_dir = output_dir / subset
	output_dir.mkdir(parents=True, exist_ok=True)
	prediction_path = output_dir / "coco_predictions_segm.json"
	prediction_path.write_text(json.dumps(predictions))
	eval_gt_paths = _filtered_gt_paths(gt_paths, rows, output_dir / "filtered_gt")
	metrics = _evaluate(prediction_path, eval_gt_paths)
	return {"queries": len(rows), "predictions": len(predictions), "metrics": metrics}