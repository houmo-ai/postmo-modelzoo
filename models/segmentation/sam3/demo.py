#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   SAM3 segmentation demo with HMM and ONNX Runtime backends.
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

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from hmatc.utils.utils import first_not_none, get_model_configs
from eval_util import GOLD_GTS, evaluate_engine, locate_dataset
from sam3_engine import SAM3Engine
from sam3_processor import DEFAULT_MODEL_DIR

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CURRENT_DIR / "config.yml"
SINGLE_BOX_XYWH = [480.0, 290.0, 110.0, 360.0]
MULTI_BOX_XYWH = [SINGLE_BOX_XYWH, [370.0, 280.0, 115.0, 375.0]]
MULTI_BOX_LABELS = [1, 0]
RESULT_COLORS = [
	(230, 159, 0),
	(86, 180, 233),
	(0, 158, 115),
	(240, 228, 66),
	(0, 114, 178),
	(213, 94, 0),
	(204, 121, 167),
]


def parse_bool(value: str) -> bool:
	"""Parse common command-line boolean values."""
	normalized = value.lower()
	if normalized in {"1", "true", "yes", "on"}:
		return True
	if normalized in {"0", "false", "no", "off"}:
		return False
	raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def default_image_path() -> str:
	root = Path(os.getenv("HOUMO_EXAMPLES_PATH", CURRENT_DIR.parents[2]))
	return str(root / "data" / "pic" / "sam3_test_image.jpg")


def get_args() -> argparse.Namespace:
	"""Parse commandline and resolve defaults from config.yml."""
	parser = argparse.ArgumentParser(description="SAM3 HMM/ONNX inference demo")
	parser.add_argument("--config", dest="config_path", type=str, default=str(DEFAULT_CONFIG_PATH))
	parser.add_argument(
		"--backend",
		type=str,
		default="hmm",
		choices=["hmm", "xh2", "onnx", "ort"],
		help="inference backend; xh2 aliases hmm and ort aliases onnx",
	)
	parser.add_argument("--model_name", type=str, default=None, help="model name")
	parser.add_argument("--model_size", type=str, default=None, help="model size")
	parser.add_argument("--model", type=str, default=None, help="HMM or ONNX model path")
	parser.add_argument(
		"--eval",
		action="store_true",
		help="evaluate the selected subset on SA-Co Gold",
	)
	parser.add_argument(
		"--dataset_path",
		type=str,
		default=None,
		help="path to saco_gold or saco_gold.zip; auto-detected when omitted",
	)
	parser.add_argument(
		"--subset",
		type=str,
		default="sa1b_nps",
		choices=list(GOLD_GTS),
		help="SA-Co Gold subset to evaluate",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=200,
		help="number of queries to evaluate; 0 means the complete subset",
	)
	parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR, help="local SAM3 model directory")
	parser.add_argument("--image", type=str, default=None, help="input image path")
	parser.add_argument("--prompt", type=str, default="shoe", help="text prompt")
	parser.add_argument("--ndevice", type=int, default=None, help="device number")
	parser.add_argument("--max_size_w", type=int, default=None, help="maximum image width")
	parser.add_argument("--max_size_h", type=int, default=None, help="maximum image height")
	parser.add_argument("--threshold", type=float, default=0.5, help="confidence threshold")
	parser.add_argument("--mode", type=int, default=0, choices=[0, 1], help="0: result only, 1: all five results")
	parser.add_argument("--output", type=str, default=None, help="result image path")
	parser.add_argument("--output_dir", type=str, default=None, help="directory for all five results")
	parser.add_argument(
		"--perf",
		type=parse_bool,
		nargs="?",
		const=True,
		default=True,
		help="run backend performance test (true/false)",
	)
	parser.add_argument("--warmup", type=int, default=1, help="performance warmup count")
	parser.add_argument("--repeat", type=int, default=1, help="performance repeat count")
	args = parser.parse_args()

	default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
	args.model_name = first_not_none(args.model_name, default_model_name)
	args.model_size = first_not_none(args.model_size, default_model_size)
	model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
	args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
	args.max_size_w = first_not_none(args.max_size_w, model_config.get("max_size_w", 1008))
	args.max_size_h = first_not_none(args.max_size_h, model_config.get("max_size_h", 1008))
	args.image = first_not_none(args.image, default_image_path())
	model_shape_name = (
		f"{args.model_name}_{args.model_size}_"
		f"{args.max_size_w}x{args.max_size_h}"
	)
	if args.backend in ("onnx", "ort"):
		default_model = CURRENT_DIR / "work_dirs" / f"{model_shape_name}_sim.onnx"
		default_output = "demo_onnx_result.png"
	else:
		default_model = CURRENT_DIR / "output" / HOUMO_TARGET / f"{model_shape_name}.hmm"
		default_output = "demo_hmm_result.png"
	args.model = first_not_none(
		args.model,
		str(default_model),
	)
	args.output = first_not_none(args.output, default_output)
	return args


def run_evaluation(args: argparse.Namespace) -> None:
	"""Evaluate ONNX first when available, then always evaluate the HMM model."""
	if args.limit < 0:
		raise ValueError("--limit must be greater than or equal to 0")
	dataset_root = locate_dataset(args.dataset_path, CURRENT_DIR)
	model_shape_name = (
		f"{args.model_name}_{args.model_size}_"
		f"{args.max_size_w}x{args.max_size_h}"
	)
	hmm_model = CURRENT_DIR / "output" / HOUMO_TARGET / f"{model_shape_name}.hmm"
	if args.backend in ("hmm", "xh2") and args.model is not None:
		hmm_model = Path(args.model)
	if not hmm_model.is_file():
		raise FileNotFoundError(f"HMM evaluation model not found: {hmm_model}")

	onnx_model = CURRENT_DIR / "work_dirs" / f"{model_shape_name}_sim.onnx"
	if onnx_model.is_file():
		print(f"[info] Evaluating floating-point ONNX model: {onnx_model}")
		onnx_engine = SAM3Engine(
			backend="onnx",
			model_dir=args.model_dir,
			ndevice=args.ndevice,
			threshold=args.threshold,
			max_size_w=args.max_size_w,
			max_size_h=args.max_size_h,
		).load(str(onnx_model))
		onnx_result = evaluate_engine(
			onnx_engine,
			dataset_root,
			args.subset,
			args.limit,
			CURRENT_DIR / "work_dirs" / "eval_onnx",
		)
		print(f"[result] ONNX: {onnx_result}")
	else:
		print(
			"[warning] Floating-point ONNX model does not exist.\n"
			"[warning] Please run the quantization step to generate it.\n"
			"[warning] The evaluation will continue with the hardware HMM model."
		)

	print(f"[info] Evaluating HMM model: {hmm_model}")
	hmm_engine = SAM3Engine(
		backend="hmm",
		model_dir=args.model_dir,
		ndevice=args.ndevice,
		threshold=args.threshold,
		max_size_w=args.max_size_w,
		max_size_h=args.max_size_h,
	).load(str(hmm_model))
	hmm_result = evaluate_engine(
		hmm_engine,
		dataset_root,
		args.subset,
		args.limit,
		CURRENT_DIR / "work_dirs" / "eval_hmm",
	)
	print(f"[result] HMM: {hmm_result}")


def draw_results(
	image: np.ndarray,
	results,
	alpha: float = 0.5,
) -> np.ndarray:
	"""Draw masks, matching boxes, and labeled confidence text."""
	result = image.copy()
	for index, item in enumerate(results[:20]):
		mask = item["mask"] > 0.5
		color = np.asarray(
			RESULT_COLORS[index % len(RESULT_COLORS)], dtype=np.uint8
		)
		result[mask] = (
			result[mask].astype(np.float32) * (1.0 - alpha)
			+ color.astype(np.float32) * alpha
		).astype(np.uint8)
		box = np.rint(item["box"]).astype(int)
		box[[0, 2]] = np.clip(box[[0, 2]], 0, image.shape[1] - 1)
		box[[1, 3]] = np.clip(box[[1, 3]], 0, image.shape[0] - 1)
		box_color = tuple(int(value) for value in color)
		cv2.rectangle(result, tuple(box[:2]), tuple(box[2:]), box_color, 2)

		text = f"id={index}, prob={item['score']:.2f}"
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = max(0.3, min(0.4, min(image.shape[:2]) / 1800.0))
		font_thickness = 1
		(text_width, text_height), baseline = cv2.getTextSize(
			text, font, font_scale, font_thickness
		)
		padding = 3
		text_x = int(box[0])
		text_y = int(box[1]) - 5
		if text_y - text_height - padding < 0:
			text_y = min(int(box[1]) + text_height + padding + 5, image.shape[0] - 1)
		background_right = min(
			text_x + text_width + padding * 2, image.shape[1] - 1
		)
		background_top = max(text_y - text_height - padding, 0)
		background_bottom = min(text_y + baseline + padding, image.shape[0] - 1)
		overlay = result.copy()
		cv2.rectangle(
			overlay,
			(text_x, background_top),
			(background_right, background_bottom),
			(255, 255, 255),
			-1,
			cv2.LINE_8,
		)
		cv2.addWeighted(overlay, 0.75, result, 0.25, 0.0, result)
		cv2.rectangle(
			result,
			(text_x, background_top),
			(background_right, background_bottom),
			(0, 0, 0),
			1,
			cv2.LINE_AA,
		)
		cv2.putText(
			result,
			text,
			(text_x + padding, text_y),
			font,
			font_scale,
			box_color,
			font_thickness,
			cv2.LINE_AA,
		)
	return result


def draw_prompt_boxes(image: np.ndarray, boxes, labels) -> np.ndarray:
	"""Draw positive and negative geometry prompts."""
	result = image.copy()
	for box_xywh, label in zip(boxes, labels):
		x, y, width, height = [int(value) for value in box_xywh]
		color = (0, 255, 0) if label > 0 else (0, 0, 255)
		cv2.rectangle(result, (x, y), (x + width, y + height), color, 3)
		cv2.putText(
			result,
			"positive" if label > 0 else "negative",
			(x, max(20, y - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			color,
			2,
		)
	return result


def save_result(path: Path, image: np.ndarray, results) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if not cv2.imwrite(str(path), draw_results(image, results)):
		raise RuntimeError(f"Failed to save result: {path}")
	print(f"[info] Result saved to: {path}")


def save_image(path: Path, image: np.ndarray) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if not cv2.imwrite(str(path), image):
		raise RuntimeError(f"Failed to save image: {path}")
	print(f"[info] Image saved to: {path}")


def run_all_results(
	engine: SAM3Engine,
	image: np.ndarray,
	prompt: str,
	output_dir: Path,
) -> None:
	"""Run text, single-box and multi-box cases and save five images."""
	output_dir.mkdir(parents=True, exist_ok=True)
	text_results = engine.infer(image, prompt)
	single_results = engine.infer(image, "visual", [SINGLE_BOX_XYWH], [1])
	multi_results = engine.infer(image, "visual", MULTI_BOX_XYWH, MULTI_BOX_LABELS)
	images = {
		"01_text_prompt_result.png": draw_results(image, text_results),
		"02_single_box_prompt.png": draw_prompt_boxes(
			image, [SINGLE_BOX_XYWH], [1]
		),
		"03_single_box_result.png": draw_results(image, single_results),
		"04_multi_box_prompt.png": draw_prompt_boxes(
			image, MULTI_BOX_XYWH, MULTI_BOX_LABELS
		),
		"05_multi_box_result.png": draw_results(image, multi_results),
	}
	for filename, result_image in images.items():
		save_image(output_dir / filename, result_image)


def print_performance(profile: dict[str, float], warmup: int, repeat: int) -> None:
	print("\n========== SAM3 Performance ==========")
	print(f"warmup={warmup}, repeat={repeat}")
	print(f"set_input : {profile['set_input_ms']:.3f} ms")
	print(f"infer     : {profile['infer_ms']:.3f} ms")
	print(f"get_output: {profile['get_output_ms']:.3f} ms")
	print(f"total     : {profile['total_ms']:.3f} ms")
	print(f"throughput: {profile['fps']:.3f} images/s")
	print("======================================")


def main() -> None:
	args = get_args()
	if args.eval:
		run_evaluation(args)
		return
	image = cv2.imread(args.image)
	if image is None:
		raise FileNotFoundError(f"Failed to read image: {args.image}")
	print(f"[info] Image: {args.image}")
	print(f"[info] Image size: {image.shape[1]}x{image.shape[0]}")
	engine = SAM3Engine(
		backend=args.backend,
		model_dir=args.model_dir,
		ndevice=args.ndevice,
		threshold=args.threshold,
		max_size_w=args.max_size_w,
		max_size_h=args.max_size_h,
	).load(args.model)
	if args.perf:
		print_performance(engine.benchmark(image, args.prompt, args.warmup, args.repeat), args.warmup, args.repeat)
	if args.mode == 0:
		results = engine.infer(image, args.prompt)
		save_result(Path(args.output), image, results)
	else:
		output_dir = Path(args.output_dir or Path(args.output).with_suffix(""))
		run_all_results(engine, image, args.prompt, output_dir)


if __name__ == "__main__":
	main()