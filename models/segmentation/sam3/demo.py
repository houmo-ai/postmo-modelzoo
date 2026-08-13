#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   SAM3 segmentation demo with HMM backend.
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
from sam3_engine import SAM3Engine
from sam3_processor import DEFAULT_MODEL_DIR

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CURRENT_DIR / "config.yml"
SINGLE_BOX_XYWH = [480.0, 290.0, 110.0, 360.0]
MULTI_BOX_XYWH = [SINGLE_BOX_XYWH, [370.0, 280.0, 115.0, 375.0]]
MULTI_BOX_LABELS = [1, 0]


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
	parser = argparse.ArgumentParser(description="SAM3 HMM inference demo")
	parser.add_argument("--config", dest="config_path", type=str, default=str(DEFAULT_CONFIG_PATH))
	parser.add_argument("--model_name", type=str, default=None, help="model name")
	parser.add_argument("--model_size", type=str, default=None, help="model size")
	parser.add_argument("--model", type=str, default=None, help="HMM model path")
	parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR, help="local SAM3 model directory")
	parser.add_argument("--image", type=str, default=None, help="input image path")
	parser.add_argument("--prompt", type=str, default="shoe", help="text prompt")
	parser.add_argument("--ndevice", type=int, default=None, help="device number")
	parser.add_argument("--max_size_w", type=int, default=None, help="maximum image width")
	parser.add_argument("--max_size_h", type=int, default=None, help="maximum image height")
	parser.add_argument("--threshold", type=float, default=0.5, help="confidence threshold")
	parser.add_argument("--mode", type=int, default=0, choices=[0, 1], help="0: result only, 1: all five results")
	parser.add_argument("--output", type=str, default="demo_hmm_result.png", help="result image path")
	parser.add_argument("--output_dir", type=str, default=None, help="directory for all five results")
	parser.add_argument(
		"--perf",
		type=parse_bool,
		nargs="?",
		const=True,
		default=True,
		help="run HMM performance test (true/false)",
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
	hmm_name = (
		f"{args.model_name}_{args.model_size}_"
		f"{args.max_size_w}x{args.max_size_h}.hmm"
	)
	args.model = first_not_none(
		args.model,
		str(CURRENT_DIR / "output" / HOUMO_TARGET / hmm_name),
	)
	return args


def draw_results(image: np.ndarray, results, alpha: float = 0.45) -> np.ndarray:
	"""Draw masks, boxes and scores on an image."""
	result = image.copy()
	rng = np.random.default_rng(0)
	for index, item in enumerate(results[:20]):
		mask = item["mask"] > 0.5
		color = rng.integers(64, 256, size=3, dtype=np.uint8)
		result[mask] = (
			result[mask].astype(np.float32) * (1.0 - alpha)
			+ color.astype(np.float32) * alpha
		).astype(np.uint8)
		box = np.asarray(item["box"]).astype(int)
		cv2.rectangle(result, tuple(box[:2]), tuple(box[2:]), (0, 255, 255), 2)
		cv2.putText(
			result,
			f"{index}:{item['score']:.2f}",
			(box[0], max(20, box[1] - 5)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 255, 255),
			2,
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


def run_all_results(engine: SAM3Engine, image: np.ndarray, prompt: str, output_dir: Path) -> None:
	"""Run text, single-box and multi-box cases and save five images."""
	output_dir.mkdir(parents=True, exist_ok=True)
	text_results = engine.infer(image, prompt)
	single_results = engine.infer(image, "visual", [SINGLE_BOX_XYWH], [1])
	multi_results = engine.infer(image, "visual", MULTI_BOX_XYWH, MULTI_BOX_LABELS)
	save_result(output_dir / "01_text_prompt_result.png", image, text_results)
	save_image(
		output_dir / "02_single_box_prompt.png",
		draw_prompt_boxes(image, [SINGLE_BOX_XYWH], [1]),
	)
	save_result(output_dir / "03_single_box_result.png", image, single_results)
	save_image(
		output_dir / "04_multi_box_prompt.png",
		draw_prompt_boxes(image, MULTI_BOX_XYWH, MULTI_BOX_LABELS),
	)
	save_result(output_dir / "05_multi_box_result.png", image, multi_results)


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
	image = cv2.imread(args.image)
	if image is None:
		raise FileNotFoundError(f"Failed to read image: {args.image}")
	print(f"[info] Image: {args.image}")
	print(f"[info] Image size: {image.shape[1]}x{image.shape[0]}")
	engine = SAM3Engine(
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