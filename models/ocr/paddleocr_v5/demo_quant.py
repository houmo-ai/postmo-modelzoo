#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: demo_quant.py
# Description:
#   PaddleOCR V5 detection + recognition demo using quantized HMONNX models.
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
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import onnx
import torch
from loguru import logger
from xhquant.api import HMONNXGoldenInference

from demo import (
    CTCLabelDecode,
    _boxes_from_bitmap,
    _crop_text_region,
    _default_dict_path,
    _draw_results,
    _expand_text_box,
    _filter_tag_det_res,
    _resize_det_image,
    _resize_rec_image,
    _sort_boxes,
)


SCRIPT_DIR = Path(__file__).resolve().parent
HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def _onnx_input_shape(model_path: str) -> List[int]:
    model = onnx.load_model(model_path)
    graph_input = model.graph.input[0]
    shape = []
    for dim in graph_input.type.tensor_type.shape.dim:
        shape.append(int(dim.dim_value) if dim.dim_value > 0 else 1)
    return shape


class QuantHMONNXModule:
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.shape = _onnx_input_shape(model_path)
        self.session = HMONNXGoldenInference(model_path)
        self.session.to(device)

    def run(self, input_data: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(input_data).to(self.device).half()
        with torch.no_grad():
            output = self.session(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        return np.asarray(output)


class PaddleOCRV5QuantDemo:
    def __init__(
        self,
        det_path: str,
        rec_path: str,
        character_dict_path: str,
        device: str = "cuda",
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.5,
        max_candidates: int = 1000,
        crop_pad_ratio: float = 0.25,
        rec_rgb: bool = False,
        debug_dir: str | None = None,
    ):
        logger.info("Loading quant det model: {}", det_path)
        self.det = QuantHMONNXModule(det_path, device)
        logger.info("Loading quant rec model: {}", rec_path)
        self.rec = QuantHMONNXModule(rec_path, device)
        self.det_shape = self.det.shape
        self.rec_shape = self.rec.shape
        self.ctc_decode = CTCLabelDecode(character_dict_path, use_space_char=True)
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.min_size = 8
        self.max_candidates = max_candidates
        self.crop_pad_ratio = crop_pad_ratio
        self.rec_rgb = rec_rgb
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info("det input shape: {}", self.det_shape)
        logger.info("rec input shape: {}", self.rec_shape)

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        _, _, input_h, input_w = self.det_shape
        det_input = _resize_det_image(image, input_h, input_w)
        pred = self.det.run(det_input).astype("float32", copy=False)
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
        return _sort_boxes(boxes)

    def recognize(self, image: np.ndarray, boxes: Sequence[np.ndarray]) -> List[Tuple[str, float]]:
        if not boxes:
            return []
        rec_batch, _, rec_h, rec_w = self.rec_shape
        results = []
        for start in range(0, len(boxes), rec_batch):
            batch_boxes = boxes[start : start + rec_batch]
            rec_inputs = []
            for local_idx, box in enumerate(batch_boxes):
                crop = _crop_text_region(image, _expand_text_box(box, image.shape, self.crop_pad_ratio))
                rec_preprocessed = _resize_rec_image(crop, rec_h, rec_w, use_rgb=self.rec_rgb)
                if self.debug_dir:
                    crop_index = start + local_idx
                    cv2.imwrite(str(self.debug_dir / f"crop_{crop_index:03d}.jpg"), crop)
                    debug_rec = rec_preprocessed.transpose(1, 2, 0).astype("float32")
                    debug_rec = ((debug_rec * 127.5) + 127.5).clip(0, 255).astype("uint8")
                    if self.rec_rgb:
                        debug_rec = cv2.cvtColor(debug_rec, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(self.debug_dir / f"rec_input_{crop_index:03d}.jpg"), debug_rec)
                rec_inputs.append(rec_preprocessed)
            while len(rec_inputs) < rec_batch:
                rec_inputs.append(np.zeros((3, rec_h, rec_w), dtype="float32"))
            rec_input = np.stack(rec_inputs, axis=0).astype("float32")
            pred = self.rec.run(rec_input)
            decoded = self.ctc_decode(pred)
            results.extend(decoded[: len(batch_boxes)])
        return results

    def run(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        start = time.time()
        det_start = time.time()
        boxes = self.detect(image)
        det_time = time.time() - det_start
        rec_start = time.time()
        rec_results = self.recognize(image, boxes)
        rec_time = time.time() - rec_start
        total_time = time.time() - start
        if self.debug_dir:
            debug_image = image.copy()
            for idx, box in enumerate(boxes):
                cv2.polylines(debug_image, [box.astype("int32").reshape(-1, 1, 2)], True, (0, 255, 255), 2)
                x = int(box[:, 0].min())
                y = int(box[:, 1].min())
                cv2.putText(debug_image, str(idx), (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(self.debug_dir / "det_boxes.jpg"), debug_image)
        return image, boxes, rec_results, {"det": det_time, "rec": rec_time, "total": total_time}


def get_args():
    parser = argparse.ArgumentParser(description="Run PaddleOCR V5 det+rec quantized HMONNX demo")
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument(
        "--det-path",
        default=str(
            SCRIPT_DIR
            / "output"
            / HOUMO_TARGET
            / "hmquant"
            / "paddleocr_v5_det"
            / "hmquant_paddleocr_v5_det_xh2_with_act.onnx"
        ),
        help="quantized detection HMONNX .onnx path",
    )
    parser.add_argument(
        "--rec-path",
        default=str(
            SCRIPT_DIR
            / "output"
            / HOUMO_TARGET
            / "hmquant"
            / "paddleocr_v5_rec"
            / "hmquant_paddleocr_v5_rec_xh2_with_act.onnx"
        ),
        help="quantized recognition HMONNX .onnx path",
    )
    parser.add_argument("--character-dict", default=_default_dict_path())
    parser.add_argument("--output", default=str(SCRIPT_DIR / "ocr_quant_result.jpg"), help="visualized result image")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--thresh", type=float, default=0.3)
    parser.add_argument("--box-thresh", type=float, default=0.6)
    parser.add_argument("--unclip-ratio", type=float, default=1.5)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--crop-pad-ratio", type=float, default=0.0, help="expand each detected text box before perspective crop")
    parser.add_argument("--rec-rgb", action="store_true", help="convert cropped rec images from BGR to RGB before recognition")
    parser.add_argument("--vis-font-path", default="/data/qianqian.zhao/NotoSansCJK-Regular.ttc", help="font path used to draw non-ASCII OCR text")
    parser.add_argument("--vis-font-size", type=int, default=24, help="font size used to draw OCR text")
    parser.add_argument("--debug-dir", default=None, help="save detection crops and recognition inputs for debugging")
    return parser.parse_args()


def main():
    args = get_args()
    demo = PaddleOCRV5QuantDemo(
        det_path=args.det_path,
        rec_path=args.rec_path,
        character_dict_path=args.character_dict,
        device=args.device,
        thresh=args.thresh,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
        max_candidates=args.max_candidates,
        crop_pad_ratio=args.crop_pad_ratio,
        rec_rgb=args.rec_rgb,
        debug_dir=args.debug_dir,
    )
    image, boxes, rec_results, timings = demo.run(args.image)
    for idx, (box, rec) in enumerate(zip(boxes, rec_results)):
        text, score = rec
        logger.info("[{}] score={:.4f} box={} text={}", idx, score, box.tolist(), text)
    logger.info(
        "PaddleOCR V5 quant demo done: boxes={}, det={:.3f}s rec={:.3f}s total={:.3f}s",
        len(boxes),
        timings["det"],
        timings["rec"],
        timings["total"],
    )
    output = _draw_results(
        image,
        boxes,
        rec_results,
        font_path=args.vis_font_path,
        font_size=args.vis_font_size,
    )
    cv2.imwrite(args.output, output)
    logger.info("Visualization saved to {}", args.output)


if __name__ == "__main__":
    main()
