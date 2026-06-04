#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""GLM-OCR full pipeline helpers for the HOUMO NPU demo."""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

import tcim_lite as tcim

DEFAULT_PPDOCLAYOUTV3_ID2LABEL = {
    0: "abstract",
    1: "algorithm",
    2: "aside_text",
    3: "chart",
    4: "content",
    5: "formula",
    6: "doc_title",
    7: "figure_title",
    8: "footer",
    9: "footer",
    10: "footnote",
    11: "formula_number",
    12: "header",
    13: "header",
    14: "image",
    15: "formula",
    16: "number",
    17: "paragraph_title",
    18: "reference",
    19: "reference_content",
    20: "seal",
    21: "table",
    22: "text",
    23: "text",
    24: "vision_footnote",
}


def find_glmocr_sdk_root() -> Optional[str]:
    candidates = [
        Path(__file__).resolve().parent / "GLM-OCR",
        Path(__file__).resolve().parent.parent / "examples" / "llm" / "glm_ocr" / "GLM-OCR",
        Path.cwd() / "examples" / "llm" / "glm_ocr" / "GLM-OCR",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return None


def ensure_glmocr_sdk() -> str:
    sdk_root = find_glmocr_sdk_root()
    if sdk_root is None:
        raise RuntimeError(
            "Official GLM-OCR SDK not found. Put it under "
            "examples/llm/glm_ocr/GLM-OCR or work_dirs/GLM-OCR."
        )
    if sdk_root not in sys.path:
        sys.path.insert(0, sdk_root)
    return sdk_root


def resolve_path(path: Optional[str], base_dir: Optional[str] = None) -> Optional[str]:
    if path is None:
        return None
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)

    candidates = [Path.cwd() / path_obj]
    if base_dir is not None:
        candidates.append(Path(base_dir) / path_obj)
    candidates.append(Path(__file__).resolve().parent.parent / path_obj)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[-1].resolve())


class NPUOCRClient:
    """SDK-compatible OCR client that calls the local VIT+prefill+decode NPU flow."""

    def __init__(self, hmglm_ocr, max_new_tokens: int, prompt: Optional[str] = None):
        self.hmglm_ocr = hmglm_ocr
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt

    def start(self):
        return None

    def stop(self):
        return None

    def process_image(
        self,
        image: Image.Image,
        task_type: str = "text",
        prompt: Optional[str] = None,
    ) -> str:
        del task_type
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.convert("RGB").save(tmp.name)
            return self.hmglm_ocr.generate(
                tmp.name,
                prompt=prompt or self.prompt or "Text Recognition:",
                max_new_tokens=self.max_new_tokens,
                stream=False,
            )


class PPDocLayoutV3NPUDetector:
    """Official SDK-compatible PP-DocLayoutV3 detector backed by an NPU HMM."""

    def __init__(
        self,
        config,
        layout_path: str,
        weight_manager,
        batch_size: Optional[int] = None,
    ):
        self.config = config
        self.layout_path = layout_path
        self.weight_manager = weight_manager
        self.batch_size = batch_size or config.batch_size
        self.threshold = config.threshold
        self.threshold_by_class = config.threshold_by_class
        self.layout_nms = config.layout_nms
        self.layout_unclip_ratio = config.layout_unclip_ratio
        self.layout_merge_bboxes_mode = config.layout_merge_bboxes_mode
        self.label_task_mapping = config.label_task_mapping
        self.id2label = getattr(config, "id2label", None)
        self._layout_model = None
        self._image_processor = None

    def start(self):
        ensure_glmocr_sdk()
        try:
            from transformers import PPDocLayoutV3ImageProcessor
        except ImportError:
            from transformers.models.pp_doclayout_v3.image_processing_pp_doclayout_v3_fast import (
                PPDocLayoutV3ImageProcessorFast as PPDocLayoutV3ImageProcessor,
            )

        self._image_processor = PPDocLayoutV3ImageProcessor()
        option = tcim.runtime.Option(self.weight_manager)
        self._layout_model = tcim.runtime.load(self.layout_path, option=option)

        if self.id2label is None:
            self.id2label = DEFAULT_PPDOCLAYOUTV3_ID2LABEL
        self.id2label = {int(key): value for key, value in self.id2label.items()}
        if self.label_task_mapping is None:
            self.label_task_mapping = {"text": list(self.id2label.values())}
        self._patch_safe_polygon_extract()

    def stop(self):
        self._layout_model = None
        self._image_processor = None

    def _patch_safe_polygon_extract(self):
        import cv2

        def _safe_extract(boxes, masks, scale_ratio):
            scale_w, scale_h = scale_ratio[0] / 4, scale_ratio[1] / 4
            mask_h, mask_w = masks.shape[1:]
            polygon_points = []
            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i].astype(np.int32)
                box_w, box_h = x_max - x_min, y_max - y_min
                rect = np.array(
                    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    dtype=np.float32,
                )
                if box_w <= 0 or box_h <= 0:
                    polygon_points.append(rect)
                    continue
                x_start = int(round((x_min * scale_w).item()))
                x_end = int(round((x_max * scale_w).item()))
                y_start = int(round((y_min * scale_h).item()))
                y_end = int(round((y_max * scale_h).item()))
                x_start, x_end = np.clip([x_start, x_end], 0, mask_w)
                y_start, y_end = np.clip([y_start, y_end], 0, mask_h)
                cropped_mask = masks[i, y_start:y_end, x_start:x_end]
                if cropped_mask.size == 0:
                    polygon_points.append(rect)
                    continue
                resized = cv2.resize(
                    cropped_mask.astype(np.uint8),
                    (box_w, box_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                polygon = self._image_processor._mask2polygon(resized)
                if polygon is not None and len(polygon) < 4:
                    polygon_points.append(rect)
                    continue
                if polygon is not None and len(polygon) > 0:
                    polygon = polygon + np.array([x_min, y_min])
                polygon_points.append(polygon)
            return polygon_points

        self._image_processor._extract_polygon_points_by_masks = _safe_extract

    def _apply_per_class_threshold(self, raw_results: List[Dict]):
        if not self.threshold_by_class:
            return raw_results
        label2id = {name: int(cls_id) for cls_id, name in self.id2label.items()}
        class_thresholds = {}
        for key, value in self.threshold_by_class.items():
            class_thresholds[label2id[key] if isinstance(key, str) and key in label2id else int(key)] = float(value)

        filtered = []
        for result in raw_results:
            scores = result["scores"]
            labels = result["labels"]
            thresholds = torch.full_like(scores, self.threshold)
            for class_id, thresh in class_thresholds.items():
                thresholds[labels == class_id] = thresh
            keep = scores >= thresholds
            item = {
                "scores": scores[keep],
                "labels": labels[keep],
                "boxes": result["boxes"][keep],
            }
            if "order_seq" in result:
                item["order_seq"] = result["order_seq"][keep]
            if "polygon_points" in result:
                keep_list = keep.tolist()
                item["polygon_points"] = [
                    p for p, k in zip(result["polygon_points"], keep_list) if k
                ]
            filtered.append(item)
        return filtered

    def _run_layout(self, pixel_values: torch.Tensor):
        self._layout_model.set_input(
            self._layout_model.get_input_name(0),
            pixel_values.detach().cpu().numpy().astype(np.float16),
        )
        self._layout_model.run()
        self._layout_model.sync()

        outputs = []
        for i in range(self._layout_model.get_num_outputs()):
            output = self._layout_model.get_output(self._layout_model.get_output_name(i))
            outputs.append(torch.tensor(output.numpy()))
        return SimpleNamespace(
            logits=outputs[0],
            pred_boxes=outputs[1],
            order_logits=outputs[2],
            out_masks=outputs[3],
        )

    def process(
        self,
        images: List[Image.Image],
        save_visualization: bool = False,
        global_start_idx: int = 0,
        use_polygon: bool = False,
    ):
        if self._layout_model is None or self._image_processor is None:
            raise RuntimeError("Layout detector not started. Call start() first.")
        from glmocr.utils.layout_postprocess_utils import apply_layout_postprocess
        from glmocr.utils.visualization_utils import draw_layout_boxes

        pil_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        all_paddle_format_results = []

        for chunk_start in range(0, len(pil_images), self.batch_size):
            chunk_pil = pil_images[chunk_start:chunk_start + self.batch_size]
            inputs = self._image_processor(images=chunk_pil, return_tensors="pt")
            outputs = self._run_layout(inputs["pixel_values"].half())
            target_sizes = torch.tensor([img.size[::-1] for img in chunk_pil])
            pre_threshold = (
                min(self.threshold, min(self.threshold_by_class.values()))
                if self.threshold_by_class
                else self.threshold
            )
            raw_results = self._image_processor.post_process_object_detection(
                outputs,
                threshold=pre_threshold,
                target_sizes=target_sizes,
            )
            raw_results = self._apply_per_class_threshold(raw_results)
            all_paddle_format_results.extend(
                apply_layout_postprocess(
                    raw_results=raw_results,
                    id2label=self.id2label,
                    img_sizes=[img.size for img in chunk_pil],
                    layout_nms=self.layout_nms,
                    layout_unclip_ratio=self.layout_unclip_ratio,
                    layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
                )
            )

        vis_images: Dict[int, Image.Image] = {}
        if save_visualization:
            for img_idx, img_results in enumerate(all_paddle_format_results):
                vis_images[global_start_idx + img_idx] = draw_layout_boxes(
                    image=np.array(pil_images[img_idx]),
                    boxes=img_results,
                    use_polygon=use_polygon,
                )

        all_results = []
        for img_idx, paddle_results in enumerate(all_paddle_format_results):
            image_width, image_height = pil_images[img_idx].size
            results = []
            valid_index = 0
            for item in paddle_results:
                label = item["label"]
                task_type = None
                for task_item, labels in self.label_task_mapping.items():
                    if isinstance(labels, list) and label in labels:
                        task_type = task_item
                        break
                if task_type is None or task_type == "abandon":
                    continue
                x1, y1, x2, y2 = item["coordinate"]
                polygon = [
                    [
                        int(float(point[0]) / image_width * 1000),
                        int(float(point[1]) / image_height * 1000),
                    ]
                    for point in item["polygon_points"]
                ]
                results.append(
                    {
                        "index": valid_index,
                        "label": label,
                        "score": float(item["score"]),
                        "bbox_2d": [
                            int(float(x1) / image_width * 1000),
                            int(float(y1) / image_height * 1000),
                            int(float(x2) / image_width * 1000),
                            int(float(y2) / image_height * 1000),
                        ],
                        "polygon": polygon,
                        "task_type": task_type,
                    }
                )
                valid_index += 1
            all_results.append(results)

        return all_results, vis_images


class LocalLayoutPipeline:
    def __init__(self, config, layout_detector, ocr_client):
        ensure_glmocr_sdk()
        from glmocr.pipeline import Pipeline

        self._pipeline = Pipeline(config=config, layout_detector=layout_detector)
        self.page_loader = self._pipeline.page_loader
        self.layout_detector = self._pipeline.layout_detector
        self.result_formatter = self._pipeline.result_formatter
        self.config = config
        self.ocr_client = ocr_client

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.layout_detector.start()
        self.ocr_client.start()

    def stop(self):
        self.ocr_client.stop()
        self.layout_detector.stop()

    def process_local(self, source: str, save_layout_visualization: bool = True):
        pages = self.page_loader.load_pages(source)
        layout_results, layout_vis_images = self.layout_detector.process(
            pages,
            save_visualization=save_layout_visualization,
            use_polygon=self.config.layout.use_polygon,
        )
        grouped_results: List[List[Dict[str, Any]]] = []
        for page, page_layout in zip(pages, layout_results):
            page_items = []
            for region in page_layout:
                item = dict(region)
                if item.get("task_type") == "skip":
                    item["content"] = None
                    page_items.append(item)
                    continue
                from glmocr.utils.image_utils import crop_image_region

                polygon = item.get("polygon") if self.config.layout.use_polygon else None
                cropped = crop_image_region(page, item["bbox_2d"], polygon)
                item["content"] = self.ocr_client.process_image(
                    cropped,
                    task_type=item.get("task_type", "text"),
                )
                page_items.append(item)
            grouped_results.append(page_items)

        json_str, markdown_str, image_files = self.result_formatter.process(grouped_results)
        from glmocr.parser_result import PipelineResult

        return PipelineResult(
            json_result=json.loads(json_str),
            markdown_result=markdown_str,
            original_images=[source],
            image_files=image_files,
            layout_vis_images=layout_vis_images,
        )


def build_pipeline_config(args, base_dir: Optional[str] = None):
    ensure_glmocr_sdk()
    from glmocr.config import PipelineConfig

    config = PipelineConfig()
    config.max_workers = 1
    config.page_loader.pdf_dpi = args.pdf_dpi
    config.page_loader.pdf_max_pages = args.pdf_max_pages
    config.page_loader.max_tokens = args.max_new_tokens
    config.page_loader.task_prompt_mapping = {
        "text": args.prompt or "Text Recognition:",
        "table": args.prompt or "Text Recognition:",
        "formula": args.prompt or "Text Recognition:",
    }
    config.layout.batch_size = args.layout_batch_size
    config.layout.threshold = args.layout_threshold
    config.layout.use_polygon = args.layout_use_polygon
    config.result_formatter.label_visualization_mapping = {
        "text": [
            "abstract",
            "algorithm",
            "aside_text",
            "content",
            "doc_title",
            "figure_title",
            "footer",
            "footnote",
            "header",
            "image",
            "number",
            "paragraph_title",
            "reference",
            "reference_content",
            "seal",
            "text",
            "vision_footnote",
        ],
        "table": ["chart", "table"],
        "formula": ["formula", "formula_number"],
    }
    return config


def save_pipeline_result(result, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.save(output_dir=str(output_path), save_layout_visualization=True)
    markdown_path = output_path / "markdown.md"
    result_json_path = output_path / "result.json"
    markdown_path.write_text(result.markdown_result, encoding="utf-8")
    result_json_path.write_text(
        json.dumps(result.json_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    region_counts = (
        [len(page) for page in result.json_result]
        if isinstance(result.json_result, list)
        else []
    )
    return {
        "markdown": str(markdown_path),
        "result_json": str(result_json_path),
        "markdown_chars": len(result.markdown_result),
        "region_counts": region_counts,
    }


def run_npu_full_pipeline(args, hmglm_ocr, logger, base_dir: Optional[str] = None) -> dict:
    ensure_glmocr_sdk()
    source = args.pdf if args.pdf is not None else args.image
    source = resolve_path(source, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir)
    layout_path = resolve_path(args.layout_path, base_dir)
    if layout_path is None or not os.path.exists(layout_path):
        raise FileNotFoundError(f"PP-DocLayoutV3 HMM not found: {layout_path}")

    config = build_pipeline_config(args, base_dir)
    layout_detector = PPDocLayoutV3NPUDetector(
        config.layout,
        layout_path=layout_path,
        weight_manager=hmglm_ocr.weight_manager,
        batch_size=args.layout_batch_size,
    )
    ocr_client = NPUOCRClient(
        hmglm_ocr,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
    )

    logger.info(f"Running GLM-OCR full pipeline: source={source}")
    logger.info(f"PP-DocLayoutV3 HMM: {layout_path}")
    with LocalLayoutPipeline(config, layout_detector, ocr_client) as pipeline:
        result = pipeline.process_local(source, save_layout_visualization=True)

    print("response:", flush=True)
    print("\033[1;95m{}\033[0m".format(result.markdown_result), flush=True)

    summary = save_pipeline_result(result, output_dir)
    summary.update(
        {
            "input": source,
            "layout_path": layout_path,
            "backend": "npu_hmm_full_pipeline",
        }
    )
    summary_path = Path(output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
