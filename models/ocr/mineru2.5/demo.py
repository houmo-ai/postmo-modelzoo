# Copyright 2025 HOUMO AI
#
# File: demo.py
# Description:
#   MinerU2.5 demo script
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
import json
import math
import os
import re
import time

import numpy as np
import tcim_lite as tcim
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2VLProcessor

from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", "Only support HOUMO_TARGET: xh2"

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

_layout_re = (
    r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$"
)

ANGLE_MAPPING = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}

BLOCK_TYPES = [
    "text",
    "title",
    "table",
    "image",
    "code",
    "algorithm",
    "header",
    "footer",
    "page_number",
    "page_footnote",
    "aside_text",
    "equation",
    "equation_block",
    "ref_text",
    "list",
    "phonetic",
    "table_caption",
    "image_caption",
    "code_caption",
    "table_footnote",
    "image_footnote",
    "unknown",
]

# Block types that are skipped during content extraction
SKIP_EXTRACT_TYPES = {"image", "list", "equation_block"}

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _parse_bucket(bucket):
    if isinstance(bucket, dict):
        return int(bucket["max_size_h"]), int(bucket["max_size_w"])
    return int(bucket[0]), int(bucket[1])


def _validate_visual_bucket_manifest(manifest: dict) -> None:
    required = {
        "buckets",
        "fallback_bucket",
        "patch_size",
        "spatial_merge_size",
        "temporal_patch_size",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(
            f"Invalid visual bucket manifest, missing keys: {sorted(missing)}"
        )

    factor = int(manifest["patch_size"]) * int(manifest["spatial_merge_size"])
    for item in manifest["buckets"]:
        bucket = _parse_bucket(item)
        if bucket[0] % factor != 0 or bucket[1] % factor != 0:
            raise ValueError(
                f"Visual bucket {bucket[0]}x{bucket[1]} must be divisible by {factor}"
            )


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _smart_resize(height: int, width: int, factor: int) -> tuple[int, int]:
    max_pixels = 16384 * factor**2
    min_pixels = 4 * factor**2
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _convert_bbox(bbox):
    bbox = tuple(map(int, bbox))
    if any(coord < 0 or coord > 1000 for coord in bbox):
        return None
    x1, y1, x2, y2 = bbox
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return [c / 1000.0 for c in (x1, y1, x2, y2)]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _unletterbox_bbox(bbox: list[float], geometry: dict) -> list[float] | None:
    bucket_h, bucket_w = geometry["bucket"]
    render_h, render_w = geometry["render"]
    pad_x, pad_y = geometry["pad"]

    x1, y1, x2, y2 = bbox
    x1 = _clamp01((x1 * bucket_w - pad_x) / render_w)
    x2 = _clamp01((x2 * bucket_w - pad_x) / render_w)
    y1 = _clamp01((y1 * bucket_h - pad_y) / render_h)
    y2 = _clamp01((y2 * bucket_h - pad_y) / render_h)
    if x1 == x2 or y1 == y2:
        return None
    return [x1, y1, x2, y2]


def _unletterbox_blocks(blocks: list, geometry: dict) -> list:
    for block in blocks:
        raw_bbox = block.get("bbox")
        if raw_bbox is None:
            continue
        block["raw_bbox"] = raw_bbox
        bbox = _unletterbox_bbox(raw_bbox, geometry)
        if bbox is None:
            block["bbox"] = raw_bbox
            logger.warning(
                "[MinerU2.5] Invalid unletterboxed bbox, keep raw bbox: {}", raw_bbox
            )
        else:
            block["bbox"] = bbox
    return blocks


def _parse_angle(tail: str):
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            return angle
    return None


def _save_layout_boxes_image(
    image: Image.Image, blocks: list, output_path: str
) -> None:
    """Draw normalized layout bboxes on the original image and save it."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    width, height = canvas.size
    line_width = max(2, round(max(width, height) / 500))

    colors = {
        "text": "red",
        "title": "blue",
        "table": "green",
        "image": "purple",
        "equation": "orange",
        "equation_block": "orange",
        "list": "brown",
    }

    for idx, block in enumerate(blocks):
        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        box = (
            round(x1 * width),
            round(y1 * height),
            round(x2 * width),
            round(y2 * height),
        )
        block_type = block.get("type", "unknown")
        color = colors.get(block_type, "yellow")
        draw.rectangle(box, outline=color, width=line_width)

        label = f"{idx}:{block_type}"
        label_box = draw.textbbox((box[0], box[1]), label, font=font)
        label_bg = (
            label_box[0] - 2,
            label_box[1] - 2,
            label_box[2] + 2,
            label_box[3] + 2,
        )
        draw.rectangle(label_bg, fill=color)
        draw.text((box[0], box[1]), label, fill="white", font=font)

    canvas.save(output_path)
    logger.info("[MinerU2.5] Layout boxes image saved: {}", output_path)


# ---------------------------------------------------------------------------
#  MinerU TCIM client
# ---------------------------------------------------------------------------


class MinerU:
    def __init__(
        self,
        visual_path,
        prefill_path,
        decode_path,
        tokenizer_config_path,
        embedding_path,
        devices=(0,),
        visual_buckets_manifest=None,
    ):
        dm = tcim.runtime.DevManager(
            devices=list(devices), backend_name="Xh2HalBackend"
        )
        wm = tcim.runtime.WeightManager(dm)

        # -- prefill / decode --------------------------------------------------
        opt1 = tcim.runtime.Option(wm)
        opt2 = tcim.runtime.Option(wm)

        self.prefill = tcim.runtime.load(prefill_path, option=opt1)
        logger.info("[MinerU2.5] Prefill  model loaded: {}", prefill_path)

        self._kv_cache_names = []
        for name in self._prefill_input_names:
            if "cache" not in name:
                continue
            self._kv_cache_names.append(name)
        opt2.set_dummy_tensors(self._kv_cache_names)
        self.decode = tcim.runtime.load(decode_path, option=opt2)
        logger.info("[MinerU2.5] Decode   model loaded: {}", decode_path)

        for name in self._kv_cache_names:
            cache = self.prefill.get_dev_input(name)
            self.decode.set_input(name, cache)
        logger.info(
            "[MinerU2.5] KV-cache bound ({} tensors)", len(self._kv_cache_names)
        )

        # -- visual (multi-bucket) --------------------------------------------
        self.patch_size = 14
        self.merge_size = 2
        self.patch_factor = self.patch_size * self.merge_size  # 28
        self.max_upscale = 2.0
        self.layout_image_size = (1036, 1036)

        self.visual_models: dict[tuple[int, int], object] = {}
        self.buckets: list[tuple[int, int]] = []
        self.fallback_bucket: tuple[int, int] = (1036, 1036)

        if visual_buckets_manifest:
            if not os.path.exists(visual_buckets_manifest):
                raise FileNotFoundError(
                    f"Visual buckets manifest not found: {visual_buckets_manifest}"
                )
            self._load_visual_buckets(visual_buckets_manifest, visual_path, wm)
            logger.info(
                "[MinerU2.5] Visual   multi-bucket mode: {} buckets, fallback {}x{}",
                len(self.visual_models),
                self.fallback_bucket[0],
                self.fallback_bucket[1],
            )
        else:
            # Single visual model (backwards compatible)
            opt0 = tcim.runtime.Option(wm)
            self.visual = tcim.runtime.load(visual_path, option=opt0)
            self.visual_models[self.fallback_bucket] = self.visual
            self.buckets = [self.fallback_bucket]
            logger.info("[MinerU2.5] Visual   single-bucket mode: {}", visual_path)

        # -- token embedding ---------------------------------------------------
        weight = torch.load(embedding_path, map_location="cpu", weights_only=True)
        self.token_embedding = nn.Embedding(
            num_embeddings=weight["weight"].shape[0],
            embedding_dim=weight["weight"].shape[1],
        )
        self.token_embedding.load_state_dict(weight)
        logger.info(
            "[MinerU2.5] Token embedding loaded: {} x {}",
            weight["weight"].shape[0],
            weight["weight"].shape[1],
        )

        # -- processor ---------------------------------------------------------
        self.processor = Qwen2VLProcessor.from_pretrained(
            tokenizer_config_path, use_fast=True
        )
        logger.info("[MinerU2.5] Processor loaded: {}", tokenizer_config_path)

        # -- special token IDs ------------------------------------------------
        self.image_token_id = 151655
        self.video_token_id = 151656
        self.vision_start_token_id = 151652
        self.vision_end_token_id = 151653
        self.vision_token_id = 151654
        self.eos_token_id = (151645, 151643)
        self.rope_deltas = None
        self.context_length = 0
        self.max_image_edge_ratio = 50
        self.min_image_edge = 28

    # -- model helpers ---------------------------------------------------------

    @property
    def _prefill_input_names(self):
        return [
            self.prefill.get_input_name(i) for i in range(self.prefill.get_num_inputs())
        ]

    @property
    def _prefill_seq_len(self):
        name = self.prefill.get_input_name(0)
        return self.prefill.input_infos[name].shape[1]

    @property
    def _max_context_len(self):
        name = self.prefill.get_input_name(6)
        return self.prefill.input_infos[name].shape[2]

    @staticmethod
    def _build_messages(text):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            },
        ]

    # -------------------------------------------------------------------
    #  Multi-bucket visual model loading
    # -------------------------------------------------------------------

    def _load_visual_buckets(
        self, manifest_path: str, fallback_vit_path: str, wm
    ) -> None:
        """Load fallback visual model + additional static buckets from manifest."""
        # 1. Always load the fallback (1036x1036) from visual_path directly
        opt = tcim.runtime.Option(wm)
        self.visual = tcim.runtime.load(fallback_vit_path, option=opt)
        self.visual_models[self.fallback_bucket] = self.visual
        self.buckets.append(self.fallback_bucket)
        logger.info(
            "[MinerU2.5] Visual   bucket {}x{} (fallback): {}",
            self.fallback_bucket[0],
            self.fallback_bucket[1],
            fallback_vit_path,
        )

        # 2. Load additional buckets from manifest (skip fallback if present)
        with open(manifest_path) as f:
            manifest = json.load(f)
        _validate_visual_bucket_manifest(manifest)

        self.patch_size = int(manifest["patch_size"])
        self.merge_size = int(manifest["spatial_merge_size"])
        self.patch_factor = self.patch_size * self.merge_size

        # Derive .hmm path pattern from the fallback visual path.
        vit_dir = os.path.dirname(fallback_vit_path)
        vit_name = os.path.basename(fallback_vit_path)
        name_pattern = re.sub(r"_\d+x\d+", "_{h}x{w}", vit_name)
        if name_pattern == vit_name:
            raise ValueError(
                f"Cannot derive visual bucket filename pattern from: {fallback_vit_path}"
            )

        for item in manifest["buckets"]:
            h, w = _parse_bucket(item)
            bucket = (h, w)
            if bucket == self.fallback_bucket:
                continue  # already loaded
            hmm_path = os.path.join(vit_dir, name_pattern.format(h=h, w=w))
            model = tcim.runtime.load(hmm_path, option=opt)
            self.visual_models[bucket] = model
            self.buckets.append(bucket)
            logger.info(
                "[MinerU2.5] Visual   bucket {}x{} loaded: {}",
                h,
                w,
                hmm_path,
            )

        # Sort by area then dimensions
        self.buckets = sorted(
            set(self.buckets),
            key=lambda item: (item[0] * item[1], item[0], item[1]),
        )

    # -------------------------------------------------------------------
    #  Bucket selection
    # -------------------------------------------------------------------

    def _select_bucket(
        self,
        native_h: int,
        native_w: int,
        candidate_buckets: list[tuple[int, int]] | None = None,
    ) -> tuple[int, int]:
        """Pick the best static visual bucket for a native image size."""
        buckets = candidate_buckets or self.buckets
        scored = [
            (self._bucket_score(native_h, native_w, bucket), bucket)
            for bucket in buckets
        ]
        score, bucket = min(
            scored, key=lambda item: (item[0], item[1][0] * item[1][1], item[1])
        )
        logger.debug(
            "[MinerU2.5] Bucket  selected {}x{} for native {}x{} (score={:.4f}, candidates={})",
            bucket[0],
            bucket[1],
            native_h,
            native_w,
            score,
            len(buckets),
        )
        return bucket

    def _bucket_score(
        self, native_h: int, native_w: int, bucket: tuple[int, int]
    ) -> float:
        """Cost function: lower is better."""
        bucket_h, bucket_w = bucket
        native_ratio = native_w / native_h
        bucket_ratio = bucket_w / bucket_h
        aspect_cost = abs(math.log(native_ratio / bucket_ratio))
        scale = min(bucket_h / native_h, bucket_w / native_w)
        effective_scale = min(scale, self.max_upscale)
        render_h = max(1, min(bucket_h, round(native_h * effective_scale)))
        render_w = max(1, min(bucket_w, round(native_w * effective_scale)))
        padding_cost = 1.0 - (render_h * render_w) / (bucket_h * bucket_w)
        downscale_cost = max(0.0, -math.log(scale))
        return 2.0 * aspect_cost + 0.8 * downscale_cost + 0.2 * padding_cost

    def _letterbox(
        self, image: Image.Image, bucket: tuple[int, int]
    ) -> tuple[Image.Image, tuple[int, int]]:
        """Resize image to fit inside bucket, then pad to bucket size."""
        bucket_h, bucket_w = bucket
        scale = min(bucket_h / image.height, bucket_w / image.width)
        scale = min(scale, self.max_upscale)
        render_w = max(1, min(bucket_w, round(image.width * scale)))
        render_h = max(1, min(bucket_h, round(image.height * scale)))
        if (render_w, render_h) != image.size:
            image = image.resize((render_w, render_h), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (bucket_w, bucket_h), (255, 255, 255))
        canvas.paste(
            image, ((bucket_w - image.width) // 2, (bucket_h - image.height) // 2)
        )
        return canvas, (render_h, render_w)

    # -- preprocessing --------------------------------------------------------

    def preprocess(
        self,
        image: Image.Image,
        text: str,
        candidate_buckets: list[tuple[int, int]] | None = None,
    ):
        """Preprocess image + text: bucket-select, letterbox, tokenize."""
        # 1. Smart resize to native dimensions (aligned to patch_factor)
        image_rgb = image.convert("RGB")
        native_h, native_w = _smart_resize(
            image.height, image.width, factor=self.patch_factor
        )

        # 2. Select best bucket
        bucket = self._select_bucket(native_h, native_w, candidate_buckets)

        # 3. Resize to native + letterbox to bucket
        native_resized = image_rgb.resize(
            (native_w, native_h), Image.Resampling.BICUBIC
        )
        bucketed_image, render_size = self._letterbox(native_resized, bucket)
        render_h, render_w = render_size
        geometry = {
            "bucket": bucket,
            "render": render_size,
            "pad": ((bucket[1] - render_w) / 2.0, (bucket[0] - render_h) / 2.0),
        }
        logger.debug(
            "[MinerU2.5] Preprocess: native={}x{} → bucket={}x{} render={}x{}",
            native_h,
            native_w,
            bucket[0],
            bucket[1],
            render_size[0],
            render_size[1],
        )

        # 4. Qwen2VL processor
        prompt = self.processor.apply_chat_template(
            self._build_messages(text), tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=[bucketed_image], padding=True, return_tensors="pt"
        )
        return (
            inputs.input_ids,
            inputs.attention_mask,
            inputs.pixel_values.contiguous().to(torch.float16),
            inputs.image_grid_thw,
            bucket,
            geometry,
        )

    # -- public API -----------------------------------------------------------

    def two_step_extract(self, image: Image.Image):
        t_start = time.time()

        block_images, blocks, indices = self.layout_detect(image)
        t_layout = time.time()
        logger.info(
            "[MinerU2.5] Layout  detect done in {:.2f}s — {} blocks ({} to extract)",
            t_layout - t_start,
            len(blocks),
            len(block_images),
        )

        for i, block_image in enumerate(block_images):
            t0 = time.time()
            context = self.text_recognition(block_image)
            blocks[indices[i]]["context"] = context
            logger.info(
                "[MinerU2.5] Extract [{}/{}] type={} size={}x{} | {:.2f}s  result: {}",
                i + 1,
                len(block_images),
                blocks[indices[i]]["type"],
                block_image.width,
                block_image.height,
                time.time() - t0,
                context[:200] + "..." if len(context) > 200 else context,
            )

        t_total = time.time()
        logger.info(
            "[MinerU2.5] Extract done in {:.2f}s ({:.2f}s layout + {:.2f}s recognition)",
            t_total - t_start,
            t_layout - t_start,
            t_total - t_layout,
        )
        return blocks

    def layout_detect(self, image: Image.Image):
        input_ids, _, pixel_values, image_grid_thw, bucket, geometry = self.preprocess(
            image, "\nLayout Detection:"
        )
        generated_ids = self._run(pixel_values, input_ids, image_grid_thw, bucket)
        texts = self.processor.batch_decode(
            [generated_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        blocks = _unletterbox_blocks(self._parse_layout(texts[0]), geometry)
        logger.info(
            "[MinerU2.5] Layout  parsed {} blocks:\n{}",
            len(blocks),
            json.dumps(blocks, ensure_ascii=False),
        )
        block_images, _, _, indices = self._prepare_for_extract(image, blocks)
        return block_images, blocks, indices

    def text_recognition(self, block_image: Image.Image):
        recognition_buckets = [
            bucket for bucket in self.buckets if bucket != self.fallback_bucket
        ]
        if not recognition_buckets:
            recognition_buckets = [self.fallback_bucket]
            logger.warning(
                "[MinerU2.5] Text recognition bucket pool is empty after excluding {}x{}; fallback is used.",
                self.fallback_bucket[0],
                self.fallback_bucket[1],
            )
        input_ids, _, pixel_values, image_grid_thw, bucket, _ = self.preprocess(
            block_image, "\nText Recognition:", candidate_buckets=recognition_buckets
        )
        generated_ids = self._run(pixel_values, input_ids, image_grid_thw, bucket)
        texts = self.processor.batch_decode(
            [generated_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return texts[0]

    # -- generation loop ------------------------------------------------------

    def _run(self, pixel_values, input_ids, image_grid_thw, bucket):
        image_embeds = self._run_visual(pixel_values, bucket)
        image_embeds = torch.from_numpy(image_embeds)

        generated_ids = []
        self.context_length = 0
        next_token, past_seq_len = self._run_prefill(
            input_ids, image_embeds, image_grid_thw
        )
        self.context_length = past_seq_len
        generated_ids.append(next_token)

        while self.context_length < self._max_context_len:
            next_token = self._run_decode(next_token)
            if next_token in self.eos_token_id:
                break
            generated_ids.append(next_token)

        return generated_ids

    def _run_visual(self, pixel_values: torch.Tensor, bucket: tuple[int, int]):
        model = self.visual_models.get(bucket)
        if model is None:
            logger.warning(
                "[MinerU2.5] Bucket {}x{} not found, fallback to {}x{}",
                bucket[0],
                bucket[1],
                self.fallback_bucket[0],
                self.fallback_bucket[1],
            )
            model = self.visual_models[self.fallback_bucket]
        name = model.get_input_name(0)
        model.set_input(name, pixel_values.detach().cpu().numpy())
        model.run()
        model.sync()
        return model.get_output(model.get_output_name(0)).numpy()

    def _run_prefill(self, input_ids, image_embeds, image_grid_thw):
        seq_len = input_ids.shape[1]
        chunk = self._prefill_seq_len
        steps = (seq_len + chunk - 1) // chunk
        pad_len = steps * chunk - seq_len
        if pad_len > 0:
            pad = torch.zeros((1, pad_len), dtype=torch.long)
            input_ids = torch.cat([input_ids, pad], dim=-1)

        inputs_embeds = self.token_embedding(input_ids).to(torch.float16)

        n_img_tokens = (input_ids == self.image_token_id).sum().item()
        assert (
            n_img_tokens == image_embeds.shape[0]
        ), f"Image tokens ({n_img_tokens}) / features ({image_embeds.shape[0]}) mismatch."
        img_mask = (
            (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        )
        inputs_embeds = inputs_embeds.masked_scatter(img_mask, image_embeds)

        position_ids, self.rope_deltas = self.get_rope_index(input_ids, image_grid_thw)
        # Split [3, 1, seq_len] into time / height / width [seq_len]
        time_pos = position_ids[0, 0, :].to(torch.int32)
        height_pos = position_ids[1, 0, :].to(torch.int32)
        width_pos = position_ids[2, 0, :].to(torch.int32)

        past_seq_len = 0
        for i in range(steps):
            start = i * chunk
            end = (i + 1) * chunk
            cur_len = min(end, seq_len) - start

            self.prefill.set_input(
                self.prefill.get_input_name(0),
                inputs_embeds[:, start:end, :].detach().cpu().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(1),
                time_pos[start:end].detach().cpu().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(2),
                height_pos[start:end].detach().cpu().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(3),
                width_pos[start:end].detach().cpu().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(4),
                np.array([past_seq_len], dtype=np.int32),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(5),
                np.array([cur_len], dtype=np.int32),
            )
            self.prefill.run()
            self.prefill.sync()
            past_seq_len += cur_len

        logits = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        next_token = int(logits.argmax(axis=-1)[0][0])
        return next_token, past_seq_len

    def _run_decode(self, input_id: int):
        input_ids = torch.tensor([[input_id]], dtype=torch.int32)
        inputs_embeds = self.token_embedding(input_ids).to(torch.float16)
        bs, slen, _ = inputs_embeds.shape

        past = torch.tensor([self.context_length], dtype=torch.int32)
        cur = torch.tensor([slen], dtype=torch.int32)
        delta = past + self.rope_deltas
        pos = torch.arange(slen, dtype=torch.int32).view(1, -1).expand(bs, -1)
        pos = pos.add(delta.repeat_interleave(bs // delta.shape[0], dim=0))
        pos = pos.to(torch.int32).squeeze()  # [1]

        self.decode.set_input(
            self.decode.get_input_name(0), inputs_embeds.detach().cpu().numpy()
        )
        self.decode.set_input(self.decode.get_input_name(1), pos.detach().cpu().numpy())
        self.decode.set_input(self.decode.get_input_name(2), pos.detach().cpu().numpy())
        self.decode.set_input(self.decode.get_input_name(3), pos.detach().cpu().numpy())
        self.decode.set_input(
            self.decode.get_input_name(4), past.detach().cpu().numpy()
        )
        self.decode.set_input(self.decode.get_input_name(5), cur.detach().cpu().numpy())
        self.decode.run()
        self.decode.sync()

        logits = self.decode.get_output(self.decode.get_output_name(0)).numpy()
        self.context_length += 1
        return int(logits.argmax(axis=-1)[0][0])

    # -- position IDs (Qwen2-VL mRoPE) ---------------------------------------

    def get_rope_index(
        self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None
    ):
        spatial_merge_size = self.merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.vision_start_token_id
        mrope_position_deltas = []

        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i].to(input_ids.device) == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    # -- layout parsing -------------------------------------------------------

    def _parse_layout(self, output: str):
        blocks = []
        for line in output.split("\n"):
            match = re.match(_layout_re, line)
            if not match:
                continue
            x1, y1, x2, y2, ref_type, tail = match.groups()
            bbox = _convert_bbox((x1, y1, x2, y2))
            if bbox is None:
                logger.warning("[MinerU2.5] Invalid bbox in layout: {}", line)
                continue
            ref_type = ref_type.lower()
            if ref_type not in BLOCK_TYPES:
                logger.warning(
                    "[MinerU2.5] Unknown block type '{}' in layout", ref_type
                )
                continue
            angle = _parse_angle(tail)
            blocks.append(
                {"type": ref_type, "bbox": bbox, "angle": angle, "context": None}
            )
        return blocks

    def _prepare_for_extract(self, image: Image.Image, blocks: list):
        width, height = image.size
        block_images, prompts, params, indices = [], [], [], []
        for idx, block in enumerate(blocks):
            _type = block["type"]
            bbox = block["bbox"]
            angle = block.get("angle")
            if _type in SKIP_EXTRACT_TYPES:
                continue
            x1, y1, x2, y2 = bbox
            scaled = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = image.crop(scaled)
            if block_image.width < 1 or block_image.height < 1:
                logger.warning(
                    "[MinerU2.5] Skipping empty crop (type={}, bbox={})", _type, bbox
                )
                continue
            if angle in (90, 180, 270):
                block_image = block_image.rotate(angle, expand=True)
            block_image = self._resize_by_need(block_image)
            block_images.append(block_image)
            indices.append(idx)
        return block_images, prompts, params, indices

    def _resize_by_need(self, image: Image.Image):
        edge_ratio = max(image.size) / min(image.size)
        if edge_ratio > self.max_image_edge_ratio:
            w, h = image.size
            if w > h:
                new_w, new_h = w, math.ceil(w / self.max_image_edge_ratio)
            else:
                new_w, new_h = math.ceil(h / self.max_image_edge_ratio), h
            canvas = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
            canvas.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
            image = canvas
        if min(image.size) < self.min_image_edge:
            scale = self.min_image_edge / min(image.size)
            new_w, new_h = round(image.width * scale), round(image.height * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return image


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", type=str, default="config.yaml", help="path to config.yaml")
    parser.add_argument("--model_name", type=str, default=None, help="model name")
    parser.add_argument("--model_size", type=str, default=None, help="model size")
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--embedding_path", type=str, default=f"output/{HOUMO_TARGET}/hmquant/quant_embedding.pt")
    parser.add_argument("--prefill_path", type=str, default=None)
    parser.add_argument("--decode_path", type=str, default=None)
    parser.add_argument("--vit_path", type=str, default=None)
    parser.add_argument("--ndevice", type=int, default=None, help="device number")
    parser.add_argument("--max_size_w", type=int, default=None)
    parser.add_argument("--max_size_h", type=int, default=None)
    parser.add_argument("--image", default="./data/0001.png")
    parser.add_argument("--visual_buckets_manifest", type=str, default=None, help="path to mineru_visual_buckets.json (enables multi-bucket mode)")
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    args.max_size_w = first_not_none(args.max_size_w, model_config.get("max_size_w", 1036))
    args.max_size_h = first_not_none(args.max_size_h, model_config.get("max_size_h", 1036))
    if args.tokenizer_dir is None:
        args.tokenizer_dir = f"output/{HOUMO_TARGET}/hmquant/hf_config"
    if args.prefill_path is None:
        args.prefill_path = f"output/{HOUMO_TARGET}/{args.model_name}-{args.model_size}_prefill.hmm"
    if args.decode_path is None:
        args.decode_path = f"output/{HOUMO_TARGET}/{args.model_name}-{args.model_size}_decode.hmm"
    if args.vit_path is None:
        args.vit_path = f"output/{HOUMO_TARGET}/{args.model_name}-{args.model_size}_visual_{args.max_size_w}x{args.max_size_h}.hmm"
    if args.visual_buckets_manifest is None:
        args.visual_buckets_manifest = f"output/{HOUMO_TARGET}/hmquant/mineru_visual_buckets.json"

    if args.ndevice > 1:
        if args.prefill_path.endswith(".hmm"):
            args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        if args.decode_path.endswith(".hmm"):
            args.decode_path = args.decode_path.replace(".hmm", ".hmms")
    # fmt: on
    return args


if __name__ == "__main__":
    args = get_args()

    logger.info("========================================")
    logger.info("[MinerU2.5] Demo — TCIM inference")
    logger.info("[MinerU2.5] HOUMO_TARGET = {}", HOUMO_TARGET)
    logger.info("[MinerU2.5] Image: {}", args.image)
    if args.visual_buckets_manifest:
        logger.info(
            "[MinerU2.5] Visual buckets manifest: {}", args.visual_buckets_manifest
        )
    logger.info("========================================")

    mineru = MinerU(
        visual_path=args.vit_path,
        prefill_path=args.prefill_path,
        decode_path=args.decode_path,
        tokenizer_config_path=args.tokenizer_dir,
        embedding_path=args.embedding_path,
        devices=(0,),
        visual_buckets_manifest=args.visual_buckets_manifest,
    )

    image = Image.open(args.image)
    blocks = mineru.two_step_extract(image)
    image_root, image_ext = os.path.splitext(args.image)
    boxes_path = f"{image_root}_layout_boxes.png"
    _save_layout_boxes_image(image, blocks, boxes_path)
    print(json.dumps(blocks, ensure_ascii=False))
    logger.info("Layout boxes saved to {}", boxes_path)
