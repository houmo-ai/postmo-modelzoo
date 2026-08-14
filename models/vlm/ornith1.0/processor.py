# Copyright (c) 2026 HOUMO AI
#
# File: processor.py
# Description:
#   Provide local tokenizer and fixed-size image processing for Ornith.
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

from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from PIL import Image

IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_MERGE_SIZE = 2
DEFAULT_TEMPORAL_PATCH_SIZE = 2
_PLACEHOLDER = "<|ornith_image_placeholder|>"


def _load_image(value):
    """Load one supported local image and return an owned RGB image."""
    if isinstance(value, Image.Image):
        return value.copy().convert("RGB")
    if not isinstance(value, (str, Path)):
        raise ValueError(f"Unsupported image input: {type(value).__name__}")

    source = str(value)
    parsed = urlparse(source)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif parsed.scheme:
        raise ValueError(f"Unsupported image source: {source}")
    else:
        path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _resize_image_element(element):
    for field in ("resized_height", "resized_width"):
        if field not in element:
            raise ValueError(f"image message requires {field}")
    height = int(element["resized_height"])
    width = int(element["resized_width"])
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {height}x{width}")
    image = _load_image(element.get("image"))
    return image.resize((width, height))


def process_visual_info(messages):
    """Return fixed-size local images and an empty video list."""
    images = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for element in content:
            if not isinstance(element, dict):
                continue
            if element.get("type") == "video":
                raise ValueError("Ornith processor does not support video inputs")
            if element.get("type") == "image":
                images.append(_resize_image_element(element))
    return images, []


def _token_id(tokenizer, token):
    token_id = getattr(tokenizer, "image_token_id", None)
    if token_id is None:
        token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0:
        raise ValueError(f"Tokenizer does not define image token {token!r}")
    return int(token_id)


class OrnithProcessor:
    """Build the text and image inputs consumed by the Ornith demo."""

    def __init__(
        self,
        tokenizer,
        image_processor,
        patch_size,
        merge_size=DEFAULT_MERGE_SIZE,
        temporal_patch_size=DEFAULT_TEMPORAL_PATCH_SIZE,
        chat_template=None,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.image_token = getattr(tokenizer, "image_token", IMAGE_TOKEN)
        self.image_token_id = _token_id(tokenizer, self.image_token)
        self.chat_template = chat_template or getattr(tokenizer, "chat_template", None)

    def apply_chat_template(self, messages, **kwargs):
        if self.chat_template is None:
            raise ValueError("Tokenizer has no chat template")
        if getattr(self.tokenizer, "chat_template", None) is None:
            self.tokenizer.chat_template = self.chat_template
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _validate_images(self, images):
        factor = self.patch_size * self.merge_size
        for image in images:
            width, height = image.size
            if height % factor or width % factor:
                raise ValueError(
                    f"image size {height}x{width} must be divisible by {factor}"
                )

    def _prepare_images(self, images):
        self._validate_images(images)
        image_inputs = self.image_processor(
            images=images,
            do_resize=False,
            return_tensors="pt",
        )
        data = dict(image_inputs)
        if "image_grid_thw" not in data:
            raise ValueError("image processor did not return image_grid_thw")
        image_grid_thw = torch.as_tensor(data["image_grid_thw"], dtype=torch.long)
        if image_grid_thw.shape[0] != len(images):
            raise ValueError(
                f"image grid count {image_grid_thw.shape[0]} does not match "
                f"image count {len(images)}"
            )
        data["image_grid_thw"] = image_grid_thw
        data["hm_pixel_values"] = [self._hm_pixels(image) for image in images]
        return data

    def _hm_pixels(self, image):
        pixels = np.asarray(image, dtype=np.uint8)
        pixels = np.ascontiguousarray(pixels.transpose(2, 0, 1))
        tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(2)
        return tensor.repeat(1, 1, self.temporal_patch_size, 1, 1)

    def __call__(
        self,
        text,
        images=None,
        padding=True,
        return_tensors="pt",
    ):
        if return_tensors != "pt":
            raise ValueError("Ornith processor only supports return_tensors='pt'")
        if not isinstance(text, list) or len(text) != 1:
            raise ValueError("Ornith processor supports one text input")

        images = list(images or [])
        placeholder_count = text[0].count(self.image_token)
        if placeholder_count != len(images):
            raise ValueError(
                f"image placeholder count {placeholder_count} does not match "
                f"image count {len(images)}"
            )

        image_inputs = self._prepare_images(images) if images else {}
        expanded = text[0]
        image_grid_thw = image_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            merge_length = self.merge_size**2
            for grid in image_grid_thw:
                grid_size = int(grid.prod().item())
                if grid_size % merge_length:
                    raise ValueError(
                        f"image grid size {grid_size} is not divisible by {merge_length}"
                    )
                count = grid_size // merge_length
                expanded = expanded.replace(
                    self.image_token,
                    _PLACEHOLDER * count,
                    1,
                )
            expanded = expanded.replace(_PLACEHOLDER, self.image_token)

        text_inputs = self.tokenizer(
            [expanded],
            padding=padding,
            return_tensors=return_tensors,
        )
        return {**dict(text_inputs), **image_inputs}


def create_processor(
    tokenizer_dir,
    max_h,
    max_w,
    patch_size,
    merge_size=DEFAULT_MERGE_SIZE,
    temporal_patch_size=DEFAULT_TEMPORAL_PATCH_SIZE,
):
    """Load the tokenizer and create the local Ornith processor."""
    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    chat_template = getattr(tokenizer, "chat_template", None)
    template_path = Path(tokenizer_dir) / "chat_template.jinja"
    if chat_template is None:
        if not template_path.exists():
            raise ValueError(f"Tokenizer has no chat template: {template_path}")
        chat_template = template_path.read_text(encoding="utf-8")

    image_processor = Qwen2VLImageProcessor(
        do_resize=False,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        min_pixels=max_h * max_w,
        max_pixels=max_h * max_w,
    )
    return OrnithProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        chat_template=chat_template,
    )
