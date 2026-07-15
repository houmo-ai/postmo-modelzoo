#!/usr/bin/env python3
# Copyright 2025 HOUMO AI
#
# File: model.py
# Description:
#   Shared SigLIP2 zero-shot inference utilities.
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
"""Shared SigLIP2 zero-shot inference utilities."""

import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_VISION_ONNX = "output/xh2/hmquant/onnx/siglip2_large_patch16_256_vision.onnx"
DEFAULT_TEXT_ONNX = "output/xh2/hmquant/onnx/siglip2_large_patch16_256_text.onnx"
DEFAULT_VISION_HMM = "output/xh2/siglip2-large-patch16-256_vision.hmm"
DEFAULT_TEXT_HMM = "output/xh2/siglip2-large-patch16-256_text.hmm"
DEFAULT_TOKENIZER_DIR = "output/xh2/hmquant/hf_config"
DEFAULT_RESIZER_INPUT_SIZE = [1080, 1920]
VISION_INPUT_NAME = "pixel_values"
VISION_EMBED_SIZE = 256
PROMPT_TEMPLATE = "a photo of a {}"


def default_imagenet_dir():
    examples_path = os.getenv("HOUMO_EXAMPLES_PATH")
    if examples_path:
        return str(Path(examples_path) / "data" / "datasets" / "imagenet")
    return "imagenet"


def load_labels(label_file):
    labels = []
    with open(label_file, "r") as f:
        for line in f:
            text = line.strip()
            labels.append(text[10:] if len(text) > 10 else text)
    if not labels:
        raise RuntimeError(f"No labels found in {label_file}")
    return labels


def normalize(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


class ONNXRunner:
    def __init__(self, model_path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_names = [x.name for x in self.session.get_inputs()]

    def __call__(self, feed):
        return self.session.run(None, {name: feed[name] for name in self.input_names})[
            0
        ]


class HMMRunner:
    def __init__(self, model_path):
        import tcim_lite

        self.module = tcim_lite.runtime.load(str(model_path))
        self.input_names = [
            self.module.get_input_name(i) for i in range(self.module.get_num_inputs())
        ]
        self.output_names = [
            self.module.get_output_name(i) for i in range(self.module.get_num_outputs())
        ]

    def __call__(self, feed):
        text_inputs_swapped = self.input_names == ["attention_mask", "input_ids"]
        for name in self.input_names:
            info = self.module.get_input_info(name)
            feed_name = name
            if text_inputs_swapped:
                feed_name = (
                    "input_ids" if name == "attention_mask" else "attention_mask"
                )
            self.module.set_input(name, feed[feed_name].astype(info.dtype))
        self.module.run()
        self.module.sync()
        return self.module.get_output(self.output_names[0]).numpy()


def make_runner(backend, onnx_path, hmm_path):
    if backend == "onnx":
        return ONNXRunner(onnx_path)
    return HMMRunner(hmm_path)


def preprocess_onnx_image(image_path, image_size=256):
    cv_image = cv2.imread(str(image_path))
    if cv_image is None:
        raise RuntimeError(f"Failed to decode image: {image_path}")
    im = cv2.resize(cv_image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32)
    im -= np.array([127.5, 127.5, 127.5], dtype=np.float32)
    im /= np.array([127.5, 127.5, 127.5], dtype=np.float32)
    pixel_values = np.ascontiguousarray(im[None].transpose(0, 3, 1, 2))
    return {"pixel_values": pixel_values.astype(np.float32)}


def pad_image_to_nchw(image, output_size):
    output_h, output_w = output_size
    image_h, image_w = image.shape[:2]
    scale = min(output_h / image_h, output_w / image_w, 1.0)
    if scale < 1.0:
        image_h = max(2, int(image_h * scale) & ~1)
        image_w = max(2, int(image_w * scale) & ~1)
        image = cv2.resize(image, (image_w, image_h), interpolation=cv2.INTER_LINEAR)

    crop_h = min(image_h, output_h) & ~1
    crop_w = min(image_w, output_w) & ~1
    padded = np.zeros((1, 3, output_h, output_w), dtype=np.uint8)
    padded[:, :, :crop_h, :crop_w] = image[:crop_h, :crop_w].transpose(2, 0, 1)
    return padded, crop_h, crop_w


def dynamic_crop_info(src_h, src_w, dst_h, dst_w):
    return np.array([[0, 0, src_h, src_w, dst_h, dst_w, 0, 0, 0, 0]], dtype=np.int32)


def bgr_to_yuv420sp_nchw(bgr):
    _, _, height, width = bgr.shape
    bgr_hwc = bgr[0].transpose(1, 2, 0)
    yuv_i420 = cv2.cvtColor(bgr_hwc, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y_size = height * width
    uv_size = y_size // 4
    y = yuv_i420[:y_size]
    u = yuv_i420[y_size : y_size + uv_size]
    v = yuv_i420[y_size + uv_size :]
    yuv420sp = np.empty(y_size + uv_size * 2, dtype=np.uint8)
    yuv420sp[:y_size] = y
    yuv420sp[y_size::2] = u
    yuv420sp[y_size + 1 :: 2] = v

    output = np.zeros_like(bgr.reshape(1, -1))
    output[:, : yuv420sp.size] = yuv420sp
    return output.reshape(bgr.shape)


def preprocess_hmm_image(image_path, input_names):
    cv_image = cv2.imread(str(image_path))
    if cv_image is None:
        raise RuntimeError(f"Failed to decode image: {image_path}")

    padded, crop_h, crop_w = pad_image_to_nchw(cv_image, DEFAULT_RESIZER_INPUT_SIZE)
    feed = {VISION_INPUT_NAME: bgr_to_yuv420sp_nchw(padded)}
    dyn_name = f"resizer_crop_{VISION_INPUT_NAME}"
    if dyn_name in input_names:
        feed[dyn_name] = dynamic_crop_info(
            crop_h, crop_w, VISION_EMBED_SIZE, VISION_EMBED_SIZE
        )
    return feed


def image_embedding(image_path, runner, backend):
    if backend == "onnx":
        feed = preprocess_onnx_image(image_path)
    else:
        feed = preprocess_hmm_image(image_path, runner.input_names)
    return normalize(runner(feed).astype(np.float32))


def load_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer

    tokenizer_dir = Path(tokenizer_dir)
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    return AutoTokenizer.from_pretrained(
        str(tokenizer_dir), trust_remote_code=True, local_files_only=True
    )


def tokenize_prompt(tokenizer, text, seq_len):
    tokens = tokenizer(
        text=[text],
        padding="max_length",
        max_length=seq_len,
        truncation=True,
        return_tensors="np",
    )
    input_ids = tokens["input_ids"].astype(np.int64)
    return {
        "input_ids": input_ids,
        "attention_mask": np.ones_like(input_ids, dtype=np.int64),
    }


def compute_text_embeds(tokenizer, text_runner, class_names, seq_len):
    embeds = []
    for name in tqdm(class_names, desc="text prompts"):
        prompt = PROMPT_TEMPLATE.format(name.split(",")[0].strip())
        feed = tokenize_prompt(tokenizer, prompt, seq_len)
        embeds.append(text_runner(feed)[0])
    return normalize(np.stack(embeds, axis=0).astype(np.float32))
