#!/usr/bin/env python3
# Copyright 2025 HOUMO AI
#
# File: eval_imagenet.py
# Description:
#   Zero-shot ImageNet evaluation for split SigLIP2 encoders.
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
"""Zero-shot ImageNet evaluation for split SigLIP2 encoders."""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from model import (
    DEFAULT_TEXT_HMM,
    DEFAULT_TEXT_ONNX,
    DEFAULT_TOKENIZER_DIR,
    DEFAULT_VISION_HMM,
    DEFAULT_VISION_ONNX,
    compute_text_embeds,
    default_imagenet_dir,
    image_embedding,
    load_labels,
    load_tokenizer,
    make_runner,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagenet_dir",
        default=default_imagenet_dir(),
        help=(
            "ImageNet root containing val.txt and ILSVRC2012_img_val. "
            "Defaults to $HOUMO_EXAMPLES_PATH/data/datasets/imagenet when set."
        ),
    )
    parser.add_argument(
        "--num",
        type=int,
        default=0,
        help="number of ImageNet validation samples to evaluate; 0 means all available samples",
    )
    parser.add_argument(
        "--tokenizer_dir",
        default=DEFAULT_TOKENIZER_DIR,
        help="local tokenizer directory saved by ptq.py",
    )
    parser.add_argument("--vision_backend", choices=["onnx", "hmm"], default="hmm")
    parser.add_argument("--text_backend", choices=["onnx", "hmm"], default="hmm")
    parser.add_argument("--vision_model", default=DEFAULT_VISION_ONNX)
    parser.add_argument("--text_model", default=DEFAULT_TEXT_ONNX)
    parser.add_argument("--vision_hmm", default=DEFAULT_VISION_HMM)
    parser.add_argument("--text_hmm", default=DEFAULT_TEXT_HMM)
    parser.add_argument("--seq_len", type=int, default=64)
    return parser.parse_args()


def load_imagenet(root):
    root = Path(root)
    image_dir = root / "ILSVRC2012_img_val"
    val_file = root / "val.txt"
    label_file = root / "synset_1000.txt"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"ImageNet image dir not found: {image_dir}")
    if not val_file.is_file():
        raise FileNotFoundError(f"ImageNet val file not found: {val_file}")
    if not label_file.is_file():
        raise FileNotFoundError(f"ImageNet label file not found: {label_file}")

    image_paths, labels = [], []
    with val_file.open("r") as f:
        for line in f:
            filename, label = line.strip().split()
            path = image_dir / filename
            if path.is_file():
                image_paths.append(path)
                labels.append(int(label))
    return load_labels(label_file), image_paths, labels


def evaluate(args):
    class_names, image_paths, labels = load_imagenet(args.imagenet_dir)
    if args.num > 0:
        image_paths = image_paths[: args.num]
        labels = labels[: args.num]

    tokenizer = load_tokenizer(args.tokenizer_dir)
    vision_runner = make_runner(args.vision_backend, args.vision_model, args.vision_hmm)
    text_runner = make_runner(args.text_backend, args.text_model, args.text_hmm)
    text_embeds = compute_text_embeds(tokenizer, text_runner, class_names, args.seq_len)

    top1 = 0
    top5 = 0
    for image_path, gt in tqdm(list(zip(image_paths, labels)), desc="images"):
        image_embed = image_embedding(image_path, vision_runner, args.vision_backend)
        logits = image_embed @ text_embeds.T
        pred = np.argsort(-logits[0])[:5]
        top1 += int(pred[0] == gt)
        top5 += int(gt in pred)

    total = len(image_paths)
    result = {
        "dataset": "ILSVRC_2012Val",
        "num": total,
        "vision_backend": args.vision_backend,
        "text_backend": args.text_backend,
        "top1_acc": f"{top1 / total:.6f}",
        "top5_acc": f"{top5 / total:.6f}",
    }
    print(result)
    return result


if __name__ == "__main__":
    evaluate(parse_args())
