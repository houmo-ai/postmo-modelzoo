#!/usr/bin/env python3
# Copyright 2025 HOUMO AI
#
# File: demo.py
# Description:
#    Run SigLIP2 zero-shot classification for one image and print top-5 labels.
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
"""Run SigLIP2 zero-shot classification for one image and print top-5 labels."""

import argparse
import os
import numpy as np
import yaml
from pathlib import Path

from model import (
    DEFAULT_TEXT_HMM,
    DEFAULT_TOKENIZER_DIR,
    DEFAULT_VISION_HMM,
    compute_text_embeds,
    default_imagenet_dir,
    image_embedding,
    load_labels,
    load_tokenizer,
    make_runner,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
HOUMO_EXAMPLES_PATH = os.getenv("HOUMO_EXAMPLES_PATH", ".")
assert HOUMO_TARGET == "xh2", "Only support HOUMO_TARGET: xh2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default="config.yaml",
        help="path to config.yaml",
    )
    parser.add_argument("--model_name", type=str, default=None, help="model name")
    parser.add_argument("--model_size", type=str, default=None, help="model size")
    parser.add_argument(
        "--image",
        type=str,
        default=f"{default_imagenet_dir()}/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG",
        help="image path to classify",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=f"{default_imagenet_dir()}/synset_1000.txt",
        help="label file; defaults to <imagenet_dir>/synset_1000.txt",
    )
    parser.add_argument(
        "--tokenizer_dir",
        default=DEFAULT_TOKENIZER_DIR,
        help="local tokenizer directory saved by ptq.py",
    )
    parser.add_argument("--vision_hmm", default=DEFAULT_VISION_HMM)
    parser.add_argument("--text_hmm", default=DEFAULT_TEXT_HMM)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--ndevice", type=int, default=1)
    return parser.parse_args()


def get_model_configs(config_path: str, config_key: str = "model_configs"):
    config = {}
    if not config_path or config_path is None or not os.path.exists(config_path):
        return "", {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    default_model_name = config.get("default_model_name", "")
    default_model_size = config.get("default_model_size", "")
    model_configs = config.get(config_key, {}) or {}
    return default_model_size, default_model_name, model_configs


def first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def main():
    args = parse_args()
    default_model_size, default_model_name, _ = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    if args.ndevice > 1:
        if args.vision_hmm.endswith(".hmm"):
            args.vision_hmm = args.vision_hmm.replace(".hmm", ".hmms")
        if args.text_hmm.endswith(".hmm"):
            args.text_hmm = args.text_hmm.replace(".hmm", ".hmms")

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    label_file = Path(args.labels)
    if not label_file.is_file():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    labels = load_labels(label_file)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    vision_runner = make_runner("hmm", "", args.vision_hmm)
    text_runner = make_runner("hmm", "", args.text_hmm)

    text_embeds = compute_text_embeds(tokenizer, text_runner, labels, args.seq_len)
    image_embed = image_embedding(image_path, vision_runner, "hmm")
    scores = (image_embed @ text_embeds.T)[0]

    for rank, idx in enumerate(np.argsort(-scores)[:5], start=1):
        print(f"top{rank}: {labels[idx]}\t{scores[idx]:.6f}")


if __name__ == "__main__":
    main()
