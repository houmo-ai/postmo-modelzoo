#!/usr/bin/env python3
# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Measure H2D, inference, and D2H latency of SigLIP2 HMMs."""

import argparse
import time
from pathlib import Path

import numpy as np

from model import (
    DEFAULT_TEXT_HMM,
    DEFAULT_TOKENIZER_DIR,
    DEFAULT_VISION_HMM,
    default_imagenet_dir,
    load_tokenizer,
    preprocess_hmm_image,
    tokenize_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=f"{default_imagenet_dir()}/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG",
    )
    parser.add_argument("--tokenizer_dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--vision_hmm", default=DEFAULT_VISION_HMM)
    parser.add_argument("--text_hmm", default=DEFAULT_TEXT_HMM)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    return parser.parse_args()


def benchmark(model_path, feed, warmup, repeat):
    import tcim_lite

    module = tcim_lite.runtime.load(str(model_path))
    input_names = [module.get_input_name(i) for i in range(module.get_num_inputs())]
    output_name = module.get_output_name(0)
    text_inputs_swapped = input_names == ["attention_mask", "input_ids"]

    def run_once():
        start = time.perf_counter()
        for name in input_names:
            feed_name = name
            if text_inputs_swapped:
                feed_name = "input_ids" if name == "attention_mask" else "attention_mask"
            info = module.get_input_info(name)
            module.set_input(name, feed[feed_name].astype(info.dtype))
        h2d = time.perf_counter() - start

        start = time.perf_counter()
        module.run()
        module.sync()
        infer = time.perf_counter() - start

        start = time.perf_counter()
        module.get_output(output_name).numpy()
        d2h = time.perf_counter() - start
        return np.array([h2d, infer, d2h]) * 1000

    for _ in range(warmup):
        run_once()
    return np.mean([run_once() for _ in range(repeat)], axis=0)


def main():
    args = parse_args()
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError("warmup must be >= 0 and repeat must be >= 1")

    image_path = Path(args.image)
    vision_hmm = Path(args.vision_hmm)
    text_hmm = Path(args.text_hmm)
    for path in (image_path, vision_hmm, text_hmm):
        if not path.is_file():
            raise FileNotFoundError(path)

    import tcim_lite

    vision_module = tcim_lite.runtime.load(str(vision_hmm))
    vision_input_names = [
        vision_module.get_input_name(i) for i in range(vision_module.get_num_inputs())
    ]
    vision_feed = preprocess_hmm_image(image_path, vision_input_names)
    del vision_module

    tokenizer = load_tokenizer(args.tokenizer_dir)
    text_feed = tokenize_prompt(tokenizer, "a photo of a cat", 64)

    results = [
        ("vision", benchmark(vision_hmm, vision_feed, args.warmup, args.repeat)),
        ("text", benchmark(text_hmm, text_feed, args.warmup, args.repeat)),
    ]

    print(f"warmup={args.warmup}, repeat={args.repeat}")
    print(f"{'Model':<8} {'H2D(ms)':>10} {'Infer(ms)':>12} {'D2H(ms)':>10} {'Total(ms)':>11}")
    for name, values in results:
        print(
            f"{name:<8} {values[0]:>10.3f} {values[1]:>12.3f} "
            f"{values[2]:>10.3f} {values.sum():>11.3f}"
        )


if __name__ == "__main__":
    main()
