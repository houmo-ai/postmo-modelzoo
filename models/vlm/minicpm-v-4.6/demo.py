#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   MiniCPM-V 4.6 inference Demo using Houmo Python Engine.
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
import sys
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent
IMODELZOO_ROOT = MODEL_DIR.parents[2]
ENGINE_SRC = IMODELZOO_ROOT / "utils" / "python"
sys.path.insert(0, str(ENGINE_SRC))

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
OUTPUT_DIR = MODEL_DIR / "output" / HOUMO_TARGET
DEFAULT_CONFIG_PATH = MODEL_DIR / "config.yaml"
DEFAULT_TOKENIZER_DIR = MODEL_DIR / "MiniCPM-V-4.6"


class HmMiniCPMV46:
    """User-facing MiniCPM-V 4.6 wrapper around one engine."""

    def __init__(
        self,
        *,
        prefill_path,
        decode_path,
        vision_path,
        embedding_path,
        tokenizer_path,
        downsample_mode: str = "16x",
        max_slice_nums: int = 36,
        ndevice: int = 1,
        batch: int = 1,
        do_sample: bool = True,
        temperature: float = 0.7,
        seed: int | None = None,
        perf: bool = False,
    ):
        from minicpm_v_4_6_engine import MiniCPMV46Engine

        self.engine = MiniCPMV46Engine(
            prefill_path=prefill_path,
            decode_path=decode_path,
            vision_path=vision_path,
            embedding_path=embedding_path,
            tokenizer_path=tokenizer_path,
            downsample_mode=downsample_mode,
            max_slice_nums=max_slice_nums,
            ndevice=ndevice,
            batch=batch,
            do_sample=do_sample,
            temperature=temperature,
            seed=seed,
            perf=perf,
        )

    def generate(
        self,
        prompt: str,
        *,
        images=None,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
    ):
        yield from self.engine.generate(
            prompt,
            images=images,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
        )

    def print_perf(self) -> None:
        self.engine.perf.print_summary()


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MiniCPM-V 4.6 image inference on Houmo XH2"
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="path to config.yaml",
    )
    parser.add_argument(
        "--question",
        dest="question",
        type=str,
        default=None,
        help="question or instruction sent to the model",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        help="model name; defaults to default_model_name in config.yaml",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size; defaults to default_model_size in config.yaml",
    )
    parser.add_argument(
        "--image-path",
        dest="image_path",
        type=str,
        nargs="+",
        default=None,
        help="optional image paths; omit for text-only inference",
    )
    parser.add_argument(
        "--system_prompt",
        "--system-prompt",
        dest="system_prompt",
        type=str,
        default=None,
        help="optional system prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=512,
        help="maximum number of generated tokens",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=0.7,
        help="sampling temperature used when greedy decoding is disabled",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help="random seed used for sampling",
    )
    parser.add_argument(
        "--greedy",
        dest="greedy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use greedy decoding; pass --no-greedy to enable sampling",
    )
    parser.add_argument(
        "--max-slice-nums",
        dest="max_slice_nums",
        type=int,
        default=36,
        help="maximum number of image slices generated by the processor",
    )
    parser.add_argument(
        "--downsample-mode",
        dest="downsample_mode",
        type=str,
        choices=["4x", "16x"],
        default="16x",
        help="vision downsample mode",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="number of Houmo devices; defaults to the model configuration",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=None,
        help="inference batch size; defaults to the model configuration",
    )
    parser.add_argument(
        "--perf",
        dest="perf",
        type=_parse_bool,
        default=True,
        nargs="?",
        const=True,
        help="enable performance reporting",
    )
    parser.add_argument(
        "--tokenizer-dir",
        dest="tokenizer_dir",
        type=str,
        default=None,
        help="path to the processor and tokenizer; defaults from config.yaml",
    )
    parser.add_argument(
        "--prefill-path",
        dest="prefill_path",
        type=str,
        default=None,
        help="path to the prefill HMM model",
    )
    parser.add_argument(
        "--decode-path",
        dest="decode_path",
        type=str,
        default=None,
        help="path to the decode HMM model",
    )
    parser.add_argument(
        "--vit_path",
        "--vit-path",
        dest="vision_path",
        type=str,
        default=None,
        help="path to the visual HMM model",
    )
    parser.add_argument(
        "--embedding-path",
        dest="embedding_path",
        type=str,
        default=str(OUTPUT_DIR / "hmquant" / "quant_embedding.pt"),
        help="path to the embedding weights",
    )
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    import yaml

    with Path(args.config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    args.model_name = args.model_name or config["default_model_name"]
    args.model_size = args.model_size or config["default_model_size"]
    try:
        model_config = config["model_configs"][args.model_name][args.model_size]
    except KeyError as error:
        raise ValueError(
            f"unsupported model configuration: {args.model_name}-{args.model_size}"
        ) from error

    args.ndevice = args.ndevice or int(model_config.get("ndevice", 1))
    args.batch = args.batch or int(model_config.get("batch", 1))
    model_prefix = f"{args.model_name}-{args.model_size}"
    model_suffix = "hmms" if args.ndevice > 1 else "hmm"
    if args.prefill_path is None:
        args.prefill_path = str(
            OUTPUT_DIR / f"{model_prefix}_prefill.{model_suffix}"
        )
    if args.decode_path is None:
        args.decode_path = str(
            OUTPUT_DIR / f"{model_prefix}_decode.{model_suffix}"
        )
    if args.question is None:
        args.question = (
            "介绍下这个图片?"
            if args.image_path
            else "介绍下存算一体的技术优势。"
        )
    if args.vision_path is None:
        args.vision_path = str(
            OUTPUT_DIR / f"{model_prefix}_visual_{args.downsample_mode}.hmm"
        )
    if args.tokenizer_dir is None:
        repo_ids = model_config.get("modelscope_repo", [])
        tokenizer_name = (
            repo_ids[0].rsplit("/", maxsplit=1)[-1]
            if repo_ids
            else DEFAULT_TOKENIZER_DIR.name
        )
        args.tokenizer_dir = str(MODEL_DIR / tokenizer_name)
    return args


def main() -> None:
    args = _resolve_args(get_args().parse_args())
    model = HmMiniCPMV46(
        prefill_path=args.prefill_path,
        decode_path=args.decode_path,
        vision_path=args.vision_path,
        embedding_path=args.embedding_path,
        tokenizer_path=args.tokenizer_dir,
        downsample_mode=args.downsample_mode,
        max_slice_nums=args.max_slice_nums,
        ndevice=args.ndevice,
        batch=args.batch,
        do_sample=not args.greedy,
        temperature=args.temperature,
        seed=args.seed,
        perf=args.perf,
    )
    print(f"\033[1;95m\nQ: {args.question}\nA: ", end="", flush=True)
    for chunk in model.generate(
        args.question,
        images=args.image_path,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt,
    ):
        print(f"\033[1;95m{chunk}", end="", flush=True)
    print()
    if args.perf:
        model.print_perf()


if __name__ == "__main__":
    main()
