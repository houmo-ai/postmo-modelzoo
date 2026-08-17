#!/usr/bin/env python3
# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   Command-line entry point for MiniCPM-V 4.5 Houmo Python Engine.
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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMODELZOO_ROOT = Path(os.getenv("HOUMO_EXAMPLES_PATH") or "../../../")
sys.path.insert(0, str(IMODELZOO_ROOT / "utils" / "python"))
sys.path.insert(0, str(ROOT))
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniCPM-V 4.5 image and video inference on Houmo XH2")
    parser.add_argument(
        "--config", dest="config_path", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.yaml"
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
    parser.add_argument("--question", dest="question", type=str, default=None, help="user question")
    parser.add_argument(
        "--image-path", dest="image_path", type=Path, nargs="+", default=None, help="optional image paths"
    )
    parser.add_argument(
        "--video-path", dest="video_path", type=Path, nargs="+", default=None, help="optional video paths"
    )
    parser.add_argument(
        "--video-fps", dest="video_fps", type=float, default=1.0, help="frames sampled per second from videos"
    )
    parser.add_argument("--system-prompt", dest="system_prompt", type=str, default=None, help="optional system prompt")
    parser.add_argument(
        "--max-new-tokens", dest="max_new_tokens", type=int, default=512, help="maximum generated tokens"
    )
    parser.add_argument("--temperature", dest="temperature", type=float, default=0.7, help="sampling temperature")
    parser.add_argument("--seed", dest="seed", type=int, default=42, help="sampling seed")
    parser.add_argument(
        "--greedy", dest="greedy", action=argparse.BooleanOptionalAction, default=True, help="use greedy decoding"
    )
    parser.add_argument("--max-slice-nums", dest="max_slice_nums", type=int, default=9, help="maximum image slices")
    parser.add_argument(
        "--ndevice", dest="ndevice", type=int, default=None, help="number of NPU devices; defaults from config.yaml"
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=None,
        help="batch size; defaults from config.yaml; only 1 is supported",
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
        type=Path,
        default=None,
        help="tokenizer directory; defaults from the modelscope repository in config.yaml",
    )
    parser.add_argument(
        "--prefill-path",
        dest="prefill_path",
        type=Path,
        default=None,
        help="prefill HMM",
    )
    parser.add_argument(
        "--decode-path",
        dest="decode_path",
        type=Path,
        default=None,
        help="decode HMM",
    )
    parser.add_argument(
        "--vit-path",
        dest="vision_path",
        type=Path,
        default=None,
        help="vision HMM",
    )
    parser.add_argument(
        "--embedding-path",
        dest="embedding_path",
        type=Path,
        default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"),
        help="embedding weights",
    )
    return parser


def _default_question(args: argparse.Namespace) -> str:
    if args.video_path:
        return "请描述这个视频的内容。"
    if args.image_path:
        return "介绍下这个图片?"
    return "介绍下存算一体的技术优势。"


def _resolve_model_paths(args: argparse.Namespace, model_prefix: str) -> None:
    model_suffix = "hmms" if args.ndevice > 1 else "hmm"
    if args.prefill_path is None:
        args.prefill_path = os.path.join("output", HOUMO_TARGET, f"{model_prefix}_prefill.{model_suffix}")
    if args.decode_path is None:
        args.decode_path = os.path.join("output", HOUMO_TARGET, f"{model_prefix}_decode.{model_suffix}")
    if args.vision_path is None:
        vision_profile = "6x" if args.video_path and not args.image_path else "1x"
        args.vision_path = os.path.join("output", HOUMO_TARGET, f"{model_prefix}_visual_{vision_profile}.hmm")
    args.video_vision_path = os.path.join("output", HOUMO_TARGET, f"{model_prefix}_visual_6x.hmm")


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    import yaml

    with args.config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    args.model_name = args.model_name or config.get("default_model_name")
    args.model_size = args.model_size or config.get("default_model_size")
    try:
        model_config = config["model_configs"][args.model_name][args.model_size]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unsupported model configuration: {args.model_name}-{args.model_size}") from error

    if args.ndevice is None:
        args.ndevice = int(model_config.get("ndevice", 1))
    if args.batch is None:
        args.batch = int(model_config.get("batch", 1))

    model_prefix = f"{args.model_name}-{args.model_size}"
    _resolve_model_paths(args, model_prefix)
    if args.tokenizer_dir is None:
        repo_ids = model_config.get("modelscope_repo", [])
        tokenizer_name = repo_ids[0].rsplit("/", maxsplit=1)[-1] if repo_ids else model_prefix
        args.tokenizer_dir = tokenizer_name
    if args.question is None:
        args.question = _default_question(args)
    return args


class HmMiniCPMV45:
    def __init__(self, args: argparse.Namespace):
        from minicpm_v45_engine import MiniCPMV45Engine
        from minicpm_v45_types import MiniCPMV45Paths

        self.engine = MiniCPMV45Engine(
            MiniCPMV45Paths(
                args.tokenizer_dir,
                args.embedding_path,
                args.prefill_path,
                args.decode_path,
                args.vision_path,
                args.video_vision_path if args.video_path else None,
            ),
            max_slice_nums=args.max_slice_nums,
            ndevice=args.ndevice,
            batch=args.batch,
            do_sample=not args.greedy,
            temperature=args.temperature,
            seed=args.seed,
            perf=args.perf,
        )

    def generate(self, request, **kwargs):
        yield from self.engine.generate(request, **kwargs)

    def print_perf(self) -> None:
        self.engine.perf.print_summary()


def main() -> None:
    args = _resolve_args(get_args().parse_args())
    if args.max_new_tokens <= 0 or args.ndevice <= 0 or args.batch != 1 or args.video_fps <= 0:
        raise ValueError("max_new_tokens, ndevice, and video_fps must be positive; batch must be 1")
    model = HmMiniCPMV45(args)
    print(f"\033[1;95m\nQ: {args.question}\nA: ", end="", flush=True)
    for chunk in model.generate(
        args.question,
        images=args.image_path,
        videos=args.video_path,
        video_fps=args.video_fps,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt,
    ):
        print(f"\033[1;95m{chunk}", end="", flush=True)
    print()
    if args.perf:
        model.print_perf()


if __name__ == "__main__":
    main()
