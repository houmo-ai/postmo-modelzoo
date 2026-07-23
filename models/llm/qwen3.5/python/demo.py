#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   Qwen3.5 Inference Demo - Run Qwen3.5 text or image generation through Houmo Python Engine.
# Inference on HOUMO AI device.
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
import argparse
import os
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
IMODELZOO_ROOT = Path(__file__).resolve().parents[4]
HOUMO_EXAMPLES_PATH = Path(os.getenv("HOUMO_EXAMPLES_PATH", str(IMODELZOO_ROOT)))
ENGINE_SRC = HOUMO_EXAMPLES_PATH / "common" / "python"
sys.path.insert(0, str(ENGINE_SRC))

from houmo_engine.sampling import GreedySamplingParams

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
DEFAULT_CONFIG_PATH = MODEL_DIR / "config.yaml"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "output" / HOUMO_TARGET
DEFAULT_IMAGE_PATHS = [str(HOUMO_EXAMPLES_PATH / "data" / "pic" / "beach.jpeg")]


class HmQwen35:
    """User-facing Qwen3.5 wrapper around one Qwen35Engine."""

    def __init__(
        self,
        *,
        prefill_path,
        decode_path,
        embedding_path,
        tokenizer_path,
        vision_path=None,
        ndevice: int = 1,
        batch: int = 1,
        max_size_w: int = 896,
        max_size_h: int = 896,
        patch_size: int = 16,
        sampling_params: GreedySamplingParams | None = None,
        perf: bool = False,
    ):
        from houmo_engine import Qwen35Engine

        self.engine = Qwen35Engine(
            prefill_path=prefill_path,
            decode_path=decode_path,
            vision_path=vision_path,
            embedding_path=embedding_path,
            tokenizer_path=tokenizer_path,
            ndevice=ndevice,
            batch=batch,
            max_size_w=max_size_w,
            max_size_h=max_size_h,
            patch_size=patch_size,
            sampling_params=sampling_params,
            perf=perf,
        )

    def generate(
        self,
        prompt: str,
        *,
        images=None,
        max_new_tokens: int | None = None,
        keep_history: bool = False,
        system_prompt: str | None = None,
    ):
        yield from self.engine.generate(
            prompt,
            images=images,
            max_new_tokens=max_new_tokens,
            keep_history=keep_history,
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


def _read_question():
    try:
        return input("Input your instruction here: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def _read_images(default_images=None):
    try:
        value = input("Image paths (comma-separated, Enter to use command-line images): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not value:
        return default_images
    return [path.strip() for path in value.split(",") if path.strip()]


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        help="model name",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size",
    )
    parser.add_argument(
        "--question",
        dest="question",
        type=str,
        default="描述这些图片",
        help="question or instruction sent to the model",
    )
    parser.add_argument(
        "--system_prompt",
        dest="system_prompt",
        type=str,
        default=None,
        help="system prompt; uses the engine default when omitted",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=None,
        help="path to the prefill HMM model",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default=None,
        help="path to the decode HMM model",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=None,
        help="path to the embedding weights",
    )
    parser.add_argument(
        "--vision_path",
        dest="vision_path",
        type=str,
        default=None,
        help="path to the vision HMM model",
    )
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default=None,
        help="path to the tokenizer and processor configuration",
    )
    parser.add_argument(
        "--image_path",
        dest="image_path",
        type=str,
        default=DEFAULT_IMAGE_PATHS,
        help="one or more image paths; the option may be repeated",
        nargs="+",
        action="extend",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="number of Houmo devices used for inference",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=1,
        help="inference batch size; Qwen3.5 currently supports 1",
    )
    parser.add_argument(
        "--max_size_w",
        dest="max_size_w",
        type=int,
        default=None,
        help="maximum resized image width",
    )
    parser.add_argument(
        "--max_size_h",
        dest="max_size_h",
        type=int,
        default=None,
        help="maximum resized image height",
    )
    parser.add_argument(
        "--max_size_t",
        dest="max_size_t",
        type=int,
        default=None,
        help="maximum temporal size used in the vision model filename",
    )
    parser.add_argument(
        "--patch_size",
        dest="patch_size",
        type=int,
        default=16,
        help="vision patch size",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="maximum number of generated tokens",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=1.0,
        help="sampling temperature",
    )
    parser.add_argument(
        "--topk",
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="top-k logits filtering value",
    )
    parser.add_argument(
        "--topp",
        "--top-p",
        dest="top_p",
        type=float,
        default=1.0,
        help="top-p probability filtering value",
    )
    parser.add_argument(
        "--presence-penalty",
        dest="presence_penalty",
        type=float,
        default=0.0,
        help="presence penalty applied to generated tokens",
    )
    parser.add_argument(
        "--repetition-penalty",
        dest="repetition_penalty",
        type=float,
        default=1.0,
        help="repetition penalty applied to generated tokens",
    )
    parser.add_argument(
        "--perf",
        dest="perf",
        type=_parse_bool,
        default=True,
        help="enable performance reporting",
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--it",
        dest="it",
        type=_parse_bool,
        default=False,
        help="enable interactive mode",
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--history",
        dest="history",
        type=_parse_bool,
        default=False,
        help="continue the conversation after the first interactive turn",
        nargs="?",
        const=True,
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
        raise ValueError(f"unsupported model configuration: {args.model_name}-{args.model_size}") from error

    args.ndevice = args.ndevice or int(model_config.get("ndevice", 1))
    args.max_size_w = args.max_size_w or int(model_config.get("max_size_w", 896))
    args.max_size_h = args.max_size_h or int(model_config.get("max_size_h", 896))
    args.max_size_t = args.max_size_t or int(model_config.get("max_size_t", 2))
    model_prefix = f"{args.model_name}-{args.model_size}"
    if args.prefill_path is None:
        args.prefill_path = str(DEFAULT_OUTPUT_DIR / f"{model_prefix}_prefill.hmm")
    if args.decode_path is None:
        args.decode_path = str(DEFAULT_OUTPUT_DIR / f"{model_prefix}_decode.hmm")
    if args.embedding_path is None:
        args.embedding_path = str(DEFAULT_OUTPUT_DIR / "hmquant" / "quant_embedding.pt")
    if args.tokenizer_dir is None:
        repo_ids = model_config.get("modelscope_repo", [])
        tokenizer_name = repo_ids[0].rsplit("/", maxsplit=1)[-1] if repo_ids else model_prefix
        args.tokenizer_dir = str(MODEL_DIR / tokenizer_name)
    if args.vision_path is None:
        sized_path = DEFAULT_OUTPUT_DIR / (
            f"{model_prefix}_visual_{args.max_size_w}x{args.max_size_h}x" f"{args.max_size_t}.hmm"
        )
        fallback_path = DEFAULT_OUTPUT_DIR / f"{model_prefix}_visual.hmm"
        args.vision_path = str(fallback_path if not sized_path.exists() and fallback_path.exists() else sized_path)
    if args.ndevice > 1:
        if args.prefill_path.endswith(".hmm"):
            args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        if args.decode_path.endswith(".hmm"):
            args.decode_path = args.decode_path.replace(".hmm", ".hmms")
    return args


def main():
    args = _resolve_args(get_args().parse_args())
    sampling = GreedySamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
    )
    model = HmQwen35(
        prefill_path=args.prefill_path,
        decode_path=args.decode_path,
        vision_path=args.vision_path,
        embedding_path=args.embedding_path,
        tokenizer_path=args.tokenizer_dir,
        ndevice=args.ndevice,
        batch=args.batch,
        max_size_w=args.max_size_w,
        max_size_h=args.max_size_h,
        patch_size=args.patch_size,
        sampling_params=sampling,
        perf=args.perf,
    )

    def run_once(question: str, *, images, keep_history: bool):
        print(f"\033[1;95m\nQ: {question}\nA: ", end="", flush=True)
        for chunk in model.generate(
            question,
            images=images,
            max_new_tokens=args.max_new_tokens,
            keep_history=keep_history,
            system_prompt=args.system_prompt,
        ):
            print(f"\033[1;95m{chunk}", end="", flush=True)
        print()
        if args.perf:
            model.print_perf()

    if not args.it:
        run_once(args.question, images=args.image_path, keep_history=False)
        return

    keep_history = False
    while True:
        question = _read_question()
        if question is None or question.lower() in {"", "stop", "exit", "quit"}:
            break
        images = _read_images(args.image_path)
        run_once(question, images=images, keep_history=keep_history)
        if args.history:
            keep_history = True


if __name__ == "__main__":
    main()
