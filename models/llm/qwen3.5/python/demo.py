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
import logging
import os
import re
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
IMODELZOO_ROOT = Path(__file__).resolve().parents[4]
HOUMO_EXAMPLES_PATH = Path(os.getenv("HOUMO_EXAMPLES_PATH", str(IMODELZOO_ROOT)))
ENGINE_SRC = HOUMO_EXAMPLES_PATH / "utils" / "python"
sys.path.insert(0, str(ENGINE_SRC))

from houmo_engine.sampling import GreedySamplingParams

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
DEFAULT_CONFIG_PATH = MODEL_DIR / "config.yaml"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "output" / HOUMO_TARGET
DEFAULT_IMAGE_PATHS = [str(HOUMO_EXAMPLES_PATH / "data" / "pic" / "beach.jpeg")]
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
DEFAULT_IMAGE_TOKEN_GEARS = (96, 196, 384, 704, 1536)


def _infer_vision_gear(path: str) -> int | None:
    parts = [Path(path).stem, *[parent.name for parent in Path(path).parents[:2]]]
    for part in parts:
        match = re.search(r"(?:^|[_-])m(\d+)(?:$|[_-])", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _discover_geared_vision_paths(model_prefix: str) -> dict[int, str]:
    paths = {}
    candidates = sorted(path for path in DEFAULT_OUTPUT_DIR.glob(f"**/{model_prefix}_visual_m*.hmm") if path.is_file())
    for path in candidates:
        gear = _infer_vision_gear(str(path))
        if gear in DEFAULT_IMAGE_TOKEN_GEARS:
            if gear in paths:
                raise ValueError(f"duplicate vision gear m{gear}: {paths[gear]} and {path}")
            paths[gear] = str(path)
    return paths


def _discover_vision_paths(model_prefix: str) -> dict[int, str]:
    paths = _discover_geared_vision_paths(model_prefix)
    if not paths:
        candidates = sorted(
            path
            for path in DEFAULT_OUTPUT_DIR.glob(f"**/{model_prefix}_visual_*.hmm")
            if path.is_file() and _infer_vision_gear(str(path)) is None
        )
        if len(candidates) > 1:
            raise ValueError(f"multiple static vision HMMs found: {candidates}")
        if candidates:
            paths[DEFAULT_IMAGE_TOKEN_GEARS[-1]] = str(candidates[0])
    if not paths:
        fallback = DEFAULT_OUTPUT_DIR / f"{model_prefix}_visual.hmm"
        if fallback.is_file():
            paths[DEFAULT_IMAGE_TOKEN_GEARS[-1]] = str(fallback)
    if not paths:
        raise FileNotFoundError(
            f"no vision HMM found under {DEFAULT_OUTPUT_DIR}; expected "
            f"{model_prefix}_visual_m<gear>.hmm, {model_prefix}_visual_<resolution>.hmm, "
            f"or {model_prefix}_visual.hmm"
        )
    return paths


def _expand_vision_specs(path_args):
    specs = [item for group in path_args for value in group for item in value.split(",") if item]
    expanded = []
    for spec in specs:
        explicit = re.match(r"^m?(\d+)[=:](.+)$", spec, flags=re.IGNORECASE)
        if explicit:
            expanded.append((int(explicit.group(1)), explicit.group(2).strip()))
        elif Path(spec).is_dir():
            directory_paths = [
                (_infer_vision_gear(str(path)), str(path)) for path in sorted(Path(spec).rglob("*_visual_m*.hmm"))
            ]
            if not directory_paths:
                static_paths = [
                    path for path in sorted(Path(spec).rglob("*_visual_*.hmm")) if _infer_vision_gear(str(path)) is None
                ]
                if not static_paths:
                    static_paths = sorted(Path(spec).rglob("*_visual.hmm"))
                directory_paths = [(DEFAULT_IMAGE_TOKEN_GEARS[-1], str(path)) for path in static_paths]
            expanded.extend(directory_paths)
        else:
            expanded.append((_infer_vision_gear(spec), spec))
    return expanded


def _resolve_vision_paths(path_args, model_prefix: str) -> dict[int, str]:
    if path_args is None:
        return _discover_vision_paths(model_prefix)
    expanded = _expand_vision_specs(path_args)
    unresolved = [path for gear, path in expanded if gear is None]
    if unresolved:
        raise ValueError(f"cannot infer gear from vision paths {unresolved}; use <gear>=<path>")
    paths = {}
    for gear, path in expanded:
        if gear not in DEFAULT_IMAGE_TOKEN_GEARS:
            raise ValueError(f"unsupported vision gear m{gear}")
        if gear in paths:
            raise ValueError(f"duplicate vision gear m{gear}")
        paths[gear] = path
    return dict(sorted(paths.items()))


class HmQwen35:
    """User-facing Qwen3.5 wrapper around one Qwen35Engine."""

    def __init__(
        self,
        *,
        prefill_path,
        decode_path,
        embedding_path,
        tokenizer_path,
        vision_paths,
        lora: bool = False,
        model_name: str | None = None,
        model_size: str | None = None,
        ndevice: int = 1,
        batch: int = 1,
        vision_min_pixels: int = 65536,
        num_position_embeddings: int = 2304,
        visual_rope_cache_length: int = 3072,
        patch_size: int = 16,
        sampling_params: GreedySamplingParams | None = None,
        perf: bool = False,
    ):
        from houmo_engine import Qwen35Engine

        self.engine = Qwen35Engine(
            prefill_path=prefill_path,
            decode_path=decode_path,
            vision_paths=vision_paths,
            embedding_path=embedding_path,
            tokenizer_path=tokenizer_path,
            ndevice=ndevice,
            batch=batch,
            vision_min_pixels=vision_min_pixels,
            num_position_embeddings=num_position_embeddings,
            visual_rope_cache_length=visual_rope_cache_length,
            patch_size=patch_size,
            sampling_params=sampling_params,
            perf=perf,
        )
        self._init_lora(
            lora=lora,
            model_name=model_name,
            model_size=model_size,
        )

    @staticmethod
    def _discover_lora_path(model_name: str | None, model_size: str | None) -> Path | None:
        if model_name is None or model_size is None:
            return None
        model_prefix = f"{model_name}-{model_size}"
        candidates = sorted(
            path for path in DEFAULT_OUTPUT_DIR.glob(f"**/{model_prefix}_*_prefill_lora_input") if path.is_dir()
        )
        if not candidates:
            candidates = sorted(
                path for path in DEFAULT_OUTPUT_DIR.glob(f"**/{model_prefix}_prefill_lora_input") if path.is_dir()
            )
        logger.info("Found %d LoRA input directories for %s", len(candidates), model_prefix)
        if not candidates:
            return None
        lora_path = candidates[0].resolve()
        logger.info("Selected LoRA input directory: %s", lora_path)
        return lora_path

    def _init_lora(
        self,
        *,
        lora: bool,
        model_name: str | None,
        model_size: str | None,
    ) -> None:
        if not lora:
            return
        lora_path = self._discover_lora_path(model_name, model_size)
        if lora_path is None:
            logger.warning("LoRA mode requested, but no LoRA input directory was found")
            return
        self.engine.module.reset_lora(lora_path)
        logger.info("Initialized LoRA with path: %s", lora_path)

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


def _switch_lora(model: HmQwen35) -> None:
    lora_input_names = model.engine.module.lora_input_names
    if not lora_input_names:
        logger.warning("This model does not support LoRA import")
        return
    while True:
        try:
            lora_path = input("Please input LoRA path (or 'base'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not lora_path:
            return
        if lora_path.lower() == "base":
            model.engine.module.reset_lora(None)
            logger.info("Successfully switched LoRA: base")
            return
        lora_path = Path(lora_path).expanduser().resolve()
        if not lora_path.is_dir():
            logger.warning("LoRA path does not exist, please input again: %s", lora_path)
            continue
        model.engine.module.reset_lora(lora_path)
        logger.info("Successfully switched LoRA: %s", lora_path)
        return


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
        nargs="+",
        action="append",
        default=None,
        help="dynamic vision HMMs: directory, files containing _m<gear>, or <gear>=<path>",
    )
    parser.add_argument(
        "--lora",
        dest="lora",
        action="store_true",
        default=False,
        help="enable LoRA mode and auto-discover LoRA weights",
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
        default=None,
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
        "--vision_min_pixels",
        dest="vision_min_pixels",
        type=int,
        default=65536,
        help="minimum dynamic image pixel budget",
    )
    parser.add_argument(
        "--num_position_embeddings",
        dest="num_position_embeddings",
        type=int,
        default=2304,
        help="number of learned ViT 2D position embeddings",
    )
    parser.add_argument(
        "--visual_rope_cache_length",
        dest="visual_rope_cache_length",
        type=int,
        default=3072,
        help="maximum dynamic visual rotary position",
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
    if args.image_path is None and not args.it:
        args.image_path = DEFAULT_IMAGE_PATHS
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
    args.vision_paths = _resolve_vision_paths(args.vision_path, model_prefix)
    if args.ndevice > 1:
        if args.prefill_path.endswith(".hmm"):
            args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        if args.decode_path.endswith(".hmm"):
            args.decode_path = args.decode_path.replace(".hmm", ".hmms")
    return args


def _build_sampling_params(args: argparse.Namespace) -> GreedySamplingParams:
    return GreedySamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
    )


def _build_model(args: argparse.Namespace) -> HmQwen35:
    return HmQwen35(
        prefill_path=args.prefill_path,
        decode_path=args.decode_path,
        vision_paths=args.vision_paths,
        lora=args.lora,
        model_name=args.model_name,
        model_size=args.model_size,
        embedding_path=args.embedding_path,
        tokenizer_path=args.tokenizer_dir,
        ndevice=args.ndevice,
        batch=args.batch,
        vision_min_pixels=args.vision_min_pixels,
        num_position_embeddings=args.num_position_embeddings,
        visual_rope_cache_length=args.visual_rope_cache_length,
        patch_size=args.patch_size,
        sampling_params=_build_sampling_params(args),
        perf=args.perf,
    )


def _run_once(model: HmQwen35, args: argparse.Namespace, question: str, *, images, keep_history: bool) -> None:
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


def _run_non_interactive(model: HmQwen35, args: argparse.Namespace, supports_vision: bool) -> None:
    _run_once(
        model,
        args,
        args.question,
        images=args.image_path if supports_vision else None,
        keep_history=False,
    )


def _run_interactive(model: HmQwen35, args: argparse.Namespace, supports_vision: bool) -> None:
    keep_history = False
    while True:
        question = _read_question()
        if question is None or question.lower() in {"", "stop", "exit", "quit"}:
            break
        if question.lower() == "switch lora":
            _switch_lora(model)
            continue
        images = _read_images(args.image_path) if supports_vision else None
        _run_once(model, args, question, images=images, keep_history=keep_history)
        if args.history:
            keep_history = True


def main():
    args = _resolve_args(get_args().parse_args())
    model = _build_model(args)
    supports_vision = bool(args.vision_paths)
    if not args.it:
        _run_non_interactive(model, args, supports_vision)
        return
    _run_interactive(model, args, supports_vision)


if __name__ == "__main__":
    main()
