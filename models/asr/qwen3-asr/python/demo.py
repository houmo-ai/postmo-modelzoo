#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   Run Qwen3-ASR generation through the Houmo Python Engine.
# model on HOUMO AI device.
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

MODEL_DIR = Path(__file__).resolve().parents[1]
IMODELZOO_ROOT = Path(__file__).resolve().parents[4]
HOUMO_EXAMPLES_PATH = Path(os.getenv("HOUMO_EXAMPLES_PATH", str(IMODELZOO_ROOT)))
ENGINE_SRC = HOUMO_EXAMPLES_PATH / "utils" / "python"
sys.path.insert(0, str(ENGINE_SRC))

from houmo_engine.sampling import GreedySamplingParams

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
DEFAULT_CONFIG_PATH = MODEL_DIR / "config.yaml"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "output" / HOUMO_TARGET
DEFAULT_AUDIO_PATH = HOUMO_EXAMPLES_PATH / "data" / "audio" / "audio.mp3"


class HmQwen3Asr:
    """User-facing Qwen3-ASR wrapper around one Qwen3AsrEngine."""

    def __init__(
        self,
        *,
        encode_path,
        prefill_path,
        decode_path,
        embedding_path,
        processor_path,
        ndevice: int = 1,
        batch: int = 1,
        sampling_params: GreedySamplingParams | None = None,
        perf: bool = False,
    ):
        from houmo_engine import Qwen3AsrEngine

        self.engine = Qwen3AsrEngine(
            encode_path=encode_path,
            prefill_path=prefill_path,
            decode_path=decode_path,
            embedding_path=embedding_path,
            processor_path=processor_path,
            ndevice=ndevice,
            batch=batch,
            sampling_params=sampling_params,
            perf=perf,
        )

    def generate(
        self,
        audio,
        *,
        max_new_tokens: int | None = None,
        sampling_params: GreedySamplingParams | None = None,
        system_prompt: str | None = None,
    ):
        yield from self.engine.generate(
            audio,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
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
        "--audio",
        dest="audio",
        type=str,
        default=os.path.join(HOUMO_EXAMPLES_PATH, "data", "audio", "audio.mp3"),
        help="path to the input audio file",
    )
    parser.add_argument(
        "--system_prompt",
        dest="system_prompt",
        type=str,
        default=None,
        help="system prompt; uses the engine default when omitted",
    )
    parser.add_argument(
        "--encode_path",
        dest="encode_path",
        type=str,
        default=None,
        help="path to the Qwen3-ASR encode HMM model",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=None,
        help="path to the Qwen3-ASR prefill HMM model",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default=None,
        help="path to the Qwen3-ASR decode HMM model",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=None,
        help="path to the Qwen3-ASR embedding weights",
    )
    parser.add_argument(
        "--processor_dir",
        dest="processor_dir",
        type=str,
        default=None,
        help="path to the Qwen3-ASR processor configuration",
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
        help="inference batch size; Qwen3-ASR currently supports 1",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="maximum generated tokens including the prefill token",
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
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    import yaml

    config_path = Path(args.config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    args.model_name = args.model_name or config["default_model_name"]
    args.model_size = args.model_size or config["default_model_size"]
    try:
        model_config = config["model_configs"][args.model_name][args.model_size]
    except KeyError as error:
        raise ValueError(f"unsupported model configuration: {args.model_name}-{args.model_size}") from error

    args.ndevice = args.ndevice or int(model_config.get("ndevice", 1))
    model_prefix = f"{args.model_name}-{args.model_size}"
    if args.encode_path is None:
        args.encode_path = str(DEFAULT_OUTPUT_DIR / f"{model_prefix}_encode.hmm")
    if args.prefill_path is None:
        args.prefill_path = str(DEFAULT_OUTPUT_DIR / f"{model_prefix}_prefill.hmm")
    if args.decode_path is None:
        args.decode_path = str(DEFAULT_OUTPUT_DIR / f"{model_prefix}_decode.hmm")
    if args.embedding_path is None:
        args.embedding_path = str(DEFAULT_OUTPUT_DIR / "hmquant" / "quant_embedding.pt")
    if args.processor_dir is None:
        repo_ids = model_config.get("modelscope_repo", [])
        processor_name = repo_ids[0].rsplit("/", maxsplit=1)[-1] if repo_ids else f"{args.model_name}-{args.model_size}"
        args.processor_dir = str(MODEL_DIR / processor_name)
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
    model = HmQwen3Asr(
        encode_path=args.encode_path,
        prefill_path=args.prefill_path,
        decode_path=args.decode_path,
        embedding_path=args.embedding_path,
        processor_path=args.processor_dir,
        ndevice=args.ndevice,
        batch=args.batch,
        sampling_params=sampling,
        perf=args.perf,
    )

    print(f"\033[1;95m\nQ: {args.audio}\nA: ", end="", flush=True)
    for chunk in model.generate(
        args.audio,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt,
    ):
        print(f"\033[1;95m{chunk}", end="", flush=True)
    print()
    if args.perf:
        model.print_perf()


if __name__ == "__main__":
    main()
