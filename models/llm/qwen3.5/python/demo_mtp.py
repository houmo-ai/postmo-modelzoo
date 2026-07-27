#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo_mtp.py
# Description:
#   Run Qwen3.6 MTP speculative generation through Houmo Python Engine.
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
ENGINE_SRC = HOUMO_EXAMPLES_PATH / "utils" / "python"
sys.path.insert(0, str(ENGINE_SRC))

from houmo_engine.sampling import GreedySamplingParams

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
DEFAULT_CONFIG_PATH = MODEL_DIR / "config.yaml"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "output" / HOUMO_TARGET


class HmQwen36Mtp:
    """User-facing Qwen3.6 MTP wrapper around one Qwen36MtpEngine."""

    def __init__(
        self,
        *,
        prefill_path,
        prefill_mtp_path,
        decode_mtp_path,
        decode_verify_path,
        embedding_path,
        tokenizer_path,
        ndevice: int = 1,
        batch: int = 1,
        sampling_params: GreedySamplingParams | None = None,
        perf: bool = False,
        debug: bool = False,
    ):
        from houmo_engine import Qwen36MtpEngine

        self.engine = Qwen36MtpEngine(
            prefill_path=prefill_path,
            prefill_mtp_path=prefill_mtp_path,
            decode_mtp_path=decode_mtp_path,
            decode_verify_path=decode_verify_path,
            embedding_path=embedding_path,
            tokenizer_path=tokenizer_path,
            ndevice=ndevice,
            batch=batch,
            sampling_params=sampling_params,
            perf=perf,
            debug=debug,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        keep_history: bool = False,
        system_prompt: str | None = None,
    ):
        yield from self.engine.generate(
            prompt,
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
        default="请介绍一下存算一体技术的优势",
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
        help="path to the main prefill HMM model",
    )
    parser.add_argument(
        "--prefill_mtp_path",
        dest="prefill_mtp_path",
        type=str,
        default=None,
        help="path to the MTP prefill HMM model",
    )
    parser.add_argument(
        "--decode_mtp_path",
        dest="decode_mtp_path",
        type=str,
        default=None,
        help="path to the MTP draft HMM model",
    )
    parser.add_argument(
        "--decode_verify_path",
        dest="decode_verify_path",
        type=str,
        default=None,
        help="path to the target verify HMM model",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=None,
        help="path to the embedding weights",
    )
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default=None,
        help="path to the tokenizer configuration",
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
        help="inference batch size; Qwen3.6 MTP currently supports 1",
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
    parser.add_argument(
        "--debug",
        dest="debug",
        type=_parse_bool,
        default=False,
        help="enable runtime diagnostic logs",
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
    if args.model_name != "qwen3.6":
        raise ValueError("Qwen3.6 MTP example requires --model_name qwen3.6")

    args.ndevice = args.ndevice or int(model_config.get("ndevice", 1))
    model_prefix = f"{args.model_name}-{args.model_size}"
    if args.prefill_path is None:
        args.prefill_path = str(DEFAULT_OUTPUT_DIR / f"{model_prefix}_prefill.hmm")
    if args.prefill_mtp_path is None:
        args.prefill_mtp_path = str(DEFAULT_OUTPUT_DIR / f"{model_prefix}_prefill_mtp.hmm")
    if args.decode_mtp_path is None:
        args.decode_mtp_path = str(DEFAULT_OUTPUT_DIR / f"{model_prefix}_decode_mtp.hmm")
    if args.decode_verify_path is None:
        args.decode_verify_path = str(DEFAULT_OUTPUT_DIR / f"{model_prefix}_decode.hmm")
    if args.embedding_path is None:
        args.embedding_path = str(DEFAULT_OUTPUT_DIR / "hmquant" / "quant_embedding.pt")
    if args.tokenizer_dir is None:
        repo_ids = model_config.get("modelscope_repo", [])
        tokenizer_name = repo_ids[0].rsplit("/", maxsplit=1)[-1] if repo_ids else model_prefix
        args.tokenizer_dir = str(MODEL_DIR / tokenizer_name)
    if args.ndevice > 1:
        for name in (
            "prefill_path",
            "prefill_mtp_path",
            "decode_mtp_path",
            "decode_verify_path",
        ):
            value = getattr(args, name)
            if value.endswith(".hmm"):
                setattr(args, name, value.replace(".hmm", ".hmms"))
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
    model = HmQwen36Mtp(
        prefill_path=args.prefill_path,
        prefill_mtp_path=args.prefill_mtp_path,
        decode_mtp_path=args.decode_mtp_path,
        decode_verify_path=args.decode_verify_path,
        embedding_path=args.embedding_path,
        tokenizer_path=args.tokenizer_dir,
        ndevice=args.ndevice,
        batch=args.batch,
        sampling_params=sampling,
        perf=args.perf,
        debug=args.debug,
    )

    print(f"\033[1;95m\nQ: {args.question}\nA: ", end="", flush=True)
    for chunk in model.generate(
        args.question,
        max_new_tokens=args.max_new_tokens,
        keep_history=False,
        system_prompt=args.system_prompt,
    ):
        print(f"\033[1;95m{chunk}", end="", flush=True)
    print()
    if args.perf:
        model.print_perf()


if __name__ == "__main__":
    main()
