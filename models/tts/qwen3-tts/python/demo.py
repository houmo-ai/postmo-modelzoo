#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   Run Qwen3-TTS CustomVoice synthesis through the Houmo Python Engine.
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

MODEL_DIR = Path(__file__).resolve().parents[1]
IMODELZOO_ROOT = Path(__file__).resolve().parents[4]
HOUMO_EXAMPLES_PATH = Path(os.getenv("HOUMO_EXAMPLES_PATH", str(IMODELZOO_ROOT)))
ENGINE_SRC = HOUMO_EXAMPLES_PATH / "utils" / "python"
sys.path.insert(0, str(ENGINE_SRC))

from houmo_engine.sampling import Qwen3TtsSamplingParams

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
DEFAULT_CONFIG_PATH = MODEL_DIR / "config.yaml"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "output" / HOUMO_TARGET
DEFAULT_HMQUANT_DIR = DEFAULT_OUTPUT_DIR / "hmquant"


class HmQwen3Tts:
    """User-facing Qwen3-TTS wrapper around one Qwen3TtsEngine."""

    def __init__(
        self,
        *,
        hf_model_dir,
        text_projection_path,
        talker_prefill_path,
        talker_decode_path,
        code_predictor_prefill_path,
        code_predictor_decode_path,
        talker_token_embedding_path,
        talker_text_embedding_path,
        code_predictor_token_embedding_path,
        mode: str = "oneshot",
        speech_tokenizer_path=None,
        stateful_decoder_path=None,
        decode_padding_shapes_path=None,
        chunk_size: int = 12,
        ndevice: int = 1,
        batch: int = 1,
        sampling_params: Qwen3TtsSamplingParams | None = None,
        perf: bool = False,
    ):
        from houmo_engine import Qwen3TtsEngine

        self.engine = Qwen3TtsEngine(
            hf_model_dir,
            text_projection_path,
            talker_prefill_path,
            talker_decode_path,
            code_predictor_prefill_path,
            code_predictor_decode_path,
            talker_token_embedding_path,
            talker_text_embedding_path,
            code_predictor_token_embedding_path,
            mode=mode,
            speech_tokenizer_path=speech_tokenizer_path,
            stateful_decoder_path=stateful_decoder_path,
            decode_padding_shapes_path=decode_padding_shapes_path,
            chunk_size=chunk_size,
            ndevice=ndevice,
            batch=batch,
            sampling_params=sampling_params,
            perf=perf,
        )

    def generate(
        self,
        text,
        *,
        language: str = "Chinese",
        speaker: str = "vivian",
        max_new_tokens: int = 4096,
        sampling_params: Qwen3TtsSamplingParams | None = None,
    ):
        yield from self.engine.generate(
            text,
            language=language,
            speaker=speaker,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="path to config.yaml",
    )
    parser.add_argument("--model_name", dest="model_name", type=str, default=None, help="model name")
    parser.add_argument("--model_size", dest="model_size", type=str, default=None, help="model size")
    parser.add_argument(
        "--hf_model_dir",
        dest="hf_model_dir",
        type=str,
        default=None,
        help="Modelscope HF model directory path",
    )
    parser.add_argument(
        "--text_projection_hmm",
        dest="text_projection_hmm",
        type=str,
        default=None,
        help="Text Projection HMM model file path",
    )
    parser.add_argument(
        "--code_predictor_prefill_hmm",
        dest="code_predictor_prefill_hmm",
        type=str,
        default=None,
        help="Code Predictor Prefill HMM file path",
    )
    parser.add_argument(
        "--code_predictor_decode_hmm",
        dest="code_predictor_decode_hmm",
        type=str,
        default=None,
        help="Code Predictor Decode HMM file path",
    )
    parser.add_argument(
        "--code_predictor_token_embedding",
        dest="code_predictor_token_embedding",
        type=str,
        default=None,
        help="Code Predictor Token Embedding file path",
    )
    parser.add_argument(
        "--talker_prefill_hmm",
        dest="talker_prefill_hmm",
        type=str,
        default=None,
        help="Talker Prefill HMM file path",
    )
    parser.add_argument(
        "--talker_decode_hmm",
        dest="talker_decode_hmm",
        type=str,
        default=None,
        help="Talker Decode HMM file path",
    )
    parser.add_argument(
        "--talker_token_embedding",
        dest="talker_token_embedding",
        type=str,
        default=None,
        help="Talker Token Embedding file path",
    )
    parser.add_argument(
        "--talker_text_embedding",
        dest="talker_text_embedding",
        type=str,
        default=None,
        help="Talker Text Embedding file path",
    )
    parser.add_argument(
        "--speech_tokenizer_hmm",
        dest="speech_tokenizer_hmm",
        type=str,
        default=None,
        help="Speech Tokenizer HMM model file path",
    )
    parser.add_argument(
        "--speech_tokenizer_decode_padding_shapes",
        dest="speech_tokenizer_decode_padding_shapes",
        type=str,
        default=None,
        help="Speech Tokenizer decode_padding_shapes JSON file path",
    )
    parser.add_argument(
        "--stateful_decoder_hmm",
        dest="stateful_decoder_hmm",
        type=str,
        default=None,
        help="Stateful Decoder HMM file path (required for streaming mode)",
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
        help="inference batch size; Qwen3-TTS currently supports 1",
    )
    parser.add_argument(
        "--output_wav",
        dest="output_wav",
        type=str,
        default="./output_custom_voice.wav",
        help="output wav file path",
    )
    parser.add_argument(
        "--text",
        dest="text",
        type=str,
        default="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地。",
        help="text to synthesize",
    )
    parser.add_argument(
        "--language",
        dest="language",
        type=str,
        default="Chinese",
        choices=[
            "auto",
            "Chinese",
            "English",
            "Japanese",
            "Korean",
            "French",
            "German",
            "Spanish",
            "Italian",
            "Portuguese",
            "Russian",
        ],
        help="language, default Chinese",
    )
    parser.add_argument(
        "--speaker",
        dest="speaker",
        type=str,
        default="vivian",
        choices=[
            "vivian",
            "serena",
            "uncle_fu",
            "ryan",
            "aiden",
            "ono_anna",
            "sohee",
            "eric",
            "dylan",
        ],
        help="speaker, default vivian",
    )
    parser.add_argument("--seed", dest="seed", type=int, default=1024, help="random seed")
    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="oneshot",
        choices=["oneshot", "streaming"],
        help="inference mode: oneshot=full generation; streaming=chunked streaming",
    )
    parser.add_argument(
        "--chunk_size",
        dest="chunk_size",
        type=int,
        default=12,
        help="streaming codec frames per decode chunk (default 12, i.e. 1s @12Hz)",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=4096,
        help="maximum generated codec frames",
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
        "--perf-format",
        dest="perf_format",
        type=str,
        choices=["text", "rich"],
        default="text",
        help="performance report formatter",
    )
    return parser


def _load_model_config(args: argparse.Namespace) -> tuple[dict, str]:
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
    return model_config, f"{args.model_name}-{args.model_size}"


def _resolve_model_paths(args: argparse.Namespace, model_config: dict, model_prefix: str) -> None:
    repo_ids = model_config.get("modelscope_repo", [])
    model_dir = repo_ids[0].rsplit("/", maxsplit=1)[-1] if repo_ids else model_prefix
    args.hf_model_dir = args.hf_model_dir or str(MODEL_DIR / model_dir)

    def hmm(sub_model_name: str) -> str:
        return str(DEFAULT_OUTPUT_DIR / f"{model_prefix}_{sub_model_name}.hmm")

    hmm_defaults = {
        "text_projection_hmm": "text_projection",
        "code_predictor_prefill_hmm": "code_predictor_prefill",
        "code_predictor_decode_hmm": "code_predictor_decode",
        "talker_prefill_hmm": "talker_prefill",
        "talker_decode_hmm": "talker_decode",
        "speech_tokenizer_hmm": "speech_tokenizer",
        "stateful_decoder_hmm": "stateful_decoder",
    }
    for name, model_name in hmm_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, hmm(model_name))


def _resolve_embedding_paths(args: argparse.Namespace) -> None:
    defaults = {
        "code_predictor_token_embedding": DEFAULT_HMQUANT_DIR / "quant_embedding_code_predictor.pt",
        "talker_token_embedding": DEFAULT_HMQUANT_DIR / "quant_embedding.pt",
        "talker_text_embedding": DEFAULT_HMQUANT_DIR / "text_embedding.pt",
        "speech_tokenizer_decode_padding_shapes": DEFAULT_HMQUANT_DIR / "decode_padding_shapes.json",
    }
    for name, path in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, str(path))


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    model_config, model_prefix = _load_model_config(args)
    _resolve_model_paths(args, model_config, model_prefix)
    _resolve_embedding_paths(args)
    return args


def _create_model(args: argparse.Namespace) -> HmQwen3Tts:
    return HmQwen3Tts(
        hf_model_dir=args.hf_model_dir,
        text_projection_path=args.text_projection_hmm,
        talker_prefill_path=args.talker_prefill_hmm,
        talker_decode_path=args.talker_decode_hmm,
        code_predictor_prefill_path=args.code_predictor_prefill_hmm,
        code_predictor_decode_path=args.code_predictor_decode_hmm,
        talker_token_embedding_path=args.talker_token_embedding,
        talker_text_embedding_path=args.talker_text_embedding,
        code_predictor_token_embedding_path=args.code_predictor_token_embedding,
        mode=args.mode,
        speech_tokenizer_path=args.speech_tokenizer_hmm,
        stateful_decoder_path=args.stateful_decoder_hmm,
        decode_padding_shapes_path=args.speech_tokenizer_decode_padding_shapes,
        chunk_size=args.chunk_size,
        ndevice=args.ndevice,
        batch=args.batch,
        perf=args.perf,
    )


def _generate_audio(model: HmQwen3Tts, args: argparse.Namespace, np, sf, logger) -> None:
    out_file = Path(args.output_wav)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    print(f"\033[1;95m\nQ: {args.text}\nA: {out_file}", end="", flush=True)
    chunks = []
    output_sr = None
    for audio, sr in model.generate(
        args.text,
        language=args.language,
        speaker=args.speaker,
        max_new_tokens=args.max_new_tokens,
    ):
        if output_sr is None:
            output_sr = sr
        elif sr != output_sr:
            raise ValueError(f"sample rate changed from {output_sr} to {sr}")
        chunks.append(audio)
    print()

    if chunks:
        full_audio = np.concatenate(chunks)
        sf.write(out_file, full_audio, output_sr)
        duration = len(full_audio) / output_sr
        logger.info(f"Audio saved to {out_file} | duration: {duration:.2f}s")
    else:
        logger.error("No audio generated")


def _print_performance(model: HmQwen3Tts, args: argparse.Namespace, logger) -> None:
    if args.perf:
        if args.perf_format != "rich":
            model.print_perf()
            return

        from houmo_engine.perf.rich_formatter import print_rich_report

        try:
            print_rich_report(model.engine.perf.summary())
        except RuntimeError as error:
            if not isinstance(error.__cause__, ImportError):
                raise
            logger.warning(f"{error}; falling back to text performance output")
            model.print_perf()


def main():
    args = _resolve_args(get_args().parse_args())

    import numpy as np
    import soundfile as sf
    import torch
    from loguru import logger

    torch.manual_seed(args.seed)
    model = _create_model(args)
    _generate_audio(model, args, np, sf, logger)
    _print_performance(model, args, logger)


if __name__ == "__main__":
    main()
