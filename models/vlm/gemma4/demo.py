# Copyright 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Gemma4 Inference Demo
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
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
HOUMO_EXAMPLES_PATH = os.getenv("HOUMO_EXAMPLES_PATH", ".")
assert HOUMO_TARGET == "xh2", "Only support HOUMO_TARGET: xh2"


@dataclass(frozen=True)
class ModelFiles:
    prefill: Path
    decode: Path
    embedding: Path
    tokenizer: Path
    visual: Path | None
    audio: Path | None
    assistant: Path | None
    ple: Path | None


def get_args(argv=None):
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"output/e2b/{HOUMO_TARGET}", help="model directory")
    parser.add_argument("--ndevice", type=int, default=1, help="maximum device number")
    parser.add_argument("--question", default=None, help="question to ask")
    parser.add_argument("--system_prompt", default=None, help="system prompt to control assistant behavior")
    parser.add_argument("--image", default="data/pic/beach.jpeg", help="image path")
    parser.add_argument("--audio", default="data/audio/0.wav", help="audio path")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="maximum number of new tokens to generate")
    parser.add_argument("--enable-thinking", action="store_true", help="enable thinking process")
    # fmt: on
    args = parser.parse_args(argv)
    if args.ndevice < 1:
        parser.error("--ndevice must be at least 1")
    return args


def find_model(model_dir, component, required=False):
    matches = sorted(
        path
        for suffix in (".hmm", ".hmms")
        for path in model_dir.glob(f"*_{component}{suffix}")
    )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple {component} models found: "
            + ", ".join(str(path) for path in matches)
        )
    if required and not matches:
        raise FileNotFoundError(f"{component} model not found in {model_dir}")
    return matches[0] if matches else None


def require_path(path, description):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def resolve_model_files(model_dir):
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model directory not found: {model_dir}")

    hmquant_dir = model_dir / "hmquant"
    audio = find_model(model_dir, "audio")
    ple = hmquant_dir / "per_layer_input_embedding.pt" if audio else None
    if ple:
        require_path(ple, "E model per-layer input embedding")

    return ModelFiles(
        prefill=find_model(model_dir, "prefill", required=True),
        decode=find_model(model_dir, "decode", required=True),
        embedding=require_path(hmquant_dir / "quant_embedding.pt", "Embedding"),
        tokenizer=require_path(hmquant_dir / "hf_config", "Tokenizer directory"),
        visual=find_model(model_dir, "visual"),
        audio=audio,
        assistant=find_model(model_dir, "assistant"),
        ple=ple,
    )


def create_model(files, args):
    from gemma4 import Gemma4, Gemma4E

    model_args = {
        "prefill_path": str(files.prefill),
        "decode_path": str(files.decode),
        "embedding_path": str(files.embedding),
        "tokenizer_dir": str(files.tokenizer),
        "vit_path": str(files.visual) if files.visual else None,
        "assistant_path": str(files.assistant) if files.assistant else None,
        "max_new_tokens": args.max_new_tokens,
        "devices": list(range(args.ndevice)),
    }
    if files.audio:
        return Gemma4E(
            **model_args,
            PLE_path=str(files.ple),
            audio_path=str(files.audio),
        )
    return Gemma4(**model_args)


def resolve_input(path, input_name):
    if not path:
        return None

    resolved = Path(path)
    if not resolved.is_file():
        resolved = Path(HOUMO_EXAMPLES_PATH) / path
    if not resolved.is_file():
        logger.warning(f"{input_name} not found: {resolved}, ignoring input")
        return None
    return str(resolved)


def main():
    args = get_args()
    files = resolve_model_files(args.model)
    model = create_model(files, args)

    image = resolve_input(args.image, "Image")
    if image and not files.visual:
        logger.warning("Image provided but visual model not found, ignoring image")
        image = None

    audio = resolve_input(args.audio, "Audio")
    if audio and not files.audio:
        logger.warning("Audio provided but audio model not found, ignoring audio")
        audio = None

    model.chat(
        args.question,
        image,
        audio,
        enable_thinking=args.enable_thinking,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
