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
import os
import argparse
from loguru import logger
from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
HOUMO_EXAMPLES_PATH = os.getenv("HOUMO_EXAMPLES_PATH", ".")
assert HOUMO_TARGET == "xh2", "Only support HOUMO_TARGET: xh2"


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", type=str, default="config.yaml", help="path to config.yaml")
    parser.add_argument("--model_name", type=str, default=None, help="model name")
    parser.add_argument("--model_size", type=str, default=None, help="model size")
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--embedding_path", type=str, default=f"output/{HOUMO_TARGET}/hmquant/quant_embedding.pt")
    parser.add_argument("--prefill_path", type=str, default=None)
    parser.add_argument("--decode_path", type=str, default=None)
    parser.add_argument("--vit_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--ndevice", type=int, default=None, help="device number")
    parser.add_argument("--question", default=None)
    parser.add_argument("--image", default="data/pic/beach.jpeg")
    parser.add_argument("--audio", default="data/audio/0.wav", help="only supported for e2b/e4b")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--plib_embedding_path", type=str, default=f"output/{HOUMO_TARGET}/hmquant/per_layer_input_embedding.pt")
    parser.add_argument("--mtp", action="store_true", default=False)
    parser.add_argument("--assistant_path", type=str, default=None)
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    if args.tokenizer_dir is None:
        args.tokenizer_dir = f"output/{HOUMO_TARGET}/hmquant/hf_config"
    if args.prefill_path is None:
        args.prefill_path = f"output/{HOUMO_TARGET}/{args.model_name}-{args.model_size}_prefill.hmm"
    if args.decode_path is None:
        args.decode_path = f"output/{HOUMO_TARGET}/{args.model_name}-{args.model_size}_decode.hmm"
    if args.vit_path is None:
        args.vit_path = f"output/{HOUMO_TARGET}/{args.model_name}-{args.model_size}_visual.hmm"
    if args.audio_path is None:
        args.audio_path = f"output/{HOUMO_TARGET}/{args.model_name}-{args.model_size}_audio.hmm"
    if args.mtp:
        args.assistant_path = args.assistant_path or f"output/{HOUMO_TARGET}/{args.model_name}-{args.model_size}_assistant.hmm"
    elif args.assistant_path is not None:
        logger.warning("--assistant_path is ignored because --mtp is disabled")
        args.assistant_path = None

    if args.ndevice > 1:
        if args.prefill_path.endswith(".hmm"):
            args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        if args.decode_path.endswith(".hmm"):
            args.decode_path = args.decode_path.replace(".hmm", ".hmms")
        if args.assistant_path.endswith(".hmm"):
            args.assistant_path = args.assistant_path.replace(".hmm", ".hmms")
    # fmt: on
    return args


if __name__ == "__main__":
    args = get_args()

    if args.model_size in ["26b-a4b", "31b"]:
        from gemma4 import Gemma4

        model = Gemma4(
            prefill_path=args.prefill_path,
            decode_path=args.decode_path,
            embedding_path=args.embedding_path,
            tokenizer_dir=args.tokenizer_dir,
            vit_path=args.vit_path,
            assistant_path=args.assistant_path,
            max_new_tokens=args.max_new_tokens,
            devices=list(range(args.ndevice)),
        )
    elif args.model_size in ["e2b", "e4b"]:
        from gemma4 import Gemma4E

        model = Gemma4E(
            prefill_path=args.prefill_path,
            decode_path=args.decode_path,
            embedding_path=args.embedding_path,
            tokenizer_dir=args.tokenizer_dir,
            PLE_path=args.plib_embedding_path,
            vit_path=args.vit_path,
            audio_path=args.audio_path,
            assistant_path=args.assistant_path,
            max_new_tokens=args.max_new_tokens,
            devices=list(range(args.ndevice)),
        )
    else:
        raise ValueError(f"Unsupported model size: {args.model_size}")

    image_path = args.image
    if image_path and not os.path.isfile(image_path):
        image_path = os.path.join(HOUMO_EXAMPLES_PATH, args.image)
        if not os.path.isfile(image_path):
            logger.warning(f"Image not found: {image_path}, running text-only mode")
            image_path = None
    if image_path and model.vit is None:
        logger.warning("Image provided but vit not loaded, running text-only mode")
        image_path = None

    if args.model_size in ["e2b", "e4b"]:
        audio_path = args.audio if args.audio else None
        if audio_path and not os.path.isfile(audio_path):
            audio_path = os.path.join(HOUMO_EXAMPLES_PATH, args.audio)
            if not os.path.isfile(audio_path):
                logger.warning(f"Audio not found: {audio_path}, running without audio")
                audio_path = None
        if audio_path and getattr(model, "audio", None) is None:
            logger.warning(
                "Audio provided but audio model not loaded, running without audio"
            )
            audio_path = None
    else:
        audio_path = None

    model.chat(args.question, image_path, audio_path)
