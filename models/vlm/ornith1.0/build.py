#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: build.py
# Description:
#   Compile and validate Ornith 1.0 prefill, decode, and vision models.
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
import multiprocessing
import os
import yaml
from pathlib import Path
from hmatc.exec.xh2_exec import Xh2Exec
from hmatc.utils.utils import find_hmonnx_file, get_platform, parse_context_length

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODEL_ROOT / "config.yaml"


def load_model_config(
    config_path: str, model_name: str | None, model_size: str | None
) -> tuple[str, str, dict]:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    model_configs = config.get("model_configs", {})
    selected_name = model_name or config.get("default_model_name")
    if (
        selected_name not in model_configs
        and model_name is None
        and len(model_configs) == 1
    ):
        selected_name = next(iter(model_configs))

    size_configs = model_configs.get(selected_name, {})
    selected_size = model_size or config.get("default_model_size")
    if (
        selected_size not in size_configs
        and model_size is None
        and len(size_configs) == 1
    ):
        selected_size = next(iter(size_configs))

    try:
        return selected_name, selected_size, size_configs[selected_size]
    except KeyError as error:
        raise ValueError(
            f"unsupported model configuration: {selected_name}-{selected_size}"
        ) from error


def discover_model_dirs(model_dir: Path) -> tuple[Path, Path, list[Path]]:
    prefill_dir = model_dir / "prefill"
    if not prefill_dir.is_dir():
        raise FileNotFoundError(f"prefill directory not found: {prefill_dir}")

    decode_dirs = sorted(path for path in model_dir.glob("*decode*") if path.is_dir())
    if not decode_dirs:
        raise FileNotFoundError(f"decode directory not found under: {model_dir}")

    visual_dirs = sorted(
        path
        for path in model_dir.glob("vis*")
        if path.is_dir() and any(key in path.name for key in ("vision", "visual"))
    )
    if not visual_dirs:
        raise FileNotFoundError(f"visual directory not found under: {model_dir}")
    return prefill_dir, decode_dirs[0], visual_dirs


def visual_model_name(model_prefix: str, visual_dir: Path) -> str:
    if "_" not in visual_dir.name:
        return f"{model_prefix}_visual"
    return f"{model_prefix}_visual_{visual_dir.name.split('_', maxsplit=1)[1]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Ornith 1.0 models.")
    parser.add_argument(
        "--config",
        "--config_path",
        dest="config_path",
        default=str(DEFAULT_CONFIG_PATH),
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model_dir",
        default=str(MODEL_ROOT / "output" / HOUMO_TARGET / "hmquant"),
        help="hmquant model directory",
    )
    parser.add_argument("--model_name", default=None, help="output model name")
    parser.add_argument("--model_size", default=None, help="output model size")
    parser.add_argument("--ndevice", type=int, default=None, help="device number")
    parser.add_argument("--ncore", type=int, default=None, help="core number")
    parser.add_argument("--batch", type=int, default=None, help="decode batch size")
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--prefill_length", type=int, default=None)
    parser.add_argument(
        "--j",
        type=int,
        default=max(1, int(multiprocessing.cpu_count() * 0.7)),
        help="parallel build jobs",
    )
    parser.add_argument("--stage", choices=["build", "test", "all"], default="build")
    parser.add_argument(
        "--output_dir",
        default=str(MODEL_ROOT / "output" / HOUMO_TARGET),
        help="compiled output directory",
    )
    parser.add_argument(
        "--flash_attention",
        nargs=2,
        type=int,
        default=(2, 2),
        metavar=("LLM", "VIT"),
    )
    parser.add_argument("--enable_common_subgraph", action="store_true")
    parser.add_argument("--enable_xh2_stable_output", action="store_true")
    args = parser.parse_args()

    args.model_name, args.model_size, model_config = load_model_config(
        args.config_path, args.model_name, args.model_size
    )
    args.ncore = args.ncore or int(model_config.get("ncore", 2))
    args.ndevice = args.ndevice or int(model_config.get("ndevice", 1))
    args.batch = args.batch or int(model_config.get("batch", 1))
    args.prefill_length = args.prefill_length or int(
        model_config.get("prefill_length", 256)
    )
    if args.context_length is None:
        args.context_length = parse_context_length(
            model_config.get("context_length", "256k")
        )

    llm_flash, vit_flash = args.flash_attention
    if llm_flash not in (0, 1, 2) or vit_flash not in (0, 1, 2):
        raise ValueError("flash_attention expects LLM in {0,1,2} and VIT in {0,1,2}")
    if args.context_length < 2048:
        llm_flash = 0
    args.flash_attention = (llm_flash, vit_flash)
    return args


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_prefix = f"{args.model_name}-{args.model_size}"
    prefill_dir, decode_dir, visual_dirs = discover_model_dirs(model_dir)
    llm_flash, vit_flash = args.flash_attention

    if args.stage in ("build", "all"):
        assert get_platform() == "x86_64", "Compilation only supports x86_64"
        output_dir.mkdir(parents=True, exist_ok=True)
        build_tasks = [
            {
                "is_prefill": True,
                "hmonnx": find_hmonnx_file(str(prefill_dir)),
                "hmm_name": f"{model_prefix}_prefill",
                "flash_attn": llm_flash,
                "context_length": args.context_length,
                "prefill_length": args.prefill_length,
                "ndevice": args.ndevice,
                "enable_common_subgraph": args.enable_common_subgraph,
                "enable_xh2_stable_output": args.enable_xh2_stable_output,
                "llm_opt": True,
            },
            {
                "hmonnx": find_hmonnx_file(str(decode_dir)),
                "hmm_name": f"{model_prefix}_decode",
                "llm_batch": args.batch,
                "flash_attn": llm_flash,
                "context_length": args.context_length,
                "ndevice": args.ndevice,
                "enable_xh2_stable_output": args.enable_xh2_stable_output,
                "llm_opt": True,
            },
        ]
        build_tasks.extend(
            {
                "hmonnx": find_hmonnx_file(str(visual_dir)),
                "hmm_name": visual_model_name(model_prefix, visual_dir),
                "flash_attn": vit_flash,
            }
            for visual_dir in visual_dirs
        )
        for build_kwargs in build_tasks:
            print(f'\n===> {build_kwargs["hmm_name"]} build start...')
            Xh2Exec.build_from_hmonnx(
                output=str(output_dir),
                ncore=args.ncore,
                parallel_jobs=args.j,
                **build_kwargs,
            )

    print("\n=== Build flow finished. ===")


if __name__ == "__main__":
    main()
