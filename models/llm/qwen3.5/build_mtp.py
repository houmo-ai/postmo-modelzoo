# Copyright (c) 2025 HOUMO AI
#
# File: build_mtp.py
# Description:
#   Qwen3.5 MTP model build tool - Python script for compiling the
#   target prefill/decode models and MTP draft prefill/decode models.
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
import logging
import multiprocessing
import os

from hmatc.exec.xh2_exec import Xh2Exec
from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import (
    find_hmonnx_file,
    first_not_none,
    get_model_configs,
    get_platform,
    parse_context_length,
)

logging.basicConfig(level="INFO")

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = os.getenv("HOUMO_CORE_NUM", 2)
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="path to the exported MTP onnx directory",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        help="base output houmo model name",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size",
    )
    parser.add_argument(
        "--draft_suffix",
        dest="draft_suffix",
        type=str,
        default="mtp",
        help="suffix used for draft hmm model names",
    )
    parser.add_argument(
        "--j",
        dest="j",
        type=int,
        default=multiprocessing.cpu_count(),
        help="build parallel jobs",
    )
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=None,
        help="core number",
    )
    parser.add_argument(
        "--context_length",
        dest="context_length",
        type=int,
        default=None,
        help="context length used for target prefill/decode compilation",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
        help="build output dir",
    )
    parser.add_argument(
        "--prefill_length",
        dest="prefill_length",
        type=int,
        default=None,
        help="prefill_length for target prefill model",
    )
    parser.add_argument(
        "--flash_attention",
        dest="flash_attention",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="flash attention optimization for target prefill/decode models",
    )
    parser.add_argument(
        "--stage",
        dest="stage",
        type=str,
        default="build",
        choices=["build", "test", "all"],
        help="build stage",
    )
    parser.add_argument(
        "--monitor_interval",
        dest="monitor_interval",
        type=float,
        default=1.0,
        help="memory monitor interval in seconds",
    )

    args = parser.parse_args()
    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ncore = first_not_none(args.ncore, model_config.get("ncore", HOUMO_CORE_NUM))
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    args.prefill_length = first_not_none(
        args.prefill_length, model_config.get("prefill_length", 256)
    )
    if args.context_length is None:
        args.context_length = parse_context_length(
            model_config.get("context_length", "256k")
        )
    if args.context_length < 2048:
        args.flash_attention = 0
    return args


def _first_existing_dir(model_dir: str, names: list[str]) -> str:
    for name in names:
        path = os.path.join(model_dir, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Directory not found under {model_dir}: {' or '.join(names)}"
    )


if __name__ == "__main__":
    args = get_args()
    print(args)

    if get_platform() != "x86_64":
        print(f"[error] tcim not support platform: {get_platform()}")
        raise SystemExit(0)

    model_name = args.model_name
    model_size = args.model_size
    model_dir = os.path.abspath(args.model_dir)
    prefill_dir = _first_existing_dir(model_dir, ["prefill"])
    decode_dir = _first_existing_dir(model_dir, ["decode"])
    draft_prefill_dir = _first_existing_dir(
        model_dir, ["mtp_draft_prefill", "draft_prefill"]
    )
    draft_decode_dir = _first_existing_dir(
        model_dir, ["mtp_draft_decode", "draft_decode"]
    )

    with ProcessMemoryMonitor(interval=args.monitor_interval, quiet=True) as monitor:
        if args.stage == "build" or args.stage == "all":
            Xh2Exec.build_from_hmonnx(
                is_prefill=True,
                hmonnx=find_hmonnx_file(prefill_dir),
                hmm_name=f"{model_name}-{model_size}_prefill",
                output=args.output_dir,
                ncore=args.ncore,
                flash_attn=args.flash_attention,
                context_length=args.context_length,
                prefill_length=args.prefill_length,
                ndevice=args.ndevice,
                llm_opt=True,
                parallel_jobs=args.j,
            )
            Xh2Exec.build_from_hmonnx(
                hmonnx=find_hmonnx_file(decode_dir),
                hmm_name=f"{model_name}-{model_size}_decode",
                output=args.output_dir,
                ncore=args.ncore,
                flash_attn=args.flash_attention,
                context_length=args.context_length,
                prefill_length=None,
                ndevice=args.ndevice,
                llm_opt=True,
                parallel_jobs=args.j,
            )
            Xh2Exec.build_from_hmonnx(
                is_prefill=True,
                hmonnx=find_hmonnx_file(draft_prefill_dir),
                hmm_name=f"{model_name}-{model_size}_prefill_{args.draft_suffix}",
                output=args.output_dir,
                ncore=args.ncore,
                flash_attn=args.flash_attention,
                context_length=args.context_length,
                ndevice=args.ndevice,
                parallel_jobs=args.j,
            )
            Xh2Exec.build_from_hmonnx(
                hmonnx=find_hmonnx_file(draft_decode_dir),
                hmm_name=f"{model_name}-{model_size}_decode_{args.draft_suffix}",
                output=args.output_dir,
                ncore=args.ncore,
                flash_attn=args.flash_attention,
                context_length=args.context_length,
                ndevice=args.ndevice,
                parallel_jobs=args.j,
            )

    print(
        f"\n=== Build flow finished. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )
