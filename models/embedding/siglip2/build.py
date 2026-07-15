#!/usr/bin/env python3
# Copyright 2025 HOUMO AI
#
# File: build.py
# Description:
#   Build SigLIP2 HMM artifacts from custom PTQ hmonnx outputs.
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
"""Build SigLIP2 HMM artifacts from custom PTQ hmonnx outputs."""

import argparse
import os

from hmatc.exec.xh2_exec import Xh2Exec
from hmatc.utils.utils import (
    find_hmonnx_file,
    first_not_none,
    get_model_configs,
    get_platform,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_MODEL_DIR = os.path.join("output", HOUMO_TARGET, "hmquant")
DEFAULT_OUTPUT_DIR = os.path.join("output", HOUMO_TARGET)


def get_args() -> argparse.Namespace:
    """Parse command line arguments and fill defaults from config.yaml."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help="path to the hmonnx model directory",
    )
    parser.add_argument("--model_name", default=None, help="output houmo model name")
    parser.add_argument("--model_size", default=None, help="output houmo model size")
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="output directory for built models",
    )
    parser.add_argument("--ncore", type=int, default=None, help="core number")
    parser.add_argument(
        "--ndevice", type=int, default=None, help="device number for multi-device"
    )
    parser.add_argument(
        "--j",
        dest="parallel_jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="build parallel jobs",
    )
    parser.add_argument(
        "--stage", default="build", choices=["build", "all"], help="build stage"
    )
    parser.add_argument(
        "--flash_attention",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="FlashAttention mode",
    )
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ncore = first_not_none(args.ncore, model_config.get("ncore", 1))
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    return args


def build_one(args, suffix):
    """Build one SigLIP2 tower from its hmonnx file."""
    model_dir = os.path.join(args.model_dir, suffix)
    Xh2Exec.build_from_hmonnx(
        hmonnx=find_hmonnx_file(model_dir),
        hmm_name=f"{args.model_name}-{args.model_size}_{suffix}",
        output=args.output_dir,
        ncore=args.ncore,
        ndevice=args.ndevice,
        flash_attn=args.flash_attention,
        parallel_jobs=args.parallel_jobs,
        target=HOUMO_TARGET,
        cpp_backend="v2",
    )


def main():
    args = get_args()
    if get_platform() != "x86_64":
        raise RuntimeError("Only supported for compilation on the x86_64 platform.")

    build_one(args, "text")
    build_one(args, "vision")


if __name__ == "__main__":
    main()
