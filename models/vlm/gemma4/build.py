# Copyright (c) 2025 HOUMO AI
#
# File: build.py
# Description:
#   Gemma-4-26B-A4B Model Build Tool - Python script for building
#   Gemma-4-26B-A4B models using Xh2Exec.
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
# fmt: off
import os
import argparse
import psutil
from pathlib import Path

from hmatc.exec.xh2_exec import Xh2Exec
from hmatc.utils.monitor import ProcessMemoryMonitor

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
HOUMO_CORE_NUM = int(os.getenv("HOUMO_CORE_NUM", 2))


def find_onnx_file(model_dir: str | Path, subdir: str, pattern: str) -> Path:
    """Find ONNX file in model directory.

    Args:
        model_dir: Base model directory.
        subdir: Subdirectory name (e.g., 'prefill', 'decode', 'visual').
        pattern: Glob pattern to match ONNX file (e.g., '*_prefill_with_act.onnx').

    Returns:
        Path to the found ONNX file.

    Raises:
        FileNotFoundError: If no matching ONNX file is found.
    """
    model_path = Path(model_dir)
    search_dir = model_path / subdir

    if not search_dir.exists():
        raise FileNotFoundError(f"Directory not found: {search_dir}")

    matches = list(search_dir.glob(pattern))
    if not matches:
        # Fallback: try without '_with_act' suffix
        fallback_pattern = pattern.replace("_with_act", "")
        matches = list(search_dir.glob(fallback_pattern))

    if not matches:
        raise FileNotFoundError(
            f"No ONNX file found in {search_dir} matching {pattern}"
        )

    # Return first match
    return str(matches[0])


def _validate_adjust_flash_attention(flash_vals: tuple, context_length: int) -> tuple:
    """Validates and adjusts FlashAttention parameter values."""
    llm_val, vit_val = flash_vals

    # Validate LLM (Prefill & Decode) FlashAttention parameter
    # Values: 0=off, 1/2=on
    if llm_val not in [0, 1, 2]:
        raise ValueError(
            f"Prefill&Decode FlashAttention values only support 0/1/2, current value:{llm_val}"
        )

    # Validate ViT (Vision Transformer) FlashAttention parameter
    # Values: 0=off, 1=on
    if vit_val not in [0, 1]:
        raise ValueError(
            f"ViT FlashAttention values only support 0/1, current value:{vit_val}"
        )

    if context_length < 2048:
        llm_val = 0

    return (llm_val, vit_val)

def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="output/xh2/hmquant", help="path to the model directory")
    parser.add_argument("--model_name", type=str, default="gemma4", help="output houmo model name")
    parser.add_argument("--model_size", type=str, default="26b-a4b", help="output houmo model size")
    parser.add_argument("--output_dir", type=str, default=f"output/{HOUMO_TARGET}", help="output directory for built models")
    parser.add_argument("--ncore", type=int, default=HOUMO_CORE_NUM, help="core number")
    parser.add_argument("--j", dest="parallel_jobs", type=int, default=psutil.cpu_count(logical=False), help="build parallel jobs")
    parser.add_argument("--context_length", type=int, default=2048, help="context length for LLM models")
    parser.add_argument("--prefill_length", type=int, default=256, help="prefill length for prefill model")
    parser.add_argument("--batch", type=int, default=1, help="batch size for decode model")
    parser.add_argument("--ndevice", type=int, default=1, help="device number for multi-device")
    parser.add_argument("--opt_level", type=int, default=2, help="optimization level")
    parser.add_argument("--stage", type=str, default="all", choices=["prefill", "decode", "visual", "all"], help="build stage")
    parser.add_argument("--monitor_interval", type=float, default=2.0, help="memory monitor interval in seconds")
    parser.add_argument(
        "--flash_attention",
        dest="flash_attention",
        nargs=2,
        type=int,
        default=(2, 1),
        help="FlashAttention optimization switches: "
        "1st int = prefill/decode model switch (0=off, 1/2=on), "
        "2nd int = ViT model switch (0=off, 1=on); "
        "e.g., --flash_attention 2 1 (prefill&decode=2, ViT=1)",
    )
    args = parser.parse_args()
    args.flash_attention = _validate_adjust_flash_attention(
        args.flash_attention, args.context_length
    )
    return args


def build_prefill(
    model_dir: str,
    model_name: str,
    output_dir: str,
    ncore: int,
    parallel_jobs: int,
    flash_attn: int,
    context_length: int,
    prefill_length: int,
    ndevice: int,
    opt_level: int,
):
    """Build prefill model."""
    hmonnx = find_onnx_file(model_dir, "prefill", "hmquant_*.onnx")

    print(f"\n===> Building prefill model...")
    print(f"  hmonnx: {hmonnx}")
    print(f"  ncore: {ncore}")
    print(f"  flash_attn: {flash_attn}")
    print(f"  context_length: {context_length}")
    print(f"  prefill_length: {prefill_length}")

    Xh2Exec.build_from_hmonnx(
        hmonnx=hmonnx,
        hmm_name=f"{model_name}_prefill",
        output=output_dir,
        ncore=ncore,
        opt_level=opt_level,
        batch=1,
        flash_attn=flash_attn,
        llm_opt=True,
        context_length=context_length,
        prefill_length=prefill_length,
        ndevice=ndevice,
        is_prefill=True,
        parallel_jobs=parallel_jobs,
    )
    print(f"<=== prefill build completed.\n")


def build_decode(
    model_dir: str,
    model_name: str,
    output_dir: str,
    ncore: int,
    parallel_jobs: int,
    flash_attn: int,
    context_length: int,
    batch: int,
    ndevice: int,
    opt_level: int,
):
    """Build decode model."""
    hmonnx = find_onnx_file(model_dir, "decode", "hmquant_*.onnx")

    print(f"\n===> Building decode model...")
    print(f"  hmonnx: {hmonnx}")
    print(f"  ncore: {ncore}")
    print(f"  flash_attn: {flash_attn}")
    print(f"  context_length: {context_length}")
    print(f"  batch: {batch}")

    Xh2Exec.build_from_hmonnx(
        hmonnx=hmonnx,
        hmm_name=f"{model_name}_decode",
        output=output_dir,
        ncore=ncore,
        opt_level=opt_level,
        batch=batch,
        flash_attn=flash_attn,
        llm_opt=True,
        context_length=context_length,
        ndevice=ndevice,
        is_prefill=False,
        parallel_jobs=parallel_jobs,
    )
    print(f"<=== decode build completed.\n")


def build_visual(
    model_dir: str,
    model_name: str,
    output_dir: str,
    ncore: int,
    parallel_jobs: int,
    flash_attn: int, 
    opt_level: int,
):
    """Build visual model."""
    hmonnx = find_onnx_file(model_dir, "visual", "hmquant_*.onnx")

    print(f"\n===> Building visual model...")
    print(f"  hmonnx: {hmonnx}")
    print(f"  ncore: {ncore}")
    print(f"  flash_attn: {flash_attn}")

    Xh2Exec.build_from_hmonnx(
        hmonnx=hmonnx,
        hmm_name=f"{model_name}_visual",
        output=output_dir,
        ncore=ncore,
        opt_level=opt_level,
        batch=1,
        flash_attn=flash_attn,
        llm_opt=True,
        parallel_jobs=parallel_jobs,
    )
    print(f"<=== visual build completed.\n")


if __name__ == "__main__":
    args = get_args()
    print(args)

    model_dir = args.model_dir
    output_dir = args.output_dir
    ncore = args.ncore
    parallel_jobs = args.parallel_jobs
    flash_attn, visual_flash_attn = args.flash_attention
    context_length = args.context_length
    prefill_length = args.prefill_length
    batch = args.batch
    ndevice = args.ndevice
    opt_level = args.opt_level
    with ProcessMemoryMonitor(interval=args.monitor_interval, quiet=True) as monitor:
        if args.stage in ["prefill", "all"]:
            build_prefill(
                model_dir=model_dir,
                model_name=f"{args.model_name}-{args.model_size}",
                output_dir=output_dir,
                ncore=ncore,
                parallel_jobs=parallel_jobs,
                flash_attn=flash_attn,
                context_length=context_length,
                prefill_length=prefill_length,
                ndevice=ndevice,
                opt_level=opt_level,
            )

        if args.stage in ["decode", "all"]:
            build_decode(
                model_dir=model_dir,
                model_name=f"{args.model_name}-{args.model_size}",
                output_dir=output_dir,
                ncore=ncore,
                parallel_jobs=parallel_jobs,
                flash_attn=flash_attn,
                context_length=context_length,
                batch=batch,
                ndevice=ndevice,
                opt_level=opt_level,
            )

        if args.stage in ["visual", "all"]:
            build_visual(
                model_dir=model_dir,
                model_name=f"{args.model_name}-{args.model_size}",
                output_dir=output_dir,
                ncore=ncore,
                parallel_jobs=parallel_jobs,
                flash_attn=visual_flash_attn,
                opt_level=opt_level,
            )

    print(f"\n=== All builds completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ===")
