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
import glob
import json
import logging
import multiprocessing
import os
import platform
import time

logging.basicConfig(level="INFO")

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = os.getenv("HOUMO_CORE_NUM", 2)
DEFAULT_MODEL_DIR = "/data02/datasets/qwen3_5_9b_mtp_k4_w4a8_8k"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="path to the exported MTP onnx directory",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="qwen3.5",
        help="base output houmo model name",
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
        default=HOUMO_CORE_NUM,
        help="core number",
    )
    parser.add_argument(
        "--context_length",
        dest="context_length",
        type=int,
        default=8192,
        help="context length used for target prefill/decode compilation",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=1,
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
        default=256,
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

    args = parser.parse_args()
    if args.context_length < 2048:
        args.flash_attention = 0
    return args


def _find_single_onnx(model_dir: str, pattern: str = "*.onnx") -> str:
    onnx_files = sorted(
        path
        for path in glob.glob(os.path.join(model_dir, pattern))
        if os.path.isfile(path)
    )
    if len(onnx_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one ONNX matching {pattern} under {model_dir}, found {len(onnx_files)}: {onnx_files}"
        )
    return os.path.abspath(onnx_files[0])


def build_hmonnx(
    model_name: str,
    onnx_path: str,
    output_dir: str,
    ncore: int,
    j: int,
    ndevice: int,
    llm_opt: bool = False,
    flash_attention: int = 0,
    context_length: int = 0,
    prefill_length: int = 0,
):
    import tcim

    kwargs = {}
    custom_msg = {}

    if llm_opt:
        kwargs["modify_llm"] = {}
        if context_length:
            kwargs["modify_llm"]["context-length"] = context_length
            custom_msg["context_length"] = context_length
        if prefill_length:
            kwargs["modify_llm"]["fill-length"] = prefill_length
            custom_msg["prefill_length"] = prefill_length
        if flash_attention:
            kwargs["flash_attention"] = flash_attention
            custom_msg["flash_attention"] = flash_attention

    if ndevice:
        kwargs["ndevice"] = ndevice
        custom_msg["ndevice"] = ndevice

    kwargs["custom_msg"] = json.dumps(custom_msg, ensure_ascii=False)

    start = time.time()
    print(
        f"\n===> {model_name} build start..."
        f"\n onnx: {onnx_path}"
        f"\n kwargs: {kwargs}"
    )
    tcim.build_from_hmonnx(
        onnx_path,
        output_name=model_name,
        ncore=ncore,
        target=HOUMO_TARGET,
        output_dir=output_dir,
        work_dir=os.path.join(output_dir, "tcim", model_name),
        llm_opt=llm_opt,
        j=j,
        **kwargs,
    )
    elapsed = time.time() - start
    print(f"{model_name} build completed in {elapsed:.3f} s.", flush=True)


if __name__ == "__main__":
    args = get_args()
    print(args)

    if platform.machine() != "x86_64":
        print(f"[error] tcim not support platform: {platform.machine()}")
        raise SystemExit(0)

    model_dir = os.path.abspath(args.model_dir)
    prefill_dir = os.path.join(model_dir, "prefill_onnx")
    decode_dir = os.path.join(model_dir, "decode_onnx")
    draft_dir = os.path.join(model_dir, "draft_onnx")

    for required_dir in [prefill_dir, decode_dir, draft_dir]:
        if not os.path.isdir(required_dir):
            raise FileNotFoundError(f"Directory not found: {required_dir}")

    prefill_onnx = _find_single_onnx(prefill_dir)
    decode_onnx = _find_single_onnx(decode_dir)
    draft_prefill_onnx = _find_single_onnx(draft_dir, "*prefill*.onnx")
    draft_decode_onnx = _find_single_onnx(draft_dir, "*decode*.onnx")

    build_hmonnx(
        f"{args.model_name}_prefill",
        prefill_onnx,
        args.output_dir,
        args.ncore,
        args.j,
        args.ndevice,
        llm_opt=True,
        flash_attention=args.flash_attention,
        context_length=args.context_length,
        prefill_length=args.prefill_length,
    )
    build_hmonnx(
        f"{args.model_name}_decode",
        decode_onnx,
        args.output_dir,
        args.ncore,
        args.j,
        args.ndevice,
        llm_opt=True,
        flash_attention=args.flash_attention,
        context_length=args.context_length,
    )
    build_hmonnx(
        f"{args.model_name}_{args.draft_suffix}_prefill",
        draft_prefill_onnx,
        args.output_dir,
        args.ncore,
        args.j,
        args.ndevice,
    )
    build_hmonnx(
        f"{args.model_name}_{args.draft_suffix}_decode",
        draft_decode_onnx,
        args.output_dir,
        args.ncore,
        args.j,
        args.ndevice,
    )
