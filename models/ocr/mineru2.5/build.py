# Copyright (c) 2025 HOUMO AI
#
# File: build.py
# Description:
#   Gemma4 Model Build Tool - Python script for building
#   Gemma4 models using Xh2Exec.
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
import psutil
import shutil
from hmatc.exec.xh2_exec import Xh2Exec
from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import (
    find_hmonnx_file,
    first_not_none,
    get_model_configs,
    get_platform,
    parse_context_length,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = int(os.getenv("HOUMO_CORE_NUM", 2))
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def _validate_adjust_flash_attention(flash_vals: tuple, context_length: int) -> tuple:
    llm_val, vit_val = flash_vals

    if llm_val not in [0, 1, 2]:
        raise ValueError(
            f"Prefill&Decode FlashAttention values only support 0/1/2, current value:{llm_val}"
        )

    if vit_val not in [0, 1]:
        raise ValueError(
            f"ViT FlashAttention values only support 0/1, current value:{vit_val}"
        )

    if context_length < 2048:
        llm_val = 0

    return (llm_val, vit_val)


def get_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=os.path.join("output", HOUMO_TARGET, "hmquant"), help="path to the model directory")
    parser.add_argument("--model_name", dest="model_name", type=str, default=None, help="output houmo model name")
    parser.add_argument("--model_size", dest="model_size", type=str, default=None, help="output houmo model size")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=os.path.join("output", HOUMO_TARGET), help="output directory for built models")
    parser.add_argument("--ncore", dest="ncore", type=int, default=None, help="core number")
    parser.add_argument("--j", dest="parallel_jobs", type=int, default=psutil.cpu_count(logical=False), help="build parallel jobs")
    parser.add_argument("--context_length", dest="context_length", type=int, default=None, help="context length for llm models")
    parser.add_argument("--prefill_length", dest="prefill_length", type=int, default=None, help="prefill length for prefill model")
    parser.add_argument("--batch", dest="batch", type=int, default=None, help="batch size for decode model")
    parser.add_argument("--max_size_w", type=int, default=None)
    parser.add_argument("--max_size_h", type=int, default=None)
    parser.add_argument("--ndevice", dest="ndevice", type=int, default=None, help="device number for multi-device")
    parser.add_argument("--stage", dest="stage", type=str, default="build", choices=["build", "test", "all"], help="build stage")
    parser.add_argument("--flash_attention", dest="flash_attention", nargs=2, type=int, default=(2, 1), help="FlashAttention switches: 1st=llm(0/1/2), 2nd=vit(0/1); e.g. --flash_attention 2 1")
    parser.add_argument("--enable_common_subgraph", dest="enable_common_subgraph", action="store_true", default=False, help="enable common subgraph optimization")

    args = parser.parse_args()
    default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ncore = first_not_none(args.ncore, model_config.get("ncore", HOUMO_CORE_NUM))
    args.batch = first_not_none(args.batch, model_config.get("batch", 1))
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    args.max_size_w = first_not_none(args.max_size_w, model_config.get("max_size_w", 1036))
    args.max_size_h = first_not_none(args.max_size_h, model_config.get("max_size_h", 1036))
    args.prefill_length = first_not_none(args.prefill_length, model_config.get("prefill_length", 256))
    if args.context_length is None:
        args.context_length = parse_context_length(model_config.get("context_length", "32k"))
    args.flash_attention = _validate_adjust_flash_attention(args.flash_attention, args.context_length)
    # fmt: on
    return args


if __name__ == "__main__":
    args = get_args()

    model_dir = args.model_dir
    model_name = args.model_name
    model_size = args.model_size
    output_dir = args.output_dir
    ncore = args.ncore
    ndevice = args.ndevice
    parallel_jobs = args.parallel_jobs
    llm_flash_attn, flash_attn = args.flash_attention

    if args.stage in ["build", "all"]:
        assert (
            get_platform() == "x86_64"
        ), "Only supported for compilation on the x86_64 platform."
        Xh2Exec.build_from_hmonnx(
            is_prefill=True,
            hmonnx=find_hmonnx_file(os.path.join(model_dir, "prefill")),
            hmm_name=f"{model_name}-{model_size}_prefill",
            output=output_dir,
            ncore=ncore,
            llm_batch=1,
            flash_attn=llm_flash_attn,
            llm_opt=True,
            context_length=args.context_length,
            prefill_length=args.prefill_length,
            ndevice=ndevice,
            parallel_jobs=parallel_jobs,
        )
        Xh2Exec.build_from_hmonnx(
            hmonnx=find_hmonnx_file(os.path.join(model_dir, "decode")),
            hmm_name=f"{model_name}-{model_size}_decode",
            output=output_dir,
            ncore=ncore,
            llm_batch=args.batch,
            flash_attn=llm_flash_attn,
            llm_opt=True,
            context_length=args.context_length,
            ndevice=ndevice,
            parallel_jobs=parallel_jobs,
        )
        visual_model_name = (
            f"{model_name}-{model_size}_visual_{args.max_size_w}x{args.max_size_h}"
        )
        Xh2Exec.build_from_hmonnx(
            hmonnx=find_hmonnx_file(
                os.path.join(model_dir, f"visual_{args.max_size_w}x{args.max_size_h}")
            ),
            hmm_name=visual_model_name,
            output=output_dir,
            ncore=ncore,
            flash_attn=flash_attn,
            parallel_jobs=parallel_jobs,
            enable_common_subgraph=args.enable_common_subgraph,
        )
        visual_buckets = os.path.join(model_dir, "visual_buckets")
        if not os.path.exists(visual_buckets):
            exit(1)
        buckets = os.listdir(visual_buckets)
        for bucket in buckets:
            try:
                visual_model_name = f"{model_name}-{model_size}_visual_{bucket}"
                Xh2Exec.build_from_hmonnx(
                    hmonnx=find_hmonnx_file(os.path.join(visual_buckets, bucket)),
                    hmm_name=visual_model_name,
                    output=output_dir,
                    ncore=ncore,
                    flash_attn=flash_attn,
                    parallel_jobs=parallel_jobs,
                    enable_common_subgraph=args.enable_common_subgraph,
                )
            except Exception as e:
                print(f"Error occurred while processing bucket {bucket}: {e}")
