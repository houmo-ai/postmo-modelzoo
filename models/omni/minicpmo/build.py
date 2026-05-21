#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: build.py
# Description:
#   minicpmo model compilation and consistency comparison.
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
import numpy as np
import time
import argparse
import multiprocessing

from hmatc.exec.xh2_exec import Xh2Exec
from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import (
    find_hmonnx_file,
    first_not_none,
    get_model_configs,
    get_platform,
    parse_context_length,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = int(os.getenv("HOUMO_CORE_NUM", 2))
GOLDEN_THRESH = 0.98
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def sanitize_name(name: str):
    return name.replace(":", "_").replace("/", "_")


def cosine_distance(data1, data2):
    if data1.shape != data2.shape:
        print(f"[error] shape not equal {data1.shape} vs {data2.shape}")
        return -1
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)
    cosine_dist = np.dot(v1_norm, v2_norm)
    if np.isnan(cosine_dist):
        return -1
    return cosine_dist


def _validate_adjust_flash_attention(flash_vals: tuple, context_length: int) -> tuple:
    """Validates and adjusts FlashAttention parameter values."""
    llm_val, other_val = flash_vals

    # Validate LLM (Prefill & Decode) FlashAttention parameter
    # Values: 0=off, 1/2=on
    if llm_val not in [0, 1, 2]:
        raise ValueError(
            f"Prefill&Decode FlashAttention values only support 0/1/2, current value:{llm_val}"
        )

    # Validate ViT & Audio Models FlashAttention parameter
    # Values: 0=off, 1=on
    if other_val not in [0, 1]:
        raise ValueError(
            f"ViT&Audio FlashAttention values only support 0/1, current value:{other_val}"
        )

    if context_length < 2048:
        llm_val = 0

    return (llm_val, other_val)


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
        help="path to the model dir",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        help="output houmo model name",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=None,
        help="batch size",
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
        type=str,
        default=None,
        help="context_length",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size",
    )
    parser.add_argument(
        "--j",
        dest="j",
        type=int,
        default=multiprocessing.cpu_count(),
        help="build parallel jobs",
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
        help="prefill_length",
    )
    parser.add_argument(
        "--enable_xh2_stable_output",
        dest="enable_xh2_stable_output",
        action="store_true",
        help="enable xh2 stable output",
    )
    parser.add_argument(
        "--flash_attention",
        dest="flash_attention",
        nargs=2,
        type=int,
        default=(2, 0),
        help="FlashAttention optimization switches: "
        "1st int = Prefill/Decode model switch (0=off, 1/2=on), "
        "2nd int = ViT/Audio model switch (0=off, 1=on); "
        "e.g., --flash_attention 2 0 (prefill&decode=2, ViT&Audio=0)",
    )

    args = parser.parse_args()
    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.batch = first_not_none(args.batch, model_config.get("batch", 1))
    args.ncore = first_not_none(args.ncore, model_config.get("ncore", HOUMO_CORE_NUM))
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    args.context_length = parse_context_length(
        first_not_none(args.context_length, model_config.get("context_length", "4k"))
    )
    args.prefill_length = first_not_none(
        args.prefill_length, model_config.get("prefill_length", 256)
    )
    args.flash_attention = _validate_adjust_flash_attention(
        args.flash_attention, args.context_length
    )
    return args


def test(model_name, model_dir, output_dir, profile, batch=1, prefix=None):
    import tcim_lite

    print(f"\n===> {model_name} test start...")
    # load model
    model_path = os.path.join(output_dir, f"{model_name}.hmm")
    start = time.time()
    print(model_path)
    module = tcim_lite.runtime.load(model_path)
    profile["load"] = time.time() - start
    print(f'{model_name} load completed in {profile["load"]:.3f} s.', flush=True)

    # set input
    profile["set_input"] = 0
    if prefix is None:
        prefix = model_name
    input_num = module.get_num_inputs()
    for id in range(input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name)
        print(
            f"input_info[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
        )
        input_data_path = os.path.join(
            model_dir, f"hmquant_{prefix}_{sanitize_name(input_name)}_input.npy"
        )
        input_data = np.load(input_data_path).astype(input_info.dtype)
        input_data = np.concatenate([input_data for i in range(batch)], axis=0)
        print(
            f"golden input[{input_name}] shape = {input_data.shape}, dtype = {input_data.dtype}"
        )
        start = time.time()
        module.set_input(input_name, input_data)
        profile["set_input"] += time.time() - start
    print(
        f'{model_name} set {input_num} inputs completed in {profile["set_input"]*1000:.3f} ms.'
    )

    # infer model
    start = time.time()
    module.run()
    module.sync()
    profile["infer"] = time.time() - start
    print(f'{model_name} infer completed in {profile["infer"]*1000:.3f} ms.')

    # get output and compare with golden
    profile["get_output"] = 0
    result_check = True
    output_num = module.get_num_outputs()
    for id in range(output_num):
        output_name = module.get_output_name(id)
        output_info = module.get_output_info(output_name)
        print(
            f"output_info[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}"
        )
        start = time.time()
        output_data = module.get_output(output_name).numpy()
        profile["get_output"] += time.time() - start
        print(
            f"output[{output_name}] shape = {output_data.shape}, dtype = {output_data.dtype}"
        )
        output_data_path = os.path.join(
            model_dir, f"hmquant_{prefix}_{sanitize_name(output_name)}_output.npy"
        )
        if os.path.exists(output_data_path):
            golden_output = np.load(output_data_path)
            golden_output = np.concatenate(
                [golden_output for i in range(batch)], axis=0
            )
        else:
            result_check = False
            print(
                f"[warning] compare canceled while golden data not found -> {output_data_path}"
            )
            continue
        if golden_output.shape == output_data.shape:
            cosine_dist = cosine_distance(golden_output, output_data)
            is_match = (golden_output == output_data).all()
            print(
                f"[compare] golden output [{output_name}] match={is_match}, similarity={cosine_dist:.6f}"
            )
            if is_match:
                continue
            if cosine_dist < GOLDEN_THRESH:
                result_check = False
        else:
            result_check = False
            print(
                f"[compare] golden output [{output_name}] shape not match {golden_output.shape} vs {output_data.shape}"
            )
    print(
        f'{model_name} get {output_num} ouputs completed in {profile["get_output"]*1000:.3f} ms.'
    )
    if not result_check:
        print("[error] result check failed.")
        exit(-1)
    print(f"<=== {model_name} test success.")


if __name__ == "__main__":
    args = get_args()
    print(args)
    model_dir = args.model_dir
    model_name = args.model_name
    output_dir = args.output_dir
    ncore = args.ncore
    batch = args.batch
    ndevice = args.ndevice
    context_length = args.context_length
    model_size = args.model_size
    j = args.j
    tso = args.enable_xh2_stable_output
    llm_flash_attention, other_flash_attention = args.flash_attention

    profile = {}

    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        if args.stage == "build" or args.stage == "all":
            assert (
                get_platform() == "x86_64"
            ), "Only supported for compilation on the x86_64 platform."

            Xh2Exec.build_from_hmonnx(
                is_prefill=True,
                hmonnx=find_hmonnx_file(os.path.join(model_dir, "prefill")),
                hmm_name=f"{model_name}-{model_size}_llm_prefill",
                output=output_dir,
                ncore=ncore,
                llm_opt=True,
                flash_attn=llm_flash_attention,
                context_length=context_length,
                prefill_length=args.prefill_length,
                ndevice=ndevice,
                enable_xh2_stable_output=tso,
                parallel_jobs=j,
            )
            Xh2Exec.build_from_hmonnx(
                hmonnx=find_hmonnx_file(os.path.join(model_dir, "decoder")),
                hmm_name=f"{model_name}-{model_size}_llm_decode",
                output=output_dir,
                ncore=ncore,
                llm_opt=True,
                flash_attn=llm_flash_attention,
                context_length=context_length,
                ndevice=ndevice,
                enable_xh2_stable_output=tso,
                parallel_jobs=j,
            )
            Xh2Exec.build_from_hmonnx(
                is_prefill=True,
                hmonnx=find_hmonnx_file(os.path.join(model_dir, "tts_prefill")),
                hmm_name=f"{model_name}-{model_size}_tts_prefill",
                output=output_dir,
                ncore=ncore,
                llm_opt=True,
                context_length=2048,
                ndevice=ndevice,
                enable_xh2_stable_output=tso,
                parallel_jobs=j,
            )
            Xh2Exec.build_from_hmonnx(
                hmonnx=find_hmonnx_file(os.path.join(model_dir, "tts_decoder")),
                hmm_name=f"{model_name}-{model_size}_tts_decode",
                output=output_dir,
                ncore=ncore,
                llm_opt=True,
                context_length=2048,
                ndevice=ndevice,
                enable_xh2_stable_output=tso,
                parallel_jobs=j,
            )
            Xh2Exec.build_from_hmonnx(
                hmonnx=find_hmonnx_file(os.path.join(model_dir, "visual")),
                hmm_name=f"{model_name}-{model_size}_visual",
                output=output_dir,
                ncore=ncore,
                flash_attn=other_flash_attention,
                enable_xh2_stable_output=tso,
                parallel_jobs=j,
            )
            Xh2Exec.build_from_hmonnx(
                hmonnx=find_hmonnx_file(os.path.join(model_dir, "audio")),
                hmm_name=f"{model_name}-{model_size}_audio",
                output=output_dir,
                ncore=ncore,
                flash_attn=other_flash_attention,
                enable_xh2_stable_output=tso,
                parallel_jobs=j,
            )
            Xh2Exec.build_from_hmonnx(
                hmonnx=find_hmonnx_file(
                    os.path.join(model_dir, "dvae"), pattern="hmquant_*part1*.onnx"
                ),
                hmm_name=f"{model_name}-{model_size}_dvae_part1",
                output=output_dir,
                ncore=ncore,
                flash_attn=other_flash_attention,
                enable_xh2_stable_output=tso,
                parallel_jobs=j,
            )
            Xh2Exec.build_from_hmonnx(
                hmonnx=find_hmonnx_file(
                    os.path.join(model_dir, "dvae"), pattern="hmquant_*part2*.onnx"
                ),
                hmm_name=f"{model_name}-{model_size}_dvae_part2",
                output=output_dir,
                ncore=ncore,
                flash_attn=other_flash_attention,
                enable_xh2_stable_output=tso,
                parallel_jobs=j,
            )
            Xh2Exec.build_from_hmonnx(
                hmonnx=find_hmonnx_file(os.path.join(model_dir, "vocos")),
                hmm_name=f"{model_name}-{model_size}_vocos",
                output=output_dir,
                ncore=ncore,
                flash_attn=other_flash_attention,
                enable_xh2_stable_output=tso,
                parallel_jobs=j,
            )

        if args.stage == "test" or args.stage == "all":
            part_dir = os.path.join(model_dir, "prefill")
            test(
                f"{model_name}-{model_size}_llm_prefill",
                part_dir,
                output_dir,
                profile,
                prefix=model_name,
            )
            part_dir = os.path.join(model_dir, "decoder")
            test(
                f"{model_name}-{model_size}_llm_decode",
                part_dir,
                output_dir,
                profile,
                prefix=model_name,
            )
            part_dir = os.path.join(model_dir, "visual")
            test(
                f"{model_name}-{model_size}_visual",
                part_dir,
                output_dir,
                profile,
                prefix=model_name,
            )
            part_dir = os.path.join(model_dir, "audio")
            test(
                f"{model_name}-{model_size}_audio",
                part_dir,
                output_dir,
                profile,
                prefix=model_name,
            )
            part_dir = os.path.join(model_dir, "tts_prefill")
            test(
                f"{model_name}-{model_size}_tts_prefill",
                part_dir,
                output_dir,
                profile,
                prefix=model_name,
            )
            part_dir = os.path.join(model_dir, "tts_decoder")
            test(
                f"{model_name}-{model_size}_tts_decode",
                part_dir,
                output_dir,
                profile,
                prefix=model_name,
            )
            part_dir = os.path.join(model_dir, "vocos")
            test(
                f"{model_name}-{model_size}_vocos",
                part_dir,
                output_dir,
                profile,
                prefix="vocos",
            )
            part_dir = os.path.join(model_dir, "dvae")
            test(
                f"{model_name}-{model_size}_dvae_part1",
                part_dir,
                output_dir,
                profile,
                prefix="dvae_part1",
            )
            test(
                f"{model_name}-{model_size}_dvae_part2",
                part_dir,
                output_dir,
                profile,
                prefix="dvae_part2",
            )

    print(
        f"\n=== All builds completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )
