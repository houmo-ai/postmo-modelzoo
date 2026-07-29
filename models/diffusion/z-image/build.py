# Copyright (c) 2025 HOUMO AI
#
# File: build.py
# Description:
#   Z-Image Model Build and Test Tool - Python script for building and testing
# Z-Image models (dit, vae, text encoder).
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
import glob
import multiprocessing
from loguru import logger

from hmatc.exec.xh2_exec import Xh2Exec
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


def get_default_parallel_jobs():
    return max(1, int(multiprocessing.cpu_count() * 0.75))


def sanitize_name(name: str):
    return name.replace(":", "_").replace("/", "_")


def cosine_distance(data1, data2):
    if data1.shape != data2.shape:
        logger.info(f"[error] shape not equal {data1.shape} vs {data2.shape}")
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


def validate_adjust_flash_attention(flash_vals: tuple, context_length: int) -> tuple:
    encoder_val, vae_val = flash_vals

    if encoder_val not in [0, 1, 2]:
        raise ValueError(
            f"DIT/text_encoder FlashAttention values only support 0/1/2, current value: {encoder_val}"
        )
    if vae_val not in [0, 1]:
        raise ValueError(
            f"VAE FlashAttention values only support 0/1, current value: {vae_val}"
        )
    if context_length < 2048:
        encoder_val = 0

    return encoder_val, vae_val


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=os.path.join("output", HOUMO_TARGET, "hmquant"), help="path to the model dir")
    parser.add_argument("--model_name", dest="model_name", type=str, default=None, help="output houmo model name")
    parser.add_argument("--model_size", dest="model_size", type=str, default=None, help="output houmo model size")
    parser.add_argument("--models", dest="models", type=str, nargs="+", default=None, choices=["dit", "vae", "text_encoder"], help="models to build or test")
    parser.add_argument("--j", dest="j", type=int, default=get_default_parallel_jobs(), help="build parallel jobs. Default is about 75%% of CPU count.")
    parser.add_argument("--ncore", dest="ncore", type=int, default=HOUMO_CORE_NUM, help="core number")
    parser.add_argument("--ndevice", dest="ndevice", type=int, default=None, help="device number")
    parser.add_argument("--stage", dest="stage", type=str, default="build", choices=["build", "test", "all"], help="build stage")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=os.path.join("output", HOUMO_TARGET), help="build output dir")
    parser.add_argument("--context_length", dest="context_length", type=int, default=None, help="context length")
    parser.add_argument("--prefill_length", dest="prefill_length", type=int, default=None, help="prefill length")
    parser.add_argument("--enable_common_subgraph", dest="enable_common_subgraph", action="store_true", default=False, help="enable common subgraph optimization")
    parser.add_argument("--enable_xh2_stable_output", dest="enable_xh2_stable_output", action="store_true", default=False, help="enable stable output")
    parser.add_argument("--flash_attention", dest="flash_attention", type=int, nargs=2, default=(2, 1), metavar=("DIT_TEXT_ENCODER", "VAE"), help="flash attention optimization switches. First value is used by dit/text_encoder (0/1/2), second value is used by vae (0/1).")
    args = parser.parse_args()
    # fmt: on

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ncore = first_not_none(args.ncore, model_config.get("ncore", HOUMO_CORE_NUM))
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    args.context_length = first_not_none(
        args.context_length,
        parse_context_length(model_config.get("context_length", "4k")),
    )
    args.prefill_length = first_not_none(
        args.prefill_length, model_config.get("prefill_length", 256)
    )
    args.flash_attention = validate_adjust_flash_attention(
        args.flash_attention, args.context_length
    )
    return args


def test(model_name, model_dir, output_dir, profile, batch=1):
    import tcim_lite

    logger.info(f"\n===> {model_name} test start...")
    # load model
    model_path = os.path.join(output_dir, f"{model_name}.hmm")
    start = time.time()
    module = tcim_lite.runtime.load(model_path)
    profile["load"] = time.time() - start
    logger.info(f'{model_name} load completed in {profile["load"]:.3f} s.', flush=True)

    # set input
    profile["set_input"] = 0
    input_num = module.get_num_inputs()
    for idx in range(input_num):
        input_name = module.get_input_name(idx)
        input_info = module.get_input_info(input_name)
        logger.info(
            f"input_info[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
        )
        input_files = sorted(
            glob.glob(
                f"{model_dir}/**/hmquant_*{sanitize_name(input_name)}*.npy",
                recursive=True,
            )
        )
        if not input_files:
            raise FileNotFoundError(
                f"Golden input not found for {input_name} under {model_dir}"
            )
        input_data_path = os.path.abspath(input_files[0])
        input_data = np.load(input_data_path).astype(input_info.dtype)
        input_data = np.concatenate([input_data] * batch, axis=0)
        logger.info(
            f"golden input[{input_name}] shape = {input_data.shape}, dtype = {input_data.dtype}"
        )
        start = time.time()
        module.set_input(input_name, input_data)
        profile["set_input"] += time.time() - start
    logger.info(
        f'{model_name} set {input_num} inputs completed in {profile["set_input"]*1000:.3f} ms.'
    )

    # infer model
    start = time.time()
    module.run()
    module.sync()
    profile["infer"] = time.time() - start
    logger.info(f'{model_name} infer completed in {profile["infer"]*1000:.3f} ms.')

    # get output and compare with golden
    profile["get_output"] = 0
    result_check = True
    output_num = module.get_num_outputs()
    for idx in range(output_num):
        output_name = module.get_output_name(idx)
        output_info = module.get_output_info(output_name)
        logger.info(
            f"output_info[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}"
        )
        start = time.time()
        output_data = module.get_output(output_name).numpy()
        profile["get_output"] += time.time() - start
        logger.info(
            f"output[{output_name}] shape = {output_data.shape}, dtype = {output_data.dtype}"
        )
        output_files = sorted(
            glob.glob(
                f"{model_dir}/**/hmquant_*{sanitize_name(output_name)}*.npy",
                recursive=True,
            )
        )
        output_data_path = os.path.abspath(output_files[0]) if output_files else ""
        if os.path.exists(output_data_path):
            golden_output = np.load(output_data_path)
            golden_output = np.concatenate([golden_output] * batch, axis=0)
        else:
            result_check = False
            logger.info(
                f"[warning] compare canceled while golden data not found -> {output_data_path}"
            )
            continue
        if golden_output.shape == output_data.shape:
            cosine_dist = cosine_distance(golden_output, output_data)
            is_match = (golden_output == output_data).all()
            logger.info(
                f"[compare] golden output [{output_name}] match={is_match}, similarity={cosine_dist:.6f}"
            )
            if is_match:
                continue
            if cosine_dist < GOLDEN_THRESH:
                result_check = False
        else:
            result_check = False
            logger.info(
                f"[compare] golden output [{output_name}] shape not match {golden_output.shape} vs {output_data.shape}"
            )
    logger.info(
        f'{model_name} get {output_num} ouputs completed in {profile["get_output"]*1000:.3f} ms.'
    )
    if not result_check:
        logger.info("[error] result check failed.")
        raise SystemExit(1)
    logger.info(f"<=== {model_name} test success.")


if __name__ == "__main__":
    args = get_args()
    logger.info(args)
    profile = {}
    model_specs = [
        {
            "name": "dit",
            "sub_dir": "dit",
            "suffix": "dit",
            "flash_attention": args.flash_attention[0],
            "build_kwargs": {
                "enable_common_subgraph": args.enable_common_subgraph,
            },
        },
        {
            "name": "vae",
            "sub_dir": "vae",
            "suffix": "vae",
            "flash_attention": args.flash_attention[1],
            "build_kwargs": {},
        },
        {
            "name": "text_encoder",
            "sub_dir": "text_encoder",
            "suffix": "encoder",
            "flash_attention": args.flash_attention[0],
            "build_kwargs": {
                "enable_common_subgraph": args.enable_common_subgraph,
                "context_length": args.context_length,
                "prefill_length": args.prefill_length,
                "llm_opt": True,
                "is_prefill": True,
            },
        },
    ]

    if args.stage == "build" or args.stage == "all":
        assert (
            get_platform() == "x86_64"
        ), "Only supported for compilation on the x86_64 platform."

        for spec in model_specs:
            if args.models and spec["name"] not in args.models:
                continue
            sub_dir = spec["sub_dir"]
            suffix = spec["suffix"]
            model_dir = os.path.join(args.model_dir, sub_dir)
            build_kwargs = {
                "hmonnx": find_hmonnx_file(model_dir),
                "hmm_name": f"{args.model_name}-{args.model_size}_{suffix}",
                "output": args.output_dir,
                "ncore": args.ncore,
                "ndevice": args.ndevice,
                "enable_xh2_stable_output": args.enable_xh2_stable_output,
                "flash_attn": spec["flash_attention"],
                "parallel_jobs": args.j,
                **spec["build_kwargs"],
            }
            Xh2Exec.build_from_hmonnx(**build_kwargs)

    if args.stage == "test" or args.stage == "all":
        for spec in model_specs:
            if args.models and spec["name"] not in args.models:
                continue
            model_dir = os.path.join(args.model_dir, spec["sub_dir"])
            test(
                f"{args.model_name}-{args.model_size}_{spec['suffix']}",
                model_dir,
                args.output_dir,
                profile,
            )

    logger.info(f"\n=== Build/Test completed. ===")
