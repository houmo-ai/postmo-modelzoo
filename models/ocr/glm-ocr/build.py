# Copyright (c) 2025 HOUMO AI
#
# File: build.py
# Description:
#  glm-ocr Model Build and Test Tool - Python script for building and testing
# glm-ocr models.
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
# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np
import time
import multiprocessing
import argparse
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
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=None,
        help="batch size",
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
        "--stage",
        dest="stage",
        type=str,
        default="build",
        choices=["build", "test", "all"],
        help="build stage",
    )
    parser.add_argument(
        "--build_target",
        dest="build_target",
        type=str,
        default="all",
        choices=["all", "ocr", "layout"],
        help="model target to build; layout uses model_dir/layout/*.onnx",
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
        "--image_size_w",
        dest="image_size_w",
        type=int,
        default=None,
        help="image width",
    )
    parser.add_argument(
        "--image_size_h",
        dest="image_size_h",
        type=int,
        default=None,
        help="image height",
    )
    parser.add_argument(
        "--max_size_t",
        dest="max_size_t",
        type=int,
        default=None,
        help="max temporal size",
    )
    parser.add_argument(
        "--flash_attention",
        dest="flash_attention",
        nargs=2,
        type=int,
        default=(0, 1),
        help="FlashAttention optimization switches: 1st int = prefill/decode switch, 2nd int = ViT switch",
    )
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ncore = first_not_none(args.ncore, model_config.get("ncore", HOUMO_CORE_NUM))
    args.batch = first_not_none(args.batch, model_config.get("batch", 1))
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    args.prefill_length = first_not_none(
        args.prefill_length, model_config.get("prefill_length", 256)
    )
    args.context_length = first_not_none(
        args.context_length,
        parse_context_length(model_config.get("context_length", "8k")),
    )
    args.image_size_w = first_not_none(
        args.image_size_w, model_config.get("image_size_w", 672)
    )
    args.image_size_h = first_not_none(
        args.image_size_h, model_config.get("image_size_h", 672)
    )
    args.max_size_t = first_not_none(args.max_size_t, model_config.get("max_size_t", 2))
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
    for idx in range(input_num):
        input_name = module.get_input_name(idx)
        input_info = module.get_input_info(input_name)
        print(
            f"input_info[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
        )
        input_data_path = os.path.join(
            model_dir, f"hmquant_{prefix}_{sanitize_name(input_name)}_input.npy"
        )
        input_data = np.load(input_data_path).astype(input_info.dtype)
        input_data = np.concatenate([input_data for _ in range(batch)], axis=0)
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
    for idx in range(output_num):
        output_name = module.get_output_name(idx)
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
                [golden_output for _ in range(batch)], axis=0
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


def build_ocr_models(args, model_dir, output_dir, visual_model_name):
    llm_flash_attention, vit_flash_attention = args.flash_attention
    Xh2Exec.build_from_hmonnx(
        hmonnx=find_hmonnx_file(os.path.join(model_dir, "visual")),
        hmm_name=visual_model_name,
        output=output_dir,
        ncore=args.ncore,
        flash_attn=vit_flash_attention,
        parallel_jobs=args.j,
    )
    Xh2Exec.build_from_hmonnx(
        is_prefill=True,
        hmonnx=find_hmonnx_file(os.path.join(model_dir, "prefill")),
        hmm_name=f"{args.model_name}-{args.model_size}_prefill",
        output=output_dir,
        context_length=args.context_length,
        prefill_length=args.prefill_length,
        ndevice=args.ndevice,
        ncore=args.ncore,
        flash_attn=llm_flash_attention,
        llm_opt=True,
        parallel_jobs=args.j,
    )
    Xh2Exec.build_from_hmonnx(
        hmonnx=find_hmonnx_file(os.path.join(model_dir, "decode")),
        hmm_name=f"{args.model_name}-{args.model_size}_decode",
        output=output_dir,
        context_length=args.context_length,
        llm_batch=args.batch,
        ndevice=args.ndevice,
        ncore=args.ncore,
        flash_attn=llm_flash_attention,
        llm_opt=True,
        parallel_jobs=args.j,
    )


def build_layout_model(args, model_dir, output_dir):
    Xh2Exec.build_from_hmonnx(
        hmonnx=find_hmonnx_file(os.path.join(model_dir, "layout")),
        hmm_name="ppdoclayoutv3",
        output=output_dir,
        ncore=args.ncore,
        parallel_jobs=args.j,
    )


if __name__ == "__main__":
    args = get_args()
    print(args)

    model_dir = args.model_dir
    model_name = args.model_name
    model_size = args.model_size
    output_dir = args.output_dir
    profile = {}
    visual_model_name = (
        f"{model_name}-{model_size}_visual_"
        f"{args.image_size_w}x{args.image_size_h}x{args.max_size_t}"
    )

    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        if args.stage == "build" or args.stage == "all":
            assert (
                get_platform() == "x86_64"
            ), "Only supported for compilation on the x86_64 platform."
            if args.build_target in ["all", "ocr"]:
                build_ocr_models(args, model_dir, output_dir, visual_model_name)
            if args.build_target in ["all", "layout"]:
                build_layout_model(args, model_dir, output_dir)

        if (args.stage == "test" or args.stage == "all") and args.build_target in [
            "all",
            "ocr",
        ]:
            test(
                f"{model_name}-{model_size}_prefill",
                os.path.join(model_dir, "prefill"),
                output_dir,
                profile,
                prefix=model_name,
            )
            test(
                f"{model_name}-{model_size}_decode",
                os.path.join(model_dir, "decode"),
                output_dir,
                profile,
                prefix=model_name,
            )
            test(
                visual_model_name,
                os.path.join(model_dir, "visual"),
                output_dir,
                profile,
                prefix=model_name,
            )

    print(
        f"\n=== Build/Test completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )
