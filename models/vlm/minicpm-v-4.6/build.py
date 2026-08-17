# Copyright (c) 2026 HOUMO AI
#
# File: build.py
# Description:
#   MiniCPM-V 4.6 model compilation and golden comparison tool.
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
import multiprocessing
import argparse
import glob
from typing import Any

from hmatc.exec.xh2_exec import Xh2Exec
from hmatc.utils.bfp import cast_fp_data_to_act_hmfp_data
from hmatc.utils.utils import (
    find_hmonnx_file,
    first_not_none,
    get_model_configs,
    get_platform,
    parse_context_length,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = os.getenv("HOUMO_CORE_NUM", 2)
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


def _find_golden_file(step_dir: str, tensor_name: str, file_type: str) -> str:
    """Return the unique golden tensor file for a model input or output."""
    pattern = os.path.join(
        step_dir,
        f"hmquant_*_{sanitize_name(tensor_name)}_{file_type}.npy",
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one golden file for {tensor_name}, "
            f"found {len(matches)}: {pattern}"
        )
    return os.path.abspath(matches[0])


def _set_golden_inputs(
    module: Any,
    step_dir: str,
    batch: int,
    profile: dict[str, float],
) -> None:
    """Load, adapt, and bind every golden input for one inference step."""
    for input_id in range(module.get_num_inputs()):
        input_name = module.get_input_name(input_id)
        input_info = module.get_input_info(input_name)
        input_data_path = _find_golden_file(step_dir, input_name, "input")
        input_data = np.load(input_data_path)
        if input_data.dtype != np.dtype(input_info.dtype):
            name_lower = input_name.lower()
            is_kcache = "kcache" in name_lower
            is_vcache = "vcache" in name_lower
            if input_info.dtype == np.int8 and (is_kcache or is_vcache):
                pack_axis = -2 if is_vcache else -1
                input_data = cast_fp_data_to_act_hmfp_data(
                    input_data, "g32e8", pack_axis
                )
                print(
                    f"packed golden input[{input_name}] as g32e8 "
                    f"along axis {pack_axis}"
                )
            else:
                input_data = input_data.astype(input_info.dtype)
        input_data = np.concatenate([input_data for _ in range(batch)], axis=0)
        print(
            f"golden input[{input_name}] path = {input_data_path}, "
            f"shape = {input_data.shape}, dtype = {input_data.dtype}"
        )
        if tuple(input_data.shape) != tuple(input_info.shape):
            raise ValueError(
                f"Golden input [{input_name}] shape not match "
                f"{input_data.shape} vs {tuple(input_info.shape)}"
            )
        start = time.time()
        module.set_input(input_name, input_data)
        profile["set_input"] += time.time() - start


def _select_compared_output(
    output_name: str,
    output_data: np.ndarray,
    golden_output: np.ndarray,
) -> np.ndarray:
    """Select the compiled-model region represented by the golden output."""
    if (
        output_name == "logits"
        and golden_output.ndim == output_data.ndim
        and 0 < golden_output.shape[1] < output_data.shape[1]
    ):
        token_count = golden_output.shape[1]
        print(
            f"[compare] output [{output_name}] selected final "
            f"{token_count} token(s) from shape {output_data.shape}"
        )
        return output_data[:, -token_count:]
    return output_data


def _check_golden_outputs(
    module: Any,
    step_dir: str,
    batch: int,
    profile: dict[str, float],
) -> bool:
    """Compare every model output with its golden tensor."""
    result_check = True
    for output_id in range(module.get_num_outputs()):
        output_name = module.get_output_name(output_id)
        start = time.time()
        output_data = module.get_output(output_name).numpy()
        profile["get_output"] += time.time() - start
        output_data_path = _find_golden_file(step_dir, output_name, "output")
        golden_output = np.load(output_data_path)
        golden_output = np.concatenate([golden_output for _ in range(batch)], axis=0)
        compared_output = _select_compared_output(
            output_name, output_data, golden_output
        )

        if golden_output.shape != compared_output.shape:
            result_check = False
            print(
                f"[compare] golden output [{output_name}] shape not match "
                f"{golden_output.shape} vs {compared_output.shape}, "
                f"golden = {output_data_path}"
            )
            continue

        cosine_dist = cosine_distance(golden_output, compared_output)
        is_match = np.array_equal(golden_output, compared_output)
        print(
            f"[compare] golden output [{output_name}] match={is_match}, "
            f"similarity={cosine_dist:.6f}, golden = {output_data_path}"
        )
        if not is_match and cosine_dist < GOLDEN_THRESH:
            result_check = False
    return result_check


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
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size",
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
    parser.add_argument(
        "--enable_common_subgraph",
        dest="enable_common_subgraph",
        action="store_true",
        default=False,
        help="enable common subgraph optimization",
    )
    parser.add_argument(
        "--enable_xh2_stable_output",
        dest="enable_xh2_stable_output",
        action="store_true",
        default=False,
        help="enable stable output",
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
    if args.context_length is None:
        args.context_length = parse_context_length(
            model_config.get("context_length", "256k")
        )
    args.flash_attention = _validate_adjust_flash_attention(
        args.flash_attention, args.context_length
    )
    return args


def test(model_name, model_dir, output_dir, profile, batch=1):
    import tcim_lite

    print(f"\n===> {model_name} test start...")
    # load model
    model_path = os.path.join(output_dir, f"{model_name}.hmm")
    start = time.time()
    print(model_path)
    module = tcim_lite.runtime.load(model_path)
    profile["load"] = time.time() - start
    print(f'{model_name} load completed in {profile["load"]:.3f} s.', flush=True)

    step_dirs = [
        path
        for path in glob.glob(os.path.join(model_dir, "step_*"))
        if os.path.isdir(path)
    ]
    step_dirs.sort(key=lambda path: int(os.path.basename(path).split("_", 1)[1]))
    if not step_dirs:
        raise FileNotFoundError(f"No golden step directory found under: {model_dir}")
    step_0_dir = os.path.join(model_dir, "step_0")
    step_dirs = [step_0_dir] if os.path.isdir(step_0_dir) else step_dirs[:1]

    profile["set_input"] = 0
    profile["infer"] = 0
    profile["get_output"] = 0
    result_check = True

    for step_dir in step_dirs:
        step_name = os.path.basename(step_dir)
        print(f"\n--- {model_name} {step_name} ---")
        _set_golden_inputs(module, step_dir, batch, profile)

        start = time.time()
        module.run()
        module.sync()
        infer_time = time.time() - start
        profile["infer"] += infer_time
        print(f"{model_name} {step_name} infer completed in {infer_time*1000:.3f} ms.")

        if not _check_golden_outputs(module, step_dir, batch, profile):
            result_check = False

    print(
        f'{model_name} tested {len(step_dirs)} steps, set_input={profile["set_input"]*1000:.3f} ms, '
        f'infer={profile["infer"]*1000:.3f} ms, get_output={profile["get_output"]*1000:.3f} ms.'
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
    model_size = args.model_size
    output_dir = args.output_dir
    ncore = args.ncore
    ndevice = args.ndevice
    j = args.j
    llm_flash_attention, vit_flash_attention = args.flash_attention
    profile = {}

    decode_dirs = sorted(
        path
        for path in glob.glob(os.path.join(model_dir, "*decode*"))
        if os.path.isdir(path)
    )
    if not decode_dirs:
        raise FileNotFoundError(
            f'No subdirectory containing "decode" found under: {model_dir}'
        )
    decode_dir = os.path.abspath(decode_dirs[0])
    prefill_dir = os.path.join(model_dir, "prefill")

    visual_dirs = [
        folder_path
        for folder_path in glob.glob(os.path.join(model_dir, "vis*"))
        if os.path.isdir(folder_path)
        and any(key in os.path.basename(folder_path) for key in ["vision", "visual"])
    ]

    if args.stage == "build" or args.stage == "all":
        assert (
            get_platform() == "x86_64"
        ), f"Only supported for compilation on the x86_64 platform."

        # Build all visual models with resolution suffix
        for visual_dir in visual_dirs:
            folder_name = os.path.basename(visual_dir)
            # Extract the profile suffix from folder names such as "vision_16x".
            if "_" in folder_name:
                suffix = folder_name.split("_", 1)[1]
                vit_model_name = f"{model_name}-{model_size}_visual_{suffix}"
            else:
                vit_model_name = f"{model_name}-{model_size}_visual"
            Xh2Exec.build_from_hmonnx(
                hmonnx=find_hmonnx_file(visual_dir),
                hmm_name=vit_model_name,
                output=output_dir,
                ncore=ncore,
                flash_attn=vit_flash_attention,
                parallel_jobs=j,
            )

        Xh2Exec.build_from_hmonnx(
            is_prefill=True,
            hmonnx=find_hmonnx_file(prefill_dir),
            hmm_name=f"{model_name}-{model_size}_prefill",
            output=output_dir,
            flash_attn=llm_flash_attention,
            context_length=args.context_length,
            prefill_length=args.prefill_length,
            ndevice=ndevice,
            ncore=ncore,
            enable_common_subgraph=args.enable_common_subgraph,
            enable_xh2_stable_output=args.enable_xh2_stable_output,
            cpp_backend="v2",
            llm_opt=True,
            parallel_jobs=j,
        )

        Xh2Exec.build_from_hmonnx(
            hmonnx=find_hmonnx_file(decode_dir),
            hmm_name=f"{model_name}-{model_size}_decode",
            output=output_dir,
            llm_batch=args.batch,
            flash_attn=llm_flash_attention,
            context_length=args.context_length,
            ndevice=ndevice,
            ncore=ncore,
            enable_xh2_stable_output=args.enable_xh2_stable_output,
            cpp_backend="v2",
            llm_opt=True,
            parallel_jobs=j,
        )

    if args.stage == "test" or args.stage == "all":
        test(f"{model_name}-{model_size}_prefill", prefill_dir, output_dir, profile)
        test(
            f"{model_name}-{model_size}_decode",
            decode_dir,
            output_dir,
            profile,
            batch=args.batch,
        )
        # Test all visual models
        for visual_dir in visual_dirs:
            folder_name = os.path.basename(visual_dir)
            if "_" in folder_name:
                suffix = folder_name.split("_", 1)[1]
                vit_model_name = f"{model_name}-{model_size}_visual_{suffix}"
            else:
                vit_model_name = f"{model_name}-{model_size}_visual"
            test(vit_model_name, visual_dir, output_dir, profile)

    print("\n=== Build flow finished. ===")
