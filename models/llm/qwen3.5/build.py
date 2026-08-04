# Copyright (c) 2025 HOUMO AI
#
# File: build.py
# Description:
#   Qwen3.5 Model Build and Test Tool - Python script for building and testing
#   Qwen3.5 models.
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
import shutil
import numpy as np
import time
import multiprocessing
import argparse
import glob
from dataclasses import dataclass
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

HOUMO_CORE_NUM = os.getenv("HOUMO_CORE_NUM", 2)
GOLDEN_THRESH = 0.98
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DECODE_DIR_PATTERN = "*decode*"


def sanitize_name(name: str):
    return name.replace(":", "_").replace("/", "_")


def find_lora_dirs(model_dir: str) -> list[str]:
    lora_root = os.path.join(model_dir, "lora")
    if not os.path.isdir(lora_root):
        raise FileNotFoundError(f"LoRA directory not found: {lora_root}")

    lora_dirs = sorted(
        path
        for path in glob.glob(os.path.join(lora_root, "*"))
        if os.path.isdir(path)
    )
    if not lora_dirs:
        raise FileNotFoundError(f"No LoRA adapter directory found under: {lora_root}")

    for lora_dir in lora_dirs:
        adapter_name = os.path.basename(lora_dir)
        prefill_dir = os.path.join(lora_dir, "prefill")
        decode_dirs = sorted(
            path
            for path in glob.glob(os.path.join(lora_dir, DECODE_DIR_PATTERN))
            if os.path.isdir(path)
        )
        if not os.path.isdir(prefill_dir):
            raise FileNotFoundError(
                f"LoRA adapter {adapter_name} prefill directory not found: {prefill_dir}"
            )
        if not decode_dirs:
            raise FileNotFoundError(
                f'No LoRA adapter {adapter_name} subdirectory containing "decode" found under: {lora_dir}'
            )
    return lora_dirs


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
    """Parse commandline."""
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=os.path.join("output", HOUMO_TARGET, "hmquant"), help="path to the model dir")
    parser.add_argument("--model_name", dest="model_name", type=str, default=None, help="output houmo model name")
    parser.add_argument("--model_size", dest="model_size", type=str, default=None, help="model size")
    parser.add_argument("--ndevice", dest="ndevice", type=int, default=None, help="device number")
    parser.add_argument("--ncore", dest="ncore", type=int, default=None, help="core number")
    parser.add_argument("--batch", dest="batch", type=int, default=None, help="batch size")
    parser.add_argument("--context_length", dest="context_length", type=int, default=None, help="context_length")
    parser.add_argument("--prefill_length", dest="prefill_length", type=int, default=None, help="prefill_length")
    parser.add_argument("--j", dest="j", type=int, default=int(multiprocessing.cpu_count() * 0.7), help="build parallel jobs")
    parser.add_argument("--stage", dest="stage", type=str, default="build", choices=["build", "test", "all"], help="build stage")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=os.path.join("output", HOUMO_TARGET), help="build output dir")
    parser.add_argument("--flash_attention", dest="flash_attention", nargs=2, type=int, default=(2, 1), help="FlashAttention optimization switches: 1st int = prefill/decode model switch (0=off, 1/2=on), 2nd int = ViT model switch (0=off, 1=on); e.g., --flash_attention 2 1 (prefill&decode=2, ViT=1)")
    parser.add_argument("--enable_common_subgraph", dest="enable_common_subgraph", action="store_true", default=False, help="enable common subgraph optimization")
    parser.add_argument("--enable_xh2_stable_output", dest="enable_xh2_stable_output", action="store_true", default=False, help="enable stable output")
    parser.add_argument("--mtp", dest="mtp", action="store_true", default=False, help="enable mtp optimization")
    parser.add_argument("--lora", dest="lora", action="store_true", default=False, help="enable lora mode")
    # fmt: on

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

    # set input
    profile["set_input"] = 0
    input_num = module.get_num_inputs()
    for id in range(input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name)
        print(
            f"input_info[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
        )
        input_files = glob.glob(
            f"{model_dir}/**/hmquant_*_{sanitize_name(input_name)}_*.npy",
            recursive=True,
        )
        input_data_path = os.path.abspath(input_files[0]) if input_files else ""
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
        output_files = glob.glob(
            f"{model_dir}/**/hmquant_*{sanitize_name(output_name)}*.npy", recursive=True
        )
        output_data_path = os.path.abspath(output_files[0]) if output_files else ""
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


def _first_existing_dir(model_dir: str, names: list[str]) -> str:
    for name in names:
        path = os.path.join(model_dir, name)
        if os.path.isdir(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"Directory not found under {model_dir}: {' or '.join(names)}"
    )


@dataclass(frozen=True)
class ModelDirs:
    prefill: str
    decode: str
    visual: list[str]
    mtp_draft_prefill: str | None = None
    mtp_draft_decode: str | None = None


def discover_model_dirs(model_dir: str, include_mtp: bool = False) -> ModelDirs:
    prefill_dir = _first_existing_dir(model_dir, ["prefill"])

    decode_dirs = sorted(
        os.path.abspath(path)
        for path in glob.glob(os.path.join(model_dir, DECODE_DIR_PATTERN))
        if os.path.isdir(path)
        and not any(key in os.path.basename(path) for key in ["mtp", "draft"])
    )
    if not decode_dirs:
        raise FileNotFoundError(
            f'No non-MTP subdirectory containing "decode" found under: {model_dir}'
        )

    visual_dirs = discover_visual_dirs(model_dir)

    mtp_draft_prefill = None
    mtp_draft_decode = None
    if include_mtp:
        mtp_draft_prefill = _first_existing_dir(
            model_dir, ["mtp_draft_prefill", "draft_prefill"]
        )
        mtp_draft_decode = _first_existing_dir(
            model_dir, ["mtp_draft_decode", "draft_decode"]
        )

    return ModelDirs(
        prefill=prefill_dir,
        decode=decode_dirs[0],
        visual=visual_dirs,
        mtp_draft_prefill=mtp_draft_prefill,
        mtp_draft_decode=mtp_draft_decode,
    )


def discover_lora_model_dirs(model_dir: str) -> ModelDirs:
    lora_dirs = find_lora_dirs(model_dir)
    if len(lora_dirs) > 1:
        raise RuntimeError(
            f"Expected one LoRA adapter directory under {os.path.join(model_dir, 'lora')}, found: {lora_dirs}"
        )
    lora_dir = lora_dirs[0]
    decode_dirs = sorted(
        os.path.abspath(path)
        for path in glob.glob(os.path.join(lora_dir, DECODE_DIR_PATTERN))
        if os.path.isdir(path)
    )
    visual_dirs = discover_visual_dirs(model_dir)
    return ModelDirs(
        prefill=os.path.abspath(os.path.join(lora_dir, "prefill")),
        decode=decode_dirs[0],
        visual=visual_dirs,
    )


def discover_visual_dirs(model_dir: str) -> list[str]:
    return sorted(
        os.path.abspath(path)
        for path in glob.glob(os.path.join(model_dir, "vis*"))
        if os.path.isdir(path)
        and any(key in os.path.basename(path) for key in ["vision", "visual"])
    )


def prepare_lora_build_dirs(model_dir: str) -> tuple[list[str], list[tuple[str, str]]]:
    lora_dirs = find_lora_dirs(model_dir)
    if len(lora_dirs) <= 1:
        return lora_dirs, []

    lora_root = os.path.join(model_dir, "lora")
    backup_dirs = []
    for lora_dir in lora_dirs:
        backup_dir = os.path.join(model_dir, os.path.basename(lora_dir))
        if os.path.exists(backup_dir):
            raise FileExistsError(f"LoRA backup directory already exists: {backup_dir}")
        shutil.move(lora_dir, backup_dir)
        backup_dirs.append((backup_dir, lora_dir))

    os.makedirs(lora_root, exist_ok=True)
    return [backup_dir for backup_dir, _ in backup_dirs], backup_dirs


def restore_lora_build_dirs(backup_dirs: list[tuple[str, str]]) -> None:
    for backup_dir, lora_dir in backup_dirs:
        if os.path.exists(lora_dir):
            shutil.move(lora_dir, backup_dir)
    for backup_dir, lora_dir in backup_dirs:
        if os.path.exists(backup_dir):
            os.makedirs(os.path.dirname(lora_dir), exist_ok=True)
            shutil.move(backup_dir, lora_dir)


def rename_lora_input_dirs(
    output_dir: str, model_name: str, model_size: str, adapter_name: str
) -> None:
    adapter_suffix = sanitize_name(adapter_name)
    for stage in ["prefill", "decode"]:
        source_dir = os.path.join(output_dir, f"{model_name}-{model_size}_{stage}_lora_input")
        if not os.path.exists(source_dir):
            continue
        target_dir = os.path.join(
            output_dir, f"{model_name}-{model_size}_{adapter_suffix}_{stage}_lora_input"
        )
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.move(source_dir, target_dir)


def enable_lora_build_params(build_tasks: list[dict]) -> None:
    for build_kwargs in build_tasks:
        if build_kwargs["hmm_name"].endswith(("_prefill", "_decode")):
            build_kwargs["enable_bundle_lora_param"] = True


def append_mtp_build_tasks(
    build_tasks: list[dict],
    model_dirs: ModelDirs,
    model_name: str,
    model_size: str,
    llm_flash_attention: int,
    ndevice: int,
    context_length: int,
) -> None:
    build_tasks.extend(
        [
            {
                "is_prefill": True,
                "hmonnx": find_hmonnx_file(model_dirs.mtp_draft_prefill),
                "hmm_name": f"{model_name}-{model_size}_prefill_mtp",
                "flash_attn": llm_flash_attention,
                "context_length": context_length,
                "ndevice": ndevice,
            },
            {
                "hmonnx": find_hmonnx_file(model_dirs.mtp_draft_decode),
                "hmm_name": f"{model_name}-{model_size}_decode_mtp",
                "flash_attn": llm_flash_attention,
                "context_length": context_length,
                "ndevice": ndevice,
            },
        ]
    )


def build_lora_adapters(
    lora_build_dirs: list[str],
    lora_backup_dirs: list[tuple[str, str]],
    model_dir: str,
    output_dir: str,
    model_name: str,
    model_size: str,
    run_build,
) -> ModelDirs:
    if not lora_backup_dirs:
        model_dirs = run_build()
        adapter_name = os.path.basename(lora_build_dirs[0])
        rename_lora_input_dirs(output_dir, model_name, model_size, adapter_name)
        return model_dirs

    try:
        return build_backed_up_lora_adapters(
            lora_build_dirs, model_dir, output_dir, model_name, model_size, run_build
        )
    finally:
        restore_lora_build_dirs(lora_backup_dirs)


def build_backed_up_lora_adapters(
    lora_build_dirs: list[str],
    model_dir: str,
    output_dir: str,
    model_name: str,
    model_size: str,
    run_build,
) -> ModelDirs:
    model_dirs = None
    for lora_build_dir in lora_build_dirs:
        adapter_name = os.path.basename(lora_build_dir)
        active_lora_dir = os.path.join(model_dir, "lora", adapter_name)
        print(f"\n===> LoRA adapter build start: {lora_build_dir}")
        shutil.move(lora_build_dir, active_lora_dir)
        try:
            model_dirs = run_build()
        finally:
            rename_lora_input_dirs(output_dir, model_name, model_size, adapter_name)
            shutil.move(active_lora_dir, lora_build_dir)
    return model_dirs


def _get_visual_model_name(model_name: str, model_size: str, visual_dir: str) -> str:
    folder_name = os.path.basename(visual_dir)
    if "_" not in folder_name:
        return f"{model_name}-{model_size}_visual"

    resolution_suffix = folder_name.split("_", 1)[1]
    return f"{model_name}-{model_size}_visual_{resolution_suffix}"


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

    model_dirs = None

    if args.stage == "build" or args.stage == "all":
        assert (
            get_platform() == "x86_64"
        ), f"Only supported for compilation on the x86_64 platform."

        def build_visual() -> None:
            for visual_dir in discover_visual_dirs(model_dir):
                build_kwargs = {
                    "hmonnx": find_hmonnx_file(visual_dir),
                    "hmm_name": _get_visual_model_name(
                        model_name, model_size, visual_dir
                    ),
                    "flash_attn": vit_flash_attention,
                }
                print(f'\n===> {build_kwargs["hmm_name"]} build start...')
                Xh2Exec.build_from_hmonnx(
                    output=output_dir, ncore=ncore, parallel_jobs=j, **build_kwargs
                )

        def run_build() -> ModelDirs:
            current_model_dirs = (
                discover_lora_model_dirs(model_dir)
                if args.lora
                else discover_model_dirs(model_dir, include_mtp=args.mtp)
            )
            build_tasks = []

            build_tasks.extend(
                [
                    {
                        "is_prefill": True,
                        "hmonnx": find_hmonnx_file(current_model_dirs.prefill),
                        "hmm_name": f"{model_name}-{model_size}_prefill",
                        "flash_attn": llm_flash_attention,
                        "context_length": args.context_length,
                        "prefill_length": args.prefill_length,
                        "ndevice": ndevice,
                        "enable_common_subgraph": (
                            args.enable_common_subgraph if not args.mtp else False
                        ),
                        "enable_xh2_stable_output": args.enable_xh2_stable_output,
                        "llm_opt": True,
                    },
                    {
                        "hmonnx": find_hmonnx_file(current_model_dirs.decode),
                        "hmm_name": f"{model_name}-{model_size}_decode",
                        "llm_batch": args.batch if not args.mtp else 1,
                        "flash_attn": llm_flash_attention,
                        "context_length": args.context_length,
                        "ndevice": ndevice,
                        "enable_xh2_stable_output": args.enable_xh2_stable_output,
                        "llm_opt": True,
                    },
                ]
            )

            if args.lora:
                enable_lora_build_params(build_tasks)

            if args.mtp and not args.lora:
                append_mtp_build_tasks(
                    build_tasks,
                    current_model_dirs,
                    model_name,
                    model_size,
                    llm_flash_attention,
                    ndevice,
                    args.context_length,
                )

            for build_kwargs in build_tasks:
                print(f'\n===> {build_kwargs["hmm_name"]} build start...')
                Xh2Exec.build_from_hmonnx(
                    output=output_dir, ncore=ncore, parallel_jobs=j, **build_kwargs
                )
            return current_model_dirs

        build_visual()

        if args.lora:
            lora_build_dirs, lora_backup_dirs = prepare_lora_build_dirs(model_dir)
            model_dirs = build_lora_adapters(
                lora_build_dirs,
                lora_backup_dirs,
                model_dir,
                output_dir,
                model_name,
                model_size,
                run_build,
            )
        else:
            model_dirs = run_build()

    if args.stage == "test" or args.stage == "all":
        if model_dirs is None:
            model_dirs = (
                discover_lora_model_dirs(model_dir)
                if args.lora
                else discover_model_dirs(model_dir, include_mtp=args.mtp)
            )
        test_tasks = [
            (f"{model_name}-{model_size}_prefill", model_dirs.prefill),
            (f"{model_name}-{model_size}_decode", model_dirs.decode),
        ]
        test_tasks.extend(
            (_get_visual_model_name(model_name, model_size, visual_dir), visual_dir)
            for visual_dir in model_dirs.visual
        )
        for test_model_name, test_model_dir in test_tasks:
            test(test_model_name, test_model_dir, output_dir, profile)

    print("\n=== Build flow finished. ===")
