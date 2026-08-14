#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: ptq.py
# Description:
#   Quantize and export Ornith 1.0 with the Merak LLM workflow.
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
import importlib.util
import os
import shutil
from pathlib import Path

import yaml

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODEL_DIR / "config.yaml"
DEFAULT_GOLDEN_IMAGE = MODEL_DIR.parents[2] / "data" / "pic" / "beach.jpeg"


def load_model_config(
    config_path: str, model_name: str | None, model_size: str | None
) -> tuple[str, str, dict]:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    model_configs = config.get("model_configs", {})
    selected_name = model_name or config.get("default_model_name")
    if (
        selected_name not in model_configs
        and model_name is None
        and len(model_configs) == 1
    ):
        selected_name = next(iter(model_configs))

    size_configs = model_configs.get(selected_name, {})
    selected_size = model_size or config.get("default_model_size")
    if (
        selected_size not in size_configs
        and model_size is None
        and len(size_configs) == 1
    ):
        selected_size = next(iter(size_configs))

    try:
        return selected_name, selected_size, size_configs[selected_size]
    except KeyError as error:
        raise ValueError(
            f"unsupported model configuration: {selected_name}-{selected_size}"
        ) from error


def find_configs_merak_dir() -> Path:
    spec = importlib.util.find_spec("xhmodel_merak")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("Python package xhmodel_merak not found")

    package_dir = Path(next(iter(spec.submodule_search_locations)))
    configs_dir = package_dir.parent / "configs_merak"
    if not configs_dir.is_dir():
        raise FileNotFoundError(
            f"configs_merak directory not found next to xhmodel_merak: {configs_dir}"
        )
    return configs_dir


def default_model_dir(model_config: dict) -> Path:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return MODEL_DIR / repo_ids[0].rsplit("/", maxsplit=1)[-1]
    return MODEL_DIR / f"{model_config['model_name']}-{model_config['model_size']}"


def remove_output_dir_if_needed(output_dir: Path, overwrite: bool) -> None:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)


def move_hmquant(export_dir: Path, output_dir: Path, visual_suffix: str) -> None:
    candidates = sorted(
        path
        for path in export_dir.iterdir()
        if path.is_dir() and path.name.startswith(f"hmquant_{HOUMO_TARGET}_")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one hmquant_{HOUMO_TARGET}_* directory under {export_dir}, "
            f"found: {candidates}"
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for item in candidates[0].iterdir():
        if item.name == "golden_meta_info.json":
            continue
        name = f"visual_{visual_suffix}" if item.name == "visual" else item.name
        shutil.move(str(item), str(output_dir / name))
    print(f"hmquant contents moved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and export Ornith 1.0.")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the Ornith config.yaml",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        default=None,
        help="model name in config.yaml",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        default=None,
        help="model size in config.yaml",
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        default=None,
        help="raw or already quantized Hugging Face model directory",
    )
    parser.add_argument(
        "--workflow-config",
        dest="workflow_config",
        default=None,
        help="Merak workflow YAML; overrides workflow_config in config.yaml",
    )
    parser.add_argument(
        "--out-dir",
        dest="output_dir",
        default=str(MODEL_DIR / "output" / HOUMO_TARGET / "hmquant"),
        help="final flattened hmquant output directory",
    )
    parser.add_argument(
        "--work-dir",
        dest="work_dir",
        default=str(MODEL_DIR / "output" / HOUMO_TARGET / "work_dirs"),
        help="default workflow working directory",
    )
    parser.add_argument(
        "--quant-output-dir",
        dest="quant_output_dir",
        default=None,
        help="workflow quantization output directory",
    )
    parser.add_argument(
        "--export-output-dir",
        dest="export_output_dir",
        default=None,
        help="workflow HMONNX export output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="device for quantization, export, golden, and quick test",
    )
    parser.add_argument("--bits", type=int, default=None, help="override quant.bits")
    parser.add_argument(
        "--max-size-w",
        dest="max_size_w",
        type=int,
        default=None,
        help="vision width",
    )
    parser.add_argument(
        "--max-size-h",
        dest="max_size_h",
        type=int,
        default=None,
        help="vision height",
    )
    parser.add_argument(
        "--max-size-t",
        dest="max_size_t",
        type=int,
        default=None,
        help="vision temporal size",
    )
    parser.add_argument(
        "--export-from-quanted-model",
        dest="export_from_quanted_model",
        action="store_true",
        help="skip quantization when model_dir already contains a quantized model",
    )
    parser.add_argument(
        "--dump-golden",
        dest="dump_golden",
        action="store_true",
        help="dump golden data after export",
    )
    parser.add_argument(
        "--golden-device-map",
        dest="golden_device_map",
        nargs="+",
        default=None,
        help="HMONNX device map used by golden generation",
    )
    parser.add_argument(
        "--golden-input-text",
        dest="golden_input_text",
        default="描述这张图片",
        help="golden generation text input",
    )
    parser.add_argument(
        "--golden-input-image",
        dest="golden_input_image",
        default=str(DEFAULT_GOLDEN_IMAGE),
        help="golden generation image input",
    )
    parser.add_argument(
        "--quick-test",
        dest="quick_test",
        action="store_true",
        help="run a quick HMONNX generation test after export",
    )
    parser.add_argument(
        "--quick-test-prompt",
        dest="quick_test_prompt",
        default="用中文简单介绍 Ornith 1.0。",
        help="prompt used by the quick HMONNX test",
    )
    parser.add_argument(
        "--quick-test-max-new-tokens",
        dest="quick_test_max_new_tokens",
        type=int,
        default=64,
        help="maximum generated tokens for the quick HMONNX test",
    )
    parser.add_argument(
        "--enable-flash-attention",
        action="store_true",
        default=False,
        help="enable FlashAttention for this export run",
    )
    parser.add_argument(
        "--disable-fuse-gdr-ops",
        action="store_true",
        default=False,
        help="disable GDR op fusion for this export run",
    )
    parser.add_argument(
        "--disable-fuse-gdr-block-recurrent-ops",
        action="store_true",
        default=False,
        help="disable GDR block recurrent op fusion for this export run",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove an existing workflow export directory",
    )
    args = parser.parse_args()

    args.model_name, args.model_size, model_config = load_model_config(
        args.config_path, args.model_name, args.model_size
    )
    args.model_dir = str(
        Path(args.model_dir).resolve()
        if args.model_dir
        else default_model_dir(model_config).resolve()
    )
    args.max_size_w = args.max_size_w or int(model_config.get("max_size_w", 448))
    args.max_size_h = args.max_size_h or int(model_config.get("max_size_h", 448))
    args.max_size_t = args.max_size_t or int(model_config.get("max_size_t", 2))

    work_dir = Path(args.work_dir).resolve()
    args.work_dir = str(work_dir)
    args.quant_output_dir = str(
        Path(args.quant_output_dir).resolve()
        if args.quant_output_dir
        else work_dir / "quant"
    )
    args.export_output_dir = str(
        Path(args.export_output_dir).resolve()
        if args.export_output_dir
        else work_dir / "export"
    )
    args.output_dir = str(Path(args.output_dir).resolve())

    if args.workflow_config is None:
        workflow_config = model_config.get("workflow_config")
        if not workflow_config:
            raise ValueError("workflow_config is missing from config.yaml")
        args.workflow_config = str(find_configs_merak_dir() / workflow_config)
    args.workflow_config = str(Path(args.workflow_config).resolve())
    if not Path(args.workflow_config).is_file():
        raise FileNotFoundError(f"workflow config not found: {args.workflow_config}")
    if args.quick_test_max_new_tokens <= 0:
        raise ValueError("--quick-test-max-new-tokens must be positive")
    if args.dump_golden and not Path(args.golden_input_image).is_file():
        raise FileNotFoundError(
            f"golden input image not found: {args.golden_input_image}"
        )
    return args


def main() -> None:
    args = parse_args()

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.workflow_config,
    )

    quant_overrides = {"quant": None} if args.export_from_quanted_model else {}
    if args.bits is not None and not args.export_from_quanted_model:
        quant_overrides["quant.bits"] = args.bits
    quant_result = workflow.quant(
        output_dir=args.quant_output_dir,
        device=args.device,
        config_overrides=quant_overrides,
    )
    print(f"quant_result: {quant_result}")

    export_dir = Path(args.export_output_dir)
    remove_output_dir_if_needed(export_dir, args.overwrite)
    export_overrides = {
        "export.model.visual_config.max_size_w": args.max_size_w,
        "export.model.visual_config.max_size_h": args.max_size_h,
    }
    export_overrides["export.model.flash_attention.enable"] = (
        args.enable_flash_attention
    )
    export_overrides["export.model.fuse_gdr_ops"] = not args.disable_fuse_gdr_ops
    export_overrides["export.model.fuse_gdr_block_recurrent_ops"] = (
        not args.disable_fuse_gdr_block_recurrent_ops
    )
    print(f"export_overrides: {export_overrides}")
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.device,
        config_overrides=export_overrides,
    )
    print(f"export_result: {export_result}")

    if args.dump_golden:
        workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages={
                "text": args.golden_input_text,
                "image": args.golden_input_image,
            },
            device_map=args.golden_device_map,
        )

    if args.quick_test:
        from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import (
            print_quick_test_result,
            quick_test_hmonnx,
        )

        quick_result = quick_test_hmonnx(
            export_result,
            prompt=args.quick_test_prompt,
            device=args.device,
            max_new_tokens=args.quick_test_max_new_tokens,
            do_sample=False,
        )
        print_quick_test_result(quick_result)

    visual_suffix = f"{args.max_size_w}x{args.max_size_h}x{args.max_size_t}"
    move_hmquant(export_dir, Path(args.output_dir), visual_suffix)


if __name__ == "__main__":
    main()
