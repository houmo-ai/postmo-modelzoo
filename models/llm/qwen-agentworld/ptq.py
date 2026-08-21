# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Standard Qwen-Agentworld Merak workflow example (language-only, no visual).
#   Run this file from the repository root.  Model shape, quantization,
#   MTP/DFlash, and GDR options stay in YAML or CONFIG_OVERRIDES; the
#   workflow API only needs paths plus QuantResult.
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
"""Standard Qwen-Agentworld Merak workflow example (language-only, no visual).

Run this file from the repository root.  Model shape, quantization,
MTP/DFlash, and GDR options stay in YAML or ``CONFIG_OVERRIDES``; the
workflow API only needs paths plus ``QuantResult``.
"""

import argparse
import importlib.util
import os
import shutil
from pathlib import Path

from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "qwen-agentworld")
    model_size = model_config.get("model_size", "35b-a3b")
    return f"{model_name}-{model_size}"


def find_configs_merak_dir() -> str:
    spec = importlib.util.find_spec("xhmodel_merak")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("Python package xhmodel_merak not found")

    xhmodel_merak_dir = next(iter(spec.submodule_search_locations))
    configs_merak_dir = os.path.join(
        os.path.dirname(xhmodel_merak_dir), "configs_merak"
    )
    if not os.path.isdir(configs_merak_dir):
        raise FileNotFoundError(
            f"configs_merak directory not found next to xhmodel_merak: {configs_merak_dir}"
        )
    return configs_merak_dir


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def _move_hmquant_to_output(
    export_output_dir: str, target_dir: str
) -> None:
    export_dir = Path(export_output_dir)
    hmquant_dirs = sorted(
        path
        for path in export_dir.iterdir()
        if path.is_dir() and path.name.startswith(f"hmquant_{HOUMO_TARGET}_")
    )
    if not hmquant_dirs:
        raise FileNotFoundError(
            f"No hmquant_{HOUMO_TARGET}_* directory found under {export_dir}"
        )
    if len(hmquant_dirs) > 1:
        raise RuntimeError(
            f"Expected one hmquant_{HOUMO_TARGET}_* directory under {export_dir}, found: {hmquant_dirs}"
        )

    source_dir = hmquant_dirs[0]
    target = Path(target_dir)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)

    for item in source_dir.iterdir():
        if item.name == "golden_meta_info.json":
            continue
        shutil.move(str(item), str(target / item.name))

    print(f"hmquant contents moved to: {target}")


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="Run the Qwen-Agentworld Merak quant/export workflow.")
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model_name", type=str, default=None, help="Model name in config.yaml.")
    parser.add_argument("--model_size", type=str, default=None, help="Model size in config.yaml.")
    parser.add_argument("--model_dir", type=str, default=None, help="HF model directory; defaults to the model selected by model name and model size in config.yaml.")
    parser.add_argument("--quant_config_path", dest="quant_config_path", type=str, default=None, help="Explicit workflow YAML path; overrides workflow_config from config.yaml.")
    parser.add_argument("--output_dir", default=f"./output/{HOUMO_TARGET}/hmquant", help="output directory")
    parser.add_argument("--work_dir", type=str, default="./work_dirs", help="work directory")
    parser.add_argument("--device", default="cuda", help="Device for quant/export/golden/quick test. Default: cuda")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing export output directories before running.")
    parser.add_argument("--dump-golden", action="store_true", help="Dump golden data after export.")
    parser.add_argument("--quick-test", action="store_true", help="Run quick HMONNX test after export.")
    parser.add_argument("--export-from-quanted-model", action="store_true", help="if --model-dir is a quanted model, set this param to True")
    parser.add_argument("--bits", type=int, default=None, help="quantization bits, set this param to override config.yaml")
    parser.add_argument("--mtp", dest="mtp", action="store_true", default=False, help="enable mtp optimization")
    parser.add_argument("--lora", dest="lora", action="store_true", default=False, help="enable lora workflow")
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})

    args.model_dir = os.path.abspath(first_not_none(args.model_dir, get_default_model_dir(model_config)))
    if args.quant_config_path is None:
        workflow_config = model_config.get("workflow_config", "")
        if args.lora and workflow_config:
            workflow_config = workflow_config.removesuffix("_full.yaml") + "_lora.yaml"
        elif args.mtp and workflow_config:
            workflow_config = workflow_config.removesuffix(".yaml") + "_mtp.yaml"
        # args.quant_config_path = os.path.join(find_configs_merak_dir(), workflow_config)
        args.quant_config_path = workflow_config
    if not os.path.isfile(args.quant_config_path):
        raise FileNotFoundError(f"Workflow config not found: {args.quant_config_path}")


    # fmt: on
    return args


def main() -> None:
    args = parse_args()

    # 初始化工作流
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.quant_config_path,
    )

    # quant
    quant_output_dir = f"{args.work_dir}/qwen_agentworld_quant"
    if args.export_from_quanted_model:
        quant_result = workflow.quant(
            output_dir=quant_output_dir,
            device=args.device,
            # 从已量化的 HF 模型导出，需要跳过量化阶段
            config_overrides={"quant": None},
        )
    else:
        config_overrides = {}
        if args.bits:
            config_overrides["quant.bits"] = args.bits
        quant_result = workflow.quant(
            output_dir=quant_output_dir,
            device=args.device,
            config_overrides=config_overrides,
        )
    print(f"quant_result: {quant_result}")

    # export
    export_output_dir = f"{args.work_dir}/qwen_agentworld_export"
    _remove_output_dir_if_needed(export_output_dir, args.overwrite)
    config_overrides = {}
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=export_output_dir,
        device=args.device,
        config_overrides=config_overrides,
    )
    print(f"export_result: {export_result}")

    if args.dump_golden:
        workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages={"text": "用中文简单介绍 Qwen-Agentworld。"},
        )

    if args.quick_test:
        from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import (
            print_quick_test_result,
            quick_test_hmonnx,
        )

        quick_result = quick_test_hmonnx(
            export_result,
            prompt="用中文简单介绍 Qwen-Agentworld。",
            device=args.device,
            max_new_tokens=64,
            do_sample=False,
        )
        print_quick_test_result(quick_result)

    _move_hmquant_to_output(export_output_dir, args.output_dir)


if __name__ == "__main__":
    main()
