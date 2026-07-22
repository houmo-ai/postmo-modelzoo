# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   emotion2vec model post-training quantization tool.
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

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from hmatc.utils.utils import get_model_configs
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def find_configs_merak_dir() -> Path:
    spec = importlib.util.find_spec("xhmodel_merak")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("Python package xhmodel_merak not found")

    xhmodel_merak_dir = Path(next(iter(spec.submodule_search_locations)))
    configs_merak_dir = xhmodel_merak_dir.parent / "configs_merak"
    if not configs_merak_dir.is_dir():
        raise FileNotFoundError(f"configs_merak directory not found next to xhmodel_merak: {configs_merak_dir}")
    return configs_merak_dir


def get_workflow_config_path(config_path: str, model_name: str, model_size: str) -> Path:
    _, _, model_configs = get_model_configs(config_path)
    model_config = model_configs.get(model_name, {}).get(model_size, {})
    workflow_config = model_config.get("workflow_config")
    if not workflow_config:
        raise ValueError(f"workflow_config not found for {model_name}/{model_size} in {config_path}")

    workflow_config_path = find_configs_merak_dir() / workflow_config
    if not workflow_config_path.is_file():
        raise FileNotFoundError(f"Workflow config not found: {workflow_config_path}")
    return workflow_config_path


def export_hmonnx(
    model_dir: str,
    config_path: str,
    output_dir: str,
    device: str = "cuda",
    overwrite: bool = False,
    golden_audio: str | None = None,
) -> Path:
    output_path = Path(output_dir)
    if overwrite and output_path.exists():
        shutil.rmtree(output_path)

    workflow = AutoLLMWorkflow.from_config(
        model_dir=model_dir,
        config_path=config_path,
    )
    quant_result = workflow.quant(
        output_dir=f"{output_dir}_quant",
        device=device,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=output_dir,
        device=device,
    )
    if golden_audio is not None:
        golden_dir = workflow.dump_golden(
            export_result=export_result,
            device=device,
            input_messages={"audio": golden_audio},
        )
        print(f"golden_dir: {golden_dir}")
    return Path(export_result.work_dir) / "emotion2vec_meta.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the emotion2vec Merak W8A8 export workflow.")
    parser.add_argument("--model-dir", default="emotion2vec_plus_large")
    parser.add_argument("--model-name", default="emotion2vec")
    parser.add_argument("--model-size", default="plus_large")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="Dump official PyTorch golden data after HMONNX export.",
    )
    parser.add_argument(
        "--golden-audio",
        default="emotion2vec_plus_large/example/test.wav",
        help="Audio used by workflow.dump_golden().",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    workflow_config_path = get_workflow_config_path(args.config_path, args.model_name, args.model_size)
    meta_path = export_hmonnx(
        model_dir=args.model_dir,
        config_path=str(workflow_config_path),
        output_dir=args.output_dir,
        device=args.device,
        overwrite=args.overwrite,
        golden_audio=args.golden_audio if args.dump_golden else None,
    )
    print(meta_path)


if __name__ == "__main__":
    main()
