# Copyright (c) 2026 HOUMO AI
#
# File: ptq.py
# Description:
#   Quantize and export MiniCPM-V 4.5 artifacts for Houmo devices.
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

"""Quantize and export MiniCPM-V 4.5 artifacts for Houmo devices."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
from pathlib import Path

from hmatc.utils.utils import get_model_configs

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def _configs_merak_dir() -> Path:
    spec = importlib.util.find_spec("xhmodel_merak")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("Python package xhmodel_merak not found")
    path = Path(next(iter(spec.submodule_search_locations))).parent / "configs_merak"
    if not path.is_dir():
        raise FileNotFoundError(f"configs_merak directory not found: {path}")
    return path


def _workflow_config_path(config_path: str) -> Path:
    _, _, configs = get_model_configs(config_path)
    relative_path = configs["minicpm"]["v-4.5"]["workflow_config"]
    path = _configs_merak_dir() / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Workflow config not found: {path}")
    return path


def _external_data_paths(onnx_path: Path, output_root: Path) -> set[Path]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    paths = {
        (onnx_path.parent / item.value).resolve()
        for tensor in model.graph.initializer
        for item in tensor.external_data
        if item.key == "location"
    }
    for path in paths:
        if not path.is_relative_to(output_root):
            raise ValueError(f"External data path escapes output directory: {path}")
    return paths


def _cleanup_unreferenced_external_data(output_dir: Path) -> None:
    output_root = output_dir.resolve()
    referenced_files: set[Path] = set()
    for onnx_path in output_root.rglob("*.onnx"):
        referenced_files.update(_external_data_paths(onnx_path, output_root))

    for external_path in output_root.rglob("*_external_data"):
        resolved_path = external_path.resolve()
        if external_path.is_file() and resolved_path not in referenced_files:
            size = external_path.stat().st_size
            external_path.unlink()
            print(
                "Removed unreferenced external data: "
                f"{external_path.relative_to(output_root)} ({size / (1024 ** 3):.2f} GiB)"
            )


def _validate_export_layout(output_dir: Path) -> None:
    required_dirs = ("prefill", "vision_1x", "vision_6x")
    missing_dirs = [name for name in required_dirs if not (output_dir / name).is_dir()]
    decode_dirs = sorted(path for path in output_dir.glob("*decode*") if path.is_dir())
    if missing_dirs or len(decode_dirs) != 1:
        raise FileNotFoundError(
            f"Unexpected workflow export layout under {output_dir}: "
            f"missing directories {missing_dirs}, decode directories {decode_dirs}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MiniCPM-V 4.5 Merak W8A8 export workflow.")
    parser.add_argument("--model-dir", default="MiniCPM-V-4_5")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", default=os.path.join("output", HOUMO_TARGET, "hmquant")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    importlib.import_module("xhmodel_merak.xh_llm.models.minicpm_v_4_5")
    from xhmodel_merak.workflows import AutoWorkflow

    output_path = Path(args.output_dir)
    if args.overwrite and output_path.exists():
        shutil.rmtree(output_path)
    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=str(_workflow_config_path(args.config_path)),
    )
    quant_result = workflow.quant(output_dir=f"{args.output_dir}_quant", device=args.device)
    export_result = workflow.export(quant_result=quant_result, output_dir=args.output_dir, device=args.device)
    if args.dump_golden:
        print(f"golden_dir: {workflow.dump_golden(export_result=export_result, device=args.device)}")
    meta_path = Path(export_result.work_dir) / "export_meta_info.json"
    _validate_export_layout(meta_path.parent)
    _cleanup_unreferenced_external_data(meta_path.parent)
    print(meta_path)


if __name__ == "__main__":
    main()
