# Copyright (c) 2026 HOUMO AI
#
# File: ptq.py
# Description:
#   MiniCPM-V 4.6 model post-training quantization tool.
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

"""Quantize and export MiniCPM-V 4.6 artifacts for Houmo devices."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path

from hmatc.utils.utils import get_model_configs

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
    print(workflow_config)
    if not workflow_config:
        raise ValueError(f"workflow_config not found for {model_name}/{model_size} in {config_path}")

    workflow_config_path = find_configs_merak_dir() / workflow_config
    if not workflow_config_path.is_file():
        raise FileNotFoundError(f"Workflow config not found: {workflow_config_path}")
    return workflow_config_path


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.write("\n")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _move_to_output(source: Path, output_dir: Path) -> Path:
    destination = output_dir / source.name
    _remove_path(destination)
    shutil.move(str(source), str(destination))
    return destination


def _move_llm_stages(
    llm_artifact_dir: Path,
    output_dir: Path,
    *metadata_objects: dict,
) -> None:
    for stage in ("prefill", "decode"):
        source_dir = llm_artifact_dir / stage
        if not source_dir.is_dir():
            raise FileNotFoundError(f"LLM {stage} directory not found: {source_dir}")
        _move_to_output(source_dir, output_dir)

        hmonnx_key = f"{stage}_hmonnx"
        for metadata in metadata_objects:
            hmonnx_path = metadata.get(hmonnx_key)
            if hmonnx_path:
                metadata[hmonnx_key] = str(Path(stage) / Path(hmonnx_path).name)


def _remove_empty_artifact_dirs(llm_artifact_dir: Path, output_dir: Path) -> None:
    artifact_parent = llm_artifact_dir.parent
    llm_artifact_dir.rmdir()
    if artifact_parent != output_dir:
        artifact_parent.rmdir()


def _move_vision_hmonnx_to_parent(output_dir: Path) -> None:
    for vision_name in ("vision_4x", "vision_16x"):
        vision_dir = output_dir / vision_name
        if not vision_dir.is_dir():
            continue
        hmonnx_dir = vision_dir / "hmonnx"
        if not hmonnx_dir.is_dir():
            continue

        for source in list(hmonnx_dir.iterdir()):
            _move_to_output(source, vision_dir)
        hmonnx_dir.rmdir()


def move_llm_artifacts_to_output_root(meta_path: Path) -> None:
    meta_info = _read_json(meta_path)

    output_dir = meta_path.parent
    llm_info = meta_info.get("llm")
    if not isinstance(llm_info, dict):
        raise ValueError(f"LLM export metadata not found in {meta_path}")

    artifact_dir = llm_info.get("artifact_dir")
    if not artifact_dir:
        raise ValueError(f"LLM artifact_dir not found in {meta_path}")

    llm_artifact_dir = output_dir / artifact_dir
    llm_metadata_path = output_dir / llm_info.get("metadata", Path(artifact_dir) / "golden_meta_info.json")
    if not llm_metadata_path.is_file():
        raise FileNotFoundError(f"LLM metadata not found: {llm_metadata_path}")
    llm_metadata = _read_json(llm_metadata_path)

    _move_llm_stages(llm_artifact_dir, output_dir, llm_info, llm_metadata)

    for source in list(llm_artifact_dir.iterdir()):
        _move_to_output(source, output_dir)
    _remove_empty_artifact_dirs(llm_artifact_dir, output_dir)
    _move_vision_hmonnx_to_parent(output_dir)

    moved_metadata_path = output_dir / llm_metadata_path.name
    _write_json(moved_metadata_path, llm_metadata)

    llm_info["artifact_dir"] = "."
    llm_info["metadata"] = moved_metadata_path.name
    for key in ("quant_embedding", "hf_config"):
        value = llm_info.get(key)
        if value:
            llm_info[key] = Path(value).name

    _write_json(meta_path, meta_info)


def export_hmonnx(
    model_dir: str,
    config_path: str,
    output_dir: str,
    device: str = "cuda",
    overwrite: bool = False,
    dump_golden: bool = False,
) -> Path:
    from xhmodel_merak.workflows import AutoWorkflow

    output_path = Path(output_dir)
    if overwrite and output_path.exists():
        shutil.rmtree(output_path)

    workflow = AutoWorkflow.from_config(
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
    if dump_golden:
        golden_dir = workflow.dump_golden(
            export_result=export_result,
            device="cpu",
        )
        print(f"golden_dir: {golden_dir}")
    meta_path = Path(export_result.work_dir) / "export_meta_info.json"
    move_llm_artifacts_to_output_root(meta_path)
    return meta_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minicpm-v-4.6 Merak W8A8 export workflow.")
    parser.add_argument("--model-dir", default="MiniCPM-V-4.6")
    parser.add_argument("--model-name", default="minicpm")
    parser.add_argument("--model-size", default="v-4.6")
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
        dump_golden=args.dump_golden,
    )
    print(meta_path)


if __name__ == "__main__":
    main()
