#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: get_model.py
# Description:
#   Download the raw Ornith 1.0 model selected by config.yaml.
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
import os
from pathlib import Path

import yaml
from hmatc.utils.utils import first_not_none, get_houmo_version, hmatc_get_file

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODEL_DIR / "config.yaml"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Ornith 1.0 raw model files.")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the configuration file",
    )
    parser.add_argument(
        "--type",
        dest="file_type",
        type=str,
        default="hmm",
        choices=["raw", "hmm"],
        help="which resource to get, choice in [raw, hmm]",
    )
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--model_size", default=None)
    parser.add_argument("--context_length", type=str, default=None)
    parser.add_argument("--prefill_length", type=int, default=None)
    parser.add_argument("--ndevice", type=int, default=None)
    parser.add_argument("--ncore", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--quant_type", type=str, default=None)
    parser.add_argument(
        "--download_dir", default=str(MODEL_DIR), help="where to save downloaded files"
    )
    parser.add_argument(
        "--extract_dir",
        dest="extract_dir",
        type=str,
        default=None,
        help="where to save extracted files",
    )
    parser.add_argument(
        "--source_type", choices=["jfrog", "modelscope"], default="jfrog"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name, model_size, model_config = load_model_config(
        args.config_path, args.model_name, args.model_size
    )
    context_length = first_not_none(
        args.context_length, model_config.get("context_length", "256k")
    )
    prefill_length = first_not_none(
        args.prefill_length, model_config.get("prefill_length", 256)
    )
    ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    ncore = first_not_none(args.ncore, model_config.get("ncore", 2))
    batch = first_not_none(args.batch, model_config.get("batch", 1))
    quant_type = first_not_none(args.quant_type, model_config.get("quant_type", "w4a8"))
    repo_ids = model_config.get("modelscope_repo", [])
    if not repo_ids:
        raise ValueError(f"modelscope_repo is missing for {model_name}-{model_size}")

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": model_name,
        "model_info": {
            "model_size": model_size,
            "ncore": ncore,
            "ndevice": ndevice,
            "context_len": context_length,
            "prefill_len": prefill_length,
            "batch": batch,
            "quant_type": quant_type,
        },
    }

    if args.file_type in ["raw"]:
        model_cfgs["modelscope_repo"] = {"repo_ids": repo_ids}

    _, result = hmatc_get_file(
        model_cfgs,
        file_type=args.file_type,
        download_dir=args.download_dir,
        extract_dir=args.extract_dir,
        source_type=args.source_type,
    )
    if not result.get("ret", False):
        raise RuntimeError(f"failed to download {model_name}-{model_size}: {result}")


if __name__ == "__main__":
    main()
