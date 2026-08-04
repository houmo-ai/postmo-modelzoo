# Copyright (c) 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download Qwen3.5 / Qwen3.6 models for text generation tasks.
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
import argparse
import shutil
from pathlib import Path
from hmatc.utils.utils import (
    first_not_none,
    hmatc_get_file,
    get_houmo_version,
    get_model_configs,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
LORA_DATASET_DIR = "3.5-35B-lora"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


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
        "--type",
        dest="file_type",
        type=str,
        default="hmm",
        choices=["raw", "hmm"],
        help="which resource to get, choice in [raw, hmm]",
    )
    parser.add_argument(
        "--download_dir",
        dest="download_dir",
        type=str,
        default=".",
        help="where to save downloaded model",
    )
    parser.add_argument(
        "--extract_dir",
        dest="extract_dir",
        type=str,
        default=None,
        help="where to save extracted files",
    )
    parser.add_argument(
        "--source_type",
        dest="source_type",
        type=str,
        default="jfrog",
        choices=["jfrog", "modelscope"],
        help="download the model from which source",
    )
    parser.add_argument(
        "--context_length",
        dest="context_length",
        type=str,
        default=None,
        help="context length",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=None,
        help="batch size",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        help="model name: qwen3.5 or qwen3.6",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size: 0.8b, 2b, 4b, 9b, 27b, 35b-a3b, 122b-a10b",
    )
    parser.add_argument(
        "--quant_type",
        dest="quant_type",
        type=str,
        default=None,
        help="quantization type",
    )
    parser.add_argument(
        "--mtp",
        dest="mtp",
        action="store_true",
        default=False,
        help="whether it is an mtp model",
    )
    parser.add_argument(
        "--lora",
        dest="lora",
        action="store_true",
        default=False,
        help="whether to download LoRA raw files",
    )
    args = parser.parse_args()
    return args


def _move_lora_dataset_to_work_dirs(download_dir: str) -> None:
    target = Path("work_dirs") / LORA_DATASET_DIR
    candidates = [
        Path(LORA_DATASET_DIR),
        Path(download_dir) / LORA_DATASET_DIR,
        Path(os.getenv("HOUMO_DATASETS_PATH", ".")) / LORA_DATASET_DIR,
    ]
    for source in candidates:
        source = source.resolve()
        if not source.is_dir() or source == target.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(source), str(target))
        print(f"LoRA dataset moved to: {target}")
        return


def _get_resource_model_size(model_size: str, args: argparse.Namespace) -> str:
    if args.mtp and args.lora:
        raise ValueError("--mtp and --lora cannot be enabled at the same time")
    if args.mtp:
        return f"{model_size}-mtp"
    if args.lora:
        return f"{model_size}-lora"
    return model_size


if __name__ == "__main__":
    args = get_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )

    model_name = first_not_none(args.model_name, default_model_name)
    model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(model_name, {}).get(model_size, {})

    context_length = first_not_none(
        args.context_length, model_config.get("context_length", "256k")
    )
    batch = first_not_none(args.batch, model_config.get("batch", 1))
    ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "wmix_amix")
    )

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": model_name,
        "model_info": {
            "model_size": _get_resource_model_size(model_size, args),
            "ncore": model_config.get("ncore", 2),
            "ndevice": ndevice,
            "context_len": context_length,
            "prefill_len": 256,
            "batch": batch,
            "quant_type": quant_type,
        },
        "raw_files": {
            "raw_path": "3rdparty/wikitext-2-raw-v1.zip",
            "other_files": [f"models/dataset/{LORA_DATASET_DIR}.zip"] if args.lora else [],
        },
        "modelscope_repo": {"repo_ids": model_config.get("modelscope_repo", [])},
    }

    _, ret_dict = hmatc_get_file(
        model_cfgs,
        args.file_type,
        args.download_dir,
        args.extract_dir,
        args.source_type,
    )
    if ret_dict.get("ret", False) is False:
        exit(1)
    if args.file_type == "raw" and args.lora:
        _move_lora_dataset_to_work_dirs(args.download_dir)
