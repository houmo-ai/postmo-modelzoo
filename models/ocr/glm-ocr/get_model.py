# Copyright 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download GLM-OCR model.
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
from hmatc.utils.utils import (
    first_not_none,
    hmatc_get_file,
    get_houmo_version,
    get_model_configs,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = int(os.getenv("HOUMO_CORE_NUM", 2))
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
LAYOUT_REPO_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"
LAYOUT_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "PP-DocLayoutV3_safetensors"
)
LAYOUT_REQUIRED_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]


def _layout_model_files_exist(model_dir: str) -> bool:
    return all(
        os.path.isfile(os.path.join(model_dir, name))
        for name in LAYOUT_REQUIRED_FILES
    )


def download_layout_model() -> None:
    if _layout_model_files_exist(LAYOUT_MODEL_DIR):
        print(f"PP-DocLayoutV3 model files already exist: {LAYOUT_MODEL_DIR}")
        return

    print(
        "Download PP-DocLayoutV3 raw model from ModelScope: "
        f"{LAYOUT_REPO_ID} -> {LAYOUT_MODEL_DIR}",
        flush=True,
    )
    from modelscope import snapshot_download

    os.makedirs(LAYOUT_MODEL_DIR, exist_ok=True)
    snapshot_download(
        LAYOUT_REPO_ID,
        local_dir=LAYOUT_MODEL_DIR,
        allow_patterns=LAYOUT_REQUIRED_FILES,
    )
    if not _layout_model_files_exist(LAYOUT_MODEL_DIR):
        missing = [
            name
            for name in LAYOUT_REQUIRED_FILES
            if not os.path.isfile(os.path.join(LAYOUT_MODEL_DIR, name))
        ]
        raise FileNotFoundError(
            f"PP-DocLayoutV3 download finished but files are missing: {missing}"
        )


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
        help="which resource to get, choise in [raw, hmm]",
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
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        help="model name",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size",
    )
    parser.add_argument(
        "--context_length",
        dest="context_length",
        type=str,
        default="",
        help="context length",
    )
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=None,
        help="number of cores",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="device number",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=None,
        help="batch size",
    )
    parser.add_argument(
        "--prefill_length",
        dest="prefill_length",
        type=int,
        default=None,
        help="prefill length",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    model_name = first_not_none(args.model_name, default_model_name)
    model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(model_name, {}).get(model_size, {})

    ncore = first_not_none(args.ncore, model_config.get("ncore", HOUMO_CORE_NUM))
    ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    batch = first_not_none(args.batch, model_config.get("batch", 1))
    prefill_length = first_not_none(
        args.prefill_length, model_config.get("prefill_length", 256)
    )
    context_length = args.context_length or model_config.get("context_length", "8k")

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": model_name,
        "model_info": {
            "model_size": model_config.get("model_size", model_size),
            "ncore": ncore,
            "ndevice": ndevice,
            "context_len": context_length,
            "prefill_len": prefill_length,
            "batch": batch,
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
    if args.file_type == "raw":
        download_layout_model()
