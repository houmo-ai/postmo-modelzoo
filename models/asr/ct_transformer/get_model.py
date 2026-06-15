# Copyright (c) 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Model Download Tool - Python script for downloading CT-Transformer
# models.
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
import argparse
import yaml

from hmatc.utils.utils import get_houmo_version, hmatc_get_file

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = int(os.getenv("HOUMO_CORE_NUM", 2))
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def first_not_none(*args):
    """Return the first argument that is not None."""
    for arg in args:
        if arg is not None:
            return arg
    return None


def get_model_configs(config_path: str):
    """Load model configs from yaml file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    default_model_size = config.get("default_model_size", "")
    default_model_name = config.get("default_model_name", "")
    model_configs = config.get("model_configs", {})
    return default_model_size, default_model_name, model_configs


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
        help="which resource to get, choose in [raw, hmm]",
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
        "--ncore",
        dest="ncore",
        type=int,
        default=None,
        help="number of cores",
    )
    parser.add_argument(
        "--quant_type",
        dest="quant_type",
        type=str,
        default=None,
        help="quantization type, e.g. w8a8_sefp",
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
    quant_type = first_not_none(args.quant_type, model_config.get("quant_type", "w8a8_sefp"))

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": model_name,
        "model_info": {
            "model_size": model_config.get("model_size", model_size),
            "ncore": ncore,
            "ndevice": ndevice,
            "opt_level": quant_type,
            "quant_type": quant_type,
        },
        "modelscope_repo": {
            "repo_ids": model_config.get("modelscope_repo", []),
            "ignore_patterns": ["*.pt"],
        },
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

    # Rename raw model dir to canonical name
    if args.file_type == "raw":
        repo_ids = model_config.get("modelscope_repo", [])
        if repo_ids:
            raw_dir = os.path.join(args.download_dir, repo_ids[0].rsplit("/", maxsplit=1)[-1])
            target_dir = os.path.join(args.download_dir, "ct_transformer")
            if os.path.exists(raw_dir):
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                os.rename(raw_dir, target_dir)
                print(f"Renamed {raw_dir} -> {target_dir}")
