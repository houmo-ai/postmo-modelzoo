# Copyright (c) 2025 HOUMO AI
#
# File: get_model.py
# Description:
#  Download raw or quantized Gemma-4-26B-A4B model artifacts.
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
import yaml

from hmatc.utils.utils import get_houmo_version, hmatc_get_file, first_not_none

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_gemma4_e2b.yml")

# fmt: off
def get_args(argv=None) -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config_gemma4_e2b.yml")
    parser.add_argument("--type", dest="file_type", type=str, default="hmm", choices=["raw", "hmm"], help="which resource to get, choice in [raw, hmm]")
    parser.add_argument("--download_dir", dest="download_dir", type=str, default="./models", help="where to save downloaded model")
    parser.add_argument("--extract_dir", dest="extract_dir", type=str, default=None, help="where to save extracted files")
    parser.add_argument("--source_type", dest="source_type", type=str, default="jfrog", choices=["jfrog", "modelscope"], help="download the model from which source")
    parser.add_argument("--context_length", dest="context_length", type=int, default=None)
    parser.add_argument("--ncore", dest="ncore", type=int, default=2)
    parser.add_argument("--batch", dest="batch", type=int, default=None)
    parser.add_argument("--ndevice", dest="ndevice", type=int, default=None, help="device number")
    parser.add_argument("--prefill_length", dest="prefill_length", type=int, default=None, help="prefill length")
    return parser.parse_args(argv)
# fmt: on


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve download arguments from commandline and config file."""
    with open(args.config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    model_config = config.get("model", {})
    quant_config = config.get("quant", {})
    build_config = config.get("build", {})

    model_size = model_config.get("model_size")
    args.ndevice = first_not_none(args.ndevice, build_config.get("ndevice"), 1)
    args.context_length = first_not_none(
        args.context_length, build_config.get("context_length")
    )
    args.prefill_length = first_not_none(
        args.prefill_length, build_config.get("prefill_chunk_length")
    )
    args.ncore = first_not_none(args.ncore, build_config.get("ncore"), 2)
    args.batch = first_not_none(args.batch, build_config.get("batch"), 1)
    args.modelscope_repo = model_config.get("modelscope_repo", [])
    return args, quant_config.get("speculative_decode") == "mtp", model_size


def format_context_length(context_length: int) -> str:
    if not isinstance(context_length, int) or isinstance(context_length, bool):
        raise TypeError("context_length must be an integer")
    if context_length <= 0:
        raise ValueError("context_length must be greater than 0")
    return f"{context_length / 1024:g}k"


if __name__ == "__main__":
    # fmt: off
    args, mtp, model_size = resolve_args(get_args())
    model_name = "gemma4"
    ndevice = args.ndevice
    context_length = format_context_length(args.context_length)
    prefill_length = args.prefill_length
    batch = args.batch
    ncore = args.ncore
    model_size = f"{model_size}-mtp" if mtp else model_size
    extract_dir = f"output/{model_size}/{HOUMO_TARGET}" if args.extract_dir is None else args.extract_dir

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
        }
    }
    if args.file_type in ["raw"]:
        repo_ids: list = args.modelscope_repo
        model_cfgs["modelscope_repo"] = {"repo_ids": repo_ids}
        
    _, ret_dict = hmatc_get_file(
        model_cfgs,
        args.file_type,
        args.download_dir,
        extract_dir,
        args.source_type,
    )
    if ret_dict.get("ret", False) is False:
        exit(1)
