# Copyright (c) 2026 HOUMO AI
#
# File: get_model.py
# Description:
#   Download MiniCPM-V 4.6 raw and pre-compiled model resources.
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

from hmatc.utils.utils import (
    first_not_none,
    get_houmo_version,
    get_model_configs,
    hmatc_get_file,
)


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_CORE_NUM = int(os.getenv("HOUMO_CORE_NUM", 2))
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
        default=None,
        help="context length",
    )
    parser.add_argument(
        "--prefill_length",
        dest="prefill_length",
        type=int,
        default=None,
        help="prefill length",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=None,
        help="batch size",
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
        "--quant_type",
        dest="quant_type",
        type=str,
        default=None,
        help="quantization type",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    download_dir = os.path.abspath(args.download_dir)

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )

    model_name = first_not_none(args.model_name, default_model_name)
    model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(model_name, {}).get(model_size)
    if model_config is None:
        supported = [
            f"{name}-{size}"
            for name, sizes in model_configs.items()
            for size in sizes
        ]
        raise ValueError(
            f"Unsupported model combination '{model_name}-{model_size}'. "
            f"Supported models: {', '.join(supported)}"
        )

    context_length = first_not_none(
        args.context_length, model_config.get("context_length", "256k")
    )
    prefill_length = first_not_none(
        args.prefill_length, model_config.get("prefill_length", 256)
    )
    batch = first_not_none(args.batch, model_config.get("batch", 1))
    ncore = first_not_none(
        args.ncore, model_config.get("ncore", HOUMO_CORE_NUM)
    )
    ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "w8a16")
    )

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
            "quant_type": quant_type,
        },
        "modelscope_repo": {
            "repo_ids": model_config.get("modelscope_repo", []),
        },
    }

    _, result = hmatc_get_file(
        model_cfgs,
        args.file_type,
        download_dir,
        args.extract_dir,
        args.source_type,
    )
    if result.get("ret", False) is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
