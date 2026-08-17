# Copyright (c) 2026 HOUMO AI
#
# File: get_model.py
# Description:
#   Download MiniCPM-V 4.5 raw and pre-compiled model resources.
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

"""Download MiniCPM-V 4.5 raw or precompiled model resources."""

import argparse
import os

from hmatc.utils.utils import first_not_none, get_houmo_version, get_model_configs, hmatc_get_file


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--type", dest="file_type", choices=["raw", "hmm"], default="hmm")
    parser.add_argument("--download_dir", default=".")
    parser.add_argument("--extract_dir", default=None)
    parser.add_argument("--source_type", choices=["jfrog", "modelscope"], default="jfrog")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--model_size", default=None)
    parser.add_argument("--context_length", default=None)
    parser.add_argument("--prefill_length", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--ncore", type=int, default=None)
    parser.add_argument("--ndevice", type=int, default=None)
    parser.add_argument("--quant_type", default=None)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    default_size, default_name, model_configs = get_model_configs(args.config_path)
    model_name = first_not_none(args.model_name, default_name)
    model_size = first_not_none(args.model_size, default_size)
    model_config = model_configs.get(model_name, {}).get(model_size)
    if model_config is None:
        raise ValueError(f"Unsupported model combination: {model_name}-{model_size}")

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": model_name,
        "model_info": {
            "model_size": model_size,
            "ncore": first_not_none(args.ncore, model_config.get("ncore", 2)),
            "ndevice": first_not_none(args.ndevice, model_config.get("ndevice", 1)),
            "context_len": first_not_none(args.context_length, model_config.get("context_length", "40k")),
            "prefill_len": first_not_none(args.prefill_length, model_config.get("prefill_length", 256)),
            "batch": first_not_none(args.batch, model_config.get("batch", 1)),
            "quant_type": first_not_none(args.quant_type, model_config.get("quant_type", "w8a8h1_sefp")),
        },
        "modelscope_repo": {"repo_ids": model_config.get("modelscope_repo", [])},
    }
    _, result = hmatc_get_file(
        model_cfgs,
        args.file_type,
        os.path.abspath(args.download_dir),
        args.extract_dir,
        args.source_type,
    )
    if not result.get("ret", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
