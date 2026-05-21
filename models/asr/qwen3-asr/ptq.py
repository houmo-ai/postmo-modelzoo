# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# Qwen3-ASR models using post-training quantization techniques.
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

from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import check_gpu, first_not_none, get_model_configs
from quant_pipeline import quant_asr

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "qwen3-asr")
    model_size = model_config.get("model_size", "0.6b")
    return f"{model_name}-{model_size}"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        choices=["qwen3-asr", "qwen3-forcealigner"],
        help="output hmonnx model name",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        help="model size",
    )
    parser.add_argument(
        "--out-dir", type=str, default="output/{}/hmquant".format(HOUMO_TARGET)
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--quant-type", default=None, help="quant type, default is w8a8_sefp"
    )
    parser.add_argument(
        "--gen_golden", action="store_true", help="generate golden data"
    )
    parser.add_argument(
        "--config", type=str, default="config.py", help="path of config file to save"
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        default=None,
        help="Manually fix the time dimension T of the Encoder input",
    )
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    if args.model_size is None:
        if args.model_name == default_model_name:
            args.model_size = default_model_size
        else:
            args.model_size = next(
                iter(model_configs.get(args.model_name, {})), default_model_size
            )
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model = first_not_none(args.model, get_default_model_dir(model_config))
    args.quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "w8a8_sefp")
    )
    args.max_audio_length = first_not_none(
        args.max_audio_length, model_config.get("max_audio_length", 1500)
    )
    return args


if __name__ == "__main__":
    assert check_gpu() is True, "Error: Not found GPU device."

    args = parse_args()
    print(args)

    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        quant_asr(args)
    print(
        f"\n=== Quantization completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )
