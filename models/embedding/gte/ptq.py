# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# Gte models using post-training quantization techniques.
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
from hmatc.utils.utils import (
    check_gpu,
    first_not_none,
    get_model_configs,
    parse_context_length,
)

from quant_pipline import quant_llm, export_llm, move_llm

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "gte")
    model_size = model_config.get("model_size", "1.5b")
    return f"{model_name}-{model_size}"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="raw model dir name",
    )
    parser.add_argument(
        "--model-name", type=str, default=None, help="output hmonnx model name"
    )
    parser.add_argument("--model-size", type=str, default=None, help="model size")
    parser.add_argument("--work-dir", type=str, default="work_dirs/")
    parser.add_argument("--out-dir", type=str, default="output/{}".format(HOUMO_TARGET))
    parser.add_argument("--skip-quarot", action="store_true", help="skip_quarot")
    parser.add_argument("--skip-gptq", action="store_true", help="skip_gptq")
    parser.add_argument("--w-bits", type=int, default=8)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--resume", action="store_true", help="resume from the cache")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--context-length", type=int, default=None, help="max sequence length"
    )
    parser.add_argument(
        "--input-sequence-length", type=int, default=None, help="input sequence length"
    )
    parser.add_argument("--quant-type", default=None, help="quant type")
    parser.add_argument(
        "--calibration-dataset",
        type=str,
        default=None,
        help="customized calibrate dataset, should be a json file",
    )
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    parser.add_argument("--datasets-dir", type=str, default="../../../data/datasets")
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model = first_not_none(args.model, get_default_model_dir(model_config))
    args.batch = first_not_none(args.batch, model_config.get("batch", 1))
    args.context_length = first_not_none(
        args.context_length,
        parse_context_length(model_config.get("context_length", "2k")),
    )
    args.input_sequence_length = first_not_none(
        args.input_sequence_length, model_config.get("prefill_length", 256)
    )
    args.quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "w8a8_sefp")
    )
    return args


if __name__ == "__main__":
    assert check_gpu() is True, "Error: Not found GPU device."

    args = parse_args()
    print(args)

    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        quant_llm(args)
        export_llm(args)
        move_llm(args)
    print(
        f"\n=== Quantization completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )
