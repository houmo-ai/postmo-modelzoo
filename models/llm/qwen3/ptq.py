# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# Qwen3 models (8B/14B) using post-training quantization techniques.
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
from quant_pipeline import quant_llm, export_llm, move_llm
from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import get_model_configs, check_gpu

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
HOUMO_EXAMPLES_PATH = os.getenv("HOUMO_EXAMPLES_PATH", os.path.abspath("../../../"))


def parse_args() -> argparse.Namespace:
    """
    Parse script input parameters.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default="./config.yaml",
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="model path (default: qwen3-0.6b / qwen3-1.7b / qwen3-8b / qwen3-14b based on model_size)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        help="model size: 0.6b, 1.7b, 8b or 14b",
    )
    parser.add_argument("--model_name", type=str, default=None, help="model name")
    parser.add_argument("--work_dir", type=str, default="work_dirs/")
    parser.add_argument(
        "--out_dir", type=str, default=os.path.join("output", HOUMO_TARGET)
    )
    parser.add_argument(
        "--context_length", type=int, default=2048, help="max sequence length"
    )
    parser.add_argument(
        "--input_sequence_length",
        type=int,
        default=None,
        help="input sequence length (corresponds to prefill_length in config)",
    )

    parser.add_argument("--quant_type", type=str, default=None, help="quant type")
    parser.add_argument(
        "--calib_data",
        type=str,
        default=None,
        help="calibration dataset path (default: auto-select based on model_size)",
    )
    parser.add_argument("--skip-quarot", action="store_true", help="skip_quarot")
    parser.add_argument("--skip-gptq", action="store_true", help="skip_gptq")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--resume", action="store_true", help="resume from the cache")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--mix_search", type=str, default=None, help="mix search settings"
    )
    parser.add_argument(
        "--num_logits_to_keep", type=int, default=1, help="not for test ppl"
    )
    parser.add_argument(
        "--gptqmodel", action="store_true", help="use gptqmodel to quant (14b only)"
    )
    args = parser.parse_args()

    # Load config and resolve parameters
    default_model_size, model_configs = get_model_configs(args.config_path)
    args.model_size = (
        args.model_size if args.model_size is not None else default_model_size
    )
    model_config = model_configs.get(args.model_size, {})

    args.model_name = (
        args.model_name
        if args.model_name is not None
        else model_config.get("model_name", "qwen3")
    )
    args.quant_type = (
        args.quant_type
        if args.quant_type is not None
        else model_config.get("quant_type", "w4a8h0_ssfp")
    )
    args.input_sequence_length = (
        args.input_sequence_length
        if args.input_sequence_length is not None
        else model_config.get("prefill_length", 256)
    )
    if args.model is None:
        args.model = f"{args.model_name}-{args.model_size}"
    if args.calib_data is None:
        args.calib_data = os.path.join(
            HOUMO_EXAMPLES_PATH, "hmodel/xh2/examples/xh_gen_data/gen_qwen3_8b.jsonl"
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
        f"\n=== All quantization steps completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )
