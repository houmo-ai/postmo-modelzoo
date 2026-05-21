# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# GLM-OCR models using post-training quantization techniques.
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

import argparse, os
from quant_pipeline import export_llm, export_vision, move_llm
from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import (
    check_gpu,
    first_not_none,
    get_model_configs,
    parse_context_length,
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "glm-ocr")
    model_size = model_config.get("model_size", "0.9b")
    return f"{model_name}-{model_size}"


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        default=None,
        help="HuggingFace model directory",
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="output hmonnx model name"
    )
    parser.add_argument("--model_size", type=str, default=None, help="model size")
    parser.add_argument(
        "--work_dir",
        type=str,
        default="work_dirs",
        help="output work directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="output directory",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="../../../data/pic/ocr.jpeg",
    )
    parser.add_argument("--prompt", type=str, default="Text Recognition:")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--valid", default=True, help="evaluate the model")
    parser.add_argument(
        "--profile_nodes",
        default=False,
        action="store_true",
        help="profile traced/quanted graph node outputs",
    )
    parser.add_argument(
        "--profile_decode_steps",
        type=int,
        default=3,
        help="number of decode steps to profile when --profile_nodes is set",
    )
    parser.add_argument(
        "--profile_dir",
        type=str,
        default=None,
        help="node profile output dir, default work_dir/node_profile",
    )
    parser.add_argument("--image_size_w", type=int, default=None, help="image width")
    parser.add_argument("--image_size_h", type=int, default=None, help="image height")
    parser.add_argument(
        "--max_size_t", type=int, default=None, help="max temporal size"
    )
    parser.add_argument(
        "--max_sequence_length", type=int, default=None, help="max sequence length"
    )
    parser.add_argument(
        "--input_sequence_length",
        type=int,
        default=None,
        help="prefill input sequence length",
    )
    parser.add_argument(
        "--target_device", type=str, default="XH2a", help="target device"
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument(
        "--temporal_patch_size", type=int, default=2, help="temporal patch size"
    )
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.hf_model_dir = first_not_none(
        args.hf_model_dir, get_default_model_dir(model_config)
    )
    args.max_sequence_length = first_not_none(
        args.max_sequence_length,
        parse_context_length(model_config.get("context_length", "8k")),
    )
    args.input_sequence_length = first_not_none(
        args.input_sequence_length, model_config.get("prefill_length", 256)
    )
    args.image_size_w = first_not_none(
        args.image_size_w, model_config.get("image_size_w", 672)
    )
    args.image_size_h = first_not_none(
        args.image_size_h, model_config.get("image_size_h", 672)
    )
    args.batch_size = first_not_none(args.batch_size, model_config.get("batch", 1))
    args.max_size_t = first_not_none(args.max_size_t, model_config.get("max_size_t", 2))

    return args


if __name__ == "__main__":
    assert check_gpu() is True, "Error: Not found GPU device."

    args = parse_arguments()
    print(args)

    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        export_llm(args)
        export_vision(args)
        move_llm(args)
    print(
        f"\n=== Quantization completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )
