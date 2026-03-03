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

import os
from quant_pipeline import export_llm, export_vision, move_llm

HOUMO_TARGET = os.getenv("HOUMO_TARGET")


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        default="glm-ocr",
        help="HuggingFace model directory",
    )
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
    parser.add_argument("--image_size_w", type=int, default=336, help="image width")
    parser.add_argument("--image_size_h", type=int, default=336, help="image height")
    parser.add_argument(
        "--max_sequence_length", type=int, default=2048, help="max sequence length"
    )
    parser.add_argument(
        "--input_sequence_length",
        type=int,
        default=256,
        help="prefill input sequence length",
    )
    parser.add_argument(
        "--target_device", type=str, default="XH2a", help="target device"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_size_t", type=int, default=2, help="max temporal size")
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument(
        "--temporal_patch_size", type=int, default=2, help="temporal patch size"
    )
    return parser


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    export_llm(args)
    export_vision(args)
    move_llm(args)
