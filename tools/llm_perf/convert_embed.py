# Copyright (c) 2025 HOUMO AI
#
# File: convert_embed.py
# Description:
#   Convert embedding format - Tool to convert embedding weights from PyTorch format to binary format for different model types.
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
import torch
import numpy as np

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert embedding format")
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="Embedding pt file path",
    )

    parser.add_argument(
        "--type",
        required=True,
        type=str,
        help="Embedding pt file model type, choice from ['llm', 'vlm']",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    embedding_path = args.path
    type = args.type
    if os.path.exists(embedding_path) and embedding_path.endswith(".pt"):
        if type == "llm":
            embedding_weight = torch.load(
                embedding_path, map_location="cpu", weights_only=True
            )
            embedding_weight = embedding_weight["weight"]
        if type == "vlm":
            embedding_weight = torch.load(
                embedding_path, map_location="cpu", weights_only=False
            )
            if HOUMO_TARGET == "xh2":
                embedding_weight = embedding_weight.weight
        if embedding_weight.dtype == torch.bfloat16:
            embedding_weight = embedding_weight.float().half()

        embedding_data = embedding_weight.detach().cpu().numpy()
        output_path = embedding_path.replace(".pt", ".bin")
        embedding_data.tofile(output_path)
        print(f"embeding file saved to {output_path}")
