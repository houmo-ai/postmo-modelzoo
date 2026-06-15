# Copyright (c) 2026 HOUMO AI
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
    args = parser.parse_args()
    return args


def _extract_embedding_weight(loaded_obj):
    if torch.is_tensor(loaded_obj):
        return loaded_obj

    if isinstance(loaded_obj, dict):
        if "weight" in loaded_obj and torch.is_tensor(loaded_obj["weight"]):
            return loaded_obj["weight"]

        # Some checkpoints may contain only one tensor entry.
        tensor_items = [
            value for value in loaded_obj.values() if torch.is_tensor(value)
        ]
        if len(tensor_items) == 1:
            return tensor_items[0]

        # Fall back to common key suffix used in state_dict.
        for key, value in loaded_obj.items():
            if key.endswith(".weight") and torch.is_tensor(value):
                return value

    if hasattr(loaded_obj, "weight") and torch.is_tensor(loaded_obj.weight):
        return loaded_obj.weight

    raise ValueError("Cannot infer embedding weight from checkpoint content")


def _load_embedding_weight(embedding_path):
    load_errors = []
    for weights_only in (True, False):
        try:
            loaded_obj = torch.load(
                embedding_path, map_location="cpu", weights_only=weights_only
            )
            return _extract_embedding_weight(loaded_obj)
        except Exception as exc:
            load_errors.append(f"weights_only={weights_only}: {exc}")

    raise RuntimeError(
        "Failed to load embedding checkpoint with both modes: "
        + " | ".join(load_errors)
    )


if __name__ == "__main__":
    args = parse_args()
    embedding_path = args.path
    if os.path.exists(embedding_path) and embedding_path.endswith(".pt"):
        embedding_weight = _load_embedding_weight(embedding_path)
        if embedding_weight.dtype == torch.bfloat16:
            embedding_weight = embedding_weight.float().half()

        embedding_data = embedding_weight.detach().cpu().numpy()
        output_path = embedding_path.replace(".pt", ".bin")
        embedding_data.tofile(output_path)
        print(f"embeding file saved to {output_path}")
    else:
        raise ValueError(
            f"Invalid embedding path (expect existing .pt file): {embedding_path}"
        )
