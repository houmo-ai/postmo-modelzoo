#!/usr/bin/env python3
# Copyright (c) 2026 HOUMO AI
#
# File: convert_embedding.py
# Description:
#   Convert Qwen3-TTS embedding weights and tokenizer for C++ inference.
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
from pathlib import Path

# Embedding file names
EMBEDDING_FILES = [
    ("quant_embedding.pt", "quant_embedding.bin", True),
    ("text_embedding.pt", "text_embedding.bin", True),
    ("quant_embedding_code_predictor.pt", "quant_embedding_code_predictor.bin", False),
]


def convert_embedding(input_path: str, output_path: str, is_weight: bool = False):
    """
    Convert a single embedding file from PyTorch to binary format.

    Args:
        input_path: Path to input .pt file
        output_path: Path to output .bin file
        is_weight: If True, extract 'weight' key from checkpoint (for quant_embedding)
    """
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} not found, skipping")
        return False

    import numpy as np
    import torch

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    if is_weight:
        embedding = checkpoint["weight"]
    elif isinstance(checkpoint, dict):
        # Code Predictor stores an nn.ModuleList state_dict with keys such as
        # "0.weight", "1.weight", etc. Keep the same codebook order as demo.py.
        weight_keys = [key for key in checkpoint if key.endswith(".weight")]
        if not weight_keys:
            raise ValueError(f"No embedding weights found in {input_path}")
        try:
            weight_keys.sort(key=lambda key: int(key.removesuffix(".weight")))
        except ValueError as error:
            raise ValueError(f"Unexpected Code Predictor embedding keys in {input_path}: " f"{weight_keys}") from error
        embedding = torch.stack([checkpoint[key] for key in weight_keys])
    else:
        embedding = checkpoint

    embedding = embedding.detach().cpu().numpy()

    # Convert to float16 and save
    embedding.astype(np.float16).tofile(output_path)

    # Print info
    print(f"Converted: {input_path} -> {output_path}")
    print(f"  Shape: {embedding.shape}")
    print(f"  Dtype: float16")
    print(f"  Size: {os.path.getsize(output_path)} bytes")

    return True


def convert_all_embeddings(input_dir: str, output_dir: str):
    """
    Convert all embedding files from input_dir to output_dir.

    Args:
        input_dir: Directory containing .pt embedding files
        output_dir: Directory to save .bin files
    """
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success_count = 0
    for input_name, output_name, is_weight in EMBEDDING_FILES:
        input_path = os.path.join(input_dir, input_name)
        output_path = os.path.join(output_dir, output_name)

        if convert_embedding(input_path, output_path, is_weight):
            success_count += 1

    print()
    print(f"Converted {success_count}/{len(EMBEDDING_FILES)} embedding files")

    return success_count == len(EMBEDDING_FILES)


def export_tokenizer(model_dir: str, export_dir: str = None) -> Path:
    """Export a fast Hugging Face tokenizer.json for the C++ tokenizer."""
    from transformers import AutoTokenizer

    model_path = Path(model_dir).resolve()
    export_path = Path(export_dir).resolve() if export_dir else model_path
    if not model_path.is_dir():
        raise FileNotFoundError(f"Tokenizer model directory not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        fix_mistral_regex=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required to export tokenizer.json")
    tokenizer.save_pretrained(export_path)

    tokenizer_json = export_path / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise RuntimeError(f"Failed to export tokenizer.json to {export_path}")
    print(f"Exported tokenizer: {tokenizer_json}")
    return tokenizer_json


def prepare_cpp_assets(
    input_dir: str,
    model_dir: str,
    output_dir: str = None,
) -> None:
    """Generate all binary embedding and tokenizer assets required by C++."""
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve() if output_dir else input_path
    missing_files = [
        input_path / input_name for input_name, _, _ in EMBEDDING_FILES if not (input_path / input_name).is_file()
    ]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing embedding files: {missing}")

    if not convert_all_embeddings(str(input_path), str(output_path)):
        raise RuntimeError("Failed to convert all Qwen3-TTS embedding files")
    export_tokenizer(model_dir)
