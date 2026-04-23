#!/usr/bin/env python3
# Copyright (c) 2026 HOUMO AI
#
# File: convert_embeddings.py
# Description:
#  Convert PyTorch embedding weights to binary format for CosyVoice3 C++ inference.
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

"""
Convert PyTorch embedding weights to binary format for CosyVoice3 C++ inference.

This script converts embedding weights from PyTorch .pt format to raw binary
format (float16) that can be directly loaded by the C++ HmEmbedding class.

Embedding files:
- quant_embedding.pt -> quant_embedding.bin (token embeddings for LLM)
- llm_speech_embedding.pt -> llm_speech_embedding.bin (speech token embeddings)
- llm_sos_embedding.pt -> llm_sos_embedding.bin (start-of-sequence embedding)
- llm_task_id_embedding.pt -> llm_task_id_embedding.bin (task ID embedding)
- flow_input_embedding.pt -> flow_input_embedding.bin (flow decoder input embedding)

Usage:
    python convert_embeddings.py [--input_dir <path>] [--output_dir <path>]

Default input: ../output/xh2/hmquant
Default output: same as input directory
"""

import argparse
import os
import sys

try:
    import torch
    import numpy as np
except ImportError:
    print("Error: torch and numpy are required")
    print("Install: pip install torch numpy")
    sys.exit(1)


# Embedding file names
EMBEDDING_FILES = [
    ("quant_embedding.pt", "quant_embedding.bin", True),
    ("llm_speech_embedding.pt", "llm_speech_embedding.bin", False),
    ("llm_sos_embedding.pt", "llm_sos_embedding.bin", False),
    ("llm_task_id_embedding.pt", "llm_task_id_embedding.bin", False),
    ("flow_input_embedding.pt", "flow_input_embedding.bin", False),
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

    # Load embedding
    if is_weight:
        embedding = torch.load(input_path, map_location="cpu", weights_only=True)
        embedding = embedding["weight"].cpu().numpy()
    else:
        embedding = torch.load(input_path, map_location="cpu", weights_only=True)
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch embedding weights to binary format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert with default paths
    python convert_embeddings.py

    # Specify input directory
    python convert_embeddings.py --input_dir /path/to/embeddings

    # Specify both input and output directories
    python convert_embeddings.py --input_dir ./embeddings --output_dir ./bin
        """,
    )
    parser.add_argument(
        "--input_dir",
        default="../output/xh2/hmquant",
        help="Input directory containing .pt files (default: ../output/xh2/hmquant)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for .bin files (default: same as input_dir)",
    )
    args = parser.parse_args()

    # Use input_dir as output_dir if not specified
    output_dir = args.output_dir if args.output_dir else args.input_dir

    success = convert_all_embeddings(args.input_dir, output_dir)

    if not success:
        print("Warning: Some embedding files were not converted")
        sys.exit(1)

    # export tokenizer.json
    import json
    from pathlib import Path
    from transformers import AutoTokenizer

    model_dir = Path("../Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN")
    special_tokens_path = Path("../special_tokens.json")
    export_dir = Path("../Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN")

    with special_tokens_path.open("r", encoding="utf-8") as f:
        special_tokens = json.load(f)

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.add_special_tokens(special_tokens)
    tok.save_pretrained(export_dir)


if __name__ == "__main__":
    main()