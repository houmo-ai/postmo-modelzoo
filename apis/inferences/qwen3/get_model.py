# Copyright (c) 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Model Download Script for Houmo AI LLM - Handles downloading pre-trained LLM models from specified sources
#   and converting the quantized embedding layer to binary format.
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
import sys
import argparse
import torch

# Get Houmo examples path from environment variable or use default
HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../../..")
# Add Houmo AI tools to path
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/hmatc")

from hmatc.utils.utils import hmatc_get_file, get_houmo_version, get_file_from_jfrog, _extract_files
from qwen_demo_utils import download_file
# Get and validate target accelerator type
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments for model download and extraction.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Model Download and Quantization Script"
    )

    parser.add_argument(
        "--download_dir",
        dest="download_dir",
        type=str,
        default=os.path.join(HOUMO_EXAMPLES_PATH, "apis/models"),
        help="Directory to save downloaded model files",
    )
    parser.add_argument(
        "--extract_dir",
        dest="extract_dir",
        type=str,
        default=".",
        help="Directory to extract downloaded files",
    )
    parser.add_argument(
        "--source_type",
        dest="source_type",
        type=str,
        default="jfrog",
        choices=["jfrog", "modelscope"],
        help="Source to download model from (jfrog or modelscope)",
    )
    parser.add_argument(
        "--context_length",
        dest="context_length",
        type=str,
        default="8k",
        help="context length",
    )
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=2,
        help="core number",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command line arguments
    args = get_args()

    # Define model configuration for download
    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": "qwen3",
        "model_info": {
            "model_size": "8b",
            "ncore": args.ncore,
            "ndevice": 1,
            "context_len": args.context_length,
            "prefill_len": 256,
            "batch": 1,
        },
        "modelscope_repo": {
            "repo_ids": ["qwen/qwen3-8b"],
            "local_dirs": ["./qwen3-8b"],
        },
    }

    # Download and extract the model
    _, ret_dict = hmatc_get_file(
        model_cfgs,
        "hmm",
        args.download_dir,
        args.extract_dir,
        args.source_type,
    )

    # Exit with error code if download/extraction fails
    if ret_dict.get("ret", False) is False:
        exit(1)

    # Convert quantized embedding to binary format
    embedding_path = "hmquant/quant_embedding.pt"
    if os.path.exists(embedding_path):
        # Load embedding weights
        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=True
        )["weight"]

        # Convert bfloat16 to float16 if necessary
        if embedding_weight.dtype == torch.bfloat16:
            embedding_weight = embedding_weight.float().half()

        # Save as binary file
        embedding_data = embedding_weight.cpu().numpy()
        embedding_data.tofile(embedding_path.replace(".pt", ".bin"))

        tokenizer_path = "3rdparty/qwen3-tokenizers-cpp.zip"
        target_dir = "./3rdparty"
        get_file_from_jfrog(tokenizer_path, target_dir, target_dir)
        download_file(
            "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip",
            "3rdparty/eigen-3.4.0.zip"
        )

        _extract_files("3rdparty/eigen-3.4.0.zip", target_dir)
        os.rename("3rdparty/eigen-3.4.0", "3rdparty/eigen3")
        for file in os.listdir(target_dir):
            if file.endswith(".zip"):
                os.remove(os.path.join(target_dir, file))


