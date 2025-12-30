"""
Model Download and Quantization Script for Houmo AI LLM

This script handles downloading pre-trained LLM models from specified sources
and converting the quantized embedding layer to binary format for Houmo AI 
accelerator deployment.

Copyright (c) 2025 HOUMOAI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import argparse
import torch

# Get Houmo examples path from environment variable or use default
HOUMO_EXAMPLES_PATH = os.environ.get('HOUMO_EXAMPLES_PATH', '../../..')
# Add Houmo AI tools to path
sys.path.insert(0, f'{HOUMO_EXAMPLES_PATH}/hmatc')

from hmatc.utils.utils import hmatc_get_file, get_houmo_version

# Get and validate target accelerator type
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments for model download and extraction.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Model Download and Quantization Script")

    parser.add_argument(
        '--download_dir',
        dest='download_dir',
        type=str,
        default=os.path.join(HOUMO_EXAMPLES_PATH, "apis/models"),
        help='Directory to save downloaded model files',
    )

    parser.add_argument(
        "--extract_dir",
        dest="extract_dir",
        type=str,
        default=".",
        help='Directory to extract downloaded files',
    )

    parser.add_argument(
        "--source_type",
        dest="source_type",
        type=str,
        default="jfrog",
        choices=["jfrog", "modelscope"],
        help='Source to download model from (jfrog or modelscope)',
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse command line arguments
    args = get_args()

    # Set default Houmo model zoo URL if not provided
    if "HOUMO_MODELZOO_URL" not in os.environ:
        os.environ["HOUMO_MODELZOO_URL"] = "http://139.224.0.199:8082/artifactory/houmo/release"

    # Define model configuration for download
    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": "qwen3",
        "model_info": {
            "model_size": "8b",
            "ncore": 2,
            "ndevice": 1,
            "context_len": "8k",
            "prefill_len": 256,
            "batch": 1,
        },
        "hmm_files": {
            "other_files": ["models/qwen3/3rdparty.zip"],
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
        # Print target accelerator type
        print(HOUMO_TARGET)

        # Load embedding weights
        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=True
        )['weight']

        # Convert bfloat16 to float16 if necessary
        if embedding_weight.dtype == torch.bfloat16:
            embedding_weight = embedding_weight.float().half()

        # Save as binary file
        embedding_data = embedding_weight.cpu().numpy()
        embedding_data.tofile(embedding_path.replace(".pt", ".bin"))