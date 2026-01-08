"""
* Copyright (c) 2025 HOUMO AI
*
* File: ptq.py
* Description:
*   Post-Training Quantization (PTQ) script for ResNet50 model.
*   This script performs quantization of an ONNX model to Houmo AI format (HMONNX),
*   generates golden data for validation, and prepares the model for deployment
*   on the XH2 platform.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-License-Identifier: Apache-2.0
"""

import os
import shutil
import torch
import argparse
from loguru import logger

# Import required modules from xhquant API for quantization
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
)

HOUMO_MODEL_PATH = os.getenv("HOUMO_MODEL_PATH", "../../../models")
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the PTQ script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        default=os.path.join(HOUMO_MODEL_PATH, "resnet50.onnx"),
        help="Path to the input ONNX model file",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="resnet50",
        help="Name of the model for output files",
    )
    parser.add_argument(
        "--input_shape",
        dest="input_shape",
        type=lambda s: [int(item) for item in s.split(",")],
        default=[1, 3, 224, 224],
        help="Input shape for the model (batch, channels, height, width)",
    )
    parser.add_argument(
        "--dynamic_resize",
        dest="dynamic_resize",
        action="store_true",
        help="Whether to enable dynamic crop/resize/pad operations",
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="Directory to store the quantized model and golden data",
    )

    args = parser.parse_args()
    return args


def calibrate(args=None):
    """
    Perform post-training quantization and generate golden data.

    This function performs the complete PTQ workflow:
    1. Quantizes the ONNX model to HMONNX format
    2. Generates golden data for validation
    3. Saves the quantized model and golden data

    Args:
        args (argparse.Namespace): Command-line arguments controlling the quantization process
    """
    # Extract parameters from arguments
    model_path = args.model_path
    model_name = args.model_name
    output_path = args.model_dir
    input_shape = args.input_shape
    input_name = "input.1"  # ONNX input name
    output_name = "495"  # ONNX output name
    quant_type = "w8a8h1_sefp"  # Weight 8-bit, Activation 8-bit
    device = "cpu"
    hmonnx_model_path = os.path.join(output_path, f"{model_name}.onnx")

    os.makedirs(output_path, exist_ok=True)

    # Generate random data for calibration
    random_data = torch.randn(input_shape, dtype=torch.float32)
    # Create quantization scheme for XH2 device
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    # Convert ONNX model to HMONNX format with quantization
    convert_onnx_to_hmonnx(
        model_path,
        [random_data],
        device_type=DeviceType.XH2a,
        out_hmonnx_file=hmonnx_model_path,
        quant_config=quant_config,
        input_names=[input_name],
        output_names=[output_name],
    )

    logger.info("start save model and generate golden...")
    # Create temporary directory for golden data
    golden_dir = os.path.join(output_path, "golden")
    if not os.path.exists(golden_dir):
        os.makedirs(golden_dir)
    else:
        shutil.rmtree(golden_dir)

    # Create HMONNX inference session for golden data generation
    session = HMONNXGoldenInference(hmonnx_model_path)
    session.to(device)
    session.save_golden = True
    session.golden_dir = golden_dir
    session.step = 0
    # Run inference with half-precision data to generate golden data
    session(random_data.half().to(device))
    # Clean up temporary HMONNX model file
    if os.path.exists(hmonnx_model_path):
        os.remove(hmonnx_model_path)

    # Move generated golden data to output directory
    shutil.copytree(
        os.path.join(golden_dir, "step_0"),
        output_path,
        dirs_exist_ok=True,
    )
    shutil.rmtree(golden_dir)
    logger.info("save model and generate golden completed.")


if __name__ == "__main__":
    import platform

    arch = platform.machine()
    if arch != "x86_64":
        logger.error(f"Hmquant not support platform: {arch}")
        exit(0)

    args = get_args()
    logger.info(args)
    calibrate(args)
