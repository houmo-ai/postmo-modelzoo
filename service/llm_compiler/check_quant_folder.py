# Copyright 2025 HOUMO AI
#
# File: check_quant_folder.py
# Description:
#   Check the quantized model folder structure and files for LLM compilation.
#   This script verifies that all required directories and files exist for
#   the quantized model before proceeding with the compilation process.
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
import logging
from compiler_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the quantized model folder checker."""
    parser = argparse.ArgumentParser(description="Check quant folder")
    parser.add_argument(
        "-n",
        "--model_name",
        required=True,
        type=str,
        help="(required) model name, example: qwen3",
    )
    parser.add_argument(
        "-qm",
        "--quant_model",
        required=True,
        type=str,
        help="(required) model path, example: /models/llm/hmquant_qwen3",
    )
    parser.add_argument(
        "-path",
        "--quant_model_path",
        required=True,
        type=str,
        help="(required) Quantized model path (Jfrog url)",
    )

    args = parser.parse_args()
    return args


def _check_folder(folder_path) -> bool:
    """Check if a folder exists.

    Args:
        folder_path (str): Path to the folder to check

    Returns:
        bool: True if folder exists, False otherwise
    """
    if not os.path.isdir(folder_path):
        logger.error(f"Missing folder {folder_path}.")
        return False
    return True


def _check_file(file_path) -> bool:
    """Check if a file exists.

    Args:
        file_path (str): Path to the file to check

    Returns:
        bool: True if file exists, False otherwise
    """
    if not os.path.isfile(file_path):
        logger.error(f"Missing file {file_path}")
        return False
    return True


def _check_model_source(quant_model_path) -> str:
    """Determine the source type of the quantized model.

    Args:
        quant_model_path (str): Path to the quantized model

    Returns:
        str: Source type ("jfrog" for remote, "local" for local storage)
    """
    if "http" in quant_model_path:
        return "jfrog"
    return "local"


def check_quant_model(quant_model_path: str, quant_model: str, model_name: str) -> bool:
    """Check the quantized model folder structure and required files.

    Args:
        quant_model_path (str): Path to the quantized model source (JFrog URL or local path)
        quant_model (str): Local path where the quantized model should be located
        model_name (str): Name of the model being checked

    Returns:
        bool: True if all required components exist, False otherwise
    """
    import glob

    # Get the target hardware from environment variable
    target = os.getenv("HOUMO_TARGET", None)
    if target is None:
        return False

    # Normalize model name for special cases
    model_name = "deepseek" if "deepseek" in model_name else model_name
    quant_model_src = _check_model_source(quant_model_path)
    if quant_model_src in ["jfrog"]:
        # quant model path is Jfrog url
        import sys

        sys.path.append(f"{script_dir}/../../hmatc")
        from hmatc.utils.utils import get_file_from_jfrog

        get_file_from_jfrog(quant_model_path, quant_model, quant_model)

        # Move contents from the nested hmquant directory to the parent directory
        if os.path.exists(f"{quant_model}/hmquant"):
            os.system(f"mv -f {quant_model}/hmquant/* {quant_model}/")
            os.system(f"rm -rf {quant_model}/hmquant")

    # Skip checks for certain model types
    if model_name in ["bge", "gte", "qwen3-reranker"]:
        return True

    # Define required directories
    decoder_dir = os.path.join(quant_model, "decoder")
    prefill_dir = os.path.join(quant_model, "prefill")
    folder_list = [decoder_dir, prefill_dir]

    # Define required files
    embedding_file = os.path.join(quant_model, "quant_embedding.pt")
    decoder_file = os.path.join(decoder_dir, f"hmquant_{model_name}_with_act.onnx")
    prefill_file = os.path.join(prefill_dir, f"hmquant_{model_name}_with_act.onnx")
    file_list = [decoder_file, prefill_file]
    # Add embedding file to check list unless it's whisper model
    if model_name != "whisper":
        file_list += [embedding_file]

    # Add weight file based on target hardware
    if target == "xh1":
        if model_name in ["qwen2.5-vl", "qwen3-vl"]:
            decoder_weight_file = os.path.join(decoder_dir, "weight.npy")
            prefill_weight_file = os.path.join(prefill_dir, "weight.npy")
            visual_weight_file = os.path.join(quant_model, "visual", "weight.npy")
            file_list.append(decoder_weight_file)
            file_list.append(prefill_weight_file)
            file_list.append(visual_weight_file)
        else:
            weight_file = os.path.join(quant_model, "weight.npy")
            file_list.append(weight_file)
    if all(_check_folder(ele) for ele in folder_list) is False:
        return False
    if all(_check_file(ele) for ele in file_list) is False:
        return False
    if target == "xh2":
        # For xh2 target, check for external data files
        decoder_external = list(glob.glob(decoder_dir + "/*external_data"))
        prefill_external = list(glob.glob(prefill_dir + "/*external_data"))
        if len(decoder_external) == 0 or len(prefill_external) == 0:
            logger.error("Missing external data.")
            return False

    return True


if __name__ == "__main__":
    args = parse_args()

    quant_model_path = args.quant_model_path
    quant_model = args.quant_model
    model_name = args.model_name

    if not check_quant_model(quant_model_path, quant_model, model_name):
        exit(-1)
