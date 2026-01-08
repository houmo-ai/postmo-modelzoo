# Copyright 2025 HOUMO AI
#
# File: norm_quant_folder.py
# Description:
#   Normalize and organize quantized model folders for deployment.
#
#   This script processes quantized model files and organizes them into a standardized
#   directory structure with appropriate naming conventions.
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
import sys
import shutil
import glob
import re
import subprocess
from pathlib import Path

QUANT_MODELS_URL = (
    "http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/"
)
JFROG_USER = os.getenv("JFROG_USER")
JFROG_PASSWORD = os.getenv("JFROG_PASSWORD")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the quantization folder normalization process."""
    parser = argparse.ArgumentParser(description="Normalize Quant Folder")
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="(required) model name, example: qwen3",
    )
    parser.add_argument(
        "--backend",
        required=True,
        type=str,
        help="(required) houmo backend, example: xh1, xh2",
    )
    parser.add_argument(
        "--quant_folder",
        required=True,
        type=str,
        help="(required) model name, example: qwen3",
    )
    parser.add_argument(
        "--result_folder",
        required=True,
        type=str,
        help="The path for storing the results.",
    )
    parser.add_argument(
        "--zipped_name",
        default=None,
        type=str,
        help="The name of the compressed package. If provided, it will be uploaded to Jfrog. Example: hmquant_xh2_qwen3_8b_2k_20250812",
    )

    args = parser.parse_args()
    return args


def find_file_recursive(root_dir, pattern, excludes=list()) -> list:
    """
    Recursively find files matching the given pattern while excluding specified substrings.

    Args:
        root_dir (str): Root directory to start the search
        pattern (str): Pattern to match (supports glob wildcards)
        excludes (list): List of substrings to exclude from results

    Returns:
        list: List of file paths that match the pattern and don't contain excluded substrings
    """
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filename_new = filename.lower()
            if glob.fnmatch.fnmatch(filename_new, pattern):
                flag = True
                for ex_str in excludes:
                    if ex_str in filename:
                        flag = False
                        break
                if flag is True:
                    matches.append(os.path.join(dirpath, filename))
    return matches


def _process_model_files_xh2(
    backend: str, quant_dir: str, result_dir: str, model_name: str, model_type: str
) -> bool:
    """
    Process model files for XH2 backend, organizing them into standardized structure.

    Args:
        backend (str): Backend identifier (should be 'xh2' for this function)
        quant_dir (str): Source directory containing quantized model files
        result_dir (str): Destination directory for organized files
        model_name (str): Name of the model being processed
        model_type (str): Type of model component ('prefill', 'decode', 'vision', 'encoder')

    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # 1. Create target directory structure
        model_res_dir = os.path.join(result_dir, model_type)
        if model_type == "decode":
            model_res_dir = os.path.join(result_dir, "decoder")
        elif model_type == "vision":
            model_res_dir = os.path.join(result_dir, "visual")
        os.makedirs(model_res_dir, exist_ok=True)
        print(f"Created target directory structure: {model_res_dir}")

        # if model_name == "qwen3-vl" and model_type == "vision":
        #     model_type = "visual"

        # 2. Process hmquant_*_with_act.onnx files
        if model_type == "encoder":
            onnx_files = find_file_recursive(
                quant_dir,
                f"hmquant_*with_act.onnx",
                excludes=["decode", "prefill", "vision"],
            )
        else:
            onnx_files = find_file_recursive(
                quant_dir, f"hmquant_*{model_type}*with_act.onnx"
            )
        if len(onnx_files) == 0:
            return False
        # Take the first matching file
        onnx_src = onnx_files[0]
        onnx_dst = os.path.join(model_res_dir, f"hmquant_{model_name}_with_act.onnx")
        shutil.copy(onnx_src, onnx_dst)
        print(f"Renamed and moved: {onnx_src} -> {onnx_dst}")

        # 3. Process *_external_data files
        if model_type == "encoder":
            external_files = find_file_recursive(
                quant_dir,
                "*external_data",
                excludes=["decode", "prefill", "vision"],
            )
        else:
            external_files = find_file_recursive(
                quant_dir, f"*{model_type}*external_data"
            )
        if len(external_files) == 0:
            return False
        for external_file in external_files:
            # Check if the file is in the same directory as the ONNX file
            onnx_dir = os.path.dirname(onnx_files[0]) if onnx_files else None
            external_dir = os.path.dirname(external_file)

            # If ONNX file was found, only move external_data files from the same directory
            if onnx_dir and external_dir == onnx_dir:
                file_name = os.path.basename(external_file)
                external_dst = os.path.join(model_res_dir, file_name)

                shutil.copy(external_file, external_dst)
                print(f"Moved: {external_file} -> {external_dst}")

        # 4. Process golden data
        if model_type == "encoder":
            golden = find_file_recursive(
                quant_dir,
                "hmquant_*_input.npy",
                excludes=["decode", "prefill", "vision"],
            )
            opt_golden = find_file_recursive(
                quant_dir,
                "hmquant_*_output.npy",
                excludes=["decode", "prefill", "vision"],
            )
            golden += opt_golden
        else:
            golden = find_file_recursive(quant_dir, f"hmquant_*{model_type}*_input.npy")
            opt_golden = find_file_recursive(
                quant_dir, f"hmquant_*{model_type}*_output.npy"
            )
            golden += opt_golden

        name_pattern = r"(?<=hmquant_).*?_decode"
        if model_type == "prefill":
            name_pattern = r"(?<=hmquant_).*?_prefill"
        elif model_type == "vision":
            if model_name == "qwen2.5-vl":
                name_pattern = r"(hmquant_)(qwen2\.5-vl-7b-insturct-vision_xh2a_)(.*?)(_batch_image_)(.*?)(\.npy)"
            elif model_name == "qwen3-vl":
                name_pattern = (
                    r"(hmquant_)(qwen3_vl_instruct)(_vision)(_config_)(.*?)(\.npy)"
                )
            elif model_name == "minicpmo":
                name_pattern = (
                    r"(hmquant_)(minicpmo_vision_)(7b_xh2a_)(.*?k_)(.*?)(\.npy)"
                )
        elif model_type == "encoder":
            name_pattern = r"(hmquant_)(whisper_meduim_xh2a_w8a8_sefp_)(.*?)(\.npy)"

        if model_type in ["decode", "prefill", "encoder"] and model_name == "whisper":
            name_pattern = r"(?<=hmquant_).*?_sefp"

        for data_file in golden:
            file_name = os.path.basename(data_file)
            if "image_embeds" in file_name:
                data_dst = os.path.join(model_res_dir, "image_embeds.npy")
                shutil.copy(data_file, data_dst)
                print(f"Moved: {data_file} -> {data_dst}")
                continue
            match = re.search(name_pattern, file_name)
            if match:
                if model_type != "vision":
                    original_name = match.group()
                    file_name_new = file_name.replace(original_name, model_name)
                else:
                    file_name_new = (
                        f"{match.group(1)}{model_name}_{match.group(5)}{match.group(6)}"
                    )
                data_dst = os.path.join(model_res_dir, file_name_new)
                shutil.copy(data_file, data_dst)
                print(f"Moved: {data_file} -> {data_dst}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return False

    return True


def _process_model_files_xh1(
    backend: str, quant_dir: str, result_dir: str, model_name: str, model_type: str
) -> bool:
    """
    Process model files for XH1 backend, organizing them into standardized structure.

    Args:
        backend (str): Backend identifier (should be 'xh1' for this function)
        quant_dir (str): Source directory containing quantized model files
        result_dir (str): Destination directory for organized files
        model_name (str): Name of the model being processed
        model_type (str): Type of model component ('prefill', 'decode', 'vision', 'encoder')

    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # 1. Create target directory structure
        model_res_dir = os.path.join(result_dir, model_type)
        if model_type == "decode":
            model_res_dir = os.path.join(result_dir, "decoder")
        elif model_type == "vision":
            model_res_dir = os.path.join(result_dir, "visual")
        os.makedirs(model_res_dir, exist_ok=True)
        print(f"Created target directory structure: {model_res_dir}")

        if model_name in ["qwen3-vl", "qwen2.5-vl"] and model_type == "vision":
            model_type = "visual"

        # 2. Process hmquant_*_with_act.onnx files
        onnx_files = find_file_recursive(
            quant_dir, f"hmquant_*{model_type}*with_act.onnx"
        )
        if len(onnx_files) == 0:
            return False
        # Take the first matching file
        onnx_src = onnx_files[0]
        onnx_dst = os.path.join(model_res_dir, f"hmquant_{model_name}_with_act.onnx")
        shutil.copy(onnx_src, onnx_dst)
        print(f"Renamed and moved: {onnx_src} -> {onnx_dst}")

        # 3. Process golden data
        golden = find_file_recursive(quant_dir, f"hmquant_*{model_type}*_input.npy")
        opt_golden = find_file_recursive(
            quant_dir, f"hmquant_*{model_type}*_output.npy"
        )
        golden += opt_golden

        name_pattern = r"(?<=hmquant_).*?_decode"
        if model_type == "prefill":
            name_pattern = r"(?<=hmquant_).*?_prefill"
        elif model_type in ["vision", "visual"]:
            name_pattern = r"(?<=hmquant_).*?_visual"

        if model_name in ["qwen3-vl", "qwen2.5-vl"]:
            if model_type == "decode":
                name_pattern = r"(?<=hmquant_).*?_decoder"
            elif model_type == "prefill":
                name_pattern = r"(?<=hmquant_).*?_Prefill"

        for data_file in golden:
            file_name = os.path.basename(data_file)
            if "image_embeds" in file_name:
                data_dst = os.path.join(model_res_dir, "image_embeds.npy")
                shutil.copy(data_file, data_dst)
                print(f"Moved: {data_file} -> {data_dst}")
                continue
            if model_type in ["vision", "visual"] and (
                "Prefill" in file_name
                or "decode" in file_name
                or "prefill" in file_name
            ):
                continue
            match = re.search(name_pattern, file_name)
            if match:
                original_name = match.group()
                file_name_new = file_name.replace(original_name, model_name)
                data_dst = os.path.join(model_res_dir, file_name_new)
                shutil.copy(data_file, data_dst)
                print(f"Moved: {data_file} -> {data_dst}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return False

    return True


def _process_model_files(
    backend: str, quant_dir: str, result_dir: str, model_name: str, model_type: str
) -> bool:
    """
    Process model files based on the specified backend.

    Args:
        backend (str): Backend identifier ('xh1' or 'xh2')
        quant_dir (str): Source directory containing quantized model files
        result_dir (str): Destination directory for organized files
        model_name (str): Name of the model being processed
        model_type (str): Type of model component ('prefill', 'decode', 'vision', 'encoder')

    Returns:
        bool: True if processing was successful, False otherwise
    """
    if backend == "xh1":
        return _process_model_files_xh1(
            backend, quant_dir, result_dir, model_name, model_type
        )
    elif backend == "xh2":
        return _process_model_files_xh2(
            backend, quant_dir, result_dir, model_name, model_type
        )


def _process_weight_file(quant_dir: str, result_dir: str):
    """
    Process weight files from the quantized model directory.

    Args:
        quant_dir (str): Source directory containing quantized model files
        result_dir (str): Destination directory for organized files
    """
    weight_src = f"{quant_dir}/weight.npy"
    if os.path.exists(weight_src):
        weight_dst = os.path.join(result_dir, "weight.npy")
        shutil.copy(weight_src, weight_dst)
        print(f"Moved: {weight_src} -> {weight_dst}")

    weight_src = f"{quant_dir}/decoder/weight.npy"
    if os.path.exists(weight_src):
        os.makedirs(f"{result_dir}/decoder", exist_ok=True)
        weight_dst = os.path.join(result_dir, "decoder", "weight.npy")
        shutil.copy(weight_src, weight_dst)
        print(f"Moved: {weight_src} -> {weight_dst}")

    weight_src = f"{quant_dir}/prefill/weight.npy"
    if os.path.exists(weight_src):
        os.makedirs(f"{result_dir}/prefill", exist_ok=True)
        weight_dst = os.path.join(result_dir, "prefill", "weight.npy")
        shutil.copy(weight_src, weight_dst)
        print(f"Moved: {weight_src} -> {weight_dst}")

    weight_src = f"{quant_dir}/visual/weight.npy"
    if os.path.exists(weight_src):
        os.makedirs(f"{result_dir}/visual", exist_ok=True)
        weight_dst = os.path.join(result_dir, "visual", "weight.npy")
        shutil.copy(weight_src, weight_dst)
        print(f"Moved: {weight_src} -> {weight_dst}")


if __name__ == "__main__":
    args = parse_args()

    model_name = args.model_name
    backend = args.backend
    quant_dir = args.quant_folder
    result_dir = args.result_folder
    zipped_name = args.zipped_name

    try:
        # Create target directory structure
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir, ignore_errors=True)
        os.makedirs(result_dir, exist_ok=True)
        print(f"Created target directory structure: {result_dir}")

        if backend == "xh1":
            _process_weight_file(quant_dir, result_dir)

        # Process token_embedding.pt
        for embedding_src in glob.glob(
            os.path.join(quant_dir, "**", "*embedding*.pt"), recursive=True
        ):
            file_name = os.path.basename(embedding_src)
            if "quant_embedding" not in file_name and "token_embedding" in file_name:
                file_name = file_name.replace("token_embedding", "quant_embedding")
            if "quant_embedding" not in file_name and "qembedding" in file_name:
                file_name = file_name.replace("qembedding", "quant_embedding")
            embedding_dst = os.path.join(result_dir, file_name)
            shutil.copy(embedding_src, embedding_dst)
            print(f"Moved: {embedding_src} -> {embedding_dst}")

        if not _process_model_files(
            backend, quant_dir, result_dir, model_name, "prefill"
        ):
            print("Error: Failed to process prefill folder")
            sys.exit(-1)

        if (
            os.path.exists(f"{quant_dir}/decode")
            or os.path.exists(f"{quant_dir}/decoder")
            or os.path.exists(f"{quant_dir}/golden/decode")
        ) and not _process_model_files(
            backend, quant_dir, result_dir, model_name, "decode"
        ):
            print("Error: Failed to process decode folder")
            sys.exit(-1)

        if (
            os.path.exists(f"{quant_dir}/vision")
            or os.path.exists(f"{quant_dir}/visual")
            or "-vl" in model_name
        ) and not _process_model_files(
            backend, quant_dir, result_dir, model_name, "vision"
        ):
            print("Error: Failed to process visual folder")
            sys.exit(-1)

        if os.path.exists(f"{quant_dir}/encoder") and not _process_model_files(
            backend, quant_dir, result_dir, model_name, "encoder"
        ):
            print("Error: Failed to process encoder folder")
            sys.exit(-1)
        print("File processing completed!")

        # If a compressed filename is specified, compress and upload to Jfrog
        if zipped_name:
            result_dir_path = Path(result_dir)
            zip_file_path = result_dir_path / f"{zipped_name}.zip"

            try:
                # Compress the folder
                subprocess.run(
                    ["zip", "-r", str(zip_file_path), "."],
                    cwd=result_dir_path,  # specify working directory
                    check=True,
                    capture_output=True,
                    text=True,  # Convert output to string
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"Error: Failed to compress quantization folder. Details: {e.stderr}"
                )
                sys.exit(-1)

            # Determine model folder path
            model_folder = "deepseek" if "deepseek" in model_name else model_name
            jfrog_file_path = f"{QUANT_MODELS_URL}/{model_folder}/"
            if not JFROG_PASSWORD:
                print("Error: JFROG_PASSWORD environment variable is not set.")
                sys.exit(-1)

            # Build curl command
            curl_cmd = [
                "curl",
                "-u",
                f"{JFROG_USER}:{JFROG_PASSWORD}",  # Authentication information
                "-T",
                str(zip_file_path),  # File to upload
                jfrog_file_path,  # Target upload path
            ]
            try:
                subprocess.run(curl_cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(
                    f"Failed to upload quantization file archive. Details: {e.stderr}"
                )
                sys.exit(-1)

            print(f"Successfully uploaded quantization file archive {zipped_name}.zip")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(-1)
