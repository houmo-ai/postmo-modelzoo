# Copyright 2025 HOUMO AI
#
# File: execute_golden_checks.py
# Description:
#    Execute golden checks for LLM models.
#    Note: Golden checks are not supported if flash_attention
#          is non-zero during model compilation.
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
import numpy as np
import torch
import time
import logging
import json
import argparse
from xhquant.backend.xh2a import HMFP
import tcim_lite as tcim
from compiler_utils import setup_logging

script_dir = os.path.dirname(os.path.abspath(__file__))

# Threshold value for cosine similarity comparison between golden and actual outputs
GOLDEN_THRESH = 0.98


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Check Model Results")
    parser.add_argument(
        "--model_cfg",
        type=str,
        help="the path of perf_file.",
    )
    parser.add_argument(
        "-log",
        "--log_file",
        type=str,
        default="",
        help="the path of log.",
    )

    args = parser.parse_args()
    return args


def sanitize_name(name: str):
    """
    Sanitize a string by replacing problematic characters with underscores.

    Args:
        name (str): Input string to sanitize

    Returns:
        str: Sanitized string with problematic characters replaced
    """
    return name.replace(":", "_").replace("/", "_")


def cosine_distance(data1, data2):
    """
    Calculate cosine similarity between two arrays/tensors.

    Args:
        data1: First data array
        data2: Second data array

    Returns:
        float: Cosine similarity value between -1 and 1, or -1 if shapes don't match
    """
    if data1.shape != data2.shape:
        logger.error(f"Shape not equal {data1.shape} vs {data2.shape}")
        return -1

    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)
    cosine_dist = np.dot(v1_norm, v2_norm)

    if np.isnan(cosine_dist):
        return -1
    return cosine_dist


def fp16tohmfp_xh2a_default(
    x: torch.Tensor,
    mode: str = "sefp",
    bit: int = 8,
    dim: int = -1,
    hidden_bit: bool = True,
    max_exp_boost: int = 0,
    keep_unpack_format: bool = False,
    **kwargs,
):
    """
    Convert a PyTorch tensor from FP16 to HMFP (Hardware Mixed Format Precision) format.

    Args:
        x (torch.Tensor): Input tensor in FP16 format
        mode (str): Quantization mode (default: "sefp")
        bit (int): Number of bits for quantization (default: 8)
        dim (int): Dimension along which to apply quantization (default: -1)
        hidden_bit (bool): Whether to use hidden bit (default: True)
        max_exp_boost (int): Maximum exponent boost value (default: 0)
        keep_unpack_format (bool): Whether to keep unpacked format (default: False)
        **kwargs: Additional keyword arguments

    Returns:
        Converted tensor in HMFP format
    """
    return HMFP.from_tensor(
        x,
        mode=mode,
        bit=bit,
        dim=dim,
        hidden_bit=hidden_bit,
        max_exp_boost=max_exp_boost,
        keep_unpack_format=keep_unpack_format,
        **kwargs,
    )


def merge_last_two_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Merge the last two dimensions of a tensor while preserving all preceding dimensions.

    Args:
        tensor (torch.Tensor): Input tensor with at least 2 dimensions

    Returns:
        torch.Tensor: Tensor with the last two dimensions merged

    Raises:
        ValueError: If tensor has fewer than 2 dimensions
    """
    if tensor.dim() < 2:
        raise ValueError(
            f"Tensor must have ≥2 dimensions, current dimensions: {tensor.dim()}"
        )

    # Calculate new shape: first n-2 dims + product of last two dims
    new_shape = list(tensor.shape[:-2]) + [tensor.shape[-2] * tensor.shape[-1]]

    return tensor.reshape(new_shape)


def convert_f16_to_hmfp(npy_path):
    """
    Convert an FP16 numpy array to HMFP format and return processed values.

    Args:
        npy_path (str): Path to the input numpy file containing FP16 data

    Returns:
        numpy.ndarray: Processed data in HMFP format as numpy array
    """
    kv_arr = np.load(npy_path)
    # logger.info(
    #     f"Numpy Arr Info, data type: {kv_arr.dtype}, "
    #     f"shape: {kv_arr.shape},  data type descriptor: {kv_arr.dtype.descr}"
    # )

    tensor = torch.from_numpy(kv_arr)
    kcache_hmfp = fp16tohmfp_xh2a_default(tensor, keep_unpack_format=False)
    kcache_hmfp_val = merge_last_two_dims(kcache_hmfp.psum)

    return kcache_hmfp_val.numpy()


def _pad_or_truncate_input(input_data: np.ndarray, target_dim2: int) -> np.ndarray:
    """
    Unified logic for padding/truncating the 3rd dimension (index 2) of input data to eliminate code duplication.

    Args:
        input_data (np.ndarray): Input data array to be adjusted
        target_dim2 (int): Target size for the 3rd dimension

    Returns:
        np.ndarray: Adjusted input data with the 3rd dimension matching target_dim2
    """
    current_dim2 = input_data.shape[2]
    if current_dim2 == target_dim2:
        return input_data

    # Pad if the current dimension is smaller than target
    if current_dim2 < target_dim2:
        pad_length = target_dim2 - current_dim2
        pad_width = ((0, 0), (0, 0), (0, pad_length), (0, 0))
        return np.pad(
            input_data, pad_width=pad_width, mode="constant", constant_values=0
        )

    # Truncate if the current dimension is larger than target
    return input_data[:, :, :target_dim2, :]


def _set_module_inputs(
    module: tcim.Module, quant_dir: str, model_name: str, batch: int, profile: dict
) -> None:
    """
    Set inputs for the model module by loading and processing input data files.

    Args:
        module: The model module to set inputs for
        quant_dir (str): Directory containing quantized model data
        model_name (str): Name of the model
        batch (int): Batch size multiplier
        profile (dict): Dictionary to store timing profiles
    """
    profile["set_input"] = 0
    input_num = module.get_num_inputs()
    for idx in range(input_num):
        input_name = module.get_input_name(idx)
        input_info = module.get_input_info(input_name)
        logger.info(
            f"input_info[{input_name}] shape = {input_info.shape}, "
            f"dtype = {input_info.dtype}, format = {input_info.format.name}"
        )
        input_data_path = os.path.join(
            quant_dir, f"hmquant_{model_name}_{sanitize_name(input_name)}_input.npy"
        )

        if "kcache" in input_name:
            input_data = convert_f16_to_hmfp(input_data_path)
            input_data = _pad_or_truncate_input(input_data, input_info.shape[2])
        else:
            input_data = np.load(input_data_path).astype(input_info.dtype)
            if "vcache" in input_name:
                input_data = _pad_or_truncate_input(input_data, input_info.shape[2])
        input_data = np.concatenate([input_data for i in range(batch)], axis=0)
        # logger.info(
        #     f"golden input[{input_name}] shape = {input_data.shape}, dtype = {input_data.dtype}"
        # )
        start = time.time()
        module.set_input(input_name, input_data)
        profile["set_input"] += time.time() - start


def check_golden(
    model_name: str,
    model_type: str,
    model_path: str,
    quant_dir: str,
    profile: dict,
    batch: int = 1,
    ndevice: int = 1,
):
    """
    Run golden test comparison for a specific model.

    Args:
        model_name (str): Name of the model being tested
        model_type (str): Type of model (e.g., prefill, decode, visual)
        model_path (str): Path to the model file
        quant_dir (str): Directory containing quantized models
        profile (dict): Dictionary to store performance metrics
        batch (int): Batch size multiplier (default: 1)
        ndevice (int): Number of devices to use (default: 1)

    Returns:
        bool: True if all outputs pass the similarity threshold, False otherwise
    """
    logger.info(f"===> {model_name} {model_type} test start...")
    logger.info(
        f"ndevice = {ndevice}, model_path = {model_path}, quant_dir = {quant_dir}"
    )
    # Load model
    start = time.time()
    device_list = list(range(ndevice))
    # Initialize device manager and weight manager for TCIM runtime
    dev_manager = tcim.runtime.DevManager(device_list, "Xh2HalBackend")
    weight_manager = tcim.runtime.WeightManager(dev_manager)
    option = tcim.runtime.Option(weight_manager)
    module = tcim.runtime.load(model_path, option=option)
    profile["load"] = time.time() - start

    _set_module_inputs(module, quant_dir, model_name, batch, profile)

    # Run model inference
    start = time.time()
    module.run()
    module.sync()
    profile["infer"] = time.time() - start

    # Get output and compare with golden
    profile["get_output"] = 0
    result_check = False
    output_num = module.get_num_outputs()
    for idx in range(output_num):
        output_name = module.get_output_name(idx)
        output_info = module.get_output_info(output_name)
        logger.info(
            f"output_info[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}"
        )
        start = time.time()
        output_data = module.get_output(output_name).numpy()
        profile["get_output"] += time.time() - start
        # logger.info(
        #     f"output[{output_name}] shape = {output_data.shape}, dtype = {output_data.dtype}"
        # )
        output_data_path = os.path.join(
            quant_dir, f"hmquant_{model_name}_{sanitize_name(output_name)}_output.npy"
        )
        if os.path.exists(output_data_path):
            golden_output = np.load(output_data_path)
            golden_output = np.concatenate(
                [golden_output for i in range(batch)], axis=0
            )
        else:
            logger.error(
                f"[warning] compare canceled while golden data not found -> {output_data_path}"
            )
            continue
        if golden_output.shape == output_data.shape:
            cosine_dist = cosine_distance(golden_output, output_data)

            if cosine_dist < GOLDEN_THRESH:
                logger.error(f"result check failed, similarity={cosine_dist:.6f}.")
            else:
                result_check = True
                logger.info(
                    f"[compare] golden output [{output_name}], similarity={cosine_dist:.6f}"
                )
        else:
            logger.error(
                f"[compare] golden output [{output_name}] shape not match {golden_output.shape} vs {output_data.shape}"
            )

    logger.info(f"<=== {model_name} {model_type} test end.")
    return result_check


if __name__ == "__main__":
    args = parse_args()

    setup_logging(log_file=args.log_file)
    logger = logging.getLogger(__name__)

    # Load model configuration from JSON file
    cfg_path = args.model_cfg
    with open(cfg_path, "r", encoding="utf-8") as f:
        model_json = json.load(f)

    # Mapping of model types to their corresponding folder names
    quant_folder_names = {
        "prefill": "prefill",
        "decode": "decoder",
        "visual": "visual",
    }
    # Iterate through each model stream in the configuration
    for model_info in model_json["Streams"]:
        profile = {}
        model_name = model_info["model_name"]
        quant_models = model_info["quant_models"]
        ndevice = model_info["ndevices"]
        batch = model_info["batch"]
        model_file_name = model_info.get("ModelName", "Unknown")

        model_path_dict = {
            "prefill": model_info.get("prefill", ""),
            "decode": model_info.get("decode", ""),
            "visual": model_info.get("visual", ""),
        }
        final_flag = True

        # Test each available model type
        for model_type, model_path in model_path_dict.items():
            if not model_path or not os.path.exists(model_path):
                continue

            profile = {}
            quant_dir = os.path.join(quant_models, quant_folder_names[model_type])
            # Run golden test for this model type
            ret = check_golden(
                model_name=model_name,
                model_type=model_type,
                model_path=model_path,
                quant_dir=quant_dir,
                profile=profile,
                batch=batch,
                ndevice=ndevice,
            )
            if ret is True:
                logger.info(
                    f"================ Verified {model_file_name} ing... ================"
                )
                logger.info(f"Model Path: {model_path}")
                logger.info(f"Model Type: {model_type}")
                logger.info(f'Load Time: {profile["load"]:.3f} s.')
                logger.info(f'Set input Time: {profile["set_input"]*1000:.3f} ms.')
                logger.info(f'Inferece Time: {profile["infer"]*1000:.3f} ms.')
                logger.info(f'Get output Time: {profile["get_output"]*1000:.3f} ms.')
            else:
                final_flag = False
                logger.error(f"Verified {model_path} FAILED")

        if final_flag is True:
            model_file_name = model_info.get("ModelName", "Unknown")
            logger.info(f"================ {model_file_name}: PASSED ================")
