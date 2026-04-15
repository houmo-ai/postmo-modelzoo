# Copyright 2025 HOUMO AI
#
# File: check.py
# Description:
#     Check config YAML
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
from PIL import Image
from . import logger


def get_image_size(filepath):
    """Get the width and height of an image.

    Args:
        filepath (str): Path to the image file

    Returns:
        tuple: A tuple containing (width, height) of the image
    """
    with Image.open(filepath) as im:
        width, height = im.size
        return width, height


def check_cfg(cfg):
    """Check configuration parameters.

    Args:
        cfg (dict): Configuration dictionary containing model settings

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    # model info
    model_cfg = cfg.get("model")
    if model_cfg is None:
        logger.fatal("[model] section not found")

    save_dir = model_cfg.get("save_dir")
    if save_dir is None:
        logger.fatal("[model.save_dir] not found")

    inputs_cfg = model_cfg.get("inputs", dict())
    if len(inputs_cfg) == 0:
        logger.fatal("[model.inputs] not found or empty")

    # Track resizer usage across all inputs
    has_resizer = False
    has_dynamic_resizer = False
    for input_name, input_cfg in inputs_cfg.items():
        resizer_mode = _check_input_cfg(input_name, input_cfg)
        if resizer_mode != 0:  # Any resizer mode (1/2/3)
            has_resizer = True
        if resizer_mode in [1, 2]:  # DYNAMIC_V2 or DYNAMIC_V1
            has_dynamic_resizer = True

    # quant config
    quant_cfg = cfg.get("quant", dict())
    if "calib_num" in quant_cfg:
        logger.warning(
            "[quant.calib_num] is currently ignored, only 1 calibration image is used"
        )

    # mix_search config (optional)
    mix_search_cfg = quant_cfg.get("mix_search")
    if mix_search_cfg is not None:
        _check_mix_search_cfg(mix_search_cfg)
        # mix_search and resizer are mutually exclusive
        # mix_search runs original ONNX for sensitivity analysis, needs float32 input
        # resizer requires YUV/uint8 input for hardware resize
        if has_resizer:
            logger.fatal(
                "[quant.mix_search] and [model.inputs.*.resizer] are mutually exclusive. "
                "mix_search runs original ONNX for sensitivity analysis and needs float32 input, "
                "while resizer requires YUV/uint8 input for hardware resize. "
                "Please remove resizer config or disable mix_search."
            )

    # build config
    build_cfg = cfg.get("build", dict())

    ncore = build_cfg.get("ncore", 1)
    if ncore not in [1, 2]:
        logger.fatal(f"[build.ncore] must be 1 or 2, got {ncore}")

    opt_level = build_cfg.get("opt_level", 2)
    if opt_level not in [0, 1, 2]:
        logger.fatal(f"[build.opt_level] must be 0/1/2, got {opt_level}")

    batch = build_cfg.get("batch", 1)
    if batch < 1:
        logger.fatal(f"[build.batch] must be >= 1, got {batch}")

    # roi_num: only valid when dynamic resizer is enabled
    if "roi_num" in build_cfg and not has_dynamic_resizer:
        logger.fatal(
            "[build.roi_num] only valid with dynamic resizer (resizer_mode=1 or 2)"
        )
    roi_num = build_cfg.get("roi_num", 1)
    if roi_num < 1:
        logger.fatal(f"[build.roi_num] must be >= 1, got {roi_num}")

    if batch > 1 and roi_num > 1:
        logger.fatal("[build.batch] and [build.roi_num] cannot both be > 1")

    parallel_jobs = build_cfg.get("parallel_jobs")
    if parallel_jobs is not None and parallel_jobs < 1:
        logger.fatal(f"[build.parallel_jobs] must be >= 1, got {parallel_jobs}")

    return True


def _check_mix_search_cfg(mix_search_cfg):
    """Check mix_search configuration for mixed precision quantization.

    Args:
        mix_search_cfg (dict): Mix search configuration dictionary

    Returns:
        bool: True if configuration is valid

    Mix search allows automatic selection of layers to use higher precision
    (e.g., w4a16, w8a16) based on sensitivity analysis.

    Config fields:
        - topk: proportion of layers to use higher precision (0-1)
        - weight_bits: candidate weight bit widths [4, 8, 16]
        - act_bits: candidate activation bit widths [4, 8, 16]
        - policy: selection strategy (threshold/topk)
        - task: task type (llm/cv)
        - metric: sensitivity metric (l1/sqnr/kl)
        - key_name: output attribute name for sensitivity calculation
    """
    prefix = "[quant.mix_search]"

    # topk: proportion of layers to use higher precision
    topk = mix_search_cfg.get("topk")
    if topk is not None:
        if not isinstance(topk, (int, float)):
            logger.fatal(f"{prefix}.topk must be int or float")
        if topk <= 0 or topk > 1:
            logger.warning(f"{prefix}.topk should be in range (0, 1], got {topk}")

    # weight_bits: candidate weight bit widths
    weight_bits = mix_search_cfg.get("weight_bits")
    if weight_bits is not None:
        if not isinstance(weight_bits, list):
            logger.fatal(f"{prefix}.weight_bits must be list")
        for v in weight_bits:
            if v not in [4, 8, 16]:
                logger.fatal(f"{prefix}.weight_bits values must be 4/8/16, got {v}")

    # act_bits: candidate activation bit widths
    act_bits = mix_search_cfg.get("act_bits")
    if act_bits is not None:
        if not isinstance(act_bits, list):
            logger.fatal(f"{prefix}.act_bits must be list")
        for v in act_bits:
            if v not in [4, 8, 16]:
                logger.fatal(f"{prefix}.act_bits values must be 4/8/16, got {v}")

    # policy: selection strategy (threshold or topk)
    policy = mix_search_cfg.get("policy")
    if policy is not None:
        if policy not in ["threshold", "topk"]:
            logger.fatal(f"{prefix}.policy must be threshold or topk, got {policy}")

    # task: task type
    task = mix_search_cfg.get("task")
    if task is not None:
        if task not in ["llm", "cv", "cv_cls"]:
            logger.fatal(f"{prefix}.task must be llm, cv or cv_cls, got {task}")

    # metric: sensitivity metric
    metric = mix_search_cfg.get("metric")
    if metric is not None:
        if metric not in ["l1", "sqnr", "kl"]:
            logger.warning(f"{prefix}.metric recommended: l1/sqnr/kl, got {metric}")

    # key_name: output attribute name
    key_name = mix_search_cfg.get("key_name")
    if key_name is not None:
        if not isinstance(key_name, str):
            logger.fatal(f"{prefix}.key_name must be string")

    return True


def _check_input_cfg(input_name, input_cfg):
    """Check input configuration.

    Args:
        input_name (str): Input name
        input_cfg (dict): Input configuration dictionary

    Returns:
        int: resizer_mode (0 if no resizer, otherwise 1/2/3)
    """
    prefix = f"[model.inputs.{input_name}]"

    # shape
    if "shape" not in input_cfg:
        logger.fatal(f"{prefix}.shape not found")
    shape = input_cfg["shape"]
    if not isinstance(shape, list):
        logger.fatal(f"{prefix}.shape must be list")

    # data_format
    if "data_format" not in input_cfg:
        logger.warning(f"{prefix}.data_format not found, using default null")
        return 0
    data_format = input_cfg["data_format"]
    if data_format is None:
        return 0
    # Image input validation
    if data_format not in ["RGB", "BGR", "GRAY"]:
        logger.fatal(f"{prefix}.data_format must be RGB/BGR/GRAY")

    if len(shape) != 4:
        logger.fatal(f"{prefix}.shape must be [N, C, H, W]")

    _, C, H, W = shape

    # mean/std
    mean = input_cfg.get("mean")
    std = input_cfg.get("std")
    if mean is None:
        logger.fatal(f"{prefix}.mean not found")
    if std is None:
        logger.fatal(f"{prefix}.std not found")
    if not isinstance(mean, list):
        logger.fatal(f"{prefix}.mean must be list")
    if not isinstance(std, list):
        logger.fatal(f"{prefix}.std must be list")

    # resize_type
    resize_type = input_cfg.get("resize_type")
    if resize_type not in [0, 1, 2]:
        logger.fatal(f"{prefix}.resize_type must be 0, 1 or 2")

    if resize_type == 1:
        # resize_type=1: aspect ratio resize with padding (padding_mode: 0-left/top, 1-center)
        padding_mode = input_cfg.get("padding_mode")
        if padding_mode is None:
            logger.fatal(f"{prefix}.padding_mode not found (resize_type=1)")
        if padding_mode not in [0, 1]:
            logger.fatal(f"{prefix}.padding_mode must be 0 or 1")

        padding_values = input_cfg.get("padding_values")
        if padding_values is None:
            logger.fatal(f"{prefix}.padding_values not found (resize_type=1)")
        if not isinstance(padding_values, list):
            logger.fatal(f"{prefix}.padding_values must be list")
        if len(padding_values) != C:
            logger.fatal(f"{prefix}.padding_values length must equal channels ({C})")

    if resize_type == 2:
        # resize_type=2: fixed height, aspect-ratio width, right padding (padding_mode fixed to 0)
        # padding_mode is optional for resize_type=2, if provided must be 0
        padding_mode = input_cfg.get("padding_mode")
        if padding_mode is not None and padding_mode != 0:
            logger.fatal(
                f"{prefix}.padding_mode must be 0 for resize_type=2 (right padding only)"
            )

        padding_values = input_cfg.get("padding_values")
        if padding_values is None:
            logger.warning(f"{prefix}.padding_values not found, using default 0")
        elif not isinstance(padding_values, list):
            logger.fatal(f"{prefix}.padding_values must be list")
        elif len(padding_values) != C:
            logger.fatal(f"{prefix}.padding_values length must equal channels ({C})")

    # resizer
    if "resizer" not in input_cfg:
        return 0
    resizer_cfg = input_cfg["resizer"]
    if resizer_cfg is None:
        resizer_cfg = {}
    if not isinstance(resizer_cfg, dict):
        logger.fatal(f"{prefix}.resizer must be dict")

    return _check_resizer_cfg(input_name, resizer_cfg, data_format, H, W)


def _check_resizer_cfg(input_name, resizer_cfg, data_format, H, W):
    """Check resizer configuration.

    Args:
        input_name (str): Input name
        resizer_cfg (dict): Resizer configuration dictionary
        data_format (str): Data format (RGB/BGR/GRAY)
        H (int): Model input height
        W (int): Model input width

    Returns:
        int: resizer_mode (1/2/3)
    """
    prefix = f"[model.inputs.{input_name}.resizer]"

    # toYUV_format
    toYUV_format = resizer_cfg.get("toYUV_format", "YUV420SP")
    if toYUV_format not in ["YUV400", "YUV420SP", "YUV422SP", "YUV444SP"]:
        logger.fatal(f"{prefix}.toYUV_format must be YUV400/YUV420SP/YUV422SP/YUV444SP")
    if data_format in ["RGB", "BGR"] and toYUV_format == "YUV400":
        logger.fatal(f"{prefix}.toYUV_format: RGB/BGR input cannot use YUV400")
    if data_format == "GRAY" and toYUV_format != "YUV400":
        logger.fatal(f"{prefix}.toYUV_format: GRAY input must use YUV400")

    # resizer_input_size
    resizer_input_size = resizer_cfg.get("resizer_input_size", [H, W])
    if not isinstance(resizer_input_size, list) or len(resizer_input_size) != 2:
        logger.fatal(f"{prefix}.resizer_input_size must be [H, W]")
    for v in resizer_input_size:
        if v % 2 != 0:
            logger.fatal(f"{prefix}.resizer_input_size must be even numbers")
    resizer_input_h, resizer_input_w = resizer_input_size
    # Size limit: H <= 4096, W <= 1024
    if resizer_input_h > 4096:
        logger.fatal(
            f"{prefix}.resizer_input_size H must be <= 4096, got {resizer_input_h}"
        )
    if resizer_input_w > 1024:
        logger.fatal(
            f"{prefix}.resizer_input_size W must be <= 1024, got {resizer_input_w}"
        )
    if resizer_input_h < H or resizer_input_w < W:
        logger.warning(
            f"{prefix}.resizer_input_size [{resizer_input_h}, {resizer_input_w}] < model input [{H}, {W}]"
        )

    # resizer_mode
    resizer_mode = resizer_cfg.get("resizer_mode", 3)
    if resizer_mode not in [1, 2, 3]:
        logger.fatal(
            f"{prefix}.resizer_mode must be 1/2/3 (DYNAMIC_V2/DYNAMIC_V1/STATIC)"
        )
    if resizer_mode == 2:
        logger.fatal(f"{prefix}.resizer_mode=2 (DYNAMIC_V1) is not supported")
    # DYNAMIC_V2 mode: padding constraint warning
    if resizer_mode == 1:
        logger.warning(
            f"{prefix}: DYNAMIC_V2 mode padding only supports one direction (left-right or top-bottom), max 32 pixels"
        )

    # resizer_crop (only valid for STATIC mode)
    if resizer_mode != 3:
        if "resizer_crop" in resizer_cfg:
            logger.fatal(
                f"{prefix}.resizer_crop only valid when resizer_mode=3 (STATIC)"
            )
        return resizer_mode

    # STATIC mode: check resizer_crop
    if "resizer_crop" not in resizer_cfg:
        logger.warning(
            f"{prefix}.resizer_crop not found, using default [0, 0, {resizer_input_h}, {resizer_input_w}]"
        )
        resizer_crop = [0, 0, resizer_input_h, resizer_input_w]
    else:
        resizer_crop = resizer_cfg["resizer_crop"]
        if not isinstance(resizer_crop, list) or len(resizer_crop) != 4:
            logger.fatal(f"{prefix}.resizer_crop must be [y, x, h, w]")

    y, x, crop_h, crop_w = resizer_crop
    for v in resizer_crop:
        if v % 2 != 0:
            logger.fatal(f"{prefix}.resizer_crop must be even numbers")
    if y < 0 or x < 0 or y + crop_h > resizer_input_h or x + crop_w > resizer_input_w:
        logger.fatal(
            f"{prefix}.resizer_crop out of bounds [0, 0, {resizer_input_h}, {resizer_input_w}]"
        )
    # Scale constraint: [1/32, 16]
    # crop -> model input, scale must be in [1/32, 16]
    scale_h = H / crop_h
    scale_w = W / crop_w
    if scale_h < 1 / 32 or scale_h > 16:
        logger.fatal(
            f"{prefix}.resizer_crop scale out of range [1/32, 16]: crop_h={crop_h} -> H={H}, scale={scale_h:.4f}"
        )
    if scale_w < 1 / 32 or scale_w > 16:
        logger.fatal(
            f"{prefix}.resizer_crop scale out of range [1/32, 16]: crop_w={crop_w} -> W={W}, scale={scale_w:.4f}"
        )
    return resizer_mode
