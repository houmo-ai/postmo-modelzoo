# Copyright 2025 HOUMO AI
#
# File: resnet50.py
# Description:
#   ResNet50 Image Classification Python Example.
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
import numpy as np
import cv2
from loguru import logger
import torch

HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../../..")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/utils/python")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/hmatc")
from image.format_converter import BGR2YUV
from hmatc.utils.postprocess import softmax

import tcim_lite as tcim

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


if __name__ == "__main__":
    logger.info("===> resnet50 python example start...")
    logger.info(
        f"houmo target: {HOUMO_TARGET}, tcim runtime version: {tcim.runtime.get_version()}"
    )

    # Discover the local model files in the example directory.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hmm_files = [
        os.path.join(current_dir, name)
        for name in os.listdir(current_dir)
        if name.endswith(".hmm")
    ]
    assert hmm_files, f"No .hmm file found in {current_dir}"

    # 1.1 Load the sample input image.
    image_data = cv2.imread("../../data/snake.jpg")
    img_height, img_width = image_data.shape[:2]
    logger.info(f"input image shape: {image_data.shape}")

    # 1.2 Load model
    model_path = hmm_files[0]
    logger.info(f"Found model file: {model_path}")
    module = tcim.runtime.load(model_path)

    # 1.3 Query model input metadata and find the maximum image canvas size.
    resizer_crop_str = "resizer_crop"
    target_height, target_width = 224, 224
    max_img_height, max_img_width = 0, 0
    input_names = []
    input_num = module.get_num_inputs()
    logger.info("Model input info:")
    for idx in range(0, input_num):
        input_name = module.get_input_name(idx)
        input_info = module.get_input_info(input_name).ascontiguous()
        input_names.append(input_name)
        logger.info(
            f"  Input[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, mem_size = {input_info.mem_size}, stride = {input_info.stride}, format = {input_info.format.name}"
        )
        if resizer_crop_str not in input_name:
            max_img_height, max_img_width = input_info.shape[2], input_info.shape[3]

    # 2. Preprocess image
    # Keep the original valid region as crop info for the resizer input.
    crop_height, crop_width = max_img_height, max_img_width
    if img_height < max_img_height and img_width <= max_img_width:
        # Pad smaller images with zeros on the bottom and right borders.
        pad_bottom = max_img_height - img_height
        pad_right = max_img_width - img_width
        image_data = cv2.copyMakeBorder(
            image_data,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        crop_height = img_height
        crop_width = img_width
        logger.info(
            f"pad input image to {image_data.shape} height = {max_img_height}, width = {max_img_width}"
        )
    else:
        # Resize images that exceed the supported canvas size.
        image_data = cv2.resize(image_data, (max_img_width, max_img_height))
        logger.info(
            f"resize input image to height = {max_img_height}, width = {max_img_width}"
        )

    # Convert HWC BGR input to CHW float32 before color space conversion.
    image_data = np.transpose(image_data, (2, 0, 1))  # CHW uint8
    input_data = torch.tensor(image_data.astype(np.float32))  # CHW float32

    # Convert BGR input to YUV420 as required by the runtime input format.
    bgr2yuv_func = BGR2YUV(fmt="YUV420")
    input_data = torch.unsqueeze(bgr2yuv_func(input_data), 0).numpy()  # NCHW float32
    input_data = input_data.astype(np.uint8)  # NCHW uint8

    # Align crop dimensions to even values for downstream processing.
    crop_height = crop_height - (crop_height % 2)
    crop_width = crop_width - (crop_width % 2)
    assert (
        crop_height % 2 == 0
        and crop_width % 2 == 0
        and crop_height > 0
        and crop_width > 2
    ), f"crop_height and crop_width must be even, got {crop_height} and {crop_width}"

    # Validate that the resize ratios stay within the accepted range.
    height_scale = target_height / crop_height
    width_scale = target_width / crop_width
    assert (
        1 / 32 <= height_scale <= 16
    ), f"{target_height} / img_height must be in [1/32, 16], got {height_scale}"
    assert (
        1 / 32 <= width_scale <= 16
    ), f"{target_width} / img_width must be in [1/32, 16], got {width_scale}"

    dyn_info = np.array(
        [0, 0, crop_height, crop_width, target_height, target_width, 0, 0, 0, 0],
        dtype=np.int32,
    )
    dyn_info = np.expand_dims(dyn_info, 0)
    logger.info(
        f"input_data shape: {input_data.shape}, dtype: {input_data.dtype}, dyn_info: {dyn_info}"
    )

    # 3. Feed image data and dynamic crop information into the model inputs.
    input_num = module.get_num_inputs()
    for input_name in input_names:
        if resizer_crop_str in input_name:
            module.set_input(input_name, dyn_info)
        else:
            module.set_input(input_name, input_data)

    # 4. Run & sync
    module.run()
    module.sync()

    # 5. Read all model outputs back to host memory.
    result_check = True
    output_num = module.get_num_outputs()
    for idx in range(0, output_num):
        output_name = module.get_output_name(idx)
        output_info = (
            module.get_output_info(output_name).ascontiguous().astype(np.float32)
        )
        logger.info(
            f"output[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}"
        )
        output_data = (
            module.get_dev_output(output_name).to_host().astype(np.float32).numpy()
        )

    # 6. Postprocess, apply softmax and report the top-k classification results.
    output_data = softmax(output_data)
    topk = 5
    pred_list = np.argsort(-output_data, axis=1, kind="quicksort").flatten()[0:topk]
    prob_list = output_data.flatten()
    for i, idx in enumerate(pred_list):
        logger.info(
            "top{}: predict cls = {}, prob = {:.6f}".format(i + 1, idx, prob_list[idx])
        )
    # check result, modify it when you change model or data
    assert pred_list[0] == 65

    logger.info("<=== resnet50 python example completed.")
