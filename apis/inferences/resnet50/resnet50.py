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

HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../../..")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/hmatc")
from hmatc.utils.postprocess import softmax
import tcim_lite as tcim

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


if __name__ == "__main__":
    logger.info("===> resnet50 python example start...")
    logger.info(
        f"tcim runtime version: {tcim.runtime.get_version()}, houmo target: {HOUMO_TARGET}"
    )

    # 1. load model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hmm_files = [
        os.path.join(current_dir, name)
        for name in os.listdir(current_dir)
        if name.endswith(".hmm")
    ]
    assert hmm_files, f"No .hmm file found in {current_dir}"

    model_path = hmm_files[0]
    logger.info(f"Found model file: {model_path}")
    module = tcim.runtime.load(model_path)

    # 2. preprocess
    input_data = cv2.imread("../../data/snake.jpg")

    image_rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224))  # HWC uint8
    mean_arr = np.array([123.675, 116.28, 103.53])
    std_arr = np.array([58.395, 57.12, 57.375])
    image_norm = (image_rgb - mean_arr) / std_arr
    image_norm = np.transpose(image_norm, (2, 0, 1))  # CHW uint8
    image_norm = np.expand_dims(image_norm, axis=0)  # NCHW uint8
    input_data = image_norm.astype(np.float16)

    # 3. set input
    input_num = module.get_num_inputs()
    for idx in range(0, input_num):
        input_name = module.get_input_name(idx)
        input_info = module.get_input_info(input_name).ascontiguous()
        logger.info(
            f"input[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
        )
        module.set_input(input_name, input_data)

    # 4. run & sync
    module.run()
    module.sync()

    # 5. get output
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
        output_data = module.get_output(output_name).astype(np.float32).numpy()

    # 6. postprocess
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
