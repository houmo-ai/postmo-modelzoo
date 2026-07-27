# Copyright 2025 HOUMO AI
#
# File: resnet50_multistreams.py
# Description:
#   ResNet50 Multi-Stream Image Classification Python Example.
#   This example demonstrates how to run multi-threaded inference
#   with the ResNet50 model on the Houmo AI platform.
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
import time
import threading
import queue
import numpy as np
import cv2
import argparse
import torch
from loguru import logger

HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../../..")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/utils/python")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/hmatc")
from image.format_converter import BGR2YUV
from hmatc.utils.postprocess import softmax
import tcim_lite as tcim

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

# Shared barrier counter used to align thread start time.
g_thread_counter = 0


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        dest="device_num",
        type=int,
        default=1,
        help="Device number",
    )
    parser.add_argument(
        "-t",
        dest="thread_num",
        type=int,
        default=4,
        help="Thread number",
    )
    parser.add_argument(
        "-s",
        dest="sample_num",
        type=int,
        default=10,
        help="Sample number",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logger.info("===> resnet50_multistreams python example start...")
    logger.info(
        f"houmo target: {HOUMO_TARGET}, tcim runtime version: {tcim.runtime.get_version()}"
    )
    if tcim.runtime.get_device_num() < 1:
        logger.warning("No available devices found.")
        exit(0)
    # Discover the local model file in the example directory.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hmm_files = [
        os.path.join(current_dir, name)
        for name in os.listdir(current_dir)
        if name.endswith(".hmm")
    ]
    assert hmm_files, f"No .hmm file found in {current_dir}"

    model_path = hmm_files[0]  # Path to the ResNet50 model file
    logger.info(f"Found model file: {model_path}")

    # Parse runtime parameters for device count, thread count, and sample count.
    args = get_args()
    device_num = args.device_num  # Number of devices to use for inference
    thread_num = args.thread_num  # Number of threads per device
    sample_num = args.sample_num  # Total number of samples to process

    logger.info(f"devices: {device_num}")
    logger.info(f"threads: {thread_num}")
    logger.info(f"samples: {sample_num}")
    logger.info(f"model: {model_path}")

    threads = []

    # Verify that the model file exists
    if not os.path.exists(model_path):
        logger.error(f"Could not find model: {model_path}")
        exit(-1)

    # 1. Define the worker function used by all inference threads.
    thread_cnt = thread_num * device_num
    count_lock = threading.Lock()

    def thread_func(tid, did, module, qin, qout, thread_cnt, cnt_lock):
        """
        Function executed by each thread for inference on a specific device.

        Args:
            tid: Thread ID
            did: Device ID
            module: Model module for inference
            qin: Input queue with tasks
            qout: Output queue for results
            thread_cnt: Total number of threads
            cnt_lock: Lock for thread counter
        """
        global g_thread_counter
        count = 0

        # 1.1 Query input metadata for this module instance.
        input_infos = {}
        input_num = module.get_num_inputs()
        for idx in range(0, input_num):
            input_name = module.get_input_name(idx)
            input_info = module.get_input_info(input_name).ascontiguous()
            logger.info(
                "input[{}] shape = {}, dtype = {}, format = {}".format(
                    input_name,
                    input_info.shape,
                    input_info.dtype,
                    input_info.format.name,
                )
            )
            input_infos[input_name] = input_info

        # 1.2 Query output metadata for this module instance.
        output_infos = {}
        output_num = module.get_num_outputs()
        for idx in range(0, output_num):
            output_name = module.get_output_name(idx)
            output_info = module.get_output_info(output_name).ascontiguous()
            logger.info(
                "output[{}] shape = {}, dtype = {}, format = {}".format(
                    output_name,
                    output_info.shape,
                    output_info.dtype,
                    output_info.format.name,
                )
            )
            output_infos[output_name] = output_info

        # 1.3 Wait until all worker threads are ready before starting inference.
        with cnt_lock:
            g_thread_counter += 1
        while g_thread_counter < thread_cnt:
            time.sleep(0.1)

        # 1.4 Consume requests from the input queue until no work remains.
        while not qin.empty():
            try:
                # Fetch one request payload from the shared input queue.
                req_id, input_payloads = qin.get(timeout=2)
            except queue.Empty:
                logger.warning(
                    f"thread {tid} failed to get task data from empty queue."
                )
                continue

            # Feed both image data and dynamic crop info into the module.
            for input_name in input_infos:
                module.set_input(input_name, input_payloads[input_name])

            # Run one inference request and wait for completion.
            module.run()
            module.sync()

            # Collect outputs and push them to the result queue.
            output_datas = {}
            for output_name in output_infos:
                output_datas[output_name] = (
                    module.get_output(output_name, output_infos[output_name])
                    .astype(np.float32)
                    .numpy()
                )
            qout.put((req_id, output_datas))
            count += 1
            logger.info(
                "thread {} on device {} run sample {} end.".format(tid, did, req_id)
            )
        logger.info(
            "thread {} on device {} completed. {} sampels tested.".format(
                tid, did, count
            )
        )

    # 2.1 Load one module instance for each device-thread pair.
    module_dict = {}
    for did in range(device_num):
        # Create weight manager and option for each device
        wm = tcim.runtime.WeightManager(did)
        option = tcim.runtime.Option(wm)
        module_dict[did] = []
        # Load model instance for each thread on this device
        for i in range(thread_num):
            module = tcim.runtime.load(model_path, option=option)
            module_dict[did].append(module)
            logger.info(
                "thread {} on device {} load model {}.".format(i, did, model_path)
            )

    # 2.2 Reuse the single-stream preprocessing flow to prepare shared inputs.
    reference_module = module_dict[0][0]
    resizer_crop_str = "resizer_crop"
    target_height, target_width = 224, 224
    max_img_height, max_img_width = 0, 0
    input_names = []
    input_num = reference_module.get_num_inputs()
    logger.info("Reference model input info:")
    for idx in range(0, input_num):
        input_name = reference_module.get_input_name(idx)
        input_info = reference_module.get_input_info(input_name).ascontiguous()
        input_names.append(input_name)
        logger.info(
            f"  Input[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, mem_size = {input_info.mem_size}, stride = {input_info.stride}, format = {input_info.format.name}"
        )
        if resizer_crop_str not in input_name:
            max_img_height, max_img_width = input_info.shape[2], input_info.shape[3]

    # Load the sample image that will be shared across all requests.
    image_data = cv2.imread("../../data/snake.jpg")
    img_height, img_width = image_data.shape[:2]
    logger.info(f"input image shape: {image_data.shape}")

    # Keep the valid image region as dynamic crop information.
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

    # Convert HWC BGR image data to packed YUV420 input expected by the runtime.
    image_data = np.transpose(image_data, (2, 0, 1))
    input_data = torch.tensor(image_data.astype(np.float32))
    bgr2yuv_func = BGR2YUV(fmt="YUV420")
    input_data = torch.unsqueeze(bgr2yuv_func(input_data), 0).numpy()
    input_data = input_data.astype(np.uint8)

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

    # Build the per-input payload map shared by all queued requests.
    input_payloads = {
        input_name: dyn_info if resizer_crop_str in input_name else input_data
        for input_name in input_names
    }

    # 2.3 Fill the work queue with repeated requests for the same prepared input.
    qin = queue.Queue()
    qout = queue.Queue()
    for i in range(sample_num):
        qin.put((i, input_payloads))

    # 2.4 Create one worker thread for each loaded module instance.
    tid = 0
    for did in range(device_num):
        for i in range(thread_num):
            threads.append(
                threading.Thread(
                    target=thread_func,
                    args=(
                        tid,
                        did,
                        module_dict[did][i],
                        qin,
                        qout,
                        thread_cnt,
                        count_lock,
                    ),
                )
            )
            tid += 1

    # 3. Start all worker threads.
    for thread in threads:
        thread.start()

    # 4. Wait until all worker threads complete.
    for thread in threads:
        thread.join()

    # 5. Post-process outputs and verify the top-1 prediction.
    while not qout.empty():
        req_id, output_datas = qout.get()
        for output_name in output_datas:
            # Apply softmax to convert logits to probabilities
            output_data = softmax(output_datas[output_name])
            # Get the predicted class (top-1 prediction)
            pred = np.argsort(-output_data, axis=1, kind="quicksort").flatten()[0]
            prob_list = output_data.flatten()
            logger.info(
                "sample {} top1: predict cls = {}, prob = {:.6f}".format(
                    req_id, pred, prob_list[pred]
                )
            )
            # Verify result (65 corresponds to snake class in ImageNet dataset)
            assert pred == 65

    logger.info("<=== resnet50_multistreams python example completed.")
