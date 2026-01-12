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
import threading
import queue
import numpy as np
import cv2
import argparse
from loguru import logger

HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../../..")
sys.path.insert(0, f"{HOUMO_EXAMPLES_PATH}/hmatc")
from hmatc.utils.postprocess import softmax
import tcim_lite as tcim

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        dest="device_num",
        type=int,
        default=1,
        help="device_num",
    )
    parser.add_argument(
        "-t",
        dest="thread_num",
        type=int,
        default=4,
        help="thread_num",
    )
    parser.add_argument(
        "-s",
        dest="sample_num",
        type=int,
        default=10,
        help="sample_num",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logger.info("===> resnet50_multistreams python example start...")
    logger.info(
        f"houmo target: {HOUMO_TARGET}, tcim runtime version: {tcim.runtime.get_version()}"
    )

    # set the parameters
    args = get_args()
    device_num = args.device_num  # Number of devices to use for inference
    thread_num = args.thread_num  # Number of threads per device
    sample_num = args.sample_num  # Total number of samples to process
    model_path = "./resnet50_xh2_b1_1core.hmm"  # Path to the ResNet50 model file

    # Limit to 1 thread if not running on ASIC platform
    if not os.environ.get("HDPL_PLATFORM") == "ASIC":
        thread_num = 1
    logger.info(f"devices: {device_num}")
    logger.info(f"threads: {thread_num}")
    logger.info(f"samples: {sample_num}")
    logger.info(f"model: {model_path}")

    modules = []
    threads = []

    # Verify that the model file exists
    if not os.path.exists(model_path):
        logger.error("[error] could not find model: {}".format(model_path))
        exit(-1)

    # 1. Preprocess input image for ResNet50 inference
    input_data = cv2.imread("../../data/snake.jpg")

    # Convert BGR to RGB and resize to the required input size (224x224)
    image_rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224))  # HWC uint8
    # Define normalization parameters (ImageNet mean and std values)
    mean_arr = np.array([123.675, 116.28, 103.53])
    std_arr = np.array([58.395, 57.12, 57.375])
    # Normalize the image using the mean and std values
    image_norm = (image_rgb - mean_arr) / std_arr
    image_norm = np.transpose(image_norm, (2, 0, 1))  # CHW uint8
    image_norm = np.expand_dims(image_norm, axis=0)  # NCHW uint8
    input_data = image_norm.astype(np.float16)

    # 2. Prepare input and output queues for multi-threading
    input_datas = []
    input_datas.append(input_data)

    # Create queues for input tasks and output results
    qin = queue.Queue()  # Input queue for tasks
    qout = queue.Queue()  # Output queue for results
    # Fill the input queue with sample_num tasks, each containing the same input data
    for i in range(sample_num):
        qin.put((i, input_datas))

    # 3. Define threading function for inference
    barrier = threading.Barrier(thread_num * device_num)
    thread_cnt = 0

    def thread_func(tid, did, module, qin, qout, barrier):
        """
        Function executed by each thread for inference on a specific device.

        Args:
            tid: Thread ID
            did: Device ID
            module: Model module for inference
            qin: Input queue with tasks
            qout: Output queue for results
            barrier: Synchronization barrier
        """
        count = 0
        # 3.1 Prepare input information by querying the module
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

        # 3.2 Prepare output information by querying the module
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

        # 3.3 Wait for all threads to be ready before starting inference
        barrier.wait()

        # 3.4 Main inference loop - process tasks from input queue until empty
        while not qin.empty():
            # 3.4.1 Get task data from the input queue
            req_id, input_datas = qin.get()

            # 3.4.2 Set input data to the module for each input name
            for input_name in input_infos:
                module.set_input(input_name, input_datas[0])

            # 3.4.3 Execute the inference and synchronize
            module.run()
            module.sync()

            # 3.4.4 Get output data and put results in output queue
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

    # 4.1 Load model for each device and thread
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

    # 4.2 Create threads for each device-thread combination
    tid = 0
    for did in range(device_num):
        for i in range(thread_num):
            threads.append(
                threading.Thread(
                    target=thread_func,
                    args=(tid, did, module_dict[did][i], qin, qout, barrier),
                )
            )
            tid += 1

    # Start all threads to begin processing
    for thread in threads:
        thread.start()

    # 5. Wait for all threads to complete before proceeding
    for thread in threads:
        thread.join()

    # 6. Post-process results and verify output
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

    logger.info("<=== resnet50_multistreams python example completed.\n")
