# Copyright (c) 2025 HOUMO AI
#
# File: computing_perf.py
# Description:
#   Computing Performance Testing Tool - Python script for measuring
# AI core computing performance using convolution operations.
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
import shutil
import math
import json
import logging
import argparse
import platform
import numpy as np
import time
import multiprocessing
from itertools import zip_longest

try:
    import tcim_lite
except ImportError:
    print("Please install tcim_lite")
    exit(-1)

import onnx

logging.basicConfig(level=logging.INFO)


class ConvUtil:
    """utils for conv2d related shapes or constraints"""

    @staticmethod
    def get_padding_for_same_fmap_size(input_feature_size, kernel_size, stride):
        total_pad_lower_bound = tuple(
            math.ceil(input_feature_size[i] / stride[i] - 1) * stride[i]
            + kernel_size[i]
            - input_feature_size[i]
            for i in range(2)
        )
        total_pad_upper_bound = tuple(
            total_pad_lower_bound[i] + stride[i] - 1 for i in range(2)
        )
        total_pad_h = (
            total_pad_lower_bound[0]
            if total_pad_lower_bound[0] >= 0
            else total_pad_upper_bound[0]
        )
        total_pad_w = (
            total_pad_lower_bound[1]
            if total_pad_lower_bound[1] >= 0
            else total_pad_upper_bound[1]
        )
        padding_low = (
            max(0, math.floor(total_pad_h / 2)),
            max(0, math.floor(total_pad_w / 2)),
        )
        padding_high = (total_pad_h - padding_low[0], total_pad_w - padding_low[1])
        assert len(set(padding_low)) == len(set(padding_high)) == 1
        return padding_low[0], padding_high[0]


def gen_conv_model_and_tops_xh2(feature_size=(64, 64)):
    """model with graph: conv2d"""
    try:
        from tcim.test_utils.onnx_builder.hmir_op_builder import make_transpose_node
        from tcim.test_utils.onnx_builder.onnx_builder import (
            make_conv2d_node,
            make_model,
            make_tensor,
        )
        from tcim.hmcc_converter.base_hmonnx import HMOnnxModelVersion
    except ImportError:
        raise ImportError("Please install tcim to use this function")

    dtype = onnx.TensorProto.FLOAT16
    batch_size = 1
    channel = 256
    kernel_shape = (3, 3)
    strides = (1, 1)
    num_layers = 64
    padding = ConvUtil.get_padding_for_same_fmap_size(
        feature_size, kernel_shape, strides
    )
    input_name = "input"

    input_tensor = make_tensor(input_name, (batch_size, *feature_size, channel), dtype)
    current_tensor = input_tensor
    initializers = []
    nodes = []
    mid_tensors = []

    weight_shape = [*kernel_shape, channel, channel]
    bias_shape = (weight_shape[-1],)
    weight_data = np.random.normal(
        loc=0.0,
        scale=np.sqrt(
            1.0 / (kernel_shape[0] * kernel_shape[1] * channel)
        ),  # Xavier initialization
        size=[
            weight_shape[3],
            weight_shape[2],
            weight_shape[0],
            weight_shape[1],
        ],  # co, ci, kh, kw
    ).astype(np.float16)
    bias_data = np.random.uniform(low=-10, high=10, size=bias_shape).astype(np.float16)
    # permute from NHWC to NCHW
    current_tensor = make_transpose_node(
        "nhwc_to_nchw",
        input_names=[current_tensor.name],
        output_names=["nhwc_to_nchw"],
        perm=[0, 3, 1, 2],
        out_shape=(batch_size, channel, *feature_size),
        nodes=nodes,
        output_dtype="float16",
    )
    mid_tensors.append(current_tensor)
    for i in range(num_layers):
        current_layer_name = f"conv{i}"
        current_tensor = make_conv2d_node(
            name=current_layer_name,
            input_names=[current_tensor.name],
            output_names=[current_layer_name],
            kernel_shape=kernel_shape,
            out_channels=channel,
            pads=[padding[0], padding[0], padding[1], padding[1]],
            strides=strides,
            weight_data=weight_data,
            bias_data=bias_data,
            batch_size=batch_size,
            out_feature_shape=feature_size,
            initializers=initializers,
            nodes=nodes,
            output_dtype="float16",
        )
        mid_tensors.append(current_tensor)

    # permute back to NHWC
    current_tensor = make_transpose_node(
        "nchw_to_nhwc",
        input_names=[current_tensor.name],
        output_names=["nchw_to_nhwc"],
        perm=[0, 2, 3, 1],
        out_shape=(batch_size, *feature_size, channel),
        nodes=nodes,
        output_dtype="float16",
    )
    mid_tensors.append(current_tensor)

    opset_import = onnx.OperatorSetIdProto()
    opset_import.version = HMOnnxModelVersion.ONNX.get_opset_version()
    opset_import.domain = "ai.onnx"

    model = make_model(
        nodes,
        "test",
        inputs=[input_tensor],
        outputs=[current_tensor],
        value_info=mid_tensors,
        initializer=initializers,
        opset_imports=[HMOnnxModelVersion.ONNX.get_opset_id()],
    )

    model_tops_info = {
        "num_layers": num_layers,
        "batch_size": batch_size,
        "in_channel": channel,
        "out_channel": channel,
        "feature_size": feature_size,
        "kernel_shape": kernel_shape,
    }
    model_tops_info["num_tops"] = (
        num_layers
        * batch_size
        * channel
        * channel
        * feature_size[0]
        * feature_size[1]
        * kernel_shape[0]
        * kernel_shape[1]
        * 2
        / (1000**4)
    )

    return model, model_tops_info


def parse_dtype(dtype: dict):
    bits = dtype.get("bits", 0)
    if dtype.get("code", "") == "int":
        return {8: "int8", 16: "int16", 32: "int32"}[bits]
    if dtype.get("code", "") == "uint":
        return {8: "uint8", 16: "uint16", 32: "uint32"}[bits]
    if dtype.get("code", "") == "float":
        return {16: "float16", 32: "float32"}[bits]
    raise RuntimeError(f"unknown dtype:{dtype}")


def gen_random_data(input_shape, input_dtype):
    if np.issubdtype(input_dtype, np.integer):
        data_size_in_bytes = np.prod(input_shape) * np.dtype(input_dtype).itemsize
        random_int8_data = np.random.randint(
            -128, 128, data_size_in_bytes, dtype=np.int8
        )
        return random_int8_data.view(input_dtype).reshape(input_shape)
    if np.issubdtype(input_dtype, np.floating):
        return np.random.random(input_shape).astype(input_dtype)
    raise ValueError(f"Unsupported input dtype: {input_dtype}")


# pylint: disable=too-many-arguments,too-many-locals
def run_model_wrapper(
    tid,
    did,
    model_path,
    wm,
    input_names,
    input_datas,
    inner_round_num,
    outer_round_num,
    barrier,
    queue_for_start_time,
    verbose,
):

    time.sleep(
        0.1 * tid
    )  # stagger thread start time a bit to avoid loading model at the same time

    # load model, create a stream and set to the model
    option = tcim_lite.runtime.Option(wm)
    run_option = tcim_lite.runtime.RunOption(inner_round_num)
    model = tcim_lite.runtime.load(model_path, option=option)
    if verbose:
        print(f"thread {tid} on device {did} load model done.")
    stream = tcim_lite.runtime.Stream(True)
    model.set_stream(stream)
    # set input to the model
    for input_name, input_data in zip(input_names, input_datas):
        input_info = model.get_input_info(input_name)
        model.set_input(input_name, tcim_lite.runtime.Tensor(input_info, input_data))

    # wait until all threads ready
    if verbose:
        print(
            f"thread {tid} on device {did} will start run when all threads are ready..."
        )
    barrier.wait()
    queue_for_start_time.put(time.time())
    # infer loop
    for _ in range(outer_round_num):
        model.run(False, run_option)
    model.sync()


# pylint: disable=too-many-arguments
def run_multi_streams(
    hmm_path: str,
    json_path: str,
    process_num: int,
    inner_round_num: int,
    outer_round_num: int,
    device_id: int,
    verbose: bool = False,
):
    # pylint: disable=unused-argument,unused-variable,too-many-locals,redefined-outer-name

    with open(json_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)
        input_shapes = [
            input_info["shape"] for input_info in model_info["Model"]["inputs"]
        ]
        input_names = [
            input_info["name"] for input_info in model_info["Model"]["inputs"]
        ]
        input_dtypes = [
            parse_dtype(input_info["dtype"])
            for input_info in model_info["Model"]["inputs"]
        ]

    queue_for_start_time = multiprocessing.Queue()

    # 1. prepare input data
    input_datas = [
        gen_random_data(input_shape, input_dtype)
        for input_shape, input_dtype in zip_longest(input_shapes, input_dtypes)
    ]

    # 2. define barrier
    barrier = multiprocessing.Barrier(process_num)

    # 3. create processes
    processes = []
    tid = 0
    for _ in range(process_num):
        wm = tcim_lite.runtime.WeightManager(device_id)
        p = multiprocessing.Process(
            target=run_model_wrapper,
            args=(
                tid,
                device_id,
                hmm_path,
                wm,
                input_names,
                input_datas,
                inner_round_num,
                outer_round_num,
                barrier,
                queue_for_start_time,
                verbose,
            ),
        )
        processes.append(p)
        tid += 1

    # 4. start processes
    for p in processes:
        p.start()

    # 5. wait for processes to finish
    for p in processes:
        p.join()
    end_time = time.time()

    # 6. get start time
    start_time_list = []
    while not queue_for_start_time.empty():
        start_time_list.append(queue_for_start_time.get())
    start_time = min(start_time_list)  # use the earliest start time

    return end_time - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, default="./output", help="work dir.")
    parser.add_argument(
        "--device-id", type=int, default=0, help="the device id to run the test on."
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="skip model building."
    )
    parser.add_argument("--skip-run", action="store_true", help="skip model running.")
    parser.add_argument(
        "--sample-num",
        type=int,
        default=512,
        help="total number of samples to run, the default value runs about 5 seconds.",
    )
    parser.add_argument(
        "--compute-mode",
        choices=("int8", "bfp16"),
        default="bfp16",
        required=False,
        help="the compute mode to run the test, default is bfp16.",
    )
    parser.add_argument(
        "--no-load-store",
        action="store_true",
        help="only emit ops using te when building the model.",
    )
    args = parser.parse_args()

    target = os.getenv("HOUMO_TARGET")
    if target not in ["xh2"]:
        raise ValueError("HOUMO_TARGET must be xh2")

    MODEL_NAME = f"{target}_conv"

    enable_xh2_sparse_feature = False
    PEAK_TOPS = 100.0
    if args.compute_mode == "int8":
        PEAK_TOPS = 160.0
        enable_xh2_sparse_feature = True
        os.environ["RUN_ON_SUBTARGET"] = "2"
        MODEL_NAME = f"{MODEL_NAME}_int8"
    mode_name = args.compute_mode
    if args.no_load_store:
        mode_name = f"{mode_name}_no_load_store"
        MODEL_NAME = f"{MODEL_NAME}_no_load_store"

    print("#########################################")
    print("##  AI core compute performance test   ##")
    print("#########################################")
    print()

    platform_name = platform.machine().lower()

    output_dir = os.path.join(args.work_dir, target)
    build_tmp_dir = os.path.join(output_dir, "tcim_temp", mode_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(build_tmp_dir, exist_ok=True)

    hmm_path = os.path.join(output_dir, f"{MODEL_NAME}.hmm")
    json_path = os.path.join(output_dir, f"{MODEL_NAME}_model.json")

    not_found_tcim = False
    try:
        import tcim
    except ImportError:
        not_found_tcim = True

    if platform_name != "x86_64" or not_found_tcim:
        print(
            f"Skipping model generation since the current platform is {platform_name} or not found tcim."
        )
        args.skip_build = True

    # build model
    print("=========================================")
    if not os.path.exists(hmm_path) and args.skip_build:
        print(f"Skipping model generation since download hmm from network.")
        args.skip_build = True
        from hmatc.utils.utils import get_file_from_jfrog

        if "HOUMO_MODELZOO_URL" not in os.environ:
            os.environ["HOUMO_MODELZOO_URL"] = (
                "http://artifactory.houmo.ai/artifactory/Dadao"
            )
        if target == "xh2":
            suffix = ""
            if args.no_load_store:
                suffix = "no_load_store_"
            if args.compute_mode == "int8":
                zipped_hmm_path = f"models/tools/computing_perf/hmm_xh2_conv_int8_1core_{suffix}20260715.zip"
            elif args.compute_mode == "bfp16":
                zipped_hmm_path = f"models/tools/computing_perf/hmm_xh2_conv_1core_{suffix}20260715.zip"

        get_file_from_jfrog(zipped_hmm_path, "./", "./")

    if not args.skip_build:
        if target == "xh2":
            if args.compute_mode == "int8":
                feature_size = (128, 128)
            elif args.compute_mode == "bfp16":
                feature_size = (64, 64)
            gen_conv_model_and_tops = lambda: gen_conv_model_and_tops_xh2(
                feature_size=feature_size
            )
        print("Generating onnx model...")
        hmonnx_model, model_tops_info = gen_conv_model_and_tops()
        model_path = os.path.join(output_dir, f"{MODEL_NAME}.onnx")
        print(f"Model generated successfully, saving to {model_path}.")
        onnx.save(hmonnx_model, model_path)
        model_tops_path = os.path.join(output_dir, f"{MODEL_NAME}_tops.json")
        with open(model_tops_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(model_tops_info, indent=4))
        print("=========================================")
        print(f"Building model with tcim in output dir: {output_dir}")
        print(f"Using tcim work dir: {build_tmp_dir}")
        import tcim

        build_kwargs = {
            "output_name": MODEL_NAME,
            "output_dir": output_dir,
            "work_dir": build_tmp_dir,
            "target": target,
            "ncore": 1,
            "opt_level": "O2",
            "io_layout": "any",
            "enable_xh2_sparse_feature": enable_xh2_sparse_feature,
            "codegen_backend": "cpp-v1",
            "march": "v1",
            "skip_check": True,
        }
        if args.no_load_store:
            build_kwargs["emit_cpp_extra_args"] = "only-emit-op-list=xh2.te,xh2.pipelined_conv2d"
        tcim.build_from_hmonnx(hmonnx_model, **build_kwargs)
        shutil.copyfile(os.path.join(build_tmp_dir, "model.json"), json_path)
        print("Model built successfully.")

    if not args.skip_run:
        if tcim_lite.runtime.get_device_num() < 1:
            print("No available devices found.")
            exit(0)
        print("=========================================")
        print(f"Running model {hmm_path}")
        if not os.path.exists(hmm_path):
            raise FileNotFoundError(
                f"Model file {hmm_path} not found, please re-run the build step."
            )
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Model json file {json_path} not found, please re-run the build step."
            )
        PROCESS_NUM = 4
        # warm up
        _ = run_multi_streams(
            hmm_path=hmm_path,
            json_path=json_path,
            process_num=PROCESS_NUM,
            inner_round_num=1,
            outer_round_num=1,
            device_id=args.device_id,
            verbose=False,
        )
        assert args.sample_num % PROCESS_NUM == 0
        SAMPLE_NUM_PER_PROCESS = args.sample_num // PROCESS_NUM
        print(
            f"Running model using {PROCESS_NUM} threads for {SAMPLE_NUM_PER_PROCESS} samples per thread"
        )
        elapsed_time = run_multi_streams(
            hmm_path=hmm_path,
            json_path=json_path,
            process_num=PROCESS_NUM,
            inner_round_num=1,
            outer_round_num=SAMPLE_NUM_PER_PROCESS,
            device_id=args.device_id,
            verbose=True,
        )
        print(
            f"Model run successfully on device {args.device_id}, elapsed time: {elapsed_time:.2f} seconds."
        )
        model_tops_path = os.path.join(output_dir, f"{MODEL_NAME}_tops.json")
        with open(model_tops_path, "r", encoding="utf-8") as f:
            model_tops_info = json.load(f)
        TOPS = (
            model_tops_info["num_tops"]
            * PROCESS_NUM
            * SAMPLE_NUM_PER_PROCESS
            / elapsed_time
        )
        COMPUTE_UNIT = "TFLOP" if args.compute_mode == "bfp16" else "TOP"
        COMPUTE_MODE = "bFP16" if args.compute_mode == "bfp16" else "INT8"
        print("\n" + "=" * 60)
        print("           Performance Test Summary")
        print("=" * 60)

        print(f"{'Model Parameter':<25} | {'Value':>15}")
        print("-" * 60)
        print(f"{'Number of Conv2d Layers':<25} | {model_tops_info['num_layers']:>15}")
        print(f"{'Batch Size':<25} | {model_tops_info['batch_size']:>15}")
        print(f"{'Input Channels':<25} | {model_tops_info['in_channel']:>15}")
        print(f"{'Output Channels':<25} | {model_tops_info['out_channel']:>15}")
        print(f"{'Feature Map Size':<25} | {str(model_tops_info['feature_size']):>15}")
        print(f"{'Kernel Shape':<25} | {str(model_tops_info['kernel_shape']):>15}")
        print("-" * 60)
        print(
            f"{'Computing amount':<25} | {model_tops_info['num_tops']:>15.4f} {COMPUTE_UNIT}s/sample"
        )
        print(f"{'Test time':<25} | {elapsed_time:>15.4f} seconds")
        print(f"{'PERFORMANCE':<25} | {TOPS:>15.2f} {COMPUTE_UNIT}S@{COMPUTE_MODE}")
        print("=" * 60 + "\n")
