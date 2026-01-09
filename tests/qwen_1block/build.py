#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: build.py
# Description:
#   Build and test script for Qwen2 1 Block model compilation and validation.
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
import time
import argparse

import logging

logging.basicConfig(level="ERROR")

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default="qwen",
        help="path to the model dir",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="qwen_decode_1block",
        help="output houmo model name",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=4,
        help="core number",
    )
    parser.add_argument(
        "--stage",
        dest="stage",
        type=str,
        default="build",
        help='build stage choise=["build", "test", "all"]',
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
        help="build output dir",
    )
    args = parser.parse_args()
    return args


def build(args=None):
    """build and test houmo model."""
    model_dir = args.model_dir
    model_name = args.model_name
    batch = args.batch
    ncore = args.ncore
    stage = args.stage
    output_dir = args.output_dir
    part_name = f"qwen_decode_1block_{ncore}cores"
    quant_name = "hmquant_qwen_with_act"
    onnx_name = quant_name + ".onnx"
    onnx_path = os.path.join(model_dir, onnx_name)
    weight_path = os.path.join(model_dir, "weight.npy")
    hmm_path = os.path.join(output_dir, f"{model_name}.hmm")
    profile = {}

    # 1. build model
    if stage == "build" or stage == "all":
        import tcim

        print(f"\n===> {model_name} build start...")
        start = time.time()
        tcim.build_from_hmonnx(
            onnx_path,
            weight_path,
            model_name=model_name,
            ncore=ncore,
            legacy=True,
            output_dir=output_dir,
            work_dir=os.path.join(output_dir, "tcim"),
        )
        profile["build_time"] = time.time() - start
        print(f'{model_name} build completed in {profile["build_time"]:.3f} s.')

    # 2. test model
    if stage == "test" or stage == "all":
        import tcim_lite

        print(f"\n===> {model_name} test start...")
        # 2.1 load model
        start = time.time()
        module = tcim_lite.runtime.load(hmm_path)
        profile["load_time"] = time.time() - start
        print(f'{model_name} load completed in {profile["load_time"]*1000:.3f} ms.')

        # 2.2 set input with golden
        profile["setinput_time"] = 0
        input_num = module.get_num_inputs()
        print("input_num:", input_num)
        for id in range(input_num):
            input_name = module.get_input_name(id)
            input_info = module.get_input_info(input_name)
            print(
                f"input_info[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
            )
            input_file_name = "hmquant_" + model_name + "_" + input_name + "_input.npy"
            input_data_path = os.path.join(model_dir, input_file_name)
            input_data = np.load(input_data_path).astype(input_info.dtype)
            if id != 0:
                input_data = [input_data]
            if input_name == "current_length":
                current_length = input_data[0]
            input_data = np.concatenate([input_data for i in range(batch)], axis=0)
            print(
                f"golden input[{input_name}] shape = {input_data.shape}, dtype = {input_data.dtype}"
            )
            start = time.time()
            module.set_input(input_name, input_data)
            profile["setinput_time"] += time.time() - start
        print(
            f'{model_name} set {input_num} inputs completed in {profile["setinput_time"]*1000:.3f} ms.'
        )

        # 2.3 infer model
        start = time.time()
        module.run()
        module.sync()
        profile["infer_time"] = time.time() - start
        print(f'{model_name} infer completed in {profile["infer_time"]*1000:.3f} ms.')

        # 2.4. get output and compare with golden
        result_check = True
        profile["getoutput_time"] = 0
        output_num = module.get_num_outputs()
        print("output_num:", output_num)
        for id in range(output_num):
            output_name = module.get_output_name(id)
            output_info = module.get_output_info(output_name)
            print(
                f"output_info[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}"
            )
            start = time.time()
            output_data = module.get_output(output_name)
            profile["getoutput_time"] += time.time() - start
            start = time.time()
            profile["dequant_time"] += time.time() - start
            output_data = output_data.numpy()
            # only compare [1,current_length,4096]
            output_data = output_data[:1, :current_length, :]
            print(
                f"output[{output_name}] shape = {output_data.shape}, dtype = {output_data.dtype}"
            )
            output_data_path = os.path.join(
                model_dir, f"hmquant_{model_name}_{output_name}_output.npy"
            )
            if os.path.exists(output_data_path):
                golden_output = np.load(output_data_path)
                golden_output = np.concatenate(
                    [golden_output for i in range(batch)], axis=0
                )
            elif not os.path.exists(output_data_path):
                print(
                    "[warning] compare canceled while golden data not found -> {output_data_path}"
                )
                result_check &= False
                continue
            if golden_output.shape == output_data.shape:
                from hmassist.utils.dist_metrics import cosine_distance

                cosine_dist1 = cosine_distance(golden_output, output_data)
                is_match = (golden_output == output_data).all()
                print(
                    f"[compare] golden output [{output_name}] match={is_match}, similarity={cosine_dist1:.6f}"
                )

                if cosine_dist1 < 0.999:
                    result_check &= False
            else:
                result_check &= False
                print(
                    f"[compare] golden output [{output_name}] shape not match {golden_output.shape} vs {output_data.shape},"
                )
            print(
                f'{model_name} get ouput completed in {profile["getoutput_time"]*1000:.3f} ms.'
            )
        if not result_check:
            print("[error] result check failed.")
            exit(-1)
        print(f"<=== {part_name} test success.")


if __name__ == "__main__":
    args = get_args()
    build(args)
