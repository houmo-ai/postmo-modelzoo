"""
* Copyright (c) 2025 HOUMO AI
*
* File: build.py
* Description:
*   Build and test script for converting ResNet50 model to Houmo AI platform.
*   This script handles the complete workflow from ONNX model to optimized
*   Houmo model (HMM format), including compilation, inference testing, and
*   output validation against golden data.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-License-Identifier: Apache-2.0
"""

import os
import numpy as np
import time
import argparse
from loguru import logger


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def cosine_distance(data1, data2):
    """
    Calculate the cosine similarity between two arrays.

    Args:
        data1 (numpy.ndarray): First input array
        data2 (numpy.ndarray): Second input array

    Returns:
        float: Cosine similarity value between -1 and 1, or -1 if shapes don't match
    """
    if data1.shape != data2.shape:
        logger.error(f"Shape not equal {data1.shape} vs {data2.shape}")
        return -1

    # Flatten arrays and convert to float64 for computation
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")

    # Replace infinite values with float16 limits to avoid computation errors
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min

    # Normalize the vectors
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)

    # Compute cosine similarity
    cosine_dist = np.dot(v1_norm, v2_norm)

    # Return -1 if result is NaN (invalid)
    if np.isnan(cosine_dist):
        return -1

    return cosine_dist


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the build script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="Path to the directory containing the quantized model files",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="resnet50",
        help="Name of the output Houmo model",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=1,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--ncore",
        dest="ncore",
        type=int,
        default=2,
        help="Number of cores",
    )
    parser.add_argument(
        "--stage",
        dest="stage",
        type=str,
        default="all",
        help='Build stage to execute. Options: ["build", "test", "all"]',
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default="output",
        help="Directory to store build outputs",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Print detailed information during execution",
    )

    args = parser.parse_args()
    return args


def build(args=None):
    """
    Build and test the Houmo model.

    This function performs the complete build and test workflow for the ResNet50 model:
    1. Compiles the ONNX model to Houmo format (.hmm)
    2. Loads the compiled model
    3. Runs inference with test inputs
    4. Compares outputs with golden data to validate correctness

    Args:
        args (argparse.Namespace): Command-line arguments controlling the build process
    """
    # Extract parameters from arguments
    model_dir = args.model_dir
    model_name = args.model_name
    batch = args.batch
    ncore = args.ncore
    stage = args.stage
    output_dir = os.path.join(args.output_dir, HOUMO_TARGET)
    verbose = args.verbose
    opt_level = "O2"
    # Define file paths for the model and related files
    hmonnx_model_path = os.path.join(model_dir, f"hmquant_{model_name}_with_act.onnx")
    hmmodel_path = os.path.join(output_dir, f"{model_name}.hmm")
    work_dir = os.path.join(output_dir, "tcim")
    # Dictionary to store performance profiling data
    profile = {}

    # 1. Build model stage
    if stage == "build" or stage == "all":
        import tcim

        logger.info(f"===> {model_name} build start...")
        start = time.time()
        # Compile the ONNX model to Houmo format
        tcim.build_from_hmonnx(
            hmonnx_model_path,
            output_name=model_name,
            ncore=ncore,
            opt_level=opt_level,
            target="xh2",
            batch=batch,
            output_dir=output_dir,
            work_dir=work_dir,
            enable_dynamic_image_resize=False,
        )
        # Record build time
        profile["build"] = time.time() - start
        logger.info(f'{model_name} build completed in {profile["build"]:.3f} s.')
        assert os.path.isfile(
            hmmodel_path
        ), f"Model file {hmmodel_path} was not generated"
    # 2. Test model stage
    if stage == "test" or stage == "all":
        import tcim_lite

        logger.info(f"===> {model_name} test start...")
        start = time.time()
        # 2.1 Load the compiled model
        module = tcim_lite.runtime.load(hmmodel_path)
        profile["load"] = time.time() - start
        logger.info(f'{model_name} load completed in {profile["load"]*1000:.3f} ms.')

        # 2.2 Set input tensors with golden data
        profile["set_input"] = 0
        input_num = module.get_num_inputs()
        logger.info(f"input_num: {input_num}")

        # Process each input of the model
        for idx in range(input_num):
            input_name = module.get_input_name(idx)
            input_info = module.get_input_info(input_name)
            logger.info(
                f"input_info[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
            )

            # Load input data from file
            input_data_path = os.path.join(
                model_dir,
                f"hmquant_{model_name}_{input_name}_input.npy",
            )
            logger.info(f"input data path: {input_data_path}")
            assert os.path.isfile(input_data_path)

            # Load and process input data
            input_data = np.load(input_data_path).astype(input_info.dtype)
            input_data = np.concatenate([input_data for i in range(batch)], axis=0)
            logger.info(
                f"golden input[{input_name}] shape = {input_data.shape}, dtype = {input_data.dtype}"
            )

            # Set input tensor and time the operation
            start = time.time()
            module.set_input(input_name, input_data)
            profile["set_input"] += time.time() - start
        logger.info(
            f'{model_name} set {input_num} inputs completed in {profile["set_input"]*1000:.3f} ms.'
        )

        # 2.3 Inference model
        start = time.time()
        module.run()
        module.sync()
        profile["infer"] = time.time() - start
        logger.info(f'{model_name} infer completed in {profile["infer"]*1000:.3f} ms.')

        # 2.4 Get outputs and compare with golden data
        result_check = True
        profile["get_output"] = 0
        output_num = module.get_num_outputs()
        logger.info(f"output_num: {output_num}")

        # Process each output of the model
        for idx in range(output_num):
            output_name = module.get_output_name(idx)
            output_info = module.get_output_info(output_name)
            logger.info(
                f"output_info[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}"
            )

            # Get output tensor and time the operation
            start = time.time()
            output_data = module.get_output(output_name)
            profile["get_output"] += time.time() - start

            output_data = output_data.numpy()
            logger.info(
                f"output[{output_name}] shape = {output_data.shape}, dtype = {output_data.dtype}"
            )
            # Define path for golden output data
            output_data_path = os.path.join(
                model_dir,
                f"hmquant_{model_name}_{output_name}_output.npy",
            )
            assert os.path.isfile(output_data_path)

            # Load golden output data and compare with model output
            if os.path.exists(output_data_path):
                golden_output = np.load(output_data_path)
                golden_output = np.concatenate(
                    [golden_output for i in range(batch)], axis=0
                )
            elif not os.path.exists(output_data_path):
                logger.warning(
                    f"Compare canceled while golden data not found -> {output_data_path}"
                )
                result_check &= False
                continue

            # Compare shapes and compute similarity
            if golden_output.shape == output_data.shape:
                cosine_dist = cosine_distance(golden_output, output_data)
                is_match = cosine_dist > 0.999
                logger.info(
                    f"[compare] golden output [{output_name}] match={is_match}, similarity={cosine_dist:.6f}"
                )
                if is_match:
                    continue
                if cosine_dist < 0.999:
                    result_check &= False
                    if verbose:
                        logger.info("output_data:\n", output_data)
                        logger.info("golden_output:\n", golden_output)
            else:
                result_check &= False
                logger.error(
                    f"[compare] golden output [{output_name}] shape not match {golden_output.shape} vs {output_data.shape},"
                )
        logger.info(
            f'{model_name} get {output_num} ouputs completed in {profile["get_output"]*1000:.3f} ms.'
        )

        # Check if all comparisons passed
        if not result_check:
            raise RuntimeError("[error] result check failed.")
        logger.info(f"<=== {model_name} test success.")


if __name__ == "__main__":
    import platform

    arch = platform.machine()
    if arch != "x86_64":
        logger.error(f"Tcim not support platform: {arch}")
        exit(0)

    args = get_args()
    logger.info(args)
    build(args)
