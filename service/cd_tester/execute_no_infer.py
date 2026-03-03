# Copyright 2025 HOUMO AI
#
# File: execute_no_infer.py
# Description:
#   Execute quantization and compilation tests without running inference.
#   This script handles test orchestration for model quantization and compilation
#   processes, setting up the appropriate environment and running model tests
#   without executing actual inference.
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
import argparse
import logging
import subprocess
from cd_tester_utils import *


script_dir = os.path.dirname(os.path.abspath(__file__))
setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the quantization and compilation execution script."""
    parser = argparse.ArgumentParser(description="Quant and Compile Models")
    parser.add_argument(
        "-log",
        "--log_file",
        type=str,
        default="./execute_no_infer_log.log",
        help="The path of log.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="use release models for testing (default is False).",
    )
    parser.add_argument(
        "-k",
        "--key_str",
        type=str,
        default="",
        help="pytest -k value",
    )
    parser.add_argument(
        "-m",
        "--model_str",
        type=str,
        default="",
        help="pytest -m value",
    )

    args = parser.parse_args()
    return args


def main(args) -> int:
    """Main function to execute quantization and compilation tests based on provided arguments.

    Args:
        args (argparse.Namespace): Parsed command line arguments

    Returns:
        int: 0 if all tests pass, 1 if any test fails
    """
    # Change to the tests directory to run tests
    test_dir = f"{script_dir}/../../tests"
    os.chdir(test_dir)
    logger.info("Current dir: %s", os.getcwd())

    # Set environment variables for quantization/compilation tests (no inference)
    os.environ["SKIP_INFER"] = "ON"
    os.environ["HDPL_PLATFORM"] = "ISIM"
    os.environ["IMODELZOO_MODELS_PATH"] = "/data02/modelzoo/"
    if args.release is False:
        os.environ["USE_RELEASED_MODELS"] = "OFF"

    root_dir = f"{script_dir}/../../"
    if HOUMO_BACKEND == "xh2":
        shutil.rmtree(
            f"{script_dir}/../../tests/model_results_{HOUMO_BACKEND}",
            ignore_errors=True,
        )

        os.chdir(root_dir)
        logger.info("Current dir: %s", os.getcwd())
        cmd = f"bash -c 'source env.sh && env'"
        result = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        # Set environment variables for xh2 from sourced environment
        for line in result.stdout.splitlines():
            if "=" in line and (
                "PYTHONPATH" in line
                or "HF_ENDPOINT" in line
                or "HOUMO_DATASETS_PATH" in line
            ):
                key, value = line.split("=", 1)
                os.environ[key] = value
        logger.info("*** Env Info ***")
        logger.info("PYTHONPATH: %s", os.getenv("PYTHONPATH"))
        logger.info("HF_ENDPOINT: %s", os.getenv("HF_ENDPOINT"))
        logger.info("HOUMO_DATASETS_PATH: %s", os.getenv("HOUMO_DATASETS_PATH"))

    os.system("pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxslim")
    # hmatc_dir = f"{root_dir}/hmatc"
    # os.chdir(hmatc_dir)
    # logger.info(f"==> [CD Test] Install latest hmatc.")
    # os.system("./install.sh")

    # Set CUDA device for tests
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    test_dir = f"{root_dir}/tests"
    os.chdir(test_dir)
    logger.info("Current dir: %s", os.getcwd())

    # Define test categories for different model types
    key_list = [
        "asr",
        "autodrive",
        "backbone",
        "detection",
        "embedding",
        "estimation",
        "llm",
        "ocr",
        "omni",
        "diffusion",
        "segmentation",
        "reranker",
    ]
    # --collect-only
    flag = True
    # Run tests for each category
    for key_str in key_list:
        if args.key_str:
            key_str = f"{key_str} and ({args.key_str})"
        logger.info(f"==> [CD Test] Start models_tests: {key_str}")
        cmds = [
            "pytest",
            "-v",
            "-s",
            "models_tests",
            "-k",
            key_str,
            f"--junitxml={script_dir}/pytest_results_no_infer_{key_str}.xml",
        ]
        # Add model-specific filter if provided
        if args.model_str:
            cmds += ["-m", f"{args.model_str}"]
        logger.info(f"execute cmds: {cmds}")
        # Execute model tests and check for success
        if not run_tests(cmds, args.log_file):
            logger.error(f"<== [CD Test] End model_tests: {key_str}, Failed.")
            flag = False

    if not flag:
        return 1

    return 0


if __name__ == "__main__":
    args = parse_args()

    ret = main(args)
    logger.info("Ret: %d", ret)

    exit(ret)
