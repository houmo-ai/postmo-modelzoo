# Copyright 2025 HOUMO AI
#
# File: execute_infer.py
# Description:
#   Execute inference tests for different model categories including APIs, models,
#   and HMATC components.
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
from cd_tester_utils import *


script_dir = os.path.dirname(os.path.abspath(__file__))
setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the inference execution script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Infer Models")
    parser.add_argument(
        "-log",
        "--log_file",
        type=str,
        default="./execute_infer_log.log",
        help="The path of log.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="use release models for testing (default is False).",
    )
    parser.add_argument(
        "--no_apis",
        action="store_true",
        help="don't execute apis testing.",
    )
    parser.add_argument(
        "--no_hmatc",
        action="store_true",
        help="don't execute apis testing.",
    )
    parser.add_argument(
        "--no_models",
        action="store_true",
        help="don't execute models testing.",
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
    """Main function to execute inference tests based on provided arguments.

    Args:
        args (argparse.Namespace): Parsed command line arguments

    Returns:
        int: 0 if all tests pass, 1 if any test fails
    """
    # Change to the tests directory to run tests
    test_dir = f"{script_dir}/../../tests"
    os.chdir(test_dir)
    logger.info("Current dir: %s", os.getcwd())

    # Set environment variables for test execution
    os.environ["IMODELZOO_MODELS_PATH"] = "/develop02/modelzoo/"
    if HOUMO_BACKEND == "xh2":
        os.environ["SKIP_INFER"] = "ON"
    os.environ["HDPL_PLATFORM"] = "ASIC"
    if args.release is False:
        os.environ["USE_RELEASED_MODELS"] = "OFF"

    root_dir = f"{script_dir}/../../"
    if HOUMO_BACKEND == "xh1":
        shutil.rmtree(
            f"{script_dir}/../../tests/model_results_{HOUMO_BACKEND}",
            ignore_errors=True,
        )

    # hmatc_dir = f"{root_dir}/hmatc"
    # os.chdir(hmatc_dir)
    # logger.info(f"==> [CD Test] Install latest hmatc.")
    # os.system("./install.sh")

    test_dir = f"{root_dir}/tests"
    os.chdir(test_dir)
    logger.info("Current dir: %s", os.getcwd())
    final_flag = True

    # Run API tests if not disabled
    if args.no_apis is False:
        # Clean up models directory before API tests
        shutil.rmtree(
            f"{root_dir}/apis/models",
            ignore_errors=True,
        )
        logger.info(f"==> [CD Test] Start apis_tests")
        cmds = [
            "pytest",
            "-v",
            "-s",
            "apis_tests",
            "-k",
            "not qwen3",
            f"--junitxml={script_dir}/pytest_results_infer_apis.xml",
        ]
        if not run_tests(cmds, args.log_file):
            logger.error(f"<== [CD Test] End apis_tests, Failed.")
            final_flag = False

    # Run model tests if not disabled
    if args.no_models is False:
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
        ]
        # --collect-only
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
                f"--junitxml={script_dir}/pytest_results_infer_{key_str}.xml",
            ]
            if args.model_str:
                cmds += ["-m", f"{args.model_str}"]
            logger.info(f"execute cmds: {cmds}")
            if not run_tests(cmds, args.log_file):
                logger.error(f"<== [CD Test] End model_tests: {key_str}, Failed.")
                final_flag = False

    # Run HMATC tests if not disabled
    if args.no_hmatc is False:
        model_list = ["resnet50", "yolov5s"]
        for model in model_list:
            logger.info(f"==> [CD Test] Start hmatc_tests: {model}")
            cmds = [
                "pytest",
                "-v",
                "-s",
                "hmatc_tests",
                "-m",
                model,
                f"--junitxml={script_dir}/pytest_results_infer_{model}.xml",
            ]
            if not run_tests(cmds, args.log_file):
                logger.error(f"<== [CD Test] End hmatc_tests {model}, Failed.")
                final_flag = False

    if not final_flag:
        return 1
    return 0


if __name__ == "__main__":
    args = parse_args()

    ret = main(args)
    logger.info("Ret: %d", ret)

    # Exit with the same code as main function
    exit(ret)
