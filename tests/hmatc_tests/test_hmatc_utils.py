# Copyright 2025 HOUMO AI
#
# File: test_hmatc_utils.py
# Description:
#   HMATC test utilities module.
#   This module provides utility functions for executing HMATC tests on different models.
#   It handles model configuration, test execution across multiple test types (quant, build,
#   demo, compare, eval, perf), and manages the complete test workflow from setup to cleanup.
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

import pytest
import os
import logging
import shutil
from glob import glob
from ..tests_utils.tests_common_utils import *


logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))


def _run_hmatc(
    model_info: dict, config_yml: str, hmatc_type: str, log_file: str
) -> bool:
    """
    Execute a specific HMATC test command for a given configuration.

    Args:
        model_info (dict): Dictionary containing model-specific information including data paths
        config_yml (str): Path to the YAML configuration file for the test
        hmatc_type (str): Type of HMATC test to run (quant, build, demo, compare, eval, perf)
        log_file (str): Path to the log file for test output

    Returns:
        bool: True if the command executed successfully, False otherwise
    """
    cmds = ["hmatc", hmatc_type, "--config", config_yml]
    if hmatc_type == "compare":
        cmds += [
            "--data_path",
            model_info["data_path"],
        ]
    elif hmatc_type == "perf":
        cmds += ["-wn", "10", "-sn", "500", "-tn", "8"]
    flag, _ = execute_test_cmd(cmds, log_file)
    if flag is False:
        logger.error(f"Execute hmatc {hmatc_type} {config_yml} failed!")

    return flag


def _perf_models(config_yml: str, log_file: str) -> bool:
    """
    Execute performance test sequence for a given configuration.

    Args:
        config_yml (str): Path to the YAML configuration file for the test
        log_file (str): Path to the log file for test output

    Returns:
        bool: True if all performance test steps executed successfully, False otherwise
    """
    # quant
    cmds = ["hmatc", "quant", "--config", config_yml]
    flag, _ = execute_test_cmd(cmds, log_file)
    if flag is False:
        logger.error(f"Perf test quant: {config_yml} failed!")
        return False
    # build
    ncore = "2" if HOUMO_BACKEND == "xh2" else "4"
    cmds = ["hmatc", "build", "--config", config_yml, "--ncore", ncore]
    flag, _ = execute_test_cmd(cmds, log_file)
    if flag is False:
        logger.error(f"Perf test build: {config_yml} failed!")
        return False
    # perf
    cmds = [
        "hmatc",
        "perf",
        "--config",
        config_yml,
        "-wn",
        "10",
        "-sn",
        "1000",
        "-tn",
        "8",
    ]
    flag, _ = execute_test_cmd(cmds, log_file)
    if flag is False:
        logger.error(f"Perf test: {config_yml} failed!")

    return flag


def execute_hmatc_cmd(model_name: str, setup_logging):
    """
    Execute HMATC tests for the specified model.

    Args:
        model_name (str): Name of the model to test (e.g., resnet50, yolov5s)
        setup_logging: pytest fixture for setting up logging configuration

    Raises:
        AssertionError: If any of the tests fail
    """
    log_file, pytest_request = setup_logging
    marker_vals = check_device_markers(pytest_request)
    dev_res_dir = f"{marker_vals[NDEVICE_MARKER]}_{marker_vals[DEVICE_MEM_MARKER]}"
    logger.info(f"log_file: {log_file}, dev_res_dir: {dev_res_dir}")

    if get_test_type() == TCaseType.SEPARATE_NO_INFER:
        skip_msg = f"Skip hmatc testcase {model_name} in the SEPARATE NO INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)

    model_dict = {
        "resnet50": {
            "model_dir": os.path.abspath(
                f"{script_dir}/../../models/backbone/resnet50"
            ),
            "data_path": "./imagenet/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG",
        },
        "yolov5s": {
            "model_dir": os.path.abspath(
                f"{script_dir}/../../models/detection/yolov5s"
            ),
            "data_path": "./coco2017/val2017/000000000139.jpg",
        },
    }
    prepare_test_folder(model_dict[model_name]["model_dir"], "hmatc")
    current_folder = os.getcwd()
    logger.info("current folder: %s.", current_folder)

    execute_test_cmd(["python3", "get_model.py", "--type", "raw"], "", True)

    test_configs = script_dir + f"/hmatc_configs/{model_name}"
    hmatc_types = ["quant", "build", "demo", "compare", "eval", "perf"]
    final_flag = True

    # Run functional tests
    for config_yml in glob(f"{test_configs}/func_test/*.yml"):
        logger.info(f"test config file: {config_yml}")
        for hmatc_type in hmatc_types:
            if not _run_hmatc(model_dict[model_name], config_yml, hmatc_type, log_file):
                final_flag = False

    # Run performance tests
    if HOUMO_BACKEND == "xh1":
        for config_yml in glob(f"{test_configs}/perf_test/*.yml"):
            if not _perf_models(config_yml, log_file):
                final_flag = False

    shutil.rmtree(os.getcwd())
    assert final_flag is True, "Hmatc Test Failed!"
    logger.info("Hmatc Test Success!")
