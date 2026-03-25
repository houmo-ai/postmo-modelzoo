# Copyright 2025 HOUMO AI
#
# File: test_apis_utils.py
# Description:
#   APIs test utilities module.
#   This module provides utility functions for executing API tests for different examples.
#   It handles example configuration loading, command generation, model downloading,
#   and execution of both Python and C++ demos with proper error handling and logging.
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
from ..tests_utils.tests_common_utils import *
from ..tests_utils.tests_pyvenv_utils import install_py_venv, VENV_NAME


logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))


def _load_example_cfg(example_name: str) -> dict:
    """
    Load example configuration from JSON file.

    Args:
        example_name (str): Name of the example whose configuration needs to be loaded

    Returns:
        dict: Configuration dictionary loaded from the JSON file, or None if file doesn't exist
    """
    example_cfg_path = script_dir + "/apis_configs/apis_cfg_" + example_name + ".json"
    return load_json(example_cfg_path)


def _generate_cmds(
    cmd_header: list,
    params_dict: dict,
    max_core_num: int = 0,
    start_idx: int = 0,
    name_prefix: str = "",
) -> list:
    """
    Generate command lists with different parameter combinations.

    Args:
        cmd_header (list): Base command elements (executable and common parameters)
        params_dict (dict): Dictionary containing parameter names and their possible values
        max_core_num (int): Maximum allowed core number (0 means no limit)

    Returns:
        list: List of command lists with different parameter combinations
    """
    cmd_list = [] if start_idx == 0 else [cmd_header]

    idx = start_idx
    flag = True
    while flag:
        flag = False
        tmp_cmd_list = []
        for param_name, param_list in params_dict.items():
            if (
                param_name in ["defines", "envs"]
                or len(param_list) <= idx
                or param_list[idx] == "default"
                or param_list[idx] is None
            ):
                continue
            if (
                max_core_num > 0
                and param_name == "ncore"
                and param_list[idx] != "default"
                and param_list[idx] is not None
                and int(param_list[idx]) > max_core_num
            ):
                continue
            if param_name == "name":
                name_val = (
                    (name_prefix + param_list[idx]) if name_prefix else param_list[idx]
                )
                tmp_cmd_list += [name_val]
            elif param_name.startswith("#"):
                tmp_cmd_list += [param_list[idx]]
            else:
                params_str = "--" + param_name
                if param_name.startswith("-"):
                    params_str = param_name
                if isinstance(param_list[idx], bool):
                    if param_list[idx] is True:
                        tmp_cmd_list += [params_str]
                else:
                    tmp_cmd_list += [params_str, param_list[idx]]
            flag = True
        if tmp_cmd_list or flag is True:
            tmp_cmd_list = cmd_header + tmp_cmd_list
            cmd_list.append(tmp_cmd_list)
        idx += 1

    return cmd_list


def _compile_cpp_exec(example_dir: str, log_file: str, defines: list) -> None:
    """
    Compile C++ executable for the example.

    Args:
        example_dir (str): Directory containing the example source code
        log_file (str): Path to the log file for compilation output
        defines (list): List of CMake definitions to pass during compilation
    """
    os.makedirs("./build", exist_ok=True)
    os.chdir(example_dir + "/build")

    cmake_prefix = "-DCMAKE_INSTALL_PREFIX=" + example_dir
    cmake_build_type = "-DCMAKE_BUILD_TYPE=Release"
    cmake_cmd_list = (
        ["cmake", "build", cmake_prefix, cmake_build_type] + defines + [".."]
    )
    execute_test_cmd(cmake_cmd_list, log_file, True)
    execute_test_cmd(["make", "-j"], log_file, True)
    execute_test_cmd(["make", "install"], log_file, True)

    os.chdir(example_dir)


def _test_get_model(
    example_info: dict, platform: str, log_file: str, model_set_dir: str
) -> bool:
    """
    Download model for the example with proper resource locking.

    Args:
        example_info (dict): Dictionary containing example configuration information
        platform (str): Target platform for the test
        log_file (str): Path to the log file for download output

    Returns:
        bool: True if model download was successful, False otherwise
    """
    get_model_flag = True
    max_core_num = 2 if platform == "aarch64" else 0
    params_dict = example_info["get_model_params"][HOUMO_BACKEND]
    cmd_header = ["python3", "get_model.py", "--model_dir", model_set_dir]
    cmd_list = _generate_cmds(cmd_header, params_dict, max_core_num, start_idx=1)
    logger.info(f"Get model cmds: {cmd_list}")

    lock_file = model_set_dir + "/lock.lock"
    with ModelResourceLock(
        lock_file, ModelResourceLock.LockMode.WRITE, "model downloading"
    ):
        for tmp_cmd_list in cmd_list:
            exec_flag, _ = execute_test_cmd(tmp_cmd_list)
            if exec_flag is False:
                get_model_flag = False
                logger.error(f"Get Model Test Failed, test cmd: {tmp_cmd_list}.")

    return get_model_flag


def execute_apis_examples(example_name: str, setup_logging):
    """
    Execute API examples for the specified example name.

    Args:
        example_name (str): Name of the example to execute
        setup_logging: pytest fixture for setting up logging configuration

    Raises:
        AssertionError: If the example folder doesn't exist or if tests fail
    """
    log_file, pytest_request = setup_logging
    marker_vals = check_device_markers(pytest_request)
    dev_res_dir = f"{marker_vals[NDEVICE_MARKER]}_{marker_vals[DEVICE_MEM_MARKER]}"
    logger.info(f"log_file: {log_file}, dev_res_dir: {dev_res_dir}")

    example_info = _load_example_cfg(example_name)
    if (
        example_info is None
        or example_info["obsolete"] is True
        or HOUMO_BACKEND not in example_info["support_backend"]
        or example_info["support_backend"][HOUMO_BACKEND] is None
    ):
        logger.warning("Not support %s testing.", example_name)
        pytest.skip("This testcase is not support.")
    if get_test_type() == TCaseType.SEPARATE_NO_INFER:
        skip_msg = f"Skip apis testcase {example_name} in the SEPARATE NO INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    platform = get_platform(example_info["support_platform"])
    if platform is None:
        logger.warning(f"Not support {example_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")
    if (
        HDPL_PLATFORM == "ASIC"
        and platform == "aarch64"
        and not check_device_info(
            example_info["support_core_num"].get(HOUMO_BACKEND, None)
        )
    ):
        logger.warning(f"Not support {example_name} testing on 2cores device.")
        pytest.skip("This testcase is not support on 2cores device.")

    if (
        example_info.get("dependency", None) is not None
        and "vpu" in example_info["dependency"]
        and (HDPL_PLATFORM == "ISIM" or check_vpu_status() is False)
    ):
        logger.warning(f"{example_name} testcase needs vpu driver.")
        pytest.skip("This testcase needs vpu driver.")

    example_dir = script_dir + "/../../" + example_info["example_dir"]
    if not os.path.exists(example_dir):
        assert False, f"The {example_name} example folder doesn't exist."
    prepare_test_folder(example_dir, "apis")
    current_folder = os.getcwd()
    logger.info("current folder: %s.", current_folder)

    run_sh_flag = True
    if HDPL_PLATFORM == "ASIC" and os.path.exists(f"{current_folder}/run.sh"):
        logger.info("Ready to execute run.sh in folder: %s.", current_folder)

        run_sh_flag, _ = execute_test_cmd(["bash", "run.sh"], log_file, False, True)

        logger.warning(f"remove folder: {current_folder}.")
        shutil.rmtree(current_folder)
        assert run_sh_flag is True, "Execute run.sh Failed!"

        prepare_test_folder(example_dir, "apis")
        current_folder = os.getcwd()
        logger.info(
            f"run.sh ret is {run_sh_flag}, change pytest folder, current folder: {current_folder}."
        )

    model_set_dir = os.path.join(MODELS_PATH, example_info["example_dir"])
    if (
        example_info.get("get_model_params", None) is None
        or example_info["get_model_params"].get(HOUMO_BACKEND, None) is None
    ):
        cmd_list = ["python3", "get_model.py"]
        if example_name not in ["qwen3", "qwen3_multibatch", "qwen3_speculative"]:
            cmd_list += ["--model_dir", model_set_dir]
        lock_file = model_set_dir + "/lock.lock"
        with ModelResourceLock(
            lock_file, ModelResourceLock.LockMode.WRITE, "model downloading"
        ):
            execute_test_cmd(cmd_list, "", True)
    else:
        get_model_flag = _test_get_model(
            example_info, platform, log_file, model_set_dir=model_set_dir
        )
        if get_model_flag is False:
            logger.error(f"{example_name} Get model Failed!")

    demo_types = example_info["support_backend"][HOUMO_BACKEND]
    # run python demo
    py_flag = True
    if "python" in demo_types:
        # install python requirements
        venv_flag = install_py_venv(current_folder, log_file)
        python_exe = "python3"
        if venv_flag:
            python_exe = f"{VENV_NAME}/bin/python3"

        params_dict = example_info["py_example_params"]
        cmd_header = [python_exe]
        cmd_list = _generate_cmds(cmd_header, params_dict)
        logger.info(f"python exe cmd_list: {cmd_list}")
        for tmp_cmd_list in cmd_list:
            exec_flag, _ = execute_test_cmd(
                tmp_cmd_list, log_file, pyvenv_flag=venv_flag
            )
            py_flag = False if exec_flag is False else py_flag
        if py_flag is False:
            logger.error("Python example execution failed!")

    # run c++ demo
    cpp_flag = True
    if "cpp" in demo_types:
        params_dict = example_info["cpp_example_params"]
        compile_defines = params_dict.get("defines", [])
        cmd_header = []
        cmd_list = _generate_cmds(cmd_header, params_dict, name_prefix="./")
        logger.info(f"cpp exe cmd_list: {cmd_list}")
        for idx, tmp_cmd_list in enumerate(cmd_list):
            defines = compile_defines[idx] if len(compile_defines) > 0 else []
            _compile_cpp_exec(current_folder, log_file, defines)
            exec_flag, _ = execute_test_cmd(tmp_cmd_list, log_file)
            cpp_flag = False if exec_flag is False else cpp_flag
        if cpp_flag is False:
            logger.error("C++ example execution failed!")

    shutil.rmtree(os.getcwd())
    assert py_flag is True and cpp_flag is True, "Apis Example Test Failed!"
    logger.info("Apis Example Test Success!")
