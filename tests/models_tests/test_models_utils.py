# Copyright 2025 HOUMO AI
#
# File: test_models_utils.py
# Description:
#   Model testing utilities module for comprehensive model validation.
#   This module provides utility functions for executing various test flows for different models.
#   It handles model configuration loading, command generation, model preparation, and execution
#   of different test types including get_model, quant, compile, demo, compare, eval, and perf tests.
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
import glob as glob_module
import re
from ..tests_utils.tests_common_utils import *
from ..tests_utils.tests_pyvenv_utils import install_py_venv, VENV_NAME


logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))


def _load_model_cfg(model_name: str) -> dict:
    """
    Load model configuration from JSON file.

    Args:
        model_name (str): Name of the model whose configuration needs to be loaded

    Returns:
        dict: Configuration dictionary loaded from the JSON file, or None if file doesn't exist
    """
    model_cfg_path = script_dir + "/model_configs/model_cfg_" + model_name + ".json"
    return load_json(model_cfg_path)


def _generate_hmatc_cmds(
    cmd_header: list,
    required_params: dict,
    optional_params: dict,
    skipped_vals: dict = None,
) -> list:
    """
    Generate HMATC command lists with different parameter combinations.

    Args:
        cmd_header (list): Base command elements (executable and common parameters)
        required_params (dict): Dictionary containing required parameter names and their values
        optional_params (dict): Dictionary containing optional parameter names and their values
        skipped_vals (dict): Dictionary of parameter values to skip during generation

    Returns:
        list: List of command lists with different parameter combinations
    """
    merged_params = required_params.copy()
    merged_params.update(optional_params)

    cmd_list = []
    # construct required test commands
    idx = 0
    flag = True
    while flag:
        flag = False
        tmp_cmd_list = []
        for param_name, param_list in merged_params.items():
            if (
                param_name == "onnx"
                or len(param_list) <= idx
                or param_list[idx] is None
                or param_list[idx] == "default"
            ):
                continue
            param_val = param_list[idx]
            if (
                skipped_vals
                and param_name in skipped_vals
                and param_val in skipped_vals[param_name]
            ):
                continue
            param_str = "--" + param_name
            tmp_cmd_list += [param_str, param_val]
            flag = True

        if tmp_cmd_list:
            tmp_cmd_list = cmd_header + tmp_cmd_list
            cmd_list.append(tmp_cmd_list)
        idx += 1

    return cmd_list


def _generate_py_cmds(
    cmd_header: list,
    params_dict: dict,
    skip_default: bool = True,
    model_dir: str = None,
    res_dir: str = None,
) -> list:
    """
    Generate Python command lists with different parameter combinations.

    Args:
        cmd_header (list): Base command elements (executable and common parameters)
        params_dict (dict): Dictionary containing parameter names and their possible values
        skip_default (bool): Whether to skip the first parameter combination (default)
        model_dir (str): Directory containing model files
        res_dir (str): Directory containing results

    Returns:
        list: List of command lists with different parameter combinations
    """

    cmd_list = [cmd_header] if skip_default else []
    idx = 1 if skip_default else 0
    flag = True
    while flag:
        flag = False
        tmp_cmd_list = []
        for param_name, param_list in params_dict.items():
            params_str = "--" + param_name
            if (
                len(param_list) <= idx
                or param_list[idx] is None
                or param_list[idx] == "default"
            ):
                continue
            param_val = param_list[idx]
            if model_dir and "cached_models" in param_list[idx]:
                param_val = param_list[idx].replace("cached_models", model_dir)
            if res_dir and "cached_results" in param_list[idx]:
                param_val = param_list[idx].replace("cached_results", res_dir)
            tmp_cmd_list += [params_str, param_val]
            flag = True
        if tmp_cmd_list:
            tmp_cmd_list = cmd_header + tmp_cmd_list
            cmd_list.append(tmp_cmd_list)
        idx += 1

    return cmd_list


def _check_compile_result(res_str: str, benchmark_val: float) -> bool:
    """
    Check compilation results against benchmark values.

    Args:
        res_str (str): Output string from compilation test
        benchmark_val (float): Benchmark value to compare against

    Returns:
        bool: True if all results meet the threshold, False otherwise
    """

    if HOUMO_BACKEND == "xh2":
        row_pattern = re.compile(r"\|\s*([\w\-/.]+)\s*\|\s*(\d+\.\d+)\s*\|$")
    else:
        row_pattern = re.compile(
            r"^\|\s*([^|]+?)\s*\|\s*(\d+\.\d+)\s*\|\s*(\w+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\w+)\s*\|$"
        )
    rows = []
    header = None
    for line in res_str.split("\n"):
        line = line.strip()
        if "cosine_dist" in line:
            logger.info(f"detect compile result headers: {line}")
            header = [col.strip() for col in line.split("|") if col and col.strip()]
        elif row_pattern.match(line):
            logger.info(f"detect compile result values: {line}")
            parts = row_pattern.match(line).groups()
            res_dict = dict()
            for idx, val in enumerate(parts):
                try:
                    final_val = float(val)
                except Exception:
                    final_val = str(val)
                res_dict[header[idx]] = final_val
            rows.append(res_dict)
            # rows.append({header[0]: str(parts[0]), header[1]: float(parts[1])})
    if not header or not rows:
        logger.error("Failed to detect the table of compilation results.")
        return False

    logger.info(f"Compilation results: {rows}")
    compile_th = 0.99
    if HOUMO_BACKEND == "xh2":
        compile_th = 0.9
    if benchmark_val > 0:
        compile_th = benchmark_val
    check_res = all(row[header[1]] >= compile_th for row in rows)
    if check_res is True and HOUMO_BACKEND == "xh1":
        check_res = all(row[header[3]] >= compile_th for row in rows)
    return check_res


def _check_compare_result(res_str: str) -> bool:
    """
    Check comparison results between ONNX and HM models.

    Args:
        res_str (str): Output string from comparison test

    Returns:
        bool: True if all results meet the threshold, False otherwise
    """
    if HOUMO_BACKEND == "xh2":
        row_pattern = re.compile(
            r"\|\s*([^|]+?)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|$"
        )
    else:
        row_pattern = re.compile(
            r"\|\s*([^|]+?)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\w+)\s*\|$"
        )
    rows = []
    header = None
    for line in res_str.split("\n"):
        line = line.strip()
        if "onnx vs hmquant" in line:
            logger.info(f"detect compare result headers: {line}")
            header = [col.strip() for col in line.split("|") if col and col.strip()]
        elif row_pattern.match(line):
            logger.info(f"detect compare result values: {line}")
            parts = row_pattern.match(line).groups()
            rows.append(
                {
                    header[0]: str(parts[0]),
                    header[1]: float(parts[1]),
                    header[2]: float(parts[2]),
                    header[3]: float(parts[3]),
                }
            )
    if not header or not rows:
        logger.error("Failed to detect the table of comparation results.")
        return False

    logger.info(f"Comparation results: {rows}")
    compare_th = 1.0
    if HOUMO_BACKEND == "xh2":
        compare_th = 0.9
    check_res = all(row[header[3]] >= compare_th for row in rows)
    return check_res


def _process_eval_result(res_str: str, perf_names: list) -> dict:
    """
    Process evaluation results and extract performance metrics.

    Args:
        res_str (str): Output string from evaluation test
        perf_names (list): List of performance metric names to extract

    Returns:
        dict: Dictionary containing extracted performance metrics
    """

    def extract_field(text, field_name):
        pattern = rf"{field_name}':\s*'?([^'\]]+)'?"
        match = re.search(pattern, text)
        return match.group(1) if match else None

    eval_res = dict()
    for line in res_str.split("\n"):
        line = line.strip()
        if any(perf_name in line for perf_name in perf_names):
            for perf_name in perf_names:
                eval_res[perf_name] = float(extract_field(line, perf_name))

    if not eval_res:
        logger.error("Failed to detect the evaluation result.")

    return eval_res


def _get_param_value(params, target_param):
    """
    Extract the value of a parameter from a command list.

    Args:
        params (list): List of parameters in the format [param, value, param, value, ...]
        target_param (str): Name of the parameter to find

    Returns:
        str or None: Value of the target parameter, or None if not found
    """
    for i in range(len(params)):
        if params[i] == target_param:
            if i + 1 < len(params):
                return params[i + 1]
            break
    return None


def _download_models(
    model_info: dict,
    file_type: str,
    download_dir: str,
    extract_dir: str,
    lock_type: str = "download",
    copy_flag: bool = False,
    assert_flag=True,
    other_params: list = [],
) -> bool:
    """
    Download models of specified type with proper resource locking.

    Args:
        model_info (dict): Dictionary containing model configuration information
        file_type (str): Type of model files to download (raw, quant, hmm)
        download_dir (str): Directory to download model files to
        extract_dir (str): Directory to extract model files to
        lock_type (str): Type of locking to use (download, extract, all)
        copy_flag (bool): Whether to copy downloaded files to current directory
        assert_flag (bool): Whether to assert on download failure
        other_params (list): Additional parameters to pass to the download command

    Returns:
        bool: True if download was successful, False otherwise
    """
    download_str = "--model_dir"
    extract_str = "--quant_model_dir"
    if "download_dir" in model_info["get_model_params"][HOUMO_BACKEND]:
        download_str = "--download_dir"
        extract_str = "--extract_dir"

    cmd_list = ["python3", "get_model.py", "--type", file_type]
    if file_type == "raw":
        cmd_list += [download_str, download_dir]
    else:
        cmd_list += [download_str, download_dir, extract_str, extract_dir]
    if len(other_params) > 0:
        cmd_list += other_params

    logger.info(f"Ready to download models using {cmd_list}")
    lock_file_src = download_dir + "/lock.lock"
    lock_file_dst = extract_dir + "/lock.lock"
    if lock_type == "all":
        with ModelResourceLock(
            lock_file_src, ModelResourceLock.LockMode.WRITE, "model downloading"
        ):
            with ModelResourceLock(
                lock_file_dst, ModelResourceLock.LockMode.WRITE, "model downloading"
            ):
                flag, _ = execute_test_cmd(cmd_list, "", assert_flag)
                if copy_flag:
                    os.system(f"cp -ar {download_dir}/* ./")
    else:
        lock_file = lock_file_src if lock_type == "download" else lock_file_dst
        with ModelResourceLock(
            lock_file, ModelResourceLock.LockMode.WRITE, "model downloading"
        ):
            flag, _ = execute_test_cmd(cmd_list, "", assert_flag)
            if copy_flag:
                os.system(f"cp -ar {download_dir}/* ./")

    return flag


def _check_existed_models(
    compile_case_id, get_model_params, model_res_dir, model_set_dir
):
    other_params = []
    if not compile_case_id:
        return other_params

    model_exist = -1
    if "extract_dir" in get_model_params:
        for idx, tmp_dir in enumerate(get_model_params["extract_dir"]):
            if (
                tmp_dir is not None
                and isinstance(tmp_dir, str)
                and compile_case_id == tmp_dir.rsplit("/", 1)[-1]
            ):
                model_exist = idx
                break
    if model_exist > -1:
        for param_key in get_model_params:
            if param_key in [
                "type",
                "download_dir",
                "extract_dir",
                "build_model_dir",
                "model_dir",
                "quant_model_dir",
            ]:
                continue
            param_val = get_model_params[param_key][model_exist]
            if param_val is None:
                continue
            tmp_str = f"--{param_key}"
            if isinstance(param_val, str) and "cached_results" in param_val:
                param_val = param_val.replace("cached_results", model_res_dir)
            elif isinstance(param_val, str) and "cached_models" in param_val:
                param_val = param_val.replace("cached_models", model_set_dir)
            other_params += [tmp_str, param_val]

    return other_params


def _prepare_quantized_llm_model(
    model_info: dict, log_file: str, model_res_dir: str, model_set_dir: str
) -> bool:
    """
    Prepare quantized LLM model for compilation.

    Args:
        model_info (dict): Dictionary containing model configuration information
        log_file (str): Path to the log file for test output

    Returns:
        bool: True if preparation was successful, False otherwise
    """
    if get_test_type() == TCaseType.SEPARATE_INFER:
        logger.warning(
            "Skip the step of preparing quantized llm model in the SPEARATE INFER stage."
        )
        return True

    lock_file_dst = model_res_dir + "/lock.lock"
    quant_params_full = model_info.get("quant_params", None)
    quant_params = (
        quant_params_full.get(HOUMO_BACKEND, None) if quant_params_full else None
    )
    flag = False

    logger.info("Start to quant llm model for compiling.")
    if (
        quant_params
        and "quant" in model_info["support_flow"][HOUMO_BACKEND]
        and check_gpu()["has_gpu"] is True
    ):
        current_folder = os.getcwd()
        for idx, tmp_model_dir in enumerate(quant_params["out-dir"]):
            quant_res_dir = tmp_model_dir.replace("cached_results", model_res_dir)
            if quant_res_dir and os.path.exists(quant_res_dir):
                logger.warning(
                    f"Skip the step of preparing quantized llm model {quant_res_dir} in the SPEARATE NO INFER stage."
                )
                flag = True
                continue
            # download raw model files
            _download_models(
                model_info,
                file_type="raw",
                download_dir=model_set_dir,
                extract_dir=model_res_dir,
            )

            # install python requirements
            venv_flag = install_py_venv(current_folder, log_file, "quant")
            python_exe = "python3"
            if venv_flag:
                python_exe = f"{VENV_NAME}/bin/python3"

            cmd_list = [python_exe, "ptq.py"]
            for param_key in quant_params:
                param_val = quant_params[param_key][idx]
                if param_val is None:
                    continue
                tmp_str = f"--{param_key}"
                if isinstance(param_val, str) and "cached_results" in param_val:
                    param_val = param_val.replace("cached_results", model_res_dir)
                elif isinstance(param_val, str) and "cached_models" in param_val:
                    param_val = param_val.replace("cached_models", model_set_dir)
                cmd_list += [tmp_str, param_val]
            with ModelResourceLock(
                lock_file_dst, ModelResourceLock.LockMode.WRITE, "model quantizing"
            ):
                flag, _ = execute_test_cmd(cmd_list, log_file, pyvenv_flag=venv_flag)
                if flag is True:
                    tmp_res_dir = f"{quant_res_dir}/hmquant"
                    os.system(f"mv {tmp_res_dir} {quant_res_dir}")
                else:
                    return flag

        return flag

    if "quant" not in model_info["get_model_params"][HOUMO_BACKEND]["type"]:
        logger.warning("Not support downloading quantized model file.")
        return False

    compile_params = model_info["compile_params"][HOUMO_BACKEND]
    compile_model_dir = set(compile_params["model_dir"])
    for tmp_model_dir in compile_model_dir:
        quant_res_dir = tmp_model_dir
        if isinstance(tmp_model_dir, str) and "cached_results" in tmp_model_dir:
            quant_res_dir = tmp_model_dir.replace("cached_results", model_res_dir)
        elif isinstance(tmp_model_dir, str) and "cached_models" in tmp_model_dir:
            quant_res_dir = tmp_model_dir.replace("cached_models", model_set_dir)
        if quant_res_dir and os.path.exists(quant_res_dir):
            logger.warning(
                f"Skip the step of preparing quantized llm model {quant_res_dir} in the SPEARATE NO INFER stage."
            )
            continue

        logger.info("Start to download quantized llm model for compiling.")
        flag = _download_models(
            model_info,
            file_type="quant",
            download_dir=model_set_dir,
            extract_dir=quant_res_dir,
            lock_type="all",
            copy_flag=False,
            assert_flag=False,
        )
        if flag is False:
            return flag

    return flag


def _prepare_quantized_cv_model(
    model_info: dict, log_file: str, model_res_dir: str, model_set_dir: str
) -> bool:
    """
    Prepare quantized computer vision model for compilation.

    Args:
        model_info (dict): Dictionary containing model configuration information
        log_file (str): Path to the log file for test output

    Returns:
        bool: True if preparation was successful, False otherwise
    """
    logger.info("Start to prepare quantized cv model for compiling.")
    flag = True
    get_model_types = model_info["get_model_params"][HOUMO_BACKEND]["type"]

    # get model
    lock_file = model_res_dir + "/lock.lock"
    if "quant" in get_model_types and "hmquant_params" not in model_info:
        compiled_ipt_dirs = model_info["compile_params"][HOUMO_BACKEND]["model_dir"]
        for idx, tmp_model_dir in enumerate(compiled_ipt_dirs):
            quant_res_dir = tmp_model_dir
            if "cached_results" in tmp_model_dir:
                quant_res_dir = tmp_model_dir.replace("cached_results", model_res_dir)
            elif "cached_models" in tmp_model_dir:
                quant_res_dir = tmp_model_dir.replace("cached_models", model_set_dir)
            if os.path.exists(quant_res_dir):
                continue

            flag = _download_models(
                model_info,
                file_type="quant",
                download_dir=model_set_dir,
                extract_dir=quant_res_dir,
                lock_type="extract",
                copy_flag=False,
                assert_flag=False,
            )
            if flag is False:
                break
        return flag

    if (
        "raw" in get_model_types
        and "quant" in model_info["support_flow"][HOUMO_BACKEND]
    ):
        flag = _download_models(
            model_info,
            file_type="raw",
            download_dir=model_set_dir,
            extract_dir=model_res_dir,
            assert_flag=False,
        )
        if flag is False:
            return False

    # copy model files to work dir
    os.system(f"cp -ar {model_set_dir}/* ./")
    # quant raw model
    if "hmquant_params" in model_info:
        flag, _ = execute_test_cmd(
            [
                "hmatc",
                "quant",
                "--target",
                HOUMO_BACKEND,
                "--config",
                "./config.yml",
            ],
            log_file,
        )
    elif "quant_params" in model_info:
        flag, _ = execute_test_cmd(["python3", "ptq.py"], log_file, True)

    return flag


def _prepare_compiled_llm_model(
    model_info: dict,
    platform: str,
    log_file: str,
    model_res_dir: str,
    model_set_dir: str,
) -> bool:
    """
    Prepare compiled LLM model for inference.

    Args:
        model_info (dict): Dictionary containing model configuration information
        platform (str): Current platform
        log_file (str): Path to the log file for test output

    Returns:
        bool: True if preparation was successful, False otherwise
    """
    if get_test_type() == TCaseType.SEPARATE_INFER:
        logger.warning(
            "Skip the step of preparing compiled model in the SPEARATE INFER stage."
        )
        return True

    compile_params = model_info["compile_params"][HOUMO_BACKEND]
    flag = True
    for idx, tmp_model_dir in enumerate(compile_params["model_dir"]):
        quant_res_dir = tmp_model_dir.replace("cached_results", model_res_dir)
        compile_res_dir = compile_params["output_dir"][idx].replace(
            "cached_results", model_res_dir
        )
        embedding_path = f"{compile_res_dir}/hmquant/quant_embedding.pt"
        if not os.path.exists(embedding_path) and os.path.exists(
            f"{quant_res_dir}/quant_embedding.pt"
        ):
            os.makedirs(f"{compile_res_dir}/hmquant/", exist_ok=True)
            os.system(f"cp -a {quant_res_dir}/quant_*.pt {compile_res_dir}/hmquant/")
        if get_test_type() in [TCaseType.SEPARATE_NO_INFER, TCaseType.DEFAULT]:
            hmm_files = glob_module.glob(os.path.join(compile_res_dir, "*.hmm"))
            if os.path.exists(compile_res_dir) and len(hmm_files) > 0:
                logger.warning(
                    f"Skip the step of preparing compiled model {compile_res_dir}."
                )
                continue

        logger.info("Start to prepare compiled llm model for inference.")
        lock_file = model_res_dir + "/lock.lock"
        if (
            platform != "aarch64"
            and is_release() is False
            and "compile" in model_info["support_flow"][HOUMO_BACKEND]
            and check_gpu()["has_gpu"] is True
            and _prepare_quantized_llm_model(
                model_info,
                log_file,
                model_res_dir=model_res_dir,
                model_set_dir=model_set_dir,
            )
        ):
            cmd_list = ["python3", "build.py"]
            for param_key in compile_params:
                param_val = compile_params[param_key][idx]
                if param_val is None:
                    continue
                tmp_str = f"--{param_key}"
                if isinstance(param_val, str) and "cached_results" in param_val:
                    param_val = param_val.replace("cached_results", model_res_dir)
                elif isinstance(param_val, str) and "cached_models" in param_val:
                    param_val = param_val.replace("cached_models", model_set_dir)
                cmd_list += [tmp_str, param_val]
            with ModelResourceLock(
                lock_file, ModelResourceLock.LockMode.WRITE, "model compiling"
            ):
                flag, _ = execute_test_cmd(cmd_list, log_file)
                if flag is False:
                    break

            if os.path.exists(f"{quant_res_dir}/quant_embedding.pt"):
                os.makedirs(f"{compile_res_dir}/hmquant/", exist_ok=True)
                os.system(
                    f"cp -a {quant_res_dir}/quant_*.pt {compile_res_dir}/hmquant/"
                )

        flag = False
        get_model_params = model_info["get_model_params"][HOUMO_BACKEND]
        if "hmm" in get_model_params["type"]:
            compile_case_id = compile_res_dir.rsplit("/", 1)[-1]
            other_params = _check_existed_models(
                compile_case_id, get_model_params, model_res_dir, model_set_dir
            )

            flag = _download_models(
                model_info,
                file_type="hmm",
                download_dir=model_set_dir,
                extract_dir=compile_res_dir,
                lock_type="all",
                copy_flag=False,
                assert_flag=False,
                other_params=other_params,
            )
            if flag is False:
                break

    return flag


def _prepare_compiled_cv_model(
    model_info: dict,
    platform: str,
    log_file: str,
    model_res_dir: str,
    model_set_dir: str,
) -> bool:
    """
    Prepare compiled computer vision model for inference.

    Args:
        model_info (dict): Dictionary containing model configuration information
        platform (str): Current platform
        log_file (str): Path to the log file for test output

    Returns:
        bool: True if preparation was successful, False otherwise
    """
    if get_test_type() == TCaseType.SEPARATE_INFER:
        logger.warning(
            "Skip the step of preparing compiled model in the SPEARATE INFER stage."
        )
        return True

    if "compile_params" in model_info:
        compile_res_dir = model_info["compile_params"][HOUMO_BACKEND]["output_dir"][0]
        compile_res_dir = compile_res_dir.replace("cached_results", model_res_dir)
    else:
        compile_res_dir = os.path.join(model_res_dir, "output", HOUMO_BACKEND)
    # if get_test_type() == TCaseType.SEPARATE_NO_INFER:
    if os.path.exists(compile_res_dir):
        logger.warning("Skip the step of preparing compiled model.")
        return True

    logger.info("Start to prepare compiled cv model for inference.")
    if platform == "aarch64":
        _download_models(
            model_info,
            file_type="hmm",
            download_dir=model_set_dir,
            extract_dir=compile_res_dir,
            lock_type="all",
        )
        os.system(f"cp -ar {compile_res_dir} ./")
        return True
    # platform != "aarch64"
    if not _prepare_quantized_cv_model(
        model_info,
        log_file,
        model_res_dir=model_res_dir,
        model_set_dir=model_set_dir,
    ):
        return False
    if "hmbuild_params" in model_info:
        execute_test_cmd(
            [
                "hmatc",
                "build",
                "--target",
                HOUMO_BACKEND,
                "--config",
                "./config.yml",
            ],
            log_file,
            True,
        )
    else:
        model_dir = model_info["compile_params"][HOUMO_BACKEND]["model_dir"][0].replace(
            "cached_results", model_res_dir
        )
        execute_test_cmd(
            [
                "python3",
                "build.py",
                "--model_dir",
                model_dir,
                "--output_dir",
                compile_res_dir,
            ],
            log_file,
            True,
        )

    return True


def _run_demo_script(
    demo_name: str,
    model_name: str,
    model_info: dict,
    model_set_dir: str,
    model_res_dir: str,
    python_exe: str,
    log_file: str,
) -> bool:
    """
    Execute demo script with different parameter combinations.

    Args:
        demo_name (str): Name of the demo script to run
        model_name (str): Name of the model being tested
        model_info (dict): Dictionary containing model configuration information
        model_set_dir (str): Directory containing model files
        model_res_dir (str): Directory containing model results
        log_file (str): Path to the log file for test output

    Returns:
        bool: True if all demo executions were successful, False otherwise
    """
    run_flag = True
    param_str = f"{demo_name}_params"
    if (
        demo_name in model_info["support_flow"][HOUMO_BACKEND]
        and param_str in model_info
    ):
        venv_flag = True if VENV_NAME in python_exe else False
        params_dict = model_info[param_str][HOUMO_BACKEND]
        cmd_header = [python_exe, f"{demo_name}.py"]
        cmd_list = _generate_py_cmds(
            cmd_header,
            params_dict,
            skip_default=False,
            model_dir=model_set_dir,
            res_dir=model_res_dir,
        )

        logger.info(f"Ready to execute {demo_name}.py, cmd list: {cmd_list}")
        lock_file_res = model_res_dir + "/lock.lock"
        with ModelResourceLock(
            lock_file_res,
            ModelResourceLock.LockMode.WRITE,
            f"execute model {demo_name}.py",
        ):
            check_flag = False if model_name == "qwen2.5-vl" else True
            for tmp_cmd_list in cmd_list:
                # [Debug]
                # if demo_name == "demo_multibatch":
                #     logger.warning(
                #         f"Skip to execute demo_multibatch.py, cmd is {tmp_cmd_list}"
                #     )
                #     continue
                exec_flag, _ = execute_test_cmd(
                    tmp_cmd_list, log_file, False, check_flag, pyvenv_flag=venv_flag
                )
                if exec_flag is False:
                    logger.error(
                        f"Failed to execute {demo_name}.py, cmd is {tmp_cmd_list}"
                    )
                    run_flag = False
    else:
        logger.warning(f"{demo_name} is not supported.")

    return run_flag


def execute_get_model_flow(model_name: str, setup_logging) -> None:
    """
    Execute the complete get model test flow for a specified model.

    Args:
        model_name (str): Name of the model to test
        setup_logging: Fixture of setup_logging
    """
    log_file, pytest_request = setup_logging
    marker_vals = check_device_markers(pytest_request)
    dev_res_dir = f"{marker_vals[NDEVICE_MARKER]}_{marker_vals[DEVICE_MEM_MARKER]}"
    logger.info(f"log_file: {log_file}, dev_res_dir: {dev_res_dir}")

    model_info = _load_model_cfg(model_name)
    if (
        model_info is None
        or model_info["obsolete"] is True
        or HOUMO_BACKEND not in model_info["support_backend"]
        or "get_model" not in model_info["support_flow"][HOUMO_BACKEND]
        or "get_model_params" not in model_info
    ):
        logger.warning("Not support %s testing.", model_name)
        pytest.skip("This testcase is not support.")
    if get_test_type() == TCaseType.SEPARATE_INFER:
        skip_msg = f"This get_model testcase of {model_name} has already been run in the SEPARATE INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    platform = get_platform(model_info["support_platform"])
    if platform is None:
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    prepare_test_folder(model_dir, "get_model")
    logger.info("current folder: %s.", os.getcwd())

    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
    model_res_dir = os.path.join(MODELS_RES_DIR, dev_res_dir, model_info["model_dir"])

    # test script: get_model.py
    params_dict = model_info["get_model_params"][HOUMO_BACKEND]
    if model_info.get("model_type", "cv") == "llm" and is_release() is True:
        # skip download raw models
        raw_indices = list()
        for i in range(len(params_dict["type"]) - 1, -1, -1):
            if params_dict["type"][i] == "raw":
                raw_indices.append(i)
        for idx in raw_indices:
            for param in params_dict:
                params_dict[param].pop(idx)
    cmd_header = ["python3", "get_model.py"]

    final_flag = True
    cmd_list = _generate_py_cmds(
        cmd_header,
        params_dict,
        skip_default=False,
        model_dir=model_set_dir,
    )
    lock_file = model_set_dir + "/lock.lock"
    with ModelResourceLock(
        lock_file, ModelResourceLock.LockMode.WRITE, "model downloading"
    ):
        for tmp_cmd_list in cmd_list:
            if is_release() is True and "modelscope" in tmp_cmd_list:
                continue
            exec_flag, _ = execute_test_cmd(tmp_cmd_list, log_file, False, False)
            final_flag = False if exec_flag is False else final_flag

    logger.warning(f"remove folder: {os.getcwd()}.")
    shutil.rmtree(os.getcwd())
    assert final_flag is True, "Get Model Test Failed!"
    logger.info("Get Model Test Success!")


def execute_quant_flow(model_name: str, setup_logging) -> None:
    """
    Execute the complete quantization test flow for a specified model.

    This function orchestrates the entire quantization test flow, including
    model preparation, quantization execution, and result validation.

    Args:
        model_name (str): Name of the model to test
        setup_logging: Fixture of setup_logging
    """
    log_file, pytest_request = setup_logging
    marker_vals = check_device_markers(pytest_request)
    dev_res_dir = f"{marker_vals[NDEVICE_MARKER]}_{marker_vals[DEVICE_MEM_MARKER]}"
    logger.info(f"log_file: {log_file}, dev_res_dir: {dev_res_dir}")

    model_info = _load_model_cfg(model_name)
    if (
        model_info is None
        or model_info["obsolete"] is True
        or HOUMO_BACKEND not in model_info["support_backend"]
        or "quant" not in model_info["support_flow"][HOUMO_BACKEND]
    ):
        logger.warning("Not support %s testing.", model_name)
        pytest.skip("This testcase is not support.")
    if get_test_type() == TCaseType.SEPARATE_INFER:
        skip_msg = f"This quant testcase of {model_name} has already been run in the SEPARATE INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    model_type = model_info.get("model_type", "cv")
    if model_type == "llm" and (
        (
            get_test_type() != TCaseType.SEPARATE_INFER
            and check_gpu()["has_gpu"] is False
        )
        or (is_release() is True)
    ):
        skip_msg = (
            f"{model_name} testcase requires GPU, release flag: {int(is_release())}."
        )
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    platform = get_platform(model_info["support_platform"])
    if platform is None or platform == "aarch64":
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    prepare_test_folder(model_dir, "quant")
    current_folder = os.getcwd()
    logger.info("current folder: %s.", current_folder)

    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
    model_res_dir = os.path.join(MODELS_RES_DIR, dev_res_dir, model_info["model_dir"])

    # download raw model for quantization
    copy_flag = True if model_type == "cv" else False
    _download_models(
        model_info,
        file_type="raw",
        download_dir=model_set_dir,
        extract_dir=model_res_dir,
        copy_flag=copy_flag,
    )

    logger.info("LD_LIBRARY_PATH: %s", os.getenv("LD_LIBRARY_PATH"))
    final_flag = True
    if "hmquant_params" in model_info:
        # test cmd: hmatc quant
        required_params = model_info["hmquant_params"]["params"]["required"]
        optional_params = model_info["hmquant_params"]["params"]["optional"]
        cmd_header = ["hmatc", "quant", "--target", HOUMO_BACKEND]

        cmd_list = _generate_hmatc_cmds(cmd_header, required_params, optional_params)
        logger.info(f"cmd list: {cmd_list}")
        for tmp_cmd_list in cmd_list:
            exec_flag, _ = execute_test_cmd(tmp_cmd_list, log_file)
            if exec_flag is False:
                final_flag = False
    else:
        # install python requirements
        venv_flag = install_py_venv(current_folder, log_file, "quant")
        python_exe = "python3"
        if venv_flag:
            python_exe = f"{VENV_NAME}/bin/python3"

        # test script: ptq.py
        params_dict = model_info["quant_params"][HOUMO_BACKEND]
        cmd_header = [python_exe, "ptq.py"]

        cmd_list = _generate_py_cmds(
            cmd_header,
            params_dict,
            skip_default=False,
            model_dir=model_set_dir,
            res_dir=model_res_dir,
        )

        lock_file_res = model_res_dir + "/lock.lock"
        with ModelResourceLock(
            lock_file_res, ModelResourceLock.LockMode.WRITE, "model quantizing"
        ):
            for tmp_cmd_list in cmd_list:
                quant_res_dir = _get_param_value(tmp_cmd_list, "--out-dir")
                if quant_res_dir is None:
                    quant_res_dir = _get_param_value(tmp_cmd_list, "--output_dir")
                if os.path.exists(quant_res_dir):
                    shutil.rmtree(quant_res_dir, ignore_errors=True)
                    # [TMP]
                    # continue
                exec_flag, _ = execute_test_cmd(
                    tmp_cmd_list, log_file, pyvenv_flag=venv_flag
                )
                if exec_flag is False:
                    final_flag = False
                else:
                    tmp_res_dir = f"{quant_res_dir}/hmquant"
                    os.system(f"mv {tmp_res_dir}/* {quant_res_dir}/")
                    os.system(f"rm -rf {tmp_res_dir}")

    logger.warning(f"remove folder: {os.getcwd()}.")
    shutil.rmtree(os.getcwd())
    assert final_flag is True, "Quantization Test Failed!"
    logger.info("Quantization Test Success!")


def execute_compile_flow(model_name: str, setup_logging) -> None:
    """
    Execute the complete compilation test flow for a specified model.

    Args:
        model_name (str): Name of the model to test
        setup_logging: Fixture of setup_logging
    """
    log_file, pytest_request = setup_logging
    marker_vals = check_device_markers(pytest_request)
    dev_res_dir = f"{marker_vals[NDEVICE_MARKER]}_{marker_vals[DEVICE_MEM_MARKER]}"
    logger.info(f"log_file: {log_file}, dev_res_dir: {dev_res_dir}")

    model_info = _load_model_cfg(model_name)
    if (
        model_info is None
        or model_info["obsolete"] is True
        or HOUMO_BACKEND not in model_info["support_backend"]
        or "compile" not in model_info["support_flow"][HOUMO_BACKEND]
    ):
        logger.warning("Not support %s testing.", model_name)
        pytest.skip("This testcase is not support.")
    if get_test_type() == TCaseType.SEPARATE_INFER:
        skip_msg = f"This compile testcase of {model_name} has already been run in the SEPARATE NO INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    model_type = model_info.get("model_type", "cv")
    if model_type == "llm" and is_release() is True:
        skip_msg = f"Skip {model_name} testcase, release flag: {int(is_release())}."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    platform = get_platform(model_info["support_platform"])
    if platform is None or platform == "aarch64":
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    prepare_test_folder(model_dir, "compile")
    logger.info("current folder: %s.", os.getcwd())

    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
    model_res_dir = os.path.join(MODELS_RES_DIR, dev_res_dir, model_info["model_dir"])

    # prepare quantized model
    if (
        model_type == "cv"
        and not _prepare_quantized_cv_model(
            model_info,
            log_file,
            model_res_dir=model_res_dir,
            model_set_dir=model_set_dir,
        )
    ) or (
        model_type == "llm"
        and not _prepare_quantized_llm_model(
            model_info,
            log_file,
            model_res_dir=model_res_dir,
            model_set_dir=model_set_dir,
        )
    ):
        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        skip_msg = f"Not support {model_name} testing on {HOUMO_BACKEND}."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)

    final_flag = True
    if "hmbuild_params" in model_info:
        # test cmd: hmatc build
        required_params = model_info["hmbuild_params"][HOUMO_BACKEND]["required"]
        optional_params = model_info["hmbuild_params"][HOUMO_BACKEND]["optional"]
        cmd_header = ["hmatc", "build", "--target", HOUMO_BACKEND]

        skipped_vals = dict()
        if HOUMO_BACKEND == "xh2":
            skipped_vals["ncore"] = ["4"]
        cmd_list = _generate_hmatc_cmds(
            cmd_header, required_params, optional_params, skipped_vals
        )
        logger.info(f"cmd list: {cmd_list}")
        for tmp_cmd_list in cmd_list:
            exec_flag, opt_str = execute_test_cmd(tmp_cmd_list, log_file)
            if exec_flag is False:
                final_flag = False
            else:
                benchmark_val = 0.76 if model_name == "ppocrv3_det" else 0
                final_flag = _check_compile_result(opt_str, benchmark_val)
                if final_flag is False:
                    logger.error(
                        f"Cosine distance exceeds 0.99, compile cmd: {tmp_cmd_list}"
                    )
    else:
        # test script: build.py
        params_dict = model_info["compile_params"][HOUMO_BACKEND]
        cmd_header = ["python3", "build.py"]

        cmd_list = _generate_py_cmds(
            cmd_header,
            params_dict,
            skip_default=False,
            model_dir=model_set_dir,
            res_dir=model_res_dir,
        )

        lock_file_res = model_res_dir + "/lock.lock"
        with ModelResourceLock(
            lock_file_res, ModelResourceLock.LockMode.WRITE, "model compiling"
        ):
            for tmp_cmd_list in cmd_list:
                compile_res_dir = _get_param_value(tmp_cmd_list, "--output_dir")
                if compile_res_dir and os.path.exists(compile_res_dir):
                    shutil.rmtree(compile_res_dir, ignore_errors=True)
                    # [TMP]
                    # continue
                model_dir = _get_param_value(tmp_cmd_list, "--model_dir")
                if not os.path.exists(model_dir):
                    logger.warning(f"Skip compilation test {tmp_cmd_list}")
                    continue
                exec_flag, _ = execute_test_cmd(tmp_cmd_list, log_file)
                if exec_flag is False:
                    final_flag = False
                if compile_res_dir and os.path.exists(
                    f"{model_dir}/quant_embedding.pt"
                ):
                    os.makedirs(f"{compile_res_dir}/hmquant/", exist_ok=True)
                    os.system(
                        f"cp {model_dir}/quant_embedding.pt {compile_res_dir}/hmquant/"
                    )

    logger.warning(f"remove folder: {os.getcwd()}.")
    shutil.rmtree(os.getcwd())
    assert final_flag is True, "Compilation Test Failed!"
    logger.info("Compilation Test Success!")


def execute_demo_flow(model_name: str, setup_logging) -> None:
    """
    Execute the complete demo test flow for a specified model.

    Args:
        model_name (str): Name of the model to test
        setup_logging: Fixture of setup_logging
    """
    log_file, pytest_request = setup_logging
    marker_vals = check_device_markers(pytest_request)
    dev_res_dir = f"{marker_vals[NDEVICE_MARKER]}_{marker_vals[DEVICE_MEM_MARKER]}"
    logger.info(f"log_file: {log_file}, dev_res_dir: {dev_res_dir}")

    model_info = _load_model_cfg(model_name)
    if (
        model_info is None
        or model_info["obsolete"] is True
        or HOUMO_BACKEND not in model_info["support_backend"]
        or "demo" not in model_info["support_flow"][HOUMO_BACKEND]
    ):
        logger.warning("Not support %s testing.", model_name)
        pytest.skip("This testcase is not support.")

    platform = get_platform(model_info["support_platform"])
    if platform is None:
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")
    # aarch64 test need compiled hmm models
    if (
        HDPL_PLATFORM == "ASIC"
        and platform == "aarch64"
        and (
            "demo_params" not in model_info
            or "hmm" not in model_info["get_model_params"][HOUMO_BACKEND]["type"]
            or not check_device_info(
                model_info["support_core_num"].get(HOUMO_BACKEND, None)
            )
        )
    ):
        logger.warning(f"Not support {model_name} testing on 2cores device.")
        pytest.skip("This testcase is not support on 2cores device.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    prepare_test_folder(model_dir, "demo")
    current_folder = os.getcwd()
    logger.info("current folder: %s.", current_folder)

    # execute test.sh
    test_sh_flag = True
    if HDPL_PLATFORM == "ASIC" and os.path.exists(f"{current_folder}/test.sh"):
        logger.info("Ready to execute test.sh in folder: %s.", current_folder)
        try:
            cmd_model_size = model_info["get_model_params"][HOUMO_BACKEND][
                "model_size"
            ][0]
        except Exception:
            cmd_model_size = None
        check_flag = False if model_name == "qwen2.5-vl" else True
        if cmd_model_size is not None and cmd_model_size in ["14b"]:
            test_sh_flag, _ = execute_test_cmd(
                ["bash", "test.sh", "-m", cmd_model_size], log_file, False, check_flag
            )
        else:
            test_sh_flag, _ = execute_test_cmd(
                ["bash", "test.sh"], log_file, False, check_flag
            )
        test_sh_folder = current_folder

        prepare_test_folder(model_dir, "demo")
        current_folder = os.getcwd()
        logger.info(
            "test.sh ret is %d, change test folder, current folder: %s.",
            test_sh_flag,
            current_folder,
        )

        logger.warning(f"remove folder: {test_sh_folder}.")
        shutil.rmtree(test_sh_folder)

    if is_release():
        logger.info("RELEASE MODE, only execute test.sh.")
        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        assert test_sh_flag is True, "Execute tesh.sh Failed!"
        return

    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
    model_res_dir = os.path.join(MODELS_RES_DIR, dev_res_dir, model_info["model_dir"])
    model_type = model_info.get("model_type", "cv")
    if (
        model_type == "cv"
        and not _prepare_compiled_cv_model(
            model_info,
            platform,
            log_file,
            model_res_dir=model_res_dir,
            model_set_dir=model_set_dir,
        )
    ) or (
        model_type == "llm"
        and not _prepare_compiled_llm_model(
            model_info,
            platform,
            log_file,
            model_res_dir=model_res_dir,
            model_set_dir=model_set_dir,
        )
    ):
        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        pytest.skip(f"Not support {model_name} testing on {HOUMO_BACKEND}.")
    if get_test_type() == TCaseType.SEPARATE_NO_INFER:
        if model_type == "cv":
            move_models_res(current_folder, model_res_dir)

        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        logger.warning(
            f"Skip demo testcase {model_name} in the SEPARATE NO INFER stage."
        )
        pytest.skip(f"Skip demo testcase {model_name} in the SEPARATE NO INFER stage.")
    if model_type == "cv" and get_test_type() == TCaseType.SEPARATE_INFER:
        restore_models_res(model_res_dir, current_folder)

    # install python requirements
    venv_flag = install_py_venv(current_folder, log_file)
    python_exe = "python3"
    if venv_flag:
        python_exe = f"{VENV_NAME}/bin/python3"

    final_flag = True
    if "hmdemo_params" in model_info and platform != "aarch64":
        # test hmatc demo
        required_params = model_info["hmdemo_params"]["params"]["required"]
        optional_params = model_info["hmdemo_params"]["params"]["optional"]
        cmd_header = ["hmatc", "demo", "--target", HOUMO_BACKEND]

        cmd_list = _generate_hmatc_cmds(cmd_header, required_params, optional_params)
        logger.info(f"cmd list: {cmd_list}")
        for tmp_cmd_list in cmd_list:
            exec_flag, _ = execute_test_cmd(
                tmp_cmd_list, log_file, pyvenv_flag=venv_flag
            )
            if exec_flag is False:
                final_flag = False
    else:
        demo_flag = _run_demo_script(
            demo_name="demo",
            model_name=model_name,
            model_info=model_info,
            model_set_dir=model_set_dir,
            model_res_dir=model_res_dir,
            python_exe=python_exe,
            log_file=log_file,
        )
        multibatch_flag = True
        multibatch_flag = _run_demo_script(
            demo_name="demo_multibatch",
            model_name=model_name,
            model_info=model_info,
            model_set_dir=model_set_dir,
            model_res_dir=model_res_dir,
            python_exe=python_exe,
            log_file=log_file,
        )
        final_flag = demo_flag and multibatch_flag

    logger.warning(f"remove folder: {os.getcwd()}.")
    shutil.rmtree(os.getcwd())
    assert test_sh_flag is True, "Execute tesh.sh Failed!"
    assert final_flag is True, "Demo Test Failed!"
    logger.info("Demo Test Success!")


def execute_compare_flow(model_name: str, setup_logging) -> None:
    """
    Execute the complete comparison test flow for a specified model.

    Args:
        model_name (str): Name of the model to test
        setup_logging: Fixture of setup_logging
    """
    if HOUMO_BACKEND == "xh1" and get_test_type() != TCaseType.DEFAULT:
        logger.warning("Not support %s (gpu) compare testing.", model_name)
        pytest.skip("This testcase is not support.")

    log_file, pytest_request = setup_logging
    marker_vals = check_device_markers(pytest_request)
    dev_res_dir = f"{marker_vals[NDEVICE_MARKER]}_{marker_vals[DEVICE_MEM_MARKER]}"
    logger.info(f"log_file: {log_file}, dev_res_dir: {dev_res_dir}")

    model_info = _load_model_cfg(model_name)
    if (
        model_info is None
        or model_info["obsolete"] is True
        or HOUMO_BACKEND not in model_info["support_backend"]
        or "compare" not in model_info["support_flow"][HOUMO_BACKEND]
        or "hmcompare_params" not in model_info
    ):
        logger.warning("Not support %s testing.", model_name)
        pytest.skip("This testcase is not support.")
    platform = get_platform(model_info["support_platform"])
    if platform is None or platform == "aarch64":
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    prepare_test_folder(model_dir, "compare")
    current_folder = os.getcwd()
    logger.info("current folder: %s.", current_folder)

    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
    model_res_dir = os.path.join(MODELS_RES_DIR, dev_res_dir, model_info["model_dir"])
    model_type = model_info.get("model_type", "cv")
    if get_test_type() != TCaseType.SEPARATE_INFER:
        _download_models(
            model_info,
            file_type="raw",
            download_dir=model_set_dir,
            extract_dir=model_res_dir,
        )

    if model_type == "cv" and not _prepare_compiled_cv_model(
        model_info,
        platform,
        log_file,
        model_res_dir=model_res_dir,
        model_set_dir=model_set_dir,
    ):
        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        pytest.skip(f"Not support {model_name} testing on {HOUMO_BACKEND}.")
    if get_test_type() == TCaseType.SEPARATE_NO_INFER:
        move_models_res(current_folder, model_res_dir)

        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        skip_msg = f"Skip compare testcase {model_name} in the SEPARATE NO INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    if get_test_type() == TCaseType.SEPARATE_INFER:
        restore_models_res(model_res_dir, current_folder)

    final_flag = True
    # test hmatc compare
    required_params = model_info["hmcompare_params"]["params"]["required"]
    optional_params = model_info["hmcompare_params"]["params"]["optional"]
    cmd_header = ["hmatc", "compare", "--target", HOUMO_BACKEND]

    cmd_list = _generate_hmatc_cmds(cmd_header, required_params, optional_params)
    logger.info(f"cmd list:{cmd_list}")
    for tmp_cmd_list in cmd_list:
        exec_flag, out_str = execute_test_cmd(tmp_cmd_list, log_file)
        if exec_flag is False:
            final_flag = False
            continue

        exec_flag = _check_compare_result(out_str)
        if exec_flag is False:
            final_flag = False

    logger.warning(f"remove folder: {os.getcwd()}.")
    shutil.rmtree(os.getcwd())
    assert final_flag is True, "HmATC Compare Test Failed!"
    logger.info("HmATC Compare Test Success!")


def execute_perf_flow(model_name: str, setup_logging) -> None:
    """
    Execute the complete performance test flow for a specified model.

    Args:
        model_name (str): Name of the model to test
        setup_logging: Fixture of setup_logging
    """
    log_file, pytest_request = setup_logging
    marker_vals = check_device_markers(pytest_request)
    dev_res_dir = f"{marker_vals[NDEVICE_MARKER]}_{marker_vals[DEVICE_MEM_MARKER]}"
    logger.info(f"log_file: {log_file}, dev_res_dir: {dev_res_dir}")

    model_info = _load_model_cfg(model_name)
    if (
        model_info is None
        or model_info["obsolete"] is True
        or HOUMO_BACKEND not in model_info["support_backend"]
        or "perf" not in model_info["support_flow"][HOUMO_BACKEND]
        or ("hmperf_params" not in model_info and "perf_params" not in model_info)
        or "perf_metrics" not in model_info
    ):
        logger.warning("Not support %s testing.", model_name)
        pytest.skip("This testcase is not support.")
    platform = get_platform(model_info["support_platform"])
    if platform is None or platform == "aarch64":
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    prepare_test_folder(model_dir, "perf")
    current_folder = os.getcwd()
    logger.info("current folder: %s.", current_folder)

    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
    model_res_dir = os.path.join(MODELS_RES_DIR, dev_res_dir, model_info["model_dir"])

    model_type = model_info.get("model_type", "cv")
    if (
        model_type == "cv"
        and not _prepare_compiled_cv_model(
            model_info,
            platform,
            log_file,
            model_res_dir=model_res_dir,
            model_set_dir=model_set_dir,
        )
    ) or (
        model_type == "llm"
        and not _prepare_compiled_llm_model(
            model_info,
            platform,
            log_file,
            model_res_dir=model_res_dir,
            model_set_dir=model_set_dir,
        )
    ):
        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        skip_msg = f"Not support {model_name} testing on {HOUMO_BACKEND}."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    if get_test_type() == TCaseType.SEPARATE_NO_INFER:
        if model_type == "cv":
            move_models_res(current_folder, model_res_dir)

        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        skip_msg = f"Skip perf testcase {model_name} in the SEPARATE NO INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    if model_type == "cv" and get_test_type() == TCaseType.SEPARATE_INFER:
        restore_models_res(model_res_dir, current_folder)

    if (
        model_type == "llm"
        and is_release()
        and get_test_type() == TCaseType.SEPARATE_INFER
    ):
        get_model_params = model_info["get_model_params"][HOUMO_BACKEND]
        demo_params = model_info["demo_params"][HOUMO_BACKEND]
        compile_case_id = ""
        for value_list in demo_params.values():
            first_path = value_list[0]
            if "cached_results" in first_path:
                path_parts = os.path.normpath(first_path).split(os.sep)
                cache_idx = path_parts.index("cached_results")
                compile_case_id = path_parts[cache_idx + 1]
                break
        other_params = _check_existed_models(
            compile_case_id, get_model_params, model_res_dir, model_set_dir
        )
        _download_models(
            model_info,
            file_type="hmm",
            download_dir=model_set_dir,
            extract_dir=model_res_dir,
            lock_type="all",
            copy_flag=False,
            other_params=other_params,
        )

    final_flag = True
    perf_threashold = 0.1 if is_release() else 0.95
    if model_info.get("perf_params", None) == "demo":
        # install python requirements
        venv_flag = install_py_venv(current_folder, log_file)
        python_exe = "python3"
        if venv_flag:
            python_exe = f"{VENV_NAME}/bin/python3"

        if model_name == "wenet":
            quant_res_dir = model_info["compile_params"][HOUMO_BACKEND]["model_dir"][0]
            quant_res_dir = quant_res_dir.replace("cached_results", model_res_dir)
            final_flag, opt_str = execute_test_cmd(
                [python_exe, "build.py", "--model_dir", quant_res_dir],
                log_file,
                pyvenv_flag=venv_flag,
            )
            infer_time = [
                float(line.rsplit(" ", 3)[-2])
                for line in opt_str.split("\n")
                if "infer completed" in line
            ]
            infer_val = 0 if len(infer_time) == 0 else infer_time[0]
            # check performance
            backend_metrics = model_info["perf_metrics"].get(HOUMO_BACKEND, None)
            benchmark = backend_metrics.get(platform, None)
            if benchmark and infer_val >= (benchmark * perf_threashold):
                logger.info(
                    f"The best performance is {infer_val} qps, benchmark time is {benchmark} ms."
                )
            else:
                final_flag = False
                error_msg = f"Performance {infer_val} degradation exceeds 5%, benchmark time is {benchmark} ms."
                logger.error(error_msg)
        elif model_name == "sdxl":
            demo_params = model_info["demo_params"][HOUMO_BACKEND]
            perf_idx = 0
            model_path = demo_params["model_path"][perf_idx].replace(
                "cached_results", model_res_dir
            )
            sdxl_ckpt = demo_params["sdxl_ckpt"][perf_idx].replace(
                "cached_models", model_set_dir
            )
            lora_weights = demo_params["lora_weights"][perf_idx].replace(
                "cached_models", model_set_dir
            )
            lock_file_res = model_res_dir + "/lock.lock"
            with ModelResourceLock(
                lock_file_res, ModelResourceLock.LockMode.WRITE, "execute model demo.py"
            ):
                final_flag, opt_str = execute_test_cmd(
                    [
                        python_exe,
                        "demo.py",
                        "--model_path",
                        model_path,
                        "--sdxl_ckpt",
                        sdxl_ckpt,
                        "--lora_weights",
                        lora_weights,
                    ],
                    log_file,
                    pyvenv_flag=venv_flag,
                )
            infer_time = [
                float(line.strip().rsplit(" ", 3)[-2])
                for line in opt_str.split("\n")
                if "ms, average" in line
            ]
            infer_val = 0 if len(infer_time) == 0 else infer_time[0]
            # check performance
            backend_metrics = model_info["perf_metrics"].get(HOUMO_BACKEND, None)
            benchmark = backend_metrics.get("avg_cost", 0)
            if benchmark and infer_val >= (benchmark * perf_threashold):
                logger.info(
                    f"The best performance is {infer_val} qps, benchmark time is {benchmark} ms."
                )
            else:
                final_flag = False
                error_msg = f"Performance {infer_val} degradation exceeds 5%, benchmark time is {benchmark} ms."
                logger.error(error_msg)
        else:
            perf_idx = 0
            demo_cmd = [python_exe, "demo.py"]
            demo_params = model_info["demo_params"][HOUMO_BACKEND]
            for param, param_list in demo_params.items():
                if param_list[perf_idx] is None:
                    continue
                param_val = param_list[perf_idx]
                if "cached_models" in param_val:
                    param_val = param_val.replace("cached_models", model_set_dir)
                elif "cached_results" in param_val:
                    param_val = param_val.replace("cached_results", model_res_dir)
                demo_cmd += [f"--{param}", param_val]
            lock_file_res = model_res_dir + "/lock.lock"
            with ModelResourceLock(
                lock_file_res, ModelResourceLock.LockMode.WRITE, "execute model demo.py"
            ):
                check_flag = False if model_name == "qwen2.5-vl" else True
                final_flag, _ = execute_test_cmd(
                    demo_cmd, log_file, False, check_flag, pyvenv_flag=venv_flag
                )
            perf_dict = {"prefill": 0, "decode": 0, "end2end": 0}
            with open(log_file, "r", encoding="utf-8") as tmp_file:
                for line in tmp_file:
                    if "Prefill Speed" in line:
                        split_idx = -1 if "Decode Speed" not in line else -2
                        perf_dict["prefill"] = float(
                            line.rsplit(":")[split_idx].strip().split(" ")[0].strip()
                        )
                    if "Decode Speed" in line:
                        perf_dict["decode"] = float(
                            line.rsplit(":", 1)[-1].strip().split(" ")[0].strip()
                        )
                    if "E2E TPS" in line:
                        perf_dict["end2end"] = float(
                            line.rsplit(":", 1)[-1].strip().split(" ")[0].strip()
                        )
            # check performance
            backend_metrics = model_info["perf_metrics"].get(HOUMO_BACKEND, None)
            benchmark = backend_metrics.get(platform, None) if backend_metrics else None
            if (
                benchmark
                and perf_dict["prefill"]
                >= (benchmark.get("prefill", 0) * perf_threashold)
                and perf_dict["decode"]
                >= (benchmark.get("decode", 0) * perf_threashold)
                and perf_dict["end2end"]
                >= (benchmark.get("end2end", 0) * perf_threashold)
            ):
                logger.info(
                    f"The best performance is {perf_dict}, benchmark is {benchmark}."
                )
            else:
                final_flag = False
                error_msg = f"Performance {perf_dict} degradation exceeds {(100-perf_threashold*100)}%, benchmark is {benchmark}."
                logger.error(error_msg)
    else:
        # use hmatc perf command to get perf metrics
        final_flag = True
        # test cmd: hmatc perf
        required_params = model_info["hmperf_params"]["params"]["required"]
        optional_params = model_info["hmperf_params"]["params"]["optional"]
        cmd_header = ["hmatc", "perf", "--target", HOUMO_BACKEND]

        max_qps = 0
        cmd_list = _generate_hmatc_cmds(cmd_header, required_params, optional_params)
        logger.info(f"cmd list: {cmd_list}")
        for tmp_cmd_list in cmd_list:
            exec_flag, opt_str = execute_test_cmd(tmp_cmd_list, log_file)
            if exec_flag is False:
                final_flag = False
            else:
                qps = [
                    float(line.split(":", 1)[-1].split("\x1b", 1)[0].strip())
                    for line in opt_str.split("\n")
                    if "[Throughput] qps" in line
                ]
                if len(qps) > 0 and max_qps < qps[0]:
                    max_qps = qps[0]
            reset_chips()
        backend_metrics = model_info["perf_metrics"].get(HOUMO_BACKEND, None)
        benchmark = backend_metrics.get(platform, None)
        if benchmark and max_qps >= (benchmark * perf_threashold):
            logger.info(
                f"The best performance is {max_qps} qps, benchmark qps is {benchmark}."
            )
        else:
            final_flag = False
            error_msg = f"Performance {max_qps} degradation exceeds 5%, benchmark qps is {benchmark}."
            logger.error(error_msg)

    logger.warning(f"remove folder: {os.getcwd()}.")
    shutil.rmtree(os.getcwd())
    assert final_flag is True, "HmATC Perf Test Failed!"
    logger.info("HmATC Perf Test Success!")


def execute_eval_flow(model_name: str, setup_logging) -> None:
    """
    Execute the complete evaluation test flow for a specified model.

    Args:
        model_name (str): Name of the model to test
        log_file (str): Path to the log file for test output
    """
    log_file, pytest_request = setup_logging
    marker_vals = check_device_markers(pytest_request)
    dev_res_dir = f"{marker_vals[NDEVICE_MARKER]}_{marker_vals[DEVICE_MEM_MARKER]}"
    logger.info(f"log_file: {log_file}, dev_res_dir: {dev_res_dir}")

    model_info = _load_model_cfg(model_name)
    if (
        model_info is None
        or model_info["obsolete"] is True
        or HOUMO_BACKEND not in model_info["support_backend"]
        or "eval" not in model_info["support_flow"][HOUMO_BACKEND]
        or "hmeval_params" not in model_info
    ):
        logger.warning("Not support %s testing.", model_name)
        pytest.skip("This testcase is not support.")

    platform = get_platform(model_info["support_platform"])
    if platform is None or platform == "aarch64":
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    prepare_test_folder(model_dir, "eval")
    current_folder = os.getcwd()
    logger.info("current folder: %s.", current_folder)

    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
    model_res_dir = os.path.join(MODELS_RES_DIR, dev_res_dir, model_info["model_dir"])
    model_type = model_info.get("model_type", "cv")
    if (
        model_type == "cv"
        and not _prepare_compiled_cv_model(
            model_info,
            platform,
            log_file,
            model_res_dir=model_res_dir,
            model_set_dir=model_set_dir,
        )
    ) or (
        model_type == "llm"
        and not _prepare_compiled_llm_model(
            model_info,
            platform,
            log_file,
            model_res_dir=model_res_dir,
            model_set_dir=model_set_dir,
        )
    ):
        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        pytest.skip(f"Not support {model_name} testing on {HOUMO_BACKEND}.")
    if get_test_type() == TCaseType.SEPARATE_NO_INFER:
        if model_type == "cv":
            move_models_res(current_folder, model_res_dir)

        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        skip_msg = f"Skip eval testcase {model_name} in the SEPARATE NO INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    if model_type == "cv" and get_test_type() == TCaseType.SEPARATE_INFER:
        restore_models_res(model_res_dir, current_folder)

    final_flag = True
    # test cmd: hmatc eval
    required_params = model_info["hmeval_params"]["params"]["required"]
    optional_params = model_info["hmeval_params"]["params"]["optional"]
    # generate onnx commands (ground truth)
    cmd_header_onnx = ["hmatc", "eval", "--target", HOUMO_BACKEND, "--onnx"]
    cmd_list_onnx = _generate_hmatc_cmds(
        cmd_header_onnx, required_params, optional_params
    )
    # generate hm model commands
    cmd_header = ["hmatc", "eval", "--target", HOUMO_BACKEND]
    cmd_list = _generate_hmatc_cmds(cmd_header, required_params, optional_params)

    logger.info(f"cmd_list_onnx: {cmd_list_onnx}")
    logger.info(f"cmd_list: {cmd_list}")

    perf_names = model_info["eval_threshold"].keys()
    hm_perf_vals = dict()
    onnx_perf_vals = dict()
    for tmp_name in perf_names:
        hm_perf_vals[tmp_name] = list()
        onnx_perf_vals[tmp_name] = list()

    for tmp_cmd_list in cmd_list_onnx:
        exec_flag, onnx_opt_str = execute_test_cmd(tmp_cmd_list, log_file)
        eval_res = _process_eval_result(onnx_opt_str, perf_names)
        for perf_name in perf_names:
            onnx_perf_vals[perf_name].append(eval_res[perf_name])

    for tmp_cmd_list in cmd_list:
        exec_flag, opt_str = execute_test_cmd(tmp_cmd_list, log_file)
        if exec_flag is False:
            final_flag = False
        else:
            eval_res = _process_eval_result(opt_str, perf_names)
            for perf_name in perf_names:
                hm_perf_vals[perf_name].append(eval_res[perf_name])

    if final_flag is False:
        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
    assert final_flag is True, "HmATC Eval Test Failed!"

    for perf_name in perf_names:
        perf_th = model_info["eval_threshold"][perf_name]
        if os.getenv("HOUMO_FULL_DATASET", None) is None:
            # lower threshold
            perf_th = perf_th * 0.5
        check_flag = all(
            perf_th * onnx_val <= hm_val
            for hm_val, onnx_val in zip(
                hm_perf_vals[perf_name], onnx_perf_vals[perf_name]
            )
        )
        if check_flag is False:
            logger.error(
                f"hm {perf_name}: {hm_perf_vals[perf_name]}, onnx {perf_name}: {onnx_perf_vals[perf_name]}"
            )
            logger.warning(f"remove folder: {os.getcwd()}.")
            shutil.rmtree(os.getcwd())
        assert (
            check_flag is True
        ), f"HmATC Eval Test Failed! The difference of {perf_name} exceeds {perf_th*100}%."

    logger.warning(f"remove folder: {os.getcwd()}.")
    shutil.rmtree(os.getcwd())
    logger.info("HmATC Eval Test Success!")
