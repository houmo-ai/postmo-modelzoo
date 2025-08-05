import pytest
import os
import subprocess
import logging
import json


HOUMO_BACKEND = os.getenv("HOUMO_TARGET", "xh1")
logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))


def _load_model_cfg(model_name: str) -> dict:
    model_cfg_path = script_dir + "/model_configs/model_cfg_" + model_name + ".json"
    if not os.path.exists(model_cfg_path):
        return None

    with open(model_cfg_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    logger.info(f"Loaded model config file {model_cfg_path}")

    return model_info


def _execute_test_cmd(
    cmd_list: list, log_file: str = "", assert_flag: bool = False, check_flag: bool = True
) -> tuple[bool, any]:
    cmd_str = " ".join(cmd_list)
    logger.info("execute command: %s", cmd_str)

    flag = True
    try:
        process = subprocess.Popen(
            cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate(timeout=7200)  # timeout: 2h
        if process.returncode != 0:
            flag = False
            logger.error(
                f"Failed to execute command: {cmd_str}, error code: {process.returncode}"
            )

    except subprocess.TimeoutExpired as e:
        flag = False
        stdout, stderr = e.stdout, e.stderr
        process.kill()
        logger.error(f"Executation timeout, command: {cmd_str}, exception info: {e}")
    except Exception as e:
        flag = False
        logger.error(f"Failed to execute command: {cmd_str}, unknown error: {e}")
    finally:
        if log_file:
            with open(log_file, "a", encoding='utf-8') as f:
                if stdout:
                    f.write(stdout)
                if stderr:
                    f.write(stderr)
        if check_flag and stdout and "fail" in stdout:
            flag = False
            logger.error(f"Result verification: FAILED!, command: {cmd_str}.")

    if assert_flag:
        assert flag is True, f"Failed to execute command: {cmd_str}."

    return flag, stdout


def _get_platform(support_list: list) -> str:
    import platform

    system = platform.system()
    machine = platform.machine()
    logger.info(f"Only supports Linux system, current system is {system}.")

    if system == 'Linux' and machine in support_list:
        return machine
    return None


def _check_device_info(support_list: list) -> bool:
    if support_list is None or len(support_list) == 0:
        logger.error("No support hmm models.")
        return False

    exec_flag, opt_str = _execute_test_cmd(["hm_smi", "-a"], True)
    lines = [
        line.split(":", 1)[-1].strip()
        for line in opt_str.split('\n')
        if "Core Num" in line
    ]
    if exec_flag and lines and len(set(lines)) == 1:
        device_core_num = int(lines[0])
        if device_core_num in support_list or any(
            device_core_num % core_num == 0 for core_num in support_list
        ):
            logger.info(f"device core num: {device_core_num}")
            return True
        logger.error(
            f"Unsupported device core num {device_core_num}, expected core num: {support_list}"
        )
    else:
        logger.error(f"Unsupported device: {lines}")

    return False


def _generate_hmassist_cmds(
    cmd_header: list, required_params: dict, optional_params: dict, skipped_vals: dict = None
) -> list:
    cmd_list = list()
    # construct required test commands
    idx = 0
    max_length = 0
    while True:
        tmp_cmd_list = list()
        for param_name, param_list in required_params.items():
            if param_name == "target" or param_name == "onnx":
                continue
            max_length = len(param_list) if max_length < len(param_list) else max_length
            param_str = "--" + param_name
            param_val = param_list[0]
            if idx < len(param_list):
                param_val = param_list[idx]
            if skipped_vals and param_name in skipped_vals and param_val in skipped_vals[param_name]:
                continue
            tmp_cmd_list += [param_str, param_val]

        if tmp_cmd_list:
            tmp_cmd_list = cmd_header + tmp_cmd_list
            cmd_list.append(tmp_cmd_list)
        idx += 1
        if idx >= max_length:
            break
    max_length = len(cmd_list)

    # construct optional test commands
    idx = 1  # skip default val
    flag = True
    while flag:
        flag = False
        tmp_cmd_list = list()
        for param_name, param_list in optional_params.items():
            if len(param_list) <= idx:
                continue
            if skipped_vals and param_name in skipped_vals and param_list[idx] in skipped_vals[param_name]:
                continue
            param_str = "--" + param_name
            tmp_cmd_list += [param_str, param_list[idx]]
            flag = True
        if tmp_cmd_list:
            cmd_list_idx = 0 if idx >= max_length else idx
            tmp_cmd_list = cmd_list[cmd_list_idx] + tmp_cmd_list
            cmd_list.append(tmp_cmd_list)
        idx += 1

    return cmd_list


def _generate_py_cmds(cmd_header: list, params_dict: dict) -> list:
    cmd_list = [cmd_header]

    idx = 1
    flag = True
    while flag:
        flag = False
        tmp_cmd_list = list()
        for param_name, param_list in params_dict.items():
            params_str = "--" + param_name
            if len(param_list) <= idx or param_list[idx] == "default":
                continue
            tmp_cmd_list += [params_str, param_list[idx]]
            flag = True
        if tmp_cmd_list:
            tmp_cmd_list = cmd_list[0] + tmp_cmd_list
            cmd_list.append(tmp_cmd_list)
        idx += 1

    return cmd_list


def _check_compile_result(res_str: str) -> bool:
    import re

    row_pattern = re.compile(
        r"\|\s*([^|]+?)\s*\|\s*(\d+\.\d+)\s*\|$"
    )
    rows = []
    header = None
    for line in res_str.split('\n'):
        line = line.strip()
        if 'cosine_dist' in line:
            logger.info(f"detect compile result headers: {line}")
            header = [col.strip() for col in line.split('|') if col and col.strip()]
        elif row_pattern.match(line):
            logger.info(f"detect compile result values: {line}")
            parts = row_pattern.match(line).groups()
            rows.append(
                {
                    header[0]: str(parts[0]),
                    header[1]: float(parts[1])
                }
            )
    if not header or not rows:
        logger.error("Failed to detect the table of compilation results.")
        return False

    logger.info(f"Compilation results: {rows}")
    check_res = all(row[header[1]] >= 0.99 for row in rows)
    return check_res


def _check_compare_result(res_str: str) -> bool:
    import re

    row_pattern = re.compile(
        r"\|\s*([^|]+?)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\w+)\s*\|$"
    )
    rows = []
    header = None
    for line in res_str.split('\n'):
        line = line.strip()
        if 'onnx vs hmquant' in line:
            logger.info(f"detect compare result headers: {line}")
            header = [col.strip() for col in line.split('|') if col and col.strip()]
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
    check_res = all(row[header[3]] == 1.0 for row in rows)
    return check_res


def _process_eval_result(res_str: str, perf_names: list) -> dict:
    import re

    def extract_field(text, field_name):
        pattern = rf"{field_name}':\s*'?([^'\]]+)'?"
        match = re.search(pattern, text)
        return match.group(1) if match else None

    eval_res = dict()
    for line in res_str.split('\n'):
        line = line.strip()
        if any(perf_name in line for perf_name in perf_names):
            for perf_name in perf_names:
                eval_res[perf_name] = float(extract_field(line, perf_name))

    if not eval_res:
        logger.error("Failed to detect the evaluation result.")

    return eval_res


def _install_py_env(env_dir: str, log_file: str) -> dict:
    """Install python env according to requirements.txt."""
    # get current python env
    pip_res = subprocess.run(
        ['pip3', 'list'],
        check=True,
        text=True,
        capture_output=True,
    )
    py_env_dict = dict()
    for line in pip_res.stdout.split('\n'):
        line = line.strip()
        if "Package" in line or "--" in line:
            continue
        split_res = line.split(" ")
        lib_name = split_res[0]
        lib_ver = split_res[-1]
        py_env_dict[lib_name] = lib_ver

    changed_libs = dict()
    rqmt_path = os.path.join(env_dir, "requirements.txt")
    if os.path.exists(rqmt_path) and os.path.isfile(rqmt_path):
        with open(rqmt_path, "r", encoding="utf-8") as f:
            for line in f:
                if "http" in line:
                    continue
                lib_name = line
                if "==" in line:
                    lib_name = line.split("==", 1)[0]
                changed_libs[lib_name] = py_env_dict[lib_name]

        os.chdir(env_dir)
        ret, _ = _execute_test_cmd(
            ['pip3', 'install', '-r', 'requirements.txt'], log_file
        )
        logger.info(
            f"Install python dependencies for the current testcase, ret: {ret}."
        )

    return changed_libs


def _prepare_quantized_model(model_info: dict, log_file: str) -> bool:
    flag = False
    get_model_types = model_info["get_model_params"][HOUMO_BACKEND]["type"]
    if "quant" in get_model_types:
        flag, _ = _execute_test_cmd(["python3", "get_model.py", "--type", "quant"], log_file)
    elif "raw" in get_model_types and "quant" in model_info["support_flow"][HOUMO_BACKEND]:
        if "hmquant_params" in model_info:
            flag1, _ = _execute_test_cmd(["python3", "get_model.py", "--type", "raw"], log_file)
            flag2, _ = _execute_test_cmd(["hmatc", "quant", "--target", HOUMO_BACKEND, "--config", "./config.yml"], log_file)
            flag = all([flag1, flag2])
        elif "quant_params" in model_info:
            flag1, _ = _execute_test_cmd(["python3", "get_model.py", "--type", "raw"], log_file, True)
            flag2, _ = _execute_test_cmd(['python3', "ptq.py"], log_file, True)
            flag = all([flag1, flag2])

    return flag


def execute_get_model_flow(model_name: str, log_file: str = "") -> None:
    """Test all the parameters of the get_model.py in all the supported models."""
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
    platform = _get_platform(model_info["support_platform"])
    if platform is None:
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    os.chdir(model_dir)
    logger.info("current folder: %s.", os.getcwd())

    # test script: get_model.py
    params_dict = model_info["get_model_params"][HOUMO_BACKEND]
    cmd_header = ['python3', 'get_model.py']

    final_flag = True
    cmd_list = _generate_py_cmds(cmd_header, params_dict)
    for tmp_cmd_list in cmd_list:
        exec_flag, _ = _execute_test_cmd(tmp_cmd_list, log_file, False, False)
        final_flag = False if exec_flag is False else final_flag

    assert final_flag is True, "Get Model Test Failed!"
    logger.info("Get Model Test Success!")


def execute_quant_flow(model_name: str, log_file: str = "") -> None:
    """
    Test all the supported model quantization functions and related parameters.
    1. Download the raw model for quantization.
    2. Check whether the model supports hmatc quant.
    3. If the model supports the hmatc tool,
       then the quantization test will be conducted using the hmatc tool.
    4. If the model doesn't support the hmatc tool,
       then execute the quantization test using the ptq.py.
    """
    model_info = _load_model_cfg(model_name)
    if (
        model_info is None
        or model_info["obsolete"] is True
        or HOUMO_BACKEND not in model_info["support_backend"]
        or "quant" not in model_info["support_flow"][HOUMO_BACKEND]
    ):
        logger.warning("Not support %s testing.", model_name)
        pytest.skip("This testcase is not support.")
    platform = _get_platform(model_info["support_platform"])
    if platform is None or platform == "aarch64":
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    os.chdir(model_dir)
    logger.info("current folder: %s.", os.getcwd())

    # download raw model for quantization
    _execute_test_cmd(["python3", "get_model.py", "--type", "raw"], log_file, True)

    final_flag = True
    if "hmquant_params" in model_info:
        # test cmd: hmatc quant
        required_params = model_info["hmquant_params"]["params"]["required"]
        optional_params = model_info["hmquant_params"]["params"]["optional"]
        cmd_header = ["hmatc", "quant", "--target", HOUMO_BACKEND]

        cmd_list = _generate_hmassist_cmds(cmd_header, required_params, optional_params)
        for tmp_cmd_list in cmd_list:
            exec_flag, _ = _execute_test_cmd(tmp_cmd_list, log_file)
            if exec_flag is False:
                final_flag = False
    else:
        # test script: ptq.py
        params_dict = model_info["compile_params"]
        cmd_header = ['python3', 'ptq.py']

        cmd_list = _generate_py_cmds(cmd_header, params_dict)
        for tmp_cmd_list in cmd_list:
            exec_flag, _ = _execute_test_cmd(tmp_cmd_list, log_file)
            if exec_flag is False:
                final_flag = False

    assert final_flag is True, "Quantization Test Failed!"
    logger.info("Quantization Test Success!")


def execute_compile_flow(
    model_name: str, log_file: str = "", clear_flag: bool = True
) -> None:
    """
    Test all the supported model compilation functions and related parameters.
    1. Download the already quantized model for compilation.
    2. Check whether the model supports 'hmatc build'.
    3. If the model supports the hmatc tool,
       then the compilation test will be conducted using the hmatc tool.
    4. If the model doesn't support the hmatc tool,
       then execute the compilation test using the build.py.
    """
    model_info = _load_model_cfg(model_name)
    if (
        model_info is None
        or model_info["obsolete"] is True
        or HOUMO_BACKEND not in model_info["support_backend"]
        or "compile" not in model_info["support_flow"][HOUMO_BACKEND]
    ):
        logger.warning("Not support %s testing.", model_name)
        pytest.skip("This testcase is not support.")
    platform = _get_platform(model_info["support_platform"])
    if platform is None or platform == "aarch64":
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    os.chdir(model_dir)
    logger.info("current folder: %s.", os.getcwd())

    if clear_flag:
        _execute_test_cmd(['rm', '-rf', "output/H30/result"], log_file, True)

    # prepare quantized model
    if not _prepare_quantized_model(model_info, log_file):
        logger.warning(f"Not support {model_name} testing on {HOUMO_BACKEND}.")
        pytest.skip(f"This testcase is not support on {HOUMO_BACKEND}.")

    final_flag = True
    if "hmbuild_params" in model_info:
        # test cmd: hmatc build
        required_params = model_info["hmbuild_params"]["params"]["required"]
        optional_params = model_info["hmbuild_params"]["params"]["optional"]
        cmd_header = ["hmatc", "build", "--target", HOUMO_BACKEND]

        skipped_vals = dict()
        if HOUMO_BACKEND == "xh2":
            skipped_vals["ncore"] = ["4"]
        cmd_list = _generate_hmassist_cmds(cmd_header, required_params, optional_params, skipped_vals)
        for tmp_cmd_list in cmd_list:
            exec_flag, opt_str = _execute_test_cmd(tmp_cmd_list, log_file)
            if exec_flag is False:
                final_flag = False
            else:
                final_flag = _check_compile_result(opt_str)
                if final_flag is False:
                    logger.error(f"Cosine distance exceeds 0.99, compile cmd: {tmp_cmd_list}")
    else:
        # test script: build.py
        params_dict = model_info["compile_params"]
        cmd_header = ['python3', 'build.py']

        cmd_list = _generate_py_cmds(cmd_header, params_dict)
        for tmp_cmd_list in cmd_list:
            exec_flag, _ = _execute_test_cmd(tmp_cmd_list, log_file)
            if exec_flag is False:
                final_flag = False

    assert final_flag is True, "Compilation Test Failed!"
    logger.info("Compilation Test Success!")


def execute_demo_flow(model_name: str, log_file: str = "") -> None:
    model_info = _load_model_cfg(model_name)
    if (
        model_info is None
        or model_info["obsolete"] is True
        or HOUMO_BACKEND not in model_info["support_backend"]
        or "demo" not in model_info["support_flow"][HOUMO_BACKEND]
    ):
        logger.warning("Not support %s testing.", model_name)
        pytest.skip("This testcase is not support.")
    platform = _get_platform(model_info["support_platform"])
    if platform is None:
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")
    # aarch64 test need compiled hmm models
    if platform == "aarch64" and (
        "demo_params" not in model_info
        or "hmm" not in model_info["get_model_params"][HOUMO_BACKEND]["type"]
        or not _check_device_info(
            model_info["support_core_num"].get(HOUMO_BACKEND, None)
        )
    ):
        logger.warning(f"Not support {model_name} testing on 2cores device.")
        pytest.skip(f"This testcase is not support on 2cores device.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."

    os.chdir(model_dir)
    logger.info("current folder: %s.", os.getcwd())

    final_flag = True
    if "hmdemo_params" in model_info and platform != "aarch64":
        if not _prepare_quantized_model(model_info, log_file):
            logger.warning(f"Not support {model_name} testing on {HOUMO_BACKEND}.")
            pytest.skip(f"This testcase is not support on {HOUMO_BACKEND}.")
        # compile quantized model
        _execute_test_cmd(
            ["hmatc", "build", "--target", HOUMO_BACKEND, "--config", "./config.yml"],
            log_file,
            True,
        )
        # install python requirements
        changed_libs = _install_py_env(model_dir, log_file)
        if changed_libs:
            logger.info(f"changed python libs: {changed_libs}.")

        # test hmatc demo
        required_params = model_info["hmdemo_params"]["params"]["required"]
        optional_params = model_info["hmdemo_params"]["params"]["optional"]
        cmd_header = ["hmatc", "demo", "--target", HOUMO_BACKEND]

        cmd_list = _generate_hmassist_cmds(cmd_header, required_params, optional_params)
        for tmp_cmd_list in cmd_list:
            exec_flag, _ = _execute_test_cmd(tmp_cmd_list, log_file)
            if exec_flag is False:
                final_flag = False
    else:
        if platform == "aarch64":
            _execute_test_cmd(
                ["python3", "get_model.py", "--type", "hmm"], log_file, True
            )
        else:
            if not _prepare_quantized_model(model_info, log_file):
                logger.warning(f"Not support {model_name} testing on {HOUMO_BACKEND}.")
                pytest.skip(f"This testcase is not support on {HOUMO_BACKEND}.")
            _execute_test_cmd(["python3", "build.py"], log_file, True)

        # install python requirements
        changed_libs = _install_py_env(model_dir, log_file)
        if changed_libs:
            logger.info(f"changed python libs: {changed_libs}.")

        # test script: demo.py
        params_dict = model_info["demo_params"]
        cmd_header = ['python3', 'demo.py']

        cmd_list = _generate_py_cmds(cmd_header, params_dict)
        for tmp_cmd_list in cmd_list:
            exec_flag, _ = _execute_test_cmd(tmp_cmd_list, log_file)
            if exec_flag is False:
                final_flag = False

    # restore python env
    for lib_name, lib_ver in changed_libs.items():
        _execute_test_cmd(
            ["pip3", "install", lib_name + "==" + lib_ver], log_file, True
        )

    assert final_flag is True, "Demo Test Failed!"
    logger.info("Demo Test Success!")


def execute_compare_flow(model_name: str, log_file: str = "") -> None:
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
    platform = _get_platform(model_info["support_platform"])
    if platform is None or platform == "aarch64":
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    os.chdir(model_dir)
    logger.info("current folder: %s.", os.getcwd())

    _execute_test_cmd(["python3", "get_model.py", "--type", "raw"], log_file, True)
    if not _prepare_quantized_model(model_info, log_file):
        logger.warning(f"Not support {model_name} testing on {HOUMO_BACKEND}.")
        pytest.skip(f"This testcase is not support on {HOUMO_BACKEND}.")
    _execute_test_cmd(
        ["hmatc", "build", "--target", HOUMO_BACKEND, "--config", "./config.yml"],
        log_file,
        True,
    )

    final_flag = True
    # test hmatc compare
    required_params = model_info["hmcompare_params"]["params"]["required"]
    optional_params = model_info["hmcompare_params"]["params"]["optional"]
    cmd_header = ["hmatc", "compare", "--target", HOUMO_BACKEND]

    cmd_list = _generate_hmassist_cmds(cmd_header, required_params, optional_params)
    logger.info(f"cmd list:{cmd_list}")
    for tmp_cmd_list in cmd_list:
        exec_flag, out_str = _execute_test_cmd(tmp_cmd_list, log_file)
        if exec_flag is False:
            final_flag = False
            continue

        exec_flag = _check_compare_result(out_str)
        if exec_flag is False:
            final_flag = False

    assert final_flag is True, "HmATC Compare Test Failed!"
    logger.info("HmATC Compare Test Success!")


def execute_perf_flow(model_name: str, log_file: str = "") -> None:
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
    platform = _get_platform(model_info["support_platform"])
    if platform is None or platform == "aarch64":
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    os.chdir(model_dir)
    logger.info("current folder: %s.", os.getcwd())

    if not _prepare_quantized_model(model_info, log_file):
        logger.warning(f"Not support {model_name} testing on {HOUMO_BACKEND}.")
        pytest.skip(f"This testcase is not support on {HOUMO_BACKEND}.")

    if model_info.get("perf_params", None) == "demo":
        final_flag, opt_str = _execute_test_cmd(["python3", "build.py"], log_file)
        if model_name == "wenet":
            infer_time = [
                float(line.rsplit(" ", 3)[-2])
                for line in opt_str.split('\n')
                if "infer completed" in line
            ]
            infer_val = 0 if len(infer_time) == 0 else infer_time[0]
            # check performance
            backend_metrics = model_info["perf_metrics"].get(HOUMO_BACKEND, None)
            benchmark = backend_metrics.get(platform, None)
            if benchmark and infer_val >= (benchmark * 0.95):
                logger.info(
                    f"The best performance is {infer_val} qps, benchmark time is {benchmark} ms."
                )
            else:
                final_flag = False
                error_msg = f"Performance {infer_val} degradation exceeds 5%, benchmark time is {benchmark} ms."
                logger.error(error_msg)
        else:
            # install python requirements
            changed_libs = _install_py_env(model_dir, log_file)
            if changed_libs:
                logger.info(f"changed python libs: {changed_libs}.")
            final_flag, _ = _execute_test_cmd(["python3", "demo.py"], log_file)
            perf_dict = {"prefill": 0, "decode": 0, "end2end": 0}
            with open(log_file, "r", encoding="utf-8") as tmp_file:
                for line in tmp_file:
                    if "prefill time" in line:
                        perf_dict["prefill"] = float(
                            line.rsplit(",", 1)[-1].split(" ")[1].strip()
                        )
                    if "decode average time" in line:
                        perf_dict["decode"] = float(
                            line.rsplit(",", 1)[-1].split(" ")[1].strip()
                        )
                    if "end2end average time" in line:
                        perf_dict["end2end"] = float(
                            line.rsplit(",", 1)[-1].split(" ")[1].strip()
                        )
            # check performance
            backend_metrics = model_info["perf_metrics"].get(HOUMO_BACKEND, None)
            benchmark = backend_metrics.get(platform, None) if backend_metrics else None
            if (
                benchmark
                and perf_dict["prefill"] >= (benchmark["prefill"] * 0.95)
                and perf_dict["decode"] >= (benchmark["decode"] * 0.95)
                and perf_dict["end2end"] >= (benchmark["end2end"] * 0.95)
            ):
                logger.info(
                    f"The best performance is {perf_dict}, benchmark is {benchmark}."
                )
            else:
                final_flag = False
                error_msg = f"Performance {perf_dict} degradation exceeds 5%, benchmark is {benchmark}."
                logger.error(error_msg)
            # restore python env
            for lib_name, lib_ver in changed_libs.items():
                _execute_test_cmd(
                    ["pip3", "install", lib_name + "==" + lib_ver], log_file, True
                )
    else:
        # use hmatc perf command to get perf metrics
        _execute_test_cmd(
            ["hmatc", "build", "--target", HOUMO_BACKEND, "--config", "./config.yml"],
            log_file,
            True,
        )

        final_flag = True
        # test cmd: hmatc perf
        required_params = model_info["hmperf_params"]["params"]["required"]
        optional_params = model_info["hmperf_params"]["params"]["optional"]
        cmd_header = ["hmatc", "perf", "--target", HOUMO_BACKEND]

        max_qps = 0
        cmd_list = _generate_hmassist_cmds(cmd_header, required_params, optional_params)
        for tmp_cmd_list in cmd_list:
            exec_flag, opt_str = _execute_test_cmd(tmp_cmd_list, log_file)
            if exec_flag is False:
                final_flag = False
            else:
                qps = [
                    float(line.split(":", 1)[-1].split("\x1b", 1)[0].strip())
                    for line in opt_str.split('\n')
                    if "[Throughput] qps" in line
                ]
                if len(qps) > 0 and max_qps < qps[0]:
                    max_qps = qps[0]
        backend_metrics = model_info["perf_metrics"].get(HOUMO_BACKEND, None)
        benchmark = backend_metrics.get(platform, None)
        if benchmark and max_qps >= (benchmark * 0.95):
            logger.info(
                f"The best performance is {max_qps} qps, benchmark qps is {benchmark}."
            )
        else:
            final_flag = False
            error_msg = f"Performance {max_qps} degradation exceeds 5%, benchmark qps is {benchmark}."
            logger.error(error_msg)

    assert final_flag is True, f"HmATC Perf Test Failed! {error_msg}"
    logger.info("HmATC Perf Test Success!")


def execute_eval_flow(model_name: str, log_file: str = "") -> None:
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
    platform = _get_platform(model_info["support_platform"])
    if platform is None or platform == "aarch64":
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    os.chdir(model_dir)
    logger.info("current folder: %s.", os.getcwd())

    # _execute_test_cmd(["python3", "get_model.py", "--type", "quant"], log_file, True)
    if not _prepare_quantized_model(model_info, log_file):
        logger.warning(f"Not support {model_name} testing on {HOUMO_BACKEND}.")
        pytest.skip(f"This testcase is not support on {HOUMO_BACKEND}.")
    _execute_test_cmd(
        ["hmatc", "build", "--target", HOUMO_BACKEND, "--config", "./config.yml"],
        log_file,
        True,
    )

    final_flag = True
    # test cmd: hmatc eval
    required_params = model_info["hmeval_params"]["params"]["required"]
    optional_params = model_info["hmeval_params"]["params"]["optional"]
    # generate onnx commands (ground truth)
    cmd_header_onnx = ["hmatc", "eval", "--target", HOUMO_BACKEND, "--onnx"]
    cmd_list_onnx = _generate_hmassist_cmds(
        cmd_header_onnx, required_params, optional_params
    )
    # generate hm model commands
    cmd_header = ["hmatc", "eval", "--target", HOUMO_BACKEND]
    cmd_list = _generate_hmassist_cmds(cmd_header, required_params, optional_params)

    logger.info(f"cmd_list_onnx: {cmd_list_onnx}")
    logger.info(f"cmd_list: {cmd_list}")

    perf_names = model_info["eval_threshold"].keys()
    hm_perf_vals = dict()
    onnx_perf_vals = dict()
    for tmp_name in perf_names:
        hm_perf_vals[tmp_name] = list()
        onnx_perf_vals[tmp_name] = list()

    for tmp_cmd_list in cmd_list_onnx:
        exec_flag, onnx_opt_str = _execute_test_cmd(tmp_cmd_list, log_file)
        eval_res = _process_eval_result(onnx_opt_str, perf_names)
        for perf_name in perf_names:
            onnx_perf_vals[perf_name].append(eval_res[perf_name])

    for tmp_cmd_list in cmd_list:
        exec_flag, opt_str = _execute_test_cmd(tmp_cmd_list, log_file)
        if exec_flag is False:
            final_flag = False
        else:
            eval_res = _process_eval_result(opt_str, perf_names)
            for perf_name in perf_names:
                hm_perf_vals[perf_name].append(eval_res[perf_name])

    assert final_flag is True, "HmATC Eval Test Failed!"

    for perf_name in perf_names:
        perf_th = model_info["eval_threshold"][perf_name]
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
        assert (
            check_flag is True
        ), f"HmATC Eval Test Failed! The difference of {perf_name} exceeds {perf_th*100}%."

    logger.info("HmATC Eval Test Success!")
