import pytest
import os
import logging
import shutil
from ..tests_utils.tests_common_utils import *


logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))


def _load_example_cfg(example_name: str) -> dict:
    example_cfg_path = script_dir + "/apis_configs/apis_cfg_" + example_name + ".json"
    return load_json(example_cfg_path)


def _generate_cmds(cmd_header: list, params_dict: dict, max_core_num: int = 0) -> list:
    cmd_list = [cmd_header]

    idx = 1
    flag = True
    while flag:
        flag = False
        tmp_cmd_list = list()
        for param_name, param_list in params_dict.items():
            if (
                param_name in ["name", "defines"]
                or len(param_list) <= idx
                or param_list[idx] == "default"
            ):
                continue
            if (
                max_core_num > 0
                and param_name == "ncore"
                and param_list[idx] != "default"
                and int(param_list[idx]) > max_core_num
            ):
                continue
            if param_name.startswith("#"):
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
            tmp_cmd_list = cmd_list[0] + tmp_cmd_list
            cmd_list.append(tmp_cmd_list)
        idx += 1

    return cmd_list


def _compile_cpp_exec(example_dir: str, log_file: str, defines: list) -> None:
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


def _test_get_model(example_info: dict, platform: str, log_file: str) -> bool:
    get_model_flag = True
    max_core_num = 2 if platform == "aarch64" else 0
    model_set_dir = os.path.join(MODELS_PATH, example_info["example_dir"])
    params_dict = example_info["get_model_params"][HOUMO_BACKEND]
    cmd_header = ["python3", "get_model.py", "--model_dir", model_set_dir]
    cmd_list = _generate_cmds(cmd_header, params_dict, max_core_num)

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


def execute_apis_examples(example_name: str, log_file: str):
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
        pytest.skip(f"This testcase is not support on 2cores device.")

    if (
        example_info.get("dependency", None) is not None
        and "vpu" in example_info["dependency"]
        and (HDPL_PLATFORM == "ISIM" or check_vpu_status() is False)
    ):
        logger.warning(f"{example_name} testcase needs vpu driver.")
        pytest.skip(f"This testcase needs vpu driver.")

    example_dir = script_dir + "/../../" + example_info["example_dir"]
    if not os.path.exists(example_dir):
        assert False, f"The {example_name} example folder doesn't exist."
    prepare_test_folder(example_dir, "apis")
    current_folder = os.getcwd()
    logger.info("current folder: %s.", current_folder)

    if (
        example_info.get("get_model_params", None) is None
        or example_info["get_model_params"].get(HOUMO_BACKEND, None) is None
    ):
        model_set_dir = os.path.join(MODELS_PATH, example_info["example_dir"])
        lock_file = model_set_dir + "/lock.lock"
        with ModelResourceLock(
            lock_file, ModelResourceLock.LockMode.WRITE, "model downloading"
        ):
            execute_test_cmd(
                ["python3", "get_model.py", "--model_dir", model_set_dir], "", True
            )
    else:
        get_model_flag = _test_get_model(example_info, platform, log_file)
        if get_model_flag is False:
            logger.error("{example_name} Get model Failed!")

    demo_types = example_info["support_backend"][HOUMO_BACKEND]
    # run python demo
    py_flag = True
    if "python" in demo_types:
        # install python requirements
        changed_libs = install_py_env(current_folder, log_file)
        if changed_libs:
            logger.info(f"changed python libs: {changed_libs}.")

        params_dict = example_info["py_example_params"]
        cmd_header = ["python3", params_dict["name"]]
        cmd_list = _generate_cmds(cmd_header, params_dict)
        for tmp_cmd_list in cmd_list:
            exec_flag, _ = execute_test_cmd(tmp_cmd_list, log_file)
            py_flag = False if exec_flag is False else py_flag
        if py_flag is False:
            logger.error("Python example execution failed!")

        # restore python env
        for lib_name, lib_ver in changed_libs.items():
            if lib_ver is None:
                execute_test_cmd(["pip3", "uninstall", lib_name, "-y"], log_file, True)
            else:
                execute_test_cmd(
                    ["pip3", "install", lib_name + "==" + lib_ver], log_file, True
                )

    # run c++ demo
    cpp_flag = True
    if "cpp" in demo_types:
        params_dict = example_info["cpp_example_params"]
        compile_defines = params_dict.get("defines", list())
        _compile_cpp_exec(current_folder, log_file, compile_defines)
        cpp_exe_str = "./" + params_dict["name"]
        cmd_header = [cpp_exe_str]
        cmd_list = _generate_cmds(cmd_header, params_dict)
        for tmp_cmd_list in cmd_list:
            exec_flag, _ = execute_test_cmd(tmp_cmd_list, log_file)
            cpp_flag = False if exec_flag is False else cpp_flag
        if cpp_flag is False:
            logger.error("C++ example execution failed!")

    shutil.rmtree(os.getcwd())
    assert py_flag is True and cpp_flag is True, "Apis Example Test Failed!"
    logger.info("Apis Example Test Success!")
