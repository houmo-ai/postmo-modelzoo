import pytest
import os
import logging
import shutil
from ..tests_utils.tests_common_utils import *
import glob


logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))


def _load_model_cfg(model_name: str) -> dict:
    model_cfg_path = script_dir + "/model_configs/model_cfg_" + model_name + ".json"
    return load_json(model_cfg_path)


def _generate_hmatc_cmds(
    cmd_header: list,
    required_params: dict,
    optional_params: dict,
    skipped_vals: dict = None,
) -> list:
    merged_params = required_params.copy()
    merged_params.update(optional_params)

    cmd_list = list()
    # construct required test commands
    idx = 0
    flag = True
    while flag:
        flag = False
        tmp_cmd_list = list()
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

    cmd_list = [cmd_header] if skip_default else list()
    idx = 1 if skip_default else 0
    flag = True
    while flag:
        flag = False
        tmp_cmd_list = list()
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


def _check_compile_result(res_str: str) -> bool:
    import re

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
                except:
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
    check_res = all(row[header[1]] >= compile_th for row in rows)
    if check_res is True and HOUMO_BACKEND == "xh1":
        check_res = all(row[header[3]] >= compile_th for row in rows)
    return check_res


def _check_compare_result(res_str: str) -> bool:
    import re

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
    # [TMP]
    if HOUMO_BACKEND == "xh2":
        compare_th = 0.9
    check_res = all(row[header[3]] >= compare_th for row in rows)
    return check_res


def _process_eval_result(res_str: str, perf_names: list) -> dict:
    import re

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
    for i in range(len(params)):
        if params[i] == target_param:
            if i + 1 < len(params):
                return params[i + 1]
            break
    return None


def _prepare_quantized_llm_model(model_info: dict, log_file: str) -> bool:
    if get_test_type() == TCaseType.SEPARATE_INFER:
        logger.warning(
            "Skip the step of preparing quantized llm model in the SPEARATE INFER stage."
        )
        return True

    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
    model_res_dir = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
    lock_file_dst = model_res_dir + "/lock.lock"
    lock_file_src = model_set_dir + "/lock.lock"
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
        for idx, tmp_model_dir in enumerate(quant_params["out-dir"]):
            quant_res_dir = tmp_model_dir.replace("cached_results", model_res_dir)
            if quant_res_dir and os.path.exists(quant_res_dir):
                logger.warning(
                    f"Skip the step of preparing quantized llm model {quant_res_dir} in the SPEARATE NO INFER stage."
                )
                flag = True
                continue
            # download raw model files
            with ModelResourceLock(
                lock_file_src, ModelResourceLock.LockMode.WRITE, "model downloading"
            ):
                execute_test_cmd(
                    [
                        "python3",
                        "get_model.py",
                        "--model_dir",
                        model_set_dir,
                        "--type",
                        "raw",
                    ],
                    "",
                    True,
                )

            with ModelResourceLock(
                lock_file_dst, ModelResourceLock.LockMode.WRITE, "model quantizing"
            ):
                # quant model
                model_dir = quant_params["model"][idx]
                if "cached_models" in quant_params["model"][idx]:
                    model_dir = quant_params["model"][idx].replace(
                        "cached_models", model_set_dir
                    )
                elif "cached_results" in quant_params["model"][idx]:
                    model_dir = quant_params["model"][idx].replace(
                        "cached_results", model_res_dir
                    )
                cmds = [
                    "python3",
                    "ptq.py",
                    "--model",
                    model_dir,
                    "--context-length",
                    quant_params["context-length"][idx],
                ]
                if HOUMO_BACKEND == "xh1":
                    cmds += ["--save_path", quant_res_dir]
                else:
                    cmds += ["--out-dir", quant_res_dir]
                flag, _ = execute_test_cmd(cmds, log_file)
                if flag is True:
                    tmp_res_dir = f"{quant_res_dir}/hmquant"
                    os.system(f"mv {tmp_res_dir} {quant_res_dir}")
                else:
                    return flag

        return flag

    compile_params = model_info["compile_params"][HOUMO_BACKEND]
    for idx, tmp_model_dir in enumerate(compile_params["model_dir"]):
        quant_res_dir = tmp_model_dir.replace("cached_results", model_res_dir)
        if quant_res_dir and os.path.exists(quant_res_dir):
            logger.warning(
                f"Skip the step of preparing quantized llm model {quant_res_dir} in the SPEARATE NO INFER stage."
            )
            continue

        logger.info("Start to download quantized llm model for compiling.")
        if "quant" in model_info["get_model_params"][HOUMO_BACKEND]["type"]:
            # download quantized model file
            with ModelResourceLock(
                lock_file_src, ModelResourceLock.LockMode.WRITE, "model downloading"
            ):
                with ModelResourceLock(
                    lock_file_dst,
                    ModelResourceLock.LockMode.WRITE,
                    "model downloading",
                ):
                    flag, _ = execute_test_cmd(
                        [
                            "python3",
                            "get_model.py",
                            "--model_dir",
                            model_set_dir,
                            "--quant_model_dir",
                            quant_res_dir,
                            "--type",
                            "quant",
                        ]
                    )
                    if flag is False:
                        return flag
        else:
            logger.warning("Not support downloading quantized model file.")
    return flag


def _prepare_quantized_cv_model(model_info: dict, log_file: str) -> bool:
    logger.info("Start to prepare quantized cv model for compiling.")
    flag = True
    model_res_dir = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
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

            with ModelResourceLock(
                lock_file, ModelResourceLock.LockMode.WRITE, "model downloading"
            ):
                flag, _ = execute_test_cmd(
                    [
                        "python3",
                        "get_model.py",
                        "--model_dir",
                        model_set_dir,
                        "--type",
                        "quant",
                        "--quant_model_dir",
                        quant_res_dir,
                    ]
                )
                if flag is False:
                    break
        return flag

    if (
        "raw" in get_model_types
        and "quant" in model_info["support_flow"][HOUMO_BACKEND]
    ):
        lock_md_file = model_set_dir + "/lock.lock"
        with ModelResourceLock(
            lock_md_file, ModelResourceLock.LockMode.WRITE, "model downloading"
        ):
            flag, _ = execute_test_cmd(
                [
                    "python3",
                    "get_model.py",
                    "--model_dir",
                    model_set_dir,
                    "--type",
                    "raw",
                ]
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


def _prepare_compiled_llm_model(model_info: dict, platform: str, log_file: str) -> bool:
    if get_test_type() == TCaseType.SEPARATE_INFER:
        logger.warning(
            "Skip the step of preparing compiled model in the SPEARATE INFER stage."
        )
        return True

    compile_params = model_info["compile_params"][HOUMO_BACKEND]
    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
    model_res_dir = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
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
            os.system(
                f"cp {quant_res_dir}/quant_embedding.pt {compile_res_dir}/hmquant/"
            )
        if get_test_type() in [TCaseType.SEPARATE_NO_INFER, TCaseType.DEFAULT]:
            hmm_files = glob.glob(os.path.join(compile_res_dir, "*.hmm"))
            if (
                os.path.exists(compile_res_dir)
                and os.path.exists(embedding_path)
                and len(hmm_files) > 0
            ):
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
            and _prepare_quantized_llm_model(model_info, log_file)
        ):
            with ModelResourceLock(
                lock_file, ModelResourceLock.LockMode.WRITE, "model compiling"
            ):
                cmd_list = [
                    "python3",
                    "build.py",
                    "--model_dir",
                    quant_res_dir,
                    "--output_dir",
                    compile_res_dir,
                    "--stage",
                    "build",
                ]
                if HOUMO_BACKEND == "xh2":
                    cmd_list += [
                        "--context_length",
                        compile_params["context_length"][idx],
                    ]
                flag, _ = execute_test_cmd(
                    cmd_list,
                    log_file,
                )
            if os.path.exists(f"{quant_res_dir}/quant_embedding.pt"):
                os.makedirs(f"{compile_res_dir}/hmquant/", exist_ok=True)
                os.system(
                    f"cp {quant_res_dir}/quant_embedding.pt {compile_res_dir}/hmquant/"
                )
            if flag is False:
                break
            else:
                continue

        flag = False
        if "hmm" in model_info["get_model_params"][HOUMO_BACKEND]["type"]:
            lock_file_src = model_set_dir + "/lock.lock"
            with ModelResourceLock(
                lock_file_src, ModelResourceLock.LockMode.WRITE, "model downloading"
            ):
                with ModelResourceLock(
                    lock_file, ModelResourceLock.LockMode.WRITE, "model downloading"
                ):
                    flag, _ = execute_test_cmd(
                        [
                            "python3",
                            "get_model.py",
                            "--model_dir",
                            model_set_dir,
                            "--build_model_dir",
                            compile_res_dir,
                            "--type",
                            "hmm",
                        ]
                    )
                if flag is False:
                    break
                else:
                    continue

    return flag


def _prepare_compiled_cv_model(model_info: dict, platform: str, log_file: str) -> bool:
    if get_test_type() == TCaseType.SEPARATE_INFER:
        logger.warning(
            "Skip the step of preparing compiled model in the SPEARATE INFER stage."
        )
        return True

    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
    model_res_dir = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
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
        lock_file_src = model_set_dir + "/lock.lock"
        lock_file_dst = model_res_dir + "/lock.lock"
        with ModelResourceLock(
            lock_file_src, ModelResourceLock.LockMode.WRITE, "model downloading"
        ):
            with ModelResourceLock(
                lock_file_dst, ModelResourceLock.LockMode.WRITE, "model downloading"
            ):
                execute_test_cmd(
                    [
                        "python3",
                        "get_model.py",
                        "--model_dir",
                        model_set_dir,
                        "--build_model_dir",
                        compile_res_dir,
                        "--type",
                        "hmm",
                    ],
                    "",
                    True,
                )
        os.system(f"cp -ar {compile_res_dir} ./")
        return True
    # platform != "aarch64"
    if not _prepare_quantized_cv_model(model_info, log_file):
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

    # test script: get_model.py
    params_dict = model_info["get_model_params"][HOUMO_BACKEND]
    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
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
            exec_flag, _ = execute_test_cmd(tmp_cmd_list, log_file, False, False)
            final_flag = False if exec_flag is False else final_flag

    logger.warning(f"remove folder: {os.getcwd()}.")
    shutil.rmtree(os.getcwd())
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
        or is_release() is True
    ):
        logger.warning(
            "Not support %s testing, release flag: %d.", model_name, int(is_release())
        )
        pytest.skip("This testcase is not support.")
    if get_test_type() == TCaseType.SEPARATE_INFER:
        skip_msg = f"This quant testcase of {model_name} has already been run in the SEPARATE INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    model_type = model_info.get("model_type", "cv")
    if (
        model_type == "llm"
        and get_test_type() != TCaseType.SEPARATE_INFER
        and check_gpu()["has_gpu"] is False
    ):
        logger.warning(f"{model_name} testcase requires GPU.")
        pytest.skip(f"{model_name} testcase requires GPU.")
    platform = get_platform(model_info["support_platform"])
    if platform is None or platform == "aarch64":
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    prepare_test_folder(model_dir, "quant")
    logger.info("current folder: %s.", os.getcwd())

    # download raw model for quantization
    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
    lock_file = model_set_dir + "/lock.lock"
    with ModelResourceLock(
        lock_file, ModelResourceLock.LockMode.WRITE, "model downloading"
    ):
        execute_test_cmd(
            ["python3", "get_model.py", "--model_dir", model_set_dir, "--type", "raw"],
            "",
            True,
        )
        if model_type == "cv":
            os.system(f"cp -ar {model_set_dir}/* ./")

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
        model_res_dir = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
        # test script: ptq.py
        params_dict = model_info["quant_params"][HOUMO_BACKEND]
        cmd_header = ["python3", "ptq.py"]

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
                if os.path.exists(quant_res_dir):
                    shutil.rmtree(quant_res_dir, ignore_errors=True)
                    # [TMP]
                    # continue
                exec_flag, _ = execute_test_cmd(tmp_cmd_list, log_file)
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
        or is_release() is True
    ):
        logger.warning(
            "Not support %s testing, release flag: %d.", model_name, int(is_release())
        )
        pytest.skip("This testcase is not support.")
    if get_test_type() == TCaseType.SEPARATE_INFER:
        skip_msg = f"This compile testcase of {model_name} has already been run in the SEPARATE NO INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    model_type = model_info.get("model_type", "cv")
    if (
        model_type == "llm"
        and get_test_type() != TCaseType.SEPARATE_INFER
        and check_gpu()["has_gpu"] is False
    ):
        logger.warning(f"{model_name} testcase requires GPU.")
        pytest.skip(f"{model_name} testcase requires GPU.")
    platform = get_platform(model_info["support_platform"])
    if platform is None or platform == "aarch64":
        logger.warning(f"Not support {model_name} testing on {platform}.")
        pytest.skip(f"This testcase is not support on {platform}.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    prepare_test_folder(model_dir, "compile")
    logger.info("current folder: %s.", os.getcwd())

    if clear_flag:
        execute_test_cmd(["rm", "-rf", "output/H30/result"], log_file, True)

    # prepare quantized model
    if (
        model_type == "cv" and not _prepare_quantized_cv_model(model_info, log_file)
    ) or (
        model_type == "llm" and not _prepare_quantized_llm_model(model_info, log_file)
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
                final_flag = _check_compile_result(opt_str)
                if final_flag is False:
                    logger.error(
                        f"Cosine distance exceeds 0.99, compile cmd: {tmp_cmd_list}"
                    )
    else:
        # test script: build.py
        params_dict = model_info["compile_params"][HOUMO_BACKEND]
        cmd_header = ["python3", "build.py"]

        model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
        model_res_dir = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
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
    model_type = model_info.get("model_type", "cv")
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
        pytest.skip(f"This testcase is not support on 2cores device.")

    model_dir = script_dir + "/../../" + model_info["model_dir"]
    if not os.path.exists(model_dir):
        assert False, f"The {model_name} model folder doesn't exist."
    prepare_test_folder(model_dir, "demo")
    current_folder = os.getcwd()
    logger.info("current folder: %s.", current_folder)

    model_type = model_info.get("model_type", "cv")
    if (
        model_type == "cv"
        and not _prepare_compiled_cv_model(model_info, platform, log_file)
    ) or (
        model_type == "llm"
        and not _prepare_compiled_llm_model(model_info, platform, log_file)
    ):
        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        pytest.skip(f"Not support {model_name} testing on {HOUMO_BACKEND}.")
    if get_test_type() == TCaseType.SEPARATE_NO_INFER:
        if model_type == "cv":
            dst_folder = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
            move_models_res(current_folder, dst_folder)

        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        logger.warning(
            f"Skip demo testcase {model_name} in the SEPARATE NO INFER stage."
        )
        pytest.skip(f"Skip demo testcase {model_name} in the SEPARATE NO INFER stage.")
    if model_type == "cv" and get_test_type() == TCaseType.SEPARATE_INFER:
        src_folder = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
        restore_models_res(src_folder, current_folder)

    # install python requirements
    changed_libs = install_py_env(current_folder, log_file)
    if changed_libs:
        logger.info(f"changed python libs: {changed_libs}.")

    final_flag = True
    if "hmdemo_params" in model_info and platform != "aarch64":
        # test hmatc demo
        required_params = model_info["hmdemo_params"]["params"]["required"]
        optional_params = model_info["hmdemo_params"]["params"]["optional"]
        cmd_header = ["hmatc", "demo", "--target", HOUMO_BACKEND]

        cmd_list = _generate_hmatc_cmds(cmd_header, required_params, optional_params)
        logger.info(f"cmd list: {cmd_list}")
        for tmp_cmd_list in cmd_list:
            exec_flag, _ = execute_test_cmd(tmp_cmd_list, log_file)
            if exec_flag is False:
                final_flag = False
    else:
        # test script: demo.py
        params_dict = model_info["demo_params"][HOUMO_BACKEND]
        cmd_header = ["python3", "demo.py"]

        model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
        model_res_dir = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
        cmd_list = _generate_py_cmds(
            cmd_header,
            params_dict,
            skip_default=False,
            model_dir=model_set_dir,
            res_dir=model_res_dir,
        )
        logger.info(f"demo flow cmd_list:{cmd_list}")
        lock_file_res = model_res_dir + "/lock.lock"
        with ModelResourceLock(
            lock_file_res, ModelResourceLock.LockMode.WRITE, "execute model demo.py"
        ):
            for tmp_cmd_list in cmd_list:
                exec_flag, _ = execute_test_cmd(tmp_cmd_list, log_file)
                if exec_flag is False:
                    final_flag = False

    # restore python env
    for lib_name, lib_ver in changed_libs.items():
        if lib_ver is None:
            execute_test_cmd(["pip3", "uninstall", lib_name, "-y"], log_file, True)
        else:
            execute_test_cmd(
                ["pip3", "install", lib_name + "==" + lib_ver], log_file, True
            )

    logger.warning(f"remove folder: {os.getcwd()}.")
    shutil.rmtree(os.getcwd())
    assert final_flag is True, "Demo Test Failed!"
    logger.info("Demo Test Success!")


def execute_compare_flow(model_name: str, log_file: str = "") -> None:
    if HOUMO_BACKEND == "xh1" and get_test_type() != TCaseType.DEFAULT:
        logger.warning("Not support %s (gpu) compare testing.", model_name)
        pytest.skip("This testcase is not support.")
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

    model_type = model_info.get("model_type", "cv")
    model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])

    lock_file = model_set_dir + "/lock.lock"
    with ModelResourceLock(
        lock_file, ModelResourceLock.LockMode.WRITE, "model downloading"
    ):
        execute_test_cmd(
            [
                "python3",
                "get_model.py",
                "--model_dir",
                model_set_dir,
                "--type",
                "raw",
            ],
            "",
            True,
        )

    if model_type == "cv" and not _prepare_compiled_cv_model(
        model_info, platform, log_file
    ):
        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        pytest.skip(f"Not support {model_name} testing on {HOUMO_BACKEND}.")
    if get_test_type() == TCaseType.SEPARATE_NO_INFER:
        dst_folder = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
        move_models_res(current_folder, dst_folder)

        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        skip_msg = f"Skip compare testcase {model_name} in the SEPARATE NO INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    if get_test_type() == TCaseType.SEPARATE_INFER:
        src_folder = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
        restore_models_res(src_folder, current_folder)

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
    model_type = model_info.get("model_type", "cv")
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

    model_type = model_info.get("model_type", "cv")
    if (
        model_type == "cv"
        and not _prepare_compiled_cv_model(model_info, platform, log_file)
    ) or (
        model_type == "llm"
        and not _prepare_compiled_llm_model(model_info, platform, log_file)
    ):
        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        skip_msg = f"Not support {model_name} testing on {HOUMO_BACKEND}."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    if get_test_type() == TCaseType.SEPARATE_NO_INFER:
        if model_type == "cv":
            dst_folder = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
            move_models_res(current_folder, dst_folder)

        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        skip_msg = f"Skip perf testcase {model_name} in the SEPARATE NO INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    if model_type == "cv" and get_test_type() == TCaseType.SEPARATE_INFER:
        src_folder = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
        restore_models_res(src_folder, current_folder)

    final_flag = True
    if model_info.get("perf_params", None) == "demo":
        # install python requirements
        changed_libs = install_py_env(current_folder, log_file)
        if changed_libs:
            logger.info(f"changed python libs: {changed_libs}.")

        model_res_dir = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
        if model_name == "wenet":
            quant_res_dir = model_info["compile_params"][HOUMO_BACKEND]["model_dir"][0]
            quant_res_dir = quant_res_dir.replace("cached_results", model_res_dir)
            final_flag, opt_str = execute_test_cmd(
                ["python3", "build.py", "--model_dir", quant_res_dir],
                log_file,
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
            if benchmark and infer_val >= (benchmark * 0.95):
                logger.info(
                    f"The best performance is {infer_val} qps, benchmark time is {benchmark} ms."
                )
            else:
                final_flag = False
                error_msg = f"Performance {infer_val} degradation exceeds 5%, benchmark time is {benchmark} ms."
                logger.error(error_msg)
        elif model_name == "sdxl":
            model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
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
                        "python3",
                        "demo.py",
                        "--model_path",
                        model_path,
                        "--sdxl_ckpt",
                        sdxl_ckpt,
                        "--lora_weights",
                        lora_weights,
                    ],
                    log_file,
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
            if benchmark and infer_val >= (benchmark * 0.95):
                logger.info(
                    f"The best performance is {infer_val} qps, benchmark time is {benchmark} ms."
                )
            else:
                final_flag = False
                error_msg = f"Performance {infer_val} degradation exceeds 5%, benchmark time is {benchmark} ms."
                logger.error(error_msg)
        else:
            model_set_dir = os.path.join(MODELS_PATH, model_info["model_dir"])
            perf_idx = 0
            demo_cmd = ["python3", "demo.py"]
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
                final_flag, _ = execute_test_cmd(demo_cmd, log_file)
            perf_dict = {"prefill": 0, "decode": 0, "end2end": 0}
            with open(log_file, "r", encoding="utf-8") as tmp_file:
                for line in tmp_file:
                    if "Prefill Speed" in line:
                        perf_dict["prefill"] = float(
                            line.rsplit(":")[-2].strip().split(" ")[0].strip()
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
            execute_test_cmd(
                ["pip3", "install", lib_name + "==" + lib_ver], log_file, True
            )
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
        if benchmark and (max_qps >= (benchmark * 0.95) or is_separate()):
            logger.info(
                f"The best performance is {max_qps} qps, benchmark qps is {benchmark}."
            )
        else:
            final_flag = False
            error_msg = f"Performance {max_qps} degradation exceeds 5%, benchmark qps is {benchmark}."
            logger.error(error_msg)

    logger.warning(f"remove folder: {os.getcwd()}.")
    shutil.rmtree(os.getcwd())
    assert final_flag is True, f"HmATC Perf Test Failed!"
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
    model_type = model_info.get("model_type", "cv")
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

    model_type = model_info.get("model_type", "cv")
    if (
        model_type == "cv"
        and not _prepare_compiled_cv_model(model_info, platform, log_file)
    ) or (
        model_type == "llm"
        and not _prepare_compiled_llm_model(model_info, platform, log_file)
    ):
        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        pytest.skip(f"Not support {model_name} testing on {HOUMO_BACKEND}.")
    if get_test_type() == TCaseType.SEPARATE_NO_INFER:
        if model_type == "cv":
            dst_folder = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
            move_models_res(current_folder, dst_folder)

        logger.warning(f"remove folder: {os.getcwd()}.")
        shutil.rmtree(os.getcwd())
        skip_msg = f"Skip perf testcase {model_name} in the SEPARATE NO INFER stage."
        logger.warning(skip_msg)
        pytest.skip(skip_msg)
    if model_type == "cv" and get_test_type() == TCaseType.SEPARATE_INFER:
        src_folder = os.path.join(MODELS_RES_DIR, model_info["model_dir"])
        restore_models_res(src_folder, current_folder)

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
