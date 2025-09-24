import pytest
import os
import logging
import shutil
from glob import glob
from ..tests_utils.tests_common_utils import *


logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))


def _run_hmatc(model_info, config_yml, hmatc_type, log_file):
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


def _perf_models(config_yml, log_file):
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


def execute_hmatc_cmd(model_name: str, log_file: str):
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
    for config_yml in glob(f"{test_configs}/func_test/*.yml"):
        logger.info(f"test config file: {config_yml}")
        for hmatc_type in hmatc_types:
            if not _run_hmatc(model_dict[model_name], config_yml, hmatc_type, log_file):
                final_flag = False

    if HOUMO_BACKEND == "xh1":
        for config_yml in glob(f"{test_configs}/perf_test/*.yml"):
            if not _perf_models(config_yml, log_file):
                final_flag = False

    shutil.rmtree(os.getcwd())
    assert final_flag is True, "Hmatc Test Failed!"
    logger.info("Hmatc Test Success!")
