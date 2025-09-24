import os
import shutil
import argparse
import logging
from cd_tester_utils import *


script_dir = os.path.dirname(os.path.abspath(__file__))
setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer Models")
    parser.add_argument(
        "-log",
        "--log_file",
        type=str,
        default="./execute_infer_log.log",
        help="The path of log.",
    )

    args = parser.parse_args()
    return args


def main(args) -> int:
    test_dir = f"{script_dir}/../../tests"
    os.chdir(test_dir)
    logger.info("Current dir: %s", os.getcwd())

    if HOUMO_BACKEND == "xh2":
        os.environ["SKIP_INFER"] = "ON"
    os.environ["HDPL_PLATFORM"] = "ASIC"

    root_dir = f"{script_dir}/../../"
    if HOUMO_BACKEND == "xh1":
        shutil.rmtree(
            f"{script_dir}/../../tests/model_results_{HOUMO_BACKEND}",
            ignore_errors=True,
        )

    hmatc_dir = f"{root_dir}/hmatc"
    os.chdir(hmatc_dir)
    logger.info(f"==> [CD Test] Install latest hmatc.")
    os.system("./install.sh")

    test_dir = f"{root_dir}/tests"
    os.chdir(test_dir)
    logger.info("Current dir: %s", os.getcwd())

    logger.info(f"==> [CD Test] Start apis_tests")
    cmds = ["pytest", "-v", "-s", "apis_tests"]
    if not run_tests(cmds, args.log_file):
        logger.error(f"<== [CD Test] End apis_tests, Failed.")
        return 1

    key_list = [
        "asr or autodrive",
        "backbone",
        "detection",
        "estimation",
        "llm",
        "diffusion",
    ]
    # --collect-only
    flag = True
    for key_str in key_list:
        logger.info(f"==> [CD Test] Start models_tests: {key_str}")
        cmds = ["pytest", "-v", "-s", "models_tests", "-k", key_str]
        if not run_tests(cmds, args.log_file):
            logger.error(f"<== [CD Test] End model_tests: {key_str}, Failed.")
            flag = False
    if not flag:
        return 1

    model_list = ["resnet50", "yolov5s"]
    flag = True
    for model in model_list:
        logger.info(f"==> [CD Test] Start hmatc_tests: {model}")
        cmds = ["pytest", "-v", "-s", "hmatc_tests", "-m", model]
        if not run_tests(cmds, args.log_file):
            logger.error(f"<== [CD Test] End hmatc_tests {model}, Failed.")
            flag = False
    if not flag:
        return 1

    return 0


if __name__ == "__main__":
    args = parse_args()

    ret = main(args)
    logger.info("Ret: %d", ret)

    exit(ret)
