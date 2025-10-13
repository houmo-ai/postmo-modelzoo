import os
import shutil
import argparse
import logging
import subprocess
from cd_tester_utils import *


HOUMO_BACKEND = os.getenv("HOUMO_TARGET", "xh1")
script_dir = os.path.dirname(os.path.abspath(__file__))
setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
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
    test_dir = f"{script_dir}/../../tests"
    os.chdir(test_dir)
    logger.info("Current dir: %s", os.getcwd())

    os.environ["SKIP_INFER"] = "ON"
    os.environ["HDPL_PLATFORM"] = "ISIM"
    os.environ["IMODELZOO_MODELS_PATH"] = "/develop02/modelzoo/"
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
        # set env for xh2
        for line in result.stdout.splitlines():
            if '=' in line and (
                'PYTHONPATH' in line
                or 'HF_ENDPOINT' in line
                or 'HOUMO_DATASETS_PATH' in line
            ):
                key, value = line.split('=', 1)
                os.environ[key] = value
        logger.info("*** Env Info ***")
        logger.info("PYTHONPATH: %s", os.getenv("PYTHONPATH"))
        logger.info("HF_ENDPOINT: %s", os.getenv("HF_ENDPOINT"))
        logger.info("HOUMO_DATASETS_PATH: %s", os.getenv("HOUMO_DATASETS_PATH"))

    # hmatc_dir = f"{root_dir}/hmatc"
    # os.chdir(hmatc_dir)
    # logger.info(f"==> [CD Test] Install latest hmatc.")
    # os.system("./install.sh")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    test_dir = f"{root_dir}/tests"
    os.chdir(test_dir)
    logger.info("Current dir: %s", os.getcwd())

    key_list = [
        "asr",
        "autodrive",
        "backbone",
        "detection",
        "estimation",
        "llm",
        "diffusion",
    ]
    # --collect-only
    flag = True
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
        if args.model_str:
            cmds += ["-m", f"{args.model_str}"]
        logger.info(f"execute cmds: {cmds}")
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
