import os
import argparse
import logging
import subprocess
from compiler_utils import setup_logging, execute_cmd


script_dir = os.path.dirname(os.path.abspath(__file__))
setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Compile LLMs")
    parser.add_argument(
        "task_id",
        nargs='?',
        default=None,
        help="Task ID (optional positional argument)",
    )

    parser.add_argument(
        "-t",
        "--target",
        required=True,
        type=str,
        help="(required) houmo backend, example: xh1, xh2",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        type=str,
        help="(required) model path, example: models/llm/qwen3",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        required=True,
        type=str,
        help="(required) model name, example: qwen3",
    )
    parser.add_argument(
        "-raw",
        "--raw_model_path",
        type=str,
        default="",
        help="raw model path, example: models/llm/qwen3-8b",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=1,
        help="Batch number, default is 1.",
    )
    parser.add_argument(
        "-pl",
        "--prefill_length",
        type=int,
        default=256,
        help="Prefill length, recommend to use the default value 256.",
    )
    parser.add_argument(
        "-cl",
        "--context_length",
        type=int,
        default=2048,
        help="Context length, default is 2k.",
    )
    parser.add_argument(
        "-r",
        "--result_dir",
        type=str,
        default="./",
        help="The path for storing the results.",
    )
    parser.add_argument(
        "-log",
        "--log_file",
        type=str,
        default="./execute_quantized_log.log",
        help="The path of log.",
    )

    args = parser.parse_args()
    return args


def main(args) -> int:
    logger.info(
        "Houmo Target: %s, Model path: %s, Model name: %s, Raw model path: %s, Batch: %d, Prefill length: %d, Context length: %d, Result Dir: %s",
        args.target,
        args.model_path,
        args.model_name,
        args.raw_model_path,
        args.batch,
        args.prefill_length,
        args.context_length,
        args.result_dir,
    )

    root_dir = f"{script_dir}/../../"
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

    model_dir = f"{root_dir}/" + args.model_path
    os.chdir(model_dir)
    logger.info("Current dir: %s", os.getcwd())

    if os.path.exists("./requirements.txt"):
        os.system("pip3 install -r requirements.txt")

    model_name = "deepseek" if "deepseek" in args.model_name else args.model_name
    raw_model = f"/modelzoo/{args.raw_model_path}"
    output_dir = args.result_dir
    os.makedirs(output_dir, exist_ok=True)
    cmds = ["python3", "ptq.py", "--model", raw_model]
    if args.target == "xh1":
        cmds += [
            "--model_name",
            model_name,
            "--cache_len",
            str(args.context_length),
            "--prefill_shape",
            "4",
            str(int(args.prefill_length / 4)),
            "--save_path",
            output_dir,
        ]
        if args.batch > 1:
            cmds += ["--multi_batch"]
    else:
        cmds += [
            "--model-name",
            model_name,
            "--context-length",
            str(args.context_length),
            "--input-sequence-length",
            str(args.prefill_length),
            "--out-dir",
            output_dir,
        ]
    ret = execute_cmd(cmds, args.log_file)
    if not ret:
        logger.error("Failed to quant model.")
        return 1

    return 0


if __name__ == "__main__":
    args = parse_args()

    ret = main(args)
    logger.info("Ret: %d", ret)

    exit(ret)
