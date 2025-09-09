import os
import argparse
import logging
from compiler_utils import setup_logging, execute_cmd


script_dir = os.path.dirname(os.path.abspath(__file__))
setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Compile LLMs")
    parser.add_argument(
        "-n",
        "--model_name",
        required=True,
        type=str,
        help="(required) model name, example: qwen3",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        type=str,
        help="(required) model path, example: models/llm/qwen3",
    )
    parser.add_argument(
        "-qm",
        "--quant_model_path",
        type=str,
        default="",
        help="Quantized model path",
    )
    parser.add_argument(
        "-cl",
        "--context_length",
        type=int,
        help="Context length.",
    )
    parser.add_argument(
        "-dn",
        "--device_num",
        type=int,
        help="The number of device, default is 1.",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=1,
        help="batch number, default is 1.",
    )
    parser.add_argument(
        "-cn",
        "--core_num",
        type=int,
        default=1,
        help="The number of core, default is 1.",
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
        default="./execute_compilation_log.log",
        help="The path of log.",
    )

    args = parser.parse_args()
    return args


def main(args) -> int:
    logger.info(
        "Model name: %s, Model path: %s, Quant model path: %s, Context length: %d, "
        "Batch: %d, Device num: %d, Core Num: %d, Result Dir: %s",
        args.model_name,
        args.model_path,
        args.quant_model_path,
        args.context_length,
        args.batch,
        args.device_num,
        args.core_num,
        args.result_dir,
    )

    model_dir = f"{script_dir}/../../" + args.model_path
    os.chdir(model_dir)
    logger.info("Current dir: %s", os.getcwd())

    output_dir = args.result_dir
    os.makedirs(output_dir, exist_ok=True)

    cmds = [
        "python3",
        "build.py",
        "--stage",
        "build",
        "--model_dir",
        args.quant_model_path,
        "--model_name",
        args.model_name,
        "--context_length",
        str(args.context_length),
        "--batch",
        str(args.batch),
        "--ncore",
        str(args.core_num),
        "--ndevice",
        str(args.device_num),
        "--output_dir",
        output_dir,
    ]

    ret = execute_cmd(cmds, args.log_file)

    if not ret:
        return 1

    embed_dir = f"{output_dir}/hmquant"
    os.makedirs(embed_dir, exist_ok=True)
    execute_cmd(
        [
            "cp",
            "-a",
            f"{args.quant_model_path}/quant_embedding.pt",
            f"{embed_dir}/",
        ],
        args.log_file,
    )
    return 0


if __name__ == "__main__":
    args = parse_args()

    ret = main(args)
    logger.info("Ret: %d", ret)

    exit(ret)
