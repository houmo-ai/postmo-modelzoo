import os
import argparse
import logging
from compiler_utils import setup_logging, execute_cmd, update_perf_values


script_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Perf LLMs")
    parser.add_argument(
        "--model",
        type=str,
        help="the description of model.",
    )
    parser.add_argument(
        "--case_dir",
        type=str,
        help="the folder path of model case.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        help="the path of raw model.",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        help="the path of embedding file.",
    )
    parser.add_argument(
        "--prefill_path",
        type=str,
        help="the path of prefill model.",
    )
    parser.add_argument(
        "--decode_path",
        type=str,
        help="the path of decode model.",
    )
    parser.add_argument(
        "--perf_id",
        type=str,
        help="the path of perf_file.",
    )
    parser.add_argument(
        "-log",
        "--log_file",
        type=str,
        default="",
        help="the path of log.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    setup_logging(log_file=args.log_file)
    logger = logging.getLogger(__name__)

    logger.info(
        "Model key: %s, Case dir: %s, Tokenizer dir: %s, Embedding path: %s, "
        "Prefill path: %s, Decode path: %s, Perf id: %s",
        args.model,
        args.case_dir,
        args.tokenizer_dir,
        args.embedding_path,
        args.prefill_path,
        args.decode_path,
        args.perf_id,
    )

    model_dir = f"{script_dir}/../../" + args.case_dir
    os.chdir(model_dir)
    logger.info("Current dir: %s", os.getcwd())
    os.environ["HDPL_PLATFORM"] = "ASIC"

    cmds = [
        "python3",
        "perf.py",
        "--tokenizer_dir",
        args.tokenizer_dir,
        "--embedding_path",
        args.embedding_path,
        "--prefill_path",
        args.prefill_path,
        "--decode_path",
        args.decode_path,
    ]
    ret, outputs = execute_cmd(cmds, args.log_file, get_outputs=True)
    if not ret:
        exit(1)

    perf_metric = {"model": args.model, "prefill": 0, "decode": 0, "e2e": 0}
    for line in outputs:
        if "Prefill Speed:" in line:
            perf_metric["prefill"] = (
                line.strip().rsplit(":", 2)[-2].strip().split(" ", 1)[0]
            )
        if "Decode Speed:" in line:
            perf_metric["decode"] = (
                line.strip().rsplit(":", 1)[-1].strip().split(" ", 1)[0]
            )
        if "E2E TPS" in line:
            perf_metric["e2e"] = (
                line.strip().rsplit(":", 1)[-1].strip().split(" ", 1)[0]
            )
    if update_perf_values(args.perf_id, perf_metric) is False:
        exit(2)
