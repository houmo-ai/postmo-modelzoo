import os
import argparse
from hmatc.utils.utils import hmatc_get_file, get_houmo_version


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='file_type',
        type=str,
        default='hmm',
        choices=["raw", "hmm"],
        help='which resource to get, choise in [raw, hmm]',
    )
    parser.add_argument(
        '--download_dir',
        dest='download_dir',
        type=str,
        default='.',
        help='where to save downloaded model',
    )
    parser.add_argument(
        "--extract_dir",
        dest="extract_dir",
        type=str,
        default=None,
        help='where to save extracted files',
    )
    parser.add_argument(
        "--source_type",
        dest="source_type",
        type=str,
        default="jfrog",
        choices=["jfrog", "modelscope"],
        help='download the model from which source',
    )
    parser.add_argument(
        "--context_length",
        dest="context_length",
        type=str,
        default="8k",
        choices=["2k", "8k"],
        help="context length",
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=1,
        choices=[1, 4],
        help='batch size',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": "qwen3",
        "model_info": {
            "model_size": "8b",
            "ncore": 2 if HOUMO_TARGET == "xh2" else 4,
            "ndevice": 1,
            "context_len": args.context_length,
            "prefill_len": 256,
            "batch": args.batch,
        },
        "raw_files": {"raw_path": "models/datasets/wikitext-2-raw-v1.zip"},
        "modelscope_repo": {"repo_ids": ["qwen/qwen3-8b"]},
    }

    _, ret_dict = hmatc_get_file(
        model_cfgs,
        args.file_type,
        args.download_dir,
        args.extract_dir,
        args.source_type,
    )
    if ret_dict.get("ret", False) is False:
        exit(1)
