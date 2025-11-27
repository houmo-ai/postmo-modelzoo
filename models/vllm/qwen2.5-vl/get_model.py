import os
import sys
import argparse
from hmatc.utils.utils import hmatc_get_file, get_houmo_version


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='file_type',
        type=str,
        default='hmm',
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
        '--model_size',
        dest='model_size',
        type=str,
        default="7b",
        choices=["3b", "7b"],
        help='model size',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    model_size = args.model_size
    date_str = "20251019" if HOUMO_TARGET == "xh2" else "20250903"
    quant_path = f"models/qwen2.5-vl/hmquant_{HOUMO_TARGET}_qwen2.5-vl_{model_size}_256_2k_{date_str}.zip"

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": "qwen2.5-vl",
        "model_info": {
            "model_size": model_size,
            "ncore": 2 if HOUMO_TARGET == "xh2" else 4,
            "ndevice": 1,
            "context_len": "8k" if HOUMO_TARGET == "xh2" else "2k",
            "prefill_len": 256,
            "batch": 1,
        },
        "quant_files": {
            "quant_path": quant_path,
        },
        "modelscope_repo": {
            "repo_ids": ["Qwen/Qwen2.5-VL-7B-Instruct"],
            "local_dirs": [f"{args.download_dir}/qwen2.5-vl"],
        },
    }

    hmatc_get_file(
        model_cfgs,
        args.file_type,
        args.download_dir,
        args.extract_dir,
        args.source_type,
    )
