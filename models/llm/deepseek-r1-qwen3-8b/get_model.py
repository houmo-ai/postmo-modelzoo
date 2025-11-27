import os
import sys
import argparse
from hmatc.utils.utils import hmatc_get_file, get_houmo_version


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", "Only support HOUMO_TARGET: xh2."


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        dest='file_type',
        type=str,
        default="hmm",
        help="which resource to get, choise in [raw, hmm]",
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    model_cfgs = {
        "target": HOUMO_TARGET,
        "version": get_houmo_version(),
        "model_type": "llm",
        "model_name": "deepseek",
        "model_info": {
            "model_size": "8b",
            "ncore": 2,
            "ndevice": 1,
            "context_len": "4k",
            "prefill_len": 256,
            "batch": 1,
        },
        "raw_files": {"raw_path": "models/datasets/wikitext-2-raw-v1.zip"},
        "modelscope_repo": {"repo_ids": ["deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]},
    }

    hmatc_get_file(
        model_cfgs,
        args.file_type,
        args.download_dir,
        args.extract_dir,
        args.source_type,
    )
