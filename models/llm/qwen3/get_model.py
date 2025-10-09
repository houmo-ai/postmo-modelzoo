import os
import sys
import argparse
from hmatc.utils.utils import get_file_from_jfrog


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='model_type',
        type=str,
        default='hmm',
        help='which resource to get, choise in [raw, hmm]',
    )
    parser.add_argument(
        "--build_model_dir",
        dest="build_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
    )
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default='.',
        help='where to save downloaded model',
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=1,
        help='batch size',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    build_model_dir = args.build_model_dir
    model_type = args.model_type
    model_dir = args.model_dir
    HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '.')
    HOUMO_MODEL_PATH = os.getenv('HOUMO_MODEL_PATH', '.')
    wiki_path = "models/datasets/wikitext-2-raw-v1.zip"
    if HOUMO_TARGET == "xh1":
        hmm_path = "models/qwen3/hmm_qwen3_256_8k_4cores_20250728.zip"
    elif HOUMO_TARGET == "xh2":
        if args.batch == 1:
            hmm_path = "models/qwen3/hmm_xh2_qwen3_8b_256_8k_2cores_20250808.zip"
        elif args.batch == 2:
            hmm_path = "models/qwen3/hmm_xh2_qwen3_8b_256_2k_2batch_2cores_20250912.zip"
        elif args.batch == 4:
            hmm_path = "models/qwen3/hmm_xh2_qwen3_8b_256_2k_4batch_2cores_20250912.zip"

    if model_type in ["raw"]:
        ignore_patterns = []
        get_file_from_jfrog(wiki_path, model_dir, HOUMO_DATASETS_PATH)
    else:
        ignore_patterns = ["*.safetensors"]

    from modelscope import snapshot_download

    snapshot_download(
        'qwen/qwen3-8b',
        local_dir=f'{model_dir}/qwen3-8b',
        ignore_patterns=ignore_patterns,
    )

    if model_type in ["hmm"] and not get_file_from_jfrog(
        hmm_path, model_dir, build_model_dir
    ):
        sys.exit(1)
