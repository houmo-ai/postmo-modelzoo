import os
import sys
import argparse

HOUMO_EXAMPLES_PATH = os.environ.get('HOUMO_EXAMPLES_PATH', '../..')
sys.path.append(f'{HOUMO_EXAMPLES_PATH}/common/python')
from utils import get_file_from_jfrog


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default='',
        help='where to save downloaded model',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if "HOUMO_MODELZOO_URL" not in os.environ:
        os.environ["HOUMO_MODELZOO_URL"] = (
            "http://139.224.0.199:8082/artifactory/houmo/release"
        )
    HOUMO_TARGET = os.getenv("HOUMO_TARGET", "houmo")

    model_dir = (
        os.path.join(HOUMO_EXAMPLES_PATH, "models")
        if not args.model_dir
        else args.model_dir
    )
    if HOUMO_TARGET == "xh1":
        hmm_path = "models/resnet50/hmm_resnet50_20250113.zip"
    elif HOUMO_TARGET == "xh2":
        hmm_path = "models/resnet50/hmm_resnet50_xh2_b1_1core_20250804.zip"

    get_file_from_jfrog(hmm_path, model_dir, "./")
