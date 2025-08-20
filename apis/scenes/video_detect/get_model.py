import os
import sys
import argparse

HOUMO_EXAMPLES_PATH = os.environ.get('HOUMO_EXAMPLES_PATH', '../..')
sys.path.append(f'{HOUMO_EXAMPLES_PATH}/apis/common/python')
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
    HOUMO_TARGET = os.environ.get('HOUMO_TARGET', 'houmo')
    model_dir = (
        os.path.join(HOUMO_EXAMPLES_PATH, "apis/models")
        if not args.model_dir
        else args.model_dir
    )
    hmm_path = "models/yolov5s/hmm_yolov5s_1080x1920_resize_20250304.zip"
    get_file_from_jfrog(hmm_path, model_dir, "./")
    hmm_path = "models/resnet50/hmm_resnet50_1080x1920_resize_20250304.zip"
    get_file_from_jfrog(hmm_path, model_dir, "./")
