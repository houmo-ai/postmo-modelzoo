import os
import sys
import argparse
from hmatc.utils.utils import get_file_from_jfrog

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh1", "xh2"], "Only support HOUMO_TARGET: xh1 or xh2."
HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '.')


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='model_type',
        type=str,
        default='raw',
        help='which model type to get, choise in [raw]',
    )
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default='.',
        help='where to save downloaded model',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    model_type = args.model_type
    model_dir = args.model_dir
    raw_path = "http://10.10.1.53:8082/artifactory/toolchain/support/custom/saimo/paddleocr_rec-sim.onnx"
    data_path = "http://10.10.1.53:8082/artifactory/toolchain/support/custom/saimo/CCPD2020_PPOCRv3_eval.tar.gz"

    if model_type in ["raw"] and not get_file_from_jfrog(raw_path, model_dir):
        sys.exit(1)

    if not os.path.exists(os.path.join(HOUMO_DATASETS_PATH, "CCPD2020_PPOCRv3_eval")):
        get_file_from_jfrog(data_path, HOUMO_DATASETS_PATH, HOUMO_DATASETS_PATH)
