import os
import sys
import onnx
import argparse

HOUMO_EXAMPLES_PATH = os.environ.get('HOUMO_EXAMPLES_PATH', '..')
sys.path.append(f'{HOUMO_EXAMPLES_PATH}/common/python')
from utils import get_file_from_jfrog


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ncore',
        dest='ncore',
        type=int,
        default=4,
        help='which resource to get, choise in [2, 4]',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if "HOUMO_MODELZOO_URL" not in os.environ:
        os.environ["HOUMO_MODELZOO_URL"] = "http://139.224.0.199:8082/artifactory/houmo/release"
    HOUMO_TARGET = os.environ.get('HOUMO_TARGET', 'houmo')
    model_dir = os.path.join(HOUMO_EXAMPLES_PATH, "models")
    hmm_path = "models/qwen3/hmm_qwen3_256_8k_"+ str(args.ncore) +"cores_20250603.zip"

    from modelscope import snapshot_download
    snapshot_download('qwen/qwen3-8b', local_dir='qwen3-8b', ignore_patterns=["*.safetensors"])

    get_file_from_jfrog(hmm_path, model_dir, "./")
