import os
import onnx
import argparse
from hmatc.utils.utils import get_file_from_jfrog


HOUMO_TARGET = os.getenv('HOUMO_TARGET')
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


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
    HOUMO_MODEL_PATH = os.getenv('HOUMO_MODEL_PATH', '.')
    if HOUMO_TARGET == "xh2":
        hmm_path = "models/gte/hmm_xh2_gte_1.5b_256_2k_2cores_20251104.zip"

    from modelscope import snapshot_download
    snapshot_download('iic/gte-Qwen2-1.5B-instruct',
                      local_dir=f'{model_dir}/gte-Qwen2-1.5B-instruct',
                      ignore_patterns=["*.safetensors"])

    print("model_type:", model_type)
    if model_type == "hmm":
        try:
            get_file_from_jfrog(hmm_path, model_dir, os.path.join('output', HOUMO_TARGET))
        except Exception as e:
            print(f"Model doesn't exist, error msg: {e}")
