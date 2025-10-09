import os
import onnx
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
    model_type = args.model_type
    model_dir = args.model_dir
    HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '.')
    HOUMO_MODEL_PATH = os.getenv('HOUMO_MODEL_PATH', '.')
    wiki_path = "models/datasets/wikitext-2-raw-v1.zip"
    if HOUMO_TARGET == "xh1":
        hmm_path = "models/qwen2.5/hmm_qwen2.5_256_8k_4cores_20250522.zip"
    elif HOUMO_TARGET == "xh2":
        hmm_path = "models/qwen2.5/hmm_xh2_qwen2.5_256_4k_2cores_20250611.zip"

    if model_type in ["raw"]:
        ignore_patterns = []
        try:
            get_file_from_jfrog(wiki_path, model_dir, HOUMO_DATASETS_PATH)
        except Exception as e:
            print(f"Model doesn't exist, error msg: {e}")
    else:
        ignore_patterns = ["*.safetensors"]

    from modelscope import snapshot_download
    snapshot_download('qwen/qwen2.5-7b-instruct',
                      local_dir='qwen2.5-7b-instruct-hf',
                      ignore_patterns=ignore_patterns)

    if model_type in ["hmm"]:
        try:
            get_file_from_jfrog(hmm_path, model_dir, os.path.join('output', HOUMO_TARGET))
        except Exception as e:
            print(f"Model doesn't exist, error msg: {e}")
