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
    quant_model_dir = args.quant_model_dir
    model_type = args.model_type
    model_dir = args.model_dir
    HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '.')
    HOUMO_MODEL_PATH = os.getenv('HOUMO_MODEL_PATH', '.')
    if HOUMO_TARGET != "xh2":
        print("Error: not support houmo target", HOUMO_TARGET)
        sys.exit(-1)
    wiki_path = "models/datasets/wikitext-2-raw-v1.zip"
    hmm_path = "models/deepseek/hmm_xh2_deepseek_8b_256_2k_2cores_20250702.zip"

    if model_type in ["raw"]:
        get_file_from_jfrog(wiki_path, model_dir, HOUMO_DATASETS_PATH)
        ignore_patterns = []
    else:
        ignore_patterns = ["*.safetensors"]
    from modelscope import snapshot_download
    snapshot_download('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
                      local_dir='DeepSeek-R1-0528-Qwen3-8B',
                      ignore_patterns=ignore_patterns)

    if model_type in ["hmm"]:
        get_file_from_jfrog(hmm_path, model_dir, os.path.join('output', HOUMO_TARGET))
