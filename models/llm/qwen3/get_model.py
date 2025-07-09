import os
import onnx
import argparse
from hmassist.utils.utils import get_file_from_jfrog

HOUMO_TARGET = os.getenv('HOUMO_TARGET', 'houmo')


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='model_type',
        type=str,
        default='quant',
        help='which resource to get, choise in [raw, quant, hmm, all]',
    )
    parser.add_argument(
        '--quant_model_dir',
        dest='quant_model_dir',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, 'hmquant'),
        help='where to save quant_model',
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
    wiki_path = "models/datasets/wikitext-2-raw-v1.zip"
    if HOUMO_TARGET == "xh1":
        quant_path = "models/qwen3/hmquant_qwen3_256_8k_20250603.zip"
        hmm_path = "models/qwen3/hmm_qwen3_256_8k_4cores_20250603.zip"
    elif HOUMO_TARGET == "xh2":
        quant_path = "models/qwen3/hmquant_xh2_qwen3_8b_256_2k_20250625.zip"
        hmm_path = "models/qwen3/hmm_xh2_qwen3_8b_256_8k_2cores_20250701.zip"

    if model_type in ["raw", "all"]:
        get_file_from_jfrog(wiki_path, model_dir, HOUMO_DATASETS_PATH)
        ignore_patterns = []
    else:
        ignore_patterns = ["*.safetensors"]

    from modelscope import snapshot_download
    snapshot_download('qwen/qwen3-8b', local_dir='qwen3-8b', ignore_patterns=ignore_patterns)

    if model_type in ["quant", "all"]:
        get_file_from_jfrog(quant_path, model_dir, quant_model_dir)

    if model_type in ["hmm", "all"]:
        get_file_from_jfrog(hmm_path, model_dir, os.path.join('output', HOUMO_TARGET))
