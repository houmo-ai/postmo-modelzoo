import os
import onnx
import argparse
from hmatc.utils.utils import get_file_from_jfrog


HOUMO_TARGET = os.getenv('HOUMO_TARGET', 'xh1')
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


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
    quant_path = "models/deepseek/hmquant_deepseek_256_8192_20250322.zip"
    hmm_path = "models/deepseek/hmm_deepseek_256_8192_4cores_20250322.zip"

    if model_type == "raw" or model_type == "all":
        get_file_from_jfrog(wiki_path, model_dir, HOUMO_DATASETS_PATH)
        from modelscope import snapshot_download
        snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', local_dir='DeepSeek-R1-Distill-Qwen-7B')

    if model_type == "quant" or model_type == "all":
        get_file_from_jfrog(quant_path, model_dir, quant_model_dir)

    if model_type == "hmm" or model_type == "all":
        get_file_from_jfrog(hmm_path, model_dir, os.path.join('output', HOUMO_TARGET))
