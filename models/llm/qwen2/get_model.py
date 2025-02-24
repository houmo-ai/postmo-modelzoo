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
        default='all',
        help='which resource to get, choise in [dataset, raw, quant, hmm, all]',
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
    DATASETS_PATH = os.getenv('DATASETS_PATH', '.')
    MODEL_PATH = os.getenv('MODEL_PATH', '.')
    wiki_path = "models/datasets/wikitext-2-raw-v1.zip"
    quant_path = "models/qwen2/hmquant_qwen2_256_4096_20250221.zip"
    hmm_path = "models/qwen2/hmm_qwen2_256_4096_20250222.zip"

    if model_type == "dataset" or model_type == "all":
        get_file_from_jfrog(wiki_path, model_dir, DATASETS_PATH)

    if model_type == "raw" or model_type == "all":
        from modelscope import snapshot_download
        snapshot_download('qwen/qwen2-7b-instruct', local_dir='qwen2-7b-instruct-hf')

    if model_type == "quant" or model_type == "all":
        get_file_from_jfrog(quant_path, model_dir, quant_model_dir)

    if model_type == "hmm" or model_type == "all":
        get_file_from_jfrog(hmm_path, model_dir, os.path.join('output', HOUMO_TARGET))
