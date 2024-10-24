import os
import onnx
import argparse
from hmassist.utils.utils import get_file_from_jfrog

DATASETS_PATH = os.getenv('DATASETS_PATH', '')

def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='model_type',
        type=str,
        default='all',
        help='which model type to get, choise in [raw, quant, all]',
    )
    parser.add_argument(
        '--quant_model_dir',
        dest='quant_model_dir',
        type=str,
        default=os.path.join('output', os.getenv('HOUMO_TARGET', ''), 'result'),
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
    # raw_path = "models/qwen2/qwen2.onnx"
    wiki_path = "models/datasets/wikitext-2-raw-v1.zip"
    quant_path = "models/qwen2/hmquant_qwen2_128_4096_20241028.zip"

    if model_type == "raw" or model_type == "all":
        file_path = get_file_from_jfrog(wiki_path, model_dir)
        os.system(f'unzip -o -d {DATASETS_PATH} {file_path}')
        # get_file_from_jfrog(raw_path, model_dir)
        os.system('huggingface-cli download --resume-download Qwen/Qwen2-7B-Instruct --local-dir qwen2-7b-instruct-hf')

    if model_type == "quant" or model_type == "all":
        file_path = get_file_from_jfrog(quant_path, model_dir)
        os.system('mkdir -p ' + quant_model_dir)
        os.system('unzip -o -d ' + quant_model_dir + ' ' + file_path)
