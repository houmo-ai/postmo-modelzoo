import os
import onnx
import argparse
from hmassist.utils.utils import get_file_from_jfrog

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
    quant_path = "models/qwen/hmquant_qwen_20240709.zip"
    weight_path = "models/qwen/hmquant_qwen_weight_20240709.zip"
    hmm_path = "models/qwen/hmm_qwen_256_2048_20241105.zip"

    if model_type == "raw" or model_type == "all":
        print("no raw model is available.")

    if model_type == "quant" or model_type == "all":
        get_file_from_jfrog(quant_path, model_dir, quant_model_dir)
        get_file_from_jfrog(weight_path, model_dir, quant_model_dir)

    if model_type == "hmm" or model_type == "all":
        get_file_from_jfrog(hmm_path, model_dir, ".")
