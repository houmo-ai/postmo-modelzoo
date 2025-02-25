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
        default='raw',
        help='which model type to get, choise in [raw, quant, all]',
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
    raw_path = "models/resnet50/resnet50.onnx"
    quant_path = "models/resnet50/hmquant_resnet50_20240827.zip"
    hmm_path = "models/resnet50/hmm_resnet50_20241112.zip"

    if model_type == "raw" or model_type == "all":
        get_file_from_jfrog(raw_path, model_dir)

    if model_type == "quant" or model_type == "all":
        get_file_from_jfrog(quant_path, model_dir, quant_model_dir)
