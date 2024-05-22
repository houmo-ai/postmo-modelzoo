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
        default='output/H30/result',
        help='where to save quant_model',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    quant_model_dir = args.quant_model_dir
    model_type = args.model_type
    raw_path = "models/yolop/yolop_384x640.onnx"
    quant_path = "models/yolop/hmquant_yolop_20240305.zip"

    if model_type == "raw" or model_type == "all":
        get_file_from_jfrog(raw_path)

    if model_type == "quant" or model_type == "all":
        get_file_from_jfrog(quant_path)
        quant_name = os.path.basename(quant_path)
        os.system('mkdir -p ' + quant_model_dir)
        os.system('unzip -o -d ' + quant_model_dir + ' ' + quant_name)