import os
import onnx
import argparse

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
    raw_name = "yolop_384x640.onnx"
    quant_name = "hmquant_yolop_20240305.zip"

    if model_type == "raw" or model_type == "all":
        if not os.path.exists(raw_name):
            url = os.path.join(os.environ.get("MODELZOO_URL"), "models/yolop", raw_name)
            os.system('wget ' + url)

    if model_type == "quant" or model_type == "all":
        if not os.path.exists(os.path.join(quant_model_dir, "hmquant_yolop_with_act.onnx")):
            if not os.path.exists(quant_name):
                url = os.path.join(os.environ.get("MODELZOO_URL"), "models/yolop", quant_name)
                os.system('wget ' + url)
            os.system('mkdir -p ' + quant_model_dir)
            os.system('unzip -d ' + quant_model_dir + ' ' + quant_name)