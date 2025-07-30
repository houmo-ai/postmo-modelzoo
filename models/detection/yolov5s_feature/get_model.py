import os
import onnx
import argparse
from hmtool.utils.utils import get_file_from_jfrog


HOUMO_TARGET = os.getenv('HOUMO_TARGET', 'xh1')
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='model_type',
        type=str,
        default='raw',
        help='which model type to get, choise in [raw, quant, build, all]',
    )
    parser.add_argument(
        '--quant_model_dir',
        dest='quant_model_dir',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, 'hmquant'),
        help='where to save quant_model',
    )
    parser.add_argument(
        '--build_model_dir',
        dest='build_model_dir',
        type=str,
        default=os.path.join('output', HOUMO_TARGET),
        help='where to save build_model',
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
    build_model_dir = args.build_model_dir
    model_type = args.model_type
    model_dir = args.model_dir

    model_name = "yolov5s_feature"
    ncore = 1
    batch = 1
    opt_level = "O2"
    version = "v2.4.2"
    target = HOUMO_TARGET
    raw_path = f"models/yolov5s/yolov5s_640x640.onnx"
    quant_path = f"models/{model_name}/hmquant_{model_name}_{target}_{version}.tar.xz"
    build_path = f"models/{model_name}/{model_name}_{target}_b{batch}_{ncore}core_{opt_level}_{version}.tar.xz"

    if model_type == "raw" or model_type == "all":
        file_path = get_file_from_jfrog(raw_path, model_dir)
        extract_path = os.path.join(os.path.dirname(file_path), "yolov5s_640x640_clip.onnx")
        onnx.utils.extract_model(file_path, extract_path, input_names=['images'], 
            output_names=['340', '378', '416'], check_model=True)

    if model_type == "quant" or model_type == "all":
        get_file_from_jfrog(quant_path, model_dir, quant_model_dir)

    if model_type == "build" or model_type == "all":
        get_file_from_jfrog(build_path, model_dir, build_model_dir)
