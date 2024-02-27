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
    quant_model_pack = 'hmquant_pointpillars_20240222.zip'
    model_type = args.model_type
    if model_type == "raw" or model_type == "all":
        if not os.path.exists("hm_yolov8m-seg_480x480.onnx"):
            os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/yolov8/hm_yolov8m-seg_480x480.onnx')
        if not os.path.exists("hm_yolov8m-seg_480x480_clip.onnx"):
            onnx.utils.extract_model("hm_yolov8m-seg_480x480.onnx", "hm_yolov8m-seg_480x480_clip.onnx", input_names=['images'], 
                output_names=['onnx::Shape_627', 'onnx::Concat_653', 'onnx::Concat_515', 'output1'], check_model=True)