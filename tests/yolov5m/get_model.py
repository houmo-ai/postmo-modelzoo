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
        help='model_type to get',
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.model_type == "raw" or args.model_type == "all":
        if not os.path.exists("yolov5m.onnx"):
            pass
            # os.system('wget http://10.10.1.53:8082/artifactory/toolchain/release/models/resnet50/resnet50.onnx')

    if args.model_type == "quant" or args.model_type == "all":
        if not os.path.exists("output/H30/result/hmquant_yolov5m_with_act.onnx"):
            if not os.path.exists("hmquant_yolov5m_20231219.zip"):
                os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/yolov5m/hmquant_yolov5m_20231219.zip')
            os.system('mkdir -p output/H30/result')
            os.system('unzip -d output/H30/result hmquant_yolov5m_20231219.zip')

    # onnx.utils.extract_model("output/H30/result/hmquant_yolov5m_with_act.onnx", "output/H30/result/hmquant_yolov5m_with_act.onnx", input_names = ['images'], output_names=['onnx::Split_471', 'onnx::Split_526', 'onnx::Split_581'], check_model=True)