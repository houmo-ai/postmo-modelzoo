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
    model_type = args.model_type
    if model_type == "raw" or model_type == "all":
        if not os.path.exists("yolov8m_640x640.onnx"):
            os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/yolov8/yolov8m_640x640.onnx')
        if not os.path.exists("yolov8m_640x640_clip.onnx"):
            onnx.utils.extract_model("yolov8m_640x640.onnx", "yolov8m_640x640_clip.onnx", input_names=['images'], 
                output_names=['/model.22/dfl/Reshape_1_output_0', '/model.22/Sigmoid_output_0'], check_model=True)
        if not os.path.exists("hm_yolov8m_736x736.onnx"):
            os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/yolov8/hm_yolov8m_736x736.onnx')
        if not os.path.exists("hm_yolov8m_736x736_clip.onnx"):
            onnx.utils.extract_model("hm_yolov8m_736x736.onnx", "hm_yolov8m_736x736_clip.onnx", input_names=['images'], 
                output_names=['onnx::Split_504', 'onnx::Concat_518'], check_model=True)
        if not os.path.exists("hm_yolov8l_640x640.onnx"):
            os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/yolov8/hm_yolov8l_640x640.onnx')
        if not os.path.exists("hm_yolov8l_640x640_clip.onnx"):
            onnx.utils.extract_model("hm_yolov8l_640x640.onnx", "hm_yolov8l_640x640_clip.onnx", input_names=['images'], 
                output_names=['/model.22/dfl/Reshape_1_output_0', '/model.22/Sigmoid_output_0'], check_model=True)