import os
import onnx

if __name__ == '__main__':
    if not os.path.exists("yolov5s_clip.onnx"):
        if not os.path.exists("yolov5s.onnx"):
            os.system('wget http://10.10.1.53:8082/artifactory/toolchain/release/models/yolov5/yolov5s.onnx')
        onnx.utils.extract_model("yolov5s.onnx", "yolov5s_clip.onnx", input_names=['images'], 
            output_names=['340', '378', '416'], check_model=True)