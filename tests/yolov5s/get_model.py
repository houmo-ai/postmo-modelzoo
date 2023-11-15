import os
import onnx

if __name__ == '__main__':
    if not os.path.exists("yolov5s_aiin.onnx"):
        os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/yolov5/yolov5s_aiin.onnx')
    
    onnx.utils.extract_model("yolov5s_aiin.onnx", "yolov5s_aiin_clip.onnx", input_names=['images'], 
        output_names=['/model.24/Sigmoid_output_0', '/model.24/Sigmoid_1_output_0', '/model.24/Sigmoid_2_output_0'], check_model=True)