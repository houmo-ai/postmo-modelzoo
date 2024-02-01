import os
import onnx

if __name__ == '__main__':
    if not os.path.exists("yolov3_clip.onnx"):
        if not os.path.exists("yolov3_416x416.onnx"):
            os.system('wget http://10.10.1.53:8082/artifactory/model_zoo2/houmo/yolov3/yolov3_416x416.onnx')
        onnx.utils.extract_model("yolov3_416x416.onnx", "yolov3_clip.onnx", input_names=['images'], 
            output_names=['onnx::Slice_497', 'onnx::Slice_431', 'onnx::Slice_557'], check_model=True)