import os
import onnx

if __name__ == '__main__':
    if not os.path.exists("ultimateDO_fp16_fuse.onnx"):
        os.system('wget http://10.10.1.53:8082/artifactory/aisolution/ai_models/ultimateDO/ultimateDO_fp16_fuse.onnx')
    # onnx.utils.extract_model("ultimateDO_fp16_fuse.onnx", "yolov5s_clip.onnx", input_names=['images'], output_names=['340', '378', '416'], check_model=True)