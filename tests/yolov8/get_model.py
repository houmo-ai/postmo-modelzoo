import os

if __name__ == '__main__':
    if not os.path.exists("hmquant_yolov8_with_act.onnx"):
        os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/yolov8/hmquant_yolov8_with_act.onnx')