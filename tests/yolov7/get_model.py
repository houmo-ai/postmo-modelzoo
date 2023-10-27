import os

if __name__ == '__main__':
    if not os.path.exists("yolov7.onnx"):
        os.system('wget http://10.10.1.53:8082/artifactory/model_zoo/yolov7/yolov7.onnx')
