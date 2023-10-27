import os

if __name__ == '__main__':
    if not os.path.exists("maskrcnn.onnx"):
        os.system('wget http://10.10.1.53:8082/artifactory/model_zoo/maskrcnn/maskrcnn.onnx')
