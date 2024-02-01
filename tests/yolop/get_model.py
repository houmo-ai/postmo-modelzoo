import os

if __name__ == '__main__':
    if not os.path.exists("yolop.onnx"):
        os.system('wget http://10.10.1.53:8082/artifactory/toolchain/release/models/yolop/yolop.onnx')
