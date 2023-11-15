import os

if __name__ == '__main__':
    if not os.path.exists("apollo_lane_1536x512.onnx"):
        os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/lane/apollo_lane_1536x512.onnx')
    if not os.path.exists("lane_golden.zip"):
        os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/lane/lane_golden.zip')
        os.system('unzip -d output/H30/result lane_golden.zip')