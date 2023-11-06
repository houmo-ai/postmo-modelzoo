import os

if __name__ == '__main__':
    if not os.path.exists("apollo_lane_1536x512.onnx"):
        os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/lane/apollo_lane_1536x512.onnx')
    if not os.path.exists("hmquant_lane_512x1536_with_act.onnx"):
        os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/lane/hmquant_lane_512x1536_with_act.onnx')