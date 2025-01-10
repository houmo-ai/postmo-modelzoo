import os
import sys
import numpy as np
import cv2
import torch

import tcim_lite as tcim


if __name__ == '__main__':
    sys.path.insert(0, "../common/python")
    print("\n===> resnet50 python example start...")
    print("tcim runtime version: {}".format(tcim.runtime.get_version()))

    # 1. load model
    module = tcim.runtime.load("resnet50.hmm")

    # 2. preprocess
    input_data = cv2.imread("../data/snake.jpg")
    input_data = cv2.resize(input_data, (224, 224))  # HWC uint8
    input_data = np.transpose(input_data, (2, 0, 1))  # CHW uint8
    input_data = np.expand_dims(input_data, axis=0)  # NCHW uint8
    input_data = torch.tensor(input_data.astype(np.float32))  # NHWC float32
    input_data = torch.squeeze(input_data, 0)  # HWC float32
    from transform import BGR2YUV
    rgb2yuv_func = BGR2YUV(fmt='YUV420')
    input_data = torch.unsqueeze(rgb2yuv_func(input_data), 0).numpy()  # NHWC float32
    input_data = input_data.astype(np.uint8)

    # 3. set input
    input_num = module.get_num_inputs()
    for id in range(0, input_num):
        input_name = module.get_input_name(id)
        if input_name.endswith(".y"):
            input_name, _ = input_name.split(".y")
        elif input_name.endswith(".uv"):
            continue
        input_info = module.get_input_info(input_name)
        print("input[{}] shape = {}, dtype = {}, format = {}".format(input_name, input_info.shape, input_info.dtype,
                                                                     input_info.format.name))
        module.set_input(input_name, tcim.runtime.Tensor(input_info, input_data))

    # 4. run & sync
    module.run()
    module.sync()

    # 5. get output
    result_check = True
    output_num = module.get_num_outputs()
    for id in range(0, output_num):
        output_name = module.get_output_name(id)
        output_info = module.get_output_info(output_name).astype(np.float32)
        print("output[{}] shape = {}, dtype = {}, format = {}".format(output_name, output_info.shape, output_info.dtype,
                                                                      output_info.format.name))
        output_data = module.get_output(output_name).cast().numpy()

    # 6. postprocess
    from postprocess import softmax
    output_data = softmax(output_data)
    topk = 5
    pred_list = np.argsort(-output_data, axis=1, kind="quicksort").flatten()[0:topk]
    prob_list = output_data.flatten()
    for i, id in enumerate(pred_list):
        print("top{}: predict cls = {}, prob = {:.6f}".format(i+1, id, prob_list[id]))

    assert(pred_list[0] == 65)

    print("<=== resnet50 python example completed.\n")
