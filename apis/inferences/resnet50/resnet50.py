import os
import sys
import numpy as np
import cv2
import torch

import tcim_lite as tcim


if __name__ == '__main__':
    sys.path.insert(0, "../../common/python")
    print("\n===> resnet50 python example start...")
    houmo_target = os.getenv("HOUMO_TARGET", "houmo")
    print(f"tcim runtime version: {tcim.runtime.get_version()}, houmo target: {houmo_target}")

    # 1. load model
    model_path = "./resnet50.hmm"
    if houmo_target == "xh2":
        model_path = "./resnet50_xh2_b1_1core.hmm"
    module = tcim.runtime.load(model_path)

    # 2. preprocess
    input_data = cv2.imread("../../data/snake.jpg")
    if houmo_target == "xh1":
        input_data = cv2.resize(input_data, (224, 224))  # HWC uint8
        input_data = np.transpose(input_data, (2, 0, 1))  # CHW uint8
        input_data = np.expand_dims(input_data, axis=0)  # NCHW uint8
        input_data = torch.tensor(input_data.astype(np.float32))  # NHWC float32
        input_data = torch.squeeze(input_data, 0)  # HWC float32
        from transform import BGR2YUV
        rgb2yuv_func = BGR2YUV(fmt='YUV420')
        input_data = torch.unsqueeze(rgb2yuv_func(input_data), 0).numpy()  # NHWC float32
        input_data = input_data.astype(np.uint8)
    elif houmo_target == "xh2":
        image_rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (224, 224))  # HWC uint8
        mean_arr = np.array([123.675, 116.28, 103.53])
        std_arr = np.array([58.395, 57.12, 57.375])
        image_norm = (image_rgb - mean_arr) / std_arr
        image_norm = np.transpose(image_norm, (2, 0, 1))  # CHW uint8
        image_norm = np.expand_dims(image_norm, axis=0)  # NCHW uint8
        input_data = image_norm.astype(np.float16)

    # 3. set input
    input_num = module.get_num_inputs()
    for id in range(0, input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name).ascontiguous()
        print("input[{}] shape = {}, dtype = {}, format = {}".format(input_name, input_info.shape, input_info.dtype,
                                                                     input_info.format.name))
        module.set_input(input_name, input_data)

    # 4. run & sync
    module.run()
    module.sync()

    # 5. get output
    result_check = True
    output_num = module.get_num_outputs()
    for id in range(0, output_num):
        output_name = module.get_output_name(id)
        output_info = module.get_output_info(output_name).ascontiguous().astype(np.float32)
        print("output[{}] shape = {}, dtype = {}, format = {}".format(output_name, output_info.shape, output_info.dtype,
                                                                      output_info.format.name))
        output_data = module.get_output(output_name).astype(np.float32).numpy()

    # 6. postprocess
    from postprocess import softmax
    output_data = softmax(output_data)
    topk = 5
    pred_list = np.argsort(-output_data, axis=1, kind="quicksort").flatten()[0:topk]
    prob_list = output_data.flatten()
    for i, id in enumerate(pred_list):
        print("top{}: predict cls = {}, prob = {:.6f}".format(i+1, id, prob_list[id]))
    # check result, modify it when you change model or data
    assert(pred_list[0] == 65)

    print("<=== resnet50 python example completed.\n")
