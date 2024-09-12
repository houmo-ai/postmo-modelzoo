import numpy as np
import cv2
import torch
import tcim


def softmax(x, axis=1):
    """
    :param x: input array
    :param axis: softmax axis
    :return: result of softmax
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


if __name__ == '__main__':
    print("resnet50 demo start...")
    print("tcim runtime version: {}".format(tcim.runtime.get_version()))

    # 1. load model
    module = tcim.runtime.load("resnet50.hmm")

    # 2. preprocess
    input_data = cv2.imread("../../../data/datasets/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_00000003.JPEG")
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    input_data = cv2.resize(input_data, (224, 224))  # HWC uint8
    input_data = np.transpose(input_data, (2, 0, 1))  # CHW uint8
    input_data = np.expand_dims(input_data, axis=0)  # NCHW uint8
    input_data = torch.tensor(input_data.astype(np.float32))  # NHWC float32
    input_data = torch.squeeze(input_data, 0)  # HWC float32
    from hmassist.utils.transform import BGR2YUV
    rgb2yuv_func = BGR2YUV(fmt='YUV422')
    input_data = torch.unsqueeze(rgb2yuv_func(input_data), 0).numpy()  # NHWC float32

    # 3. set input
    input_num = module.get_num_inputs()
    for id in range(0, input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name)
        print("input[{}] shape = {}, dtype = {}, format = {}".format(input_name, input_info.shape, input_info.dtype,
                                                                     input_info.format.name))
        module.set_input(input_name, input_data.astype(input_info.dtype))

    # 4. run
    module.run()

    # 5. get output
    result_check = True
    output_num = module.get_num_outputs()
    for id in range(0, output_num):
        output_name = module.get_output_name(id)
        output_info = module.get_output_info(output_name, is_quanted=False)
        print("output[{}] shape = {}, dtype = {}, format = {}".format(output_name, output_info.shape, output_info.dtype,
                                                                      output_info.format.name))
        output_data = module.get_output(output_name, is_quanted=False)

    # 6. postprocess
    output_data = softmax(output_data)
    topk = 5
    pred_list = np.argsort(-output_data, axis=1, kind="quicksort").flatten()[0:topk]
    prob_list = output_data.flatten()
    for i, id in enumerate(pred_list):
        print("top {}: predict cls = {}, prob = {:.6f}".format(i+1, id, prob_list[id]))

    expected = 230
    if (pred_list[0] != expected):
        print("[error] predict result != {}".format(expected))
        exit(-1)

    print("resnet50 demo completed.")