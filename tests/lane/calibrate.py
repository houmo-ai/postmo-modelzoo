import os
import time

import onnx
import cv2
import torch
import torchvision.transforms as transforms
from hmquant.api import quant_single_onnx_network, generate_golden
from hmquant.tools.dataset.preprocess.transform import BGR2YUV
from hmquant.tools.dataset.preprocess.transform import ToTensorNotNormal
from torchvision.datasets.folder import pil_loader
from hmquant.configs.api_config import *


def calibrate():
    env_dict = os.environ
    onnx_model_path = 'apollo_lane_1536x512.onnx'

    calib_dir = './lane_data'
    calib_dataset = []
    filelist = os.listdir(calib_dir)
    for filename in filelist:
        _, ext = os.path.splitext(filename)
        if ext in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP", ".bin"]:
            img_name = os.path.join(calib_dir, filename)
            image = cv2.imread(img_name)
            image_resize = cv2.resize(image, dsize=[1536, 864])
            img_crop = image_resize[352:864, :, :]
            data_tem = torch.tensor(img_crop.transpose([2,0,1])).unsqueeze(0)*1.0
            calib_dataset.append(data_tem)

    quanttool_config = {
        'inputs_cfg': {
            'ALL': {
                'data_format': 'BGR',
                'first_layer_weight_denorm_mean': [0.37254901960784315, 0.38823529411764707, 0.3764705882352941],
                'first_layer_weight_denorm_std': [0.00392156862745098,0.00392156862745098, 0.00392156862745098],
                'resizer_crop': {'top': 0, 'left': 0, 'height': 512, 'width': 1536},
                'resizer_resize': {
                    'height': 512,
                    'width': 1536,
                    'align_corners': False,
                    'method': 'bilinear',
                },
                'toYUV_format': 'YUV422',
                'insert_pad_scatter' : False,
                'dynamic_crop' : False,
            },
        },
        'graph_opt_cfg': {},
    }

    onnx_input = [{"data": calib_dataset[0].numpy().astype("float32")}]

    t0 = time.time()
    print("====> start to quant onnx network...", flush=True)
    sequencer = quant_single_onnx_network(
        quanttool_config,
        calib_dataset,
        onnx_model_path,
        device='cpu',
    )

    #sequencer.save_onnx(
    #    'quant_lane.onnx',
    #    save_special_onnx=True
    #)
    #
    #crop_data = torch.zeros(
    #[
    #    16,
    #],
    #dtype=torch.int32,
    #)

    onnx_generate_dataset = {
        'data': calib_dataset[0]
    }
    
    golden_input, golden_inter, golden_onnx = generate_golden(
        sequencer,
        calibset=onnx_generate_dataset,
        save_path="./output/H30/result",
        model_name='lane',
    )


if __name__ == '__main__':
    calibrate()
