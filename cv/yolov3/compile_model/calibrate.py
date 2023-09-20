import os
import sys

import numpy as np
import onnx
import torch
import torchvision.transforms as transforms
from hmquant.api import quant_single_onnx_network
from hmquant.tools.dataset.preprocess.transform import ToTensorNotNormal
from preprocess import HmYuvInt8YoloV3
from torchvision.datasets.folder import pil_loader


def calibrate():
    env_dict = os.environ
    onnx_model_path = os.path.join(env_dict.get('MODEL_PATH'), 'yolov3.onnx')

    def unsqueeze(x):
        x = torch.tensor(x)
        return torch.unsqueeze(x, 0).float()
    height, width = (416, 416)

    calib_transform = transforms.Compose(
        [
            HmYuvInt8YoloV3(image_size=(height, width)),
            unsqueeze,
        ],
    )
    image_root = os.path.join(env_dict.get('DATASETS_PATH'), 'COCO', 'val2017')
    calib_image_files = [
        '000000000139.jpg',
        '000000000285.jpg',
        '000000000632.jpg',
        '000000000724.jpg',
        '000000000776.jpg',
        '000000000785.jpg',
        '000000000802.jpg',
        '000000000872.jpg',
        '000000000885.jpg',
        '000000001000.jpg',
        '000000001268.jpg',
        '000000001296.jpg',
        '000000001353.jpg',
        '000000001425.jpg',
        '000000001490.jpg',
        '000000001503.jpg',
        '000000001532.jpg',
        '000000001584.jpg',
        '000000001675.jpg',
        '000000001761.jpg',
    ]
    calib_images = [
        pil_loader(os.path.join(image_root, img_path))
        for img_path in calib_image_files
    ]
    calib_dataset = [calib_transform(data) for data in calib_images]

    quanttool_config = {
        'inputs_cfg': {
            'ALL': {
                'data_format': 'RGB',
                'first_layer_weight_denorm_mean': [0, 0, 0],
                'first_layer_weight_denorm_std': [1, 1, 1],
                'resizer_crop': {'top': 0, 'left': 0, 'height': 0, 'width': 0},
                'resizer_resize': {
                    'height': height,
                    'width': width,
                    'align_corners': False,
                    'method': 'bilinear',
                },
                'toYUV_format': 'YUV422',
            },
        },
        'graph_opt_cfg': {
            'auto_quant_flag': True,
        },
    }
    sequencer = quant_single_onnx_network(
        quanttool_config, calib_dataset, onnx_model_path, device='cpu',
    )

    sequencer.save_onnx(
        'quant_yolov3.onnx',
        save_out_tensor=False,
        save_params_npy=True,
        save_special_onnx=True
    )


if __name__ == '__main__':
    calibrate()
