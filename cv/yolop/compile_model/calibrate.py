import os

import onnx
import torch
import torchvision.transforms as transforms
from hmquant.api import quant_single_input_onnx_network
from hmquant.api import quant_single_onnx_network
from hmquant.configs.api_config import BaseHoumoConfig
from hmquant.tools.dataset.preprocess.transform import RGB2YUV
from hmquant.tools.dataset.preprocess.transform import ToTensorNotNormal
from torchvision.datasets.folder import pil_loader


def calibrate():
    env_dict = os.environ
    onnx_model_path = os.path.join(
        env_dict.get('MODEL_PATH'), 'yolop.onnx',
    )

    calib_dataset = [torch.randint(0, 200, (1, 3, 192, 320)).float()]

    quanttool_config = {
        'inputs_cfg': {
            'ALL': {
                'data_format': 'RGB',
                'first_layer_weight_denorm_mean': [0.485, 0.456, 0.406],
                'first_layer_weight_denorm_std': [0.229, 0.224, 0.225],
                'resizer_crop': {'top': 0, 'left': 0, 'height': 192, 'width': 320},
                'resizer_resize': {
                    'height': 192,
                    'width': 320,
                    'align_corners': False,
                    'method': 'bilinear',
                },
                'toYUV_format': 'YUV422',
            },
        },
        'graph_opt_cfg': {},
    }
    sequencer = quant_single_onnx_network(
        quanttool_config, calib_dataset, onnx_model_path, device='cpu',
    )

    sequencer.save_onnx(
        'quant_yolop.onnx',
        save_out_tensor=False,
        save_params_npy=True,
        save_special_onnx=True,
    )


if __name__ == '__main__':
    calibrate()
