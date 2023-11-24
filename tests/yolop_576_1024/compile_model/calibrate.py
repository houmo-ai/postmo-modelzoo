import os

import torch
from hmquant.api import quant_single_onnx_network


def calibrate() -> None:
    env_dict = os.environ
    onnx_model_path = os.path.join(
        str(env_dict.get('MODEL_PATH')), 'yolop_576_1024.onnx',
    )

    height = 576
    width = 1024
    calib_dataset = [torch.randint(0, 200, (1, 3, height, width)).float()]

    quanttool_config = {
        'inputs_cfg': {
            'ALL': {
                'data_format': 'RGB',
                'first_layer_weight_denorm_mean': [0.485, 0.456, 0.406],
                'first_layer_weight_denorm_std': [0.229, 0.224, 0.225],
                'resizer_crop': {'top': 0, 'left': 0, 'height': height, 'width': width},
                'resizer_resize': {
                    'height': height,
                    'width': width,
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
