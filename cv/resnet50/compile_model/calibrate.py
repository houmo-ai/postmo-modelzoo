import os
import time

import onnx
import torch
import torchvision.transforms as transforms
from hmquant.api import quant_single_onnx_network, convert_profiling, quantize_profiling
# from hmquant.tools.dataset.preprocess.transform import RGB2YUV
from hmquant.tools.dataset.preprocess.transform import ToTensorNotNormal
from torchvision.datasets.folder import pil_loader
from hmquant.configs.api_config import *


def calibrate():
    env_dict = os.environ
    onnx_model_path = os.path.join(env_dict.get('MODEL_PATH'), 'resnet50.onnx')

    def unsqueeze(x):
        return torch.unsqueeze(x, 0)

    calib_transform = transforms.Compose(
        [
            transforms.Resize(256), transforms.CenterCrop(224),
            ToTensorNotNormal(), unsqueeze,
        ],
    )

    image_root = os.path.join(env_dict.get('DATASETS_PATH'), 'imagenet')
    calib_image_files = [
        'ILSVRC2012_val_00000016.JPEG',
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
                'first_layer_weight_denorm_mean': [0.485, 0.456, 0.406],
                'first_layer_weight_denorm_std': [0.229, 0.224, 0.225],
                'resizer_crop': {'top': 0, 'left': 0, 'height': 224, 'width': 224},
                'resizer_resize': {
                    'height': 224,
                    'width': 224,
                    'align_corners': False,
                    'method': 'bilinear',
                },
                'toYUV_format': 'YUV422',
            },
        },
        'graph_opt_cfg': {},
    }

    onnx_input = [{"input.1": calib_dataset[0].numpy()}]

    t0 = time.time()
    print("====> start to quant onnx network...", flush=True)
    sequencer = quant_single_onnx_network(
        quanttool_config,
        calib_dataset,
        onnx_model_path,
        device='cpu',
        analyze=True
    )
    t1 = time.time()
    print(f"====> complete quant onnx network, cost {(t1-t0)*1000:.3f}ms", flush=True)

    print("====> start to convert profiling...", flush=True)
    convert_profiling(onnx_model_path, onnx_input, quanttool_config, onnx_input)
    t2 = time.time()
    print(f"====> complete convert profiling, cost {(t2-t1)*1000:.3f}ms", flush=True)

    print("====> start to quantize profiling...", flush=True)
    quantize_profiling(sequencer, onnx_input)
    t3 = time.time()
    print(f"====> complete quantize profiling, cost {(t3-t2)*1000:.3f}ms", flush=True)

    t4 = time.time()
    print("====> start to save quantized model...", flush=True)
    sequencer.save_onnx(
        'quant_resnet50.onnx',
        save_out_tensor=False,
        save_params_npy=True,
        save_special_onnx=True
    )

    t5 = time.time()
    print(f"====> complete save quantized model, cost {(t5-t4)*1000:.3f}ms", flush=True)
    print(f"====> calibrate success, cost {(t5-t0)*1000:.3f}ms", flush=True)


if __name__ == '__main__':
    calibrate()
