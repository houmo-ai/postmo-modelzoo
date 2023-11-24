import os

import torch
import torchvision.transforms as transforms
from hmquant.api import quant_single_onnx_network
from hmquant.tools.dataset.preprocess.transform import ToTensorNotNormal
from torchvision.datasets.folder import pil_loader


def calibrate() -> None:
    env_dict = os.environ
    onnx_model_path = os.path.join(
        str(
            env_dict.get(
                'MODEL_PATH',
            ),
        ), 'resnet50_576_1024.onnx',
    )

    def unsqueeze(x: torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(x, 0)

    calib_transform = transforms.Compose(
        [
            transforms.Resize((576, 1024)),
            ToTensorNotNormal(), unsqueeze,
        ],
    )

    image_root = os.path.join(str(env_dict.get('DATASETS_PATH')), 'imagenet')
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
                'resizer_crop': {'top': 0, 'left': 0, 'height': 576, 'width': 1024},
                'resizer_resize': {
                    'height': 576,
                    'width': 1024,
                    'align_corners': False,
                    'method': 'bilinear',
                },
                'toYUV_format': 'YUV422',
            },
        },
        'graph_opt_cfg': {},
    }

    print('====> start to quant onnx network...', flush=True)
    sequencer = quant_single_onnx_network(
        quanttool_config,
        calib_dataset,
        onnx_model_path,
        device='cpu',
        analyze=False,
    )
    sequencer.save_onnx(
        'quant_resnet50.onnx',
        save_out_tensor=False,
        save_params_npy=True,
        save_special_onnx=True,
    )


if __name__ == '__main__':
    calibrate()
