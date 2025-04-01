import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
import argparse
from hmquant.api import quant_single_onnx_network, generate_golden, quantize_profiling
from hmquant.tools.dataset.preprocess.transform import ToTensorNotNormal

HOUMO_TARGET = os.getenv('HOUMO_TARGET', 'houmo')
HOUMO_MODEL_PATH = os.getenv("HOUMO_MODEL_PATH", '')
HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '')


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        dest='model_path',
        type=str,
        default=os.path.join(HOUMO_MODEL_PATH, 'yolov3_640x640_clip.onnx'),
        help='path to the model path',
    )
    parser.add_argument(
        '--model_name',
        dest='model_name',
        type=str,
        default='yolov3',
        help='model name',
    )
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, 'hmquant'),
        help='path to the quanted model dir',
    )
    args = parser.parse_args()
    return args


def calibrate(args=None):
    model_path = args.model_path
    model_name = args.model_name
    output_path = args.model_dir

    def preprocess(filepath):
        import cv2
        from hmassist.utils import utils
        from hmassist.utils.box_utils import letterbox
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, _, _ = letterbox(image, [640, 640], stride=64, auto=False)  # HWC
        image = np.transpose(image, (2, 0, 1))  # CHW .astype(np.float32)
        image = np.expand_dims(image, axis=0)  # NCHW
        data = torch.tensor(image.astype(np.float32))
        return data

    calib_num = 20
    calib_files = []
    calib_dir = os.path.join(HOUMO_DATASETS_PATH, 'coco2017/val2017')
    file_list = os.listdir(calib_dir)
    for filename in file_list:
        _, ext = os.path.splitext(filename)
        if ext in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP", ".bin"]:
            calib_files.append(filename)
            if len(calib_files) == calib_num:
                break
    # calib_files = [
    #     'ILSVRC2012_val_00000016.JPEG',
    # ]
    print("calib_files =", calib_files)
    calib_dataset = [
        preprocess(os.path.join(calib_dir, file_path))
        for file_path in calib_files
    ]

    quanttool_config = {
        'inputs_cfg': {
            'ALL': {
                'data_format': 'RGB',
                'first_layer_weight_denorm_mean': [0, 0, 0],
                'first_layer_weight_denorm_std': [1, 1, 1],
                'resizer_crop': {'top': 0, 'left': 0, 'height': 640, 'width': 640},
                'resizer_resize': {
                    'height': 640,
                    'width': 640,
                    'align_corners': False,
                    'method': 'bilinear',
                },
                'toYUV_format': 'YUV420',
            },
        },
        'graph_opt_cfg': {},
    }

    onnx_input = calib_dataset[0]

    print("start calibrating...")
    sequencer = quant_single_onnx_network(
        quanttool_config,
        calib_dataset,
        model_path,
        device='cpu',
    )
    
    print("start quantize profiling...")
    quantize_profiling(sequencer, [onnx_input])
    print("calibrate completed.")

    print("start save model and generate golden...")
    generate_golden(
        sequencer=sequencer,
        calibset=onnx_input,
        save_path=output_path,
        model_name=model_name,
        batch_size=1,
        device="cpu"
    )
    print("save model and generate golden completed.")


if __name__ == '__main__':
    args = get_args()
    calibrate(args)
