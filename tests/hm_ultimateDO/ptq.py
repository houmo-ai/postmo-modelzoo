import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
import argparse
from hmquant.api import quant_single_onnx_network, generate_golden
from hmquant.tools.dataset.preprocess.transform import ToTensorNotNormal


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        dest='model_path',
        type=str,
        default='ultimateDO_fp16_fuse.onnx',
        help='path to the model path',
    )
    parser.add_argument(
        '--model_name',
        dest='model_name',
        type=str,
        default='hm_ultimateDO',
        help='model name',
    )
    args = parser.parse_args()
    return args


def calibrate(args=None):
    model_path = args.model_path
    model_name = args.model_name
    output_path = 'output/H30/result'

    env_dict = os.environ

    calib_dataset = []
    calib_data = {}
    calib_data['img'] = torch.randint(0, 200, (6, 3, 256, 704)).float()
    calib_data['ranks_depth'] = torch.randint(0, 200, (1,408083))
    calib_data['ranks_feat'] = torch.randint(0, 200, (1,408083))
    calib_data['ranks_bev'] = torch.randint(0, 200, (1,408083))
    calib_data['interval_starts'] = torch.randint(0, 200, (1,32151))
    calib_data['interval_lengths'] = torch.randint(0, 200, (1,32151))
    calib_dataset.append(calib_data)

    quanttool_config = {
        'inputs_cfg': {
            'img': {
                'data_format': "Float32Feature",
                'first_layer_weight_denorm_mean': [0, 0, 0],
                'first_layer_weight_denorm_std': [1/255.0, 1/255.0, 1/255.0],
                'quantize': {
                    "quanted": False,
                    "quant_type": "int8",
                    "quant_scale": None,
                },
            },
            'ranks_depth': {
                'data_format': 'Int8Feature',
                'quantize': {
                    "quanted": True,
                    "quant_type": "int8",
                    "quant_scale": 1.0,
                    "quant_method": "Min_Max",
                },
            },
            'ranks_feat': {
                'data_format': 'Int8Feature',
                'quantize': {
                    "quanted": True,
                    "quant_type": "int8",
                    "quant_scale": 1.0,
                    "quant_method": "Min_Max",
                },
            },
            'ranks_bev': {
                'data_format': 'Int8Feature',
                'quantize': {
                    "quanted": True,
                    "quant_type": "int8",
                    "quant_scale": 1.0,
                    "quant_method": "Min_Max",
                },
            },
            'interval_starts': {
                'data_format': 'Int8Feature',
                'quantize': {
                    "quanted": True,
                    "quant_type": "int8",
                    "quant_scale": 1.0,
                    "quant_method": "Min_Max",
                },
            },
            'interval_lengths': {
                'data_format': 'Int8Feature',
                'quantize': {
                    "quanted": True,
                    "quant_type": "int8",
                    "quant_scale": 1.0,
                    "quant_method": "Min_Max",
                },
            },
        },
        'graph_opt_cfg': {},
    }

    onnx_input = {"input.1": calib_dataset[0]}

    print("start calibrating...")
    sequencer = quant_single_onnx_network(
        quanttool_config,
        calib_dataset,
        model_path,
        device='cpu',
    )

    print("start save model and generate golden...")
    generate_golden(
        sequencer=sequencer,
        calibset=calib_dataset[0],
        save_path=output_path,
        model_name=model_name,
        batch_size=1,
        device="cpu"
    )

    # print("start quantize profiling...")
    # quantize_profiling(sequencer, calib_dataset)
    print("calibrate completed")


if __name__ == '__main__':
    args = get_args()
    calibrate(args)
