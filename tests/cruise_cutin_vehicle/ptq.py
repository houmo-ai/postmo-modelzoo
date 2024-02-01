import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
import onnx
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
        default='cruise_cutin.onnx',
        help='path to the model path',
    )
    parser.add_argument(
        '--model_name',
        dest='model_name',
        type=str,
        default='cruise_cutin',
        help='model name',
    )
    args = parser.parse_args()
    return args


def calibrate(args=None):
    model_path = args.model_path
    model_name = args.model_name
    output_path = 'output/H30/result'

    env_dict = os.environ
    
    onnx_model = onnx.load(model_path)
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    input_shape = [dim.dim_value for dim in dims]

    calib_dataset = [torch.randint(low=-128, high=127, size=input_shape, dtype=torch.float32)]

    quanttool_config = {
        "inputs_cfg": {
            "ALL": {
                "data_format": "Float32Feature",
                "first_layer_weight_denorm_mean": [0, 0, 0],
                "first_layer_weight_denorm_std": [0.003921568627451, 0.003921568627451, 0.003921568627451],
                "quantize": {
                    "quanted": False,
                    "quant_type": "int8",
                    "quant_scale": 1.0,
                    "quant_method": {"type": "Min_Max", "percent": 0.99999}
                }
            }
        },
        "graph_opt_cfg": {
            "save_fx_model": False,
            "auto_quant_flag": True,
            "fuse_conv_relu": False,
            "return_fuse_onnx": False
        }
    }

    onnx_input = {"input": calib_dataset[0]}

    print("start calibrating...")
    sequencer = quant_single_onnx_network(
        quanttool_config,
        calib_dataset,
        model_path,
        device='cpu'
    )

    print("start save model and generate golden...")
    generate_golden(
        sequencer=sequencer,
        calibset=onnx_input,
        save_path=output_path,
        model_name=model_name,
        batch_size=1,
        device="cpu"
    )

    #print("start quantize profiling...")
    #quantize_profiling(sequencer, [onnx_input])
    print("calibrate completed")


if __name__ == '__main__':
    args = get_args()
    calibrate(args)
