import os
import onnx
import argparse

def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='model_type',
        type=str,
        default='all',
        help='which model type to get, choise in [raw, quant, all]',
    )
    parser.add_argument(
        '--quant_model_dir',
        dest='quant_model_dir',
        type=str,
        default='output/H30/result',
        help='where to save quant_model',
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    quant_model_dir = args.quant_model_dir
    model_type = args.model_type
    if model_type == "raw" or model_type == "all":
        # import torch
        # import torchvision
        # model = torchvision.models.resnet50(pretrained=True)
        # dummy_input = torch.rand((1, 3, 224, 224))
        # torch.onnx.export(model, dummy_input, 'resnet50.onnx', export_params=True, opset_version=13)
        if not os.path.exists("resnet50.onnx"):
            os.system('wget http://10.10.1.53:8082/artifactory/toolchain/release/models/resnet50/resnet50.onnx')

    if model_type == "quant" or model_type == "all":
        if not os.path.exists(os.path.join(quant_model_dir, "hmquant_resnet50_with_act.onnx")):
            if not os.path.exists("resnet50_golden_20231218.zip"):
                os.system('wget http://10.10.1.53:8082/artifactory/toolchain/release/models/resnet50/resnet50_golden_20231218.zip')
            os.system('mkdir -p ' + quant_model_dir)
            os.system('unzip -d ' + quant_model_dir + ' resnet50_golden_20231218.zip')

    # onnx.utils.extract_model("resnet50.onnx", "resnet50_clip.onnx", input_names=[''], output_names=[''], check_model=True)
