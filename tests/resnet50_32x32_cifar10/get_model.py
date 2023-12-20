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
        help='model_type to get',
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.model_type == "raw" or args.model_type == "all":
        # import torch
        # import torchvision
        # model = torchvision.models.resnet50(pretrained=True)
        # dummy_input = torch.rand((1, 3, 224, 224))
        # torch.onnx.export(model, dummy_input, 'resnet50.onnx', export_params=True, opset_version=13)
        if not os.path.exists("resnet50.onnx"):
            os.system('wget http://10.10.1.53:8082/artifactory/toolchain/release/models/resnet50/resnet50.onnx')

    if args.model_type == "quant" or args.model_type == "all":
        if not os.path.exists("output/H30/result/hmquant_resnet50_32x32_cifar10_with_act.onnx"):
            if not os.path.exists("hmquant_resnet50_32x32_cifar10_20231219.zip"):
                os.system('wget http://10.10.1.53:8082/artifactory/toolchain/support/models/resnet50_32x32_cifar10/hmquant_resnet50_32x32_cifar10_20231219.zip')
            os.system('mkdir -p output/H30/result')
            os.system('unzip -d output/H30/result hmquant_resnet50_32x32_cifar10_20231219.zip')

    # onnx.utils.extract_model("resnet50.onnx", "resnet50_clip.onnx", input_names=[''], output_names=[''], check_model=True)