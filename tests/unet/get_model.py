import os
import onnx
import torch

if __name__ == '__main__':
    # download raw model
    # if not os.path.exists('unet.onnx'):
    #     model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
    #     dummy_input = torch.rand(1, 3, 256, 256)
    #     torch.onnx.export(model, dummy_input, 'unet.onnx', export_params=True, opset_version=13)

    # download hmquant model
    if not os.path.exists('output/H30/result/hmquant_unet_with_act.onnx'):
        os.system('wget http://10.10.1.53:8082/artifactory/toolchain/release/models/unet/unet_golden.zip')
        os.system('mkdir -p output/H30/result')
        os.system('unzip -d output/H30/result unet_golden.zip')