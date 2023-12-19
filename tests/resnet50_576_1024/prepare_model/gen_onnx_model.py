import torch
import torchvision
import os

# model = torchvision.models.resnet50(pretrained=True)
# dummy_input = torch.rand((1, 3, 576, 1024))
# torch.onnx.export(
#     model, dummy_input, 'resnet50_576_1024.onnx', export_params=True, opset_version=13,
# )

os.system('wget -O resnet50_576_1024.onnx http://10.10.1.53:8082/artifactory/toolchain/support/models/resnet50_576x1024/resnet50_576x1024.onnx')