import torch
import torchvision
import os

# model = torchvision.models.resnet50(pretrained=True)
# dummy_input = torch.rand((1, 3, 224, 224))
# torch.onnx.export(
#     model, dummy_input, 'resnet50.onnx', export_params=True, opset_version=13,
# )

os.system('wget http://10.10.1.53:8082/artifactory/toolchain/release/models/resnet50/resnet50.onnx')
