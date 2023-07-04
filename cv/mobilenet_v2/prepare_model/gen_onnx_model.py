import torch
import torchvision

model = torchvision.models.mobilenet_v2(pretrained=True)
dummy_input = torch.rand(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, 'mobilenet_v2.onnx', export_params=True, opset_version=13,
)
