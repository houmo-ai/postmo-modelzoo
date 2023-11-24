import torch
import torchvision

model = torchvision.models.resnet50(pretrained=True)
dummy_input = torch.rand((1, 3, 576, 1024))
torch.onnx.export(
    model, dummy_input, 'resnet50_576_1024.onnx', export_params=True, opset_version=13,
)
