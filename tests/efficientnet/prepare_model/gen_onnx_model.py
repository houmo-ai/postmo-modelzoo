import torch

efficientnet = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_efficientnet_b0', pretrained=True,
)
input = torch.randn([1, 3, 224, 224])
torch.onnx.export(
    efficientnet, input,
    'efficientnet_b0_224x224.onnx', opset_version=13,
)
