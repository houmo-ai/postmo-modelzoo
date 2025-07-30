import torch


class CustomMultiInputModel(torch.nn.Module):
    def __init__(self):
        super(CustomMultiInputModel, self).__init__()

    def forward(self, x, y):
        return x + y
    

x = torch.randn(1, 4, 4, dtype=torch.float32)
y = torch.randn(1, 4, 4, dtype=torch.float32)

torch.onnx.export(
    CustomMultiInputModel(),
    (x, y),
    "custom_multi_input.onnx",
    export_params=True,
    opset_version=11,
    input_names=["x", "y"],
    output_names=["z"],
)