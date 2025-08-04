import os
import shutil
import torch
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    HMONNXInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
)


HOUMO_TARGET = "xh2"
HOUMO_MODEL_PATH = os.getenv("HOUMO_MODEL_PATH", "../../../models")
HOUMO_DATASETS_PATH = os.getenv("HOUMO_DATASETS_PATH", ".")

model_path = os.path.join(HOUMO_MODEL_PATH, "resnet50.onnx")
input_name = "input.1"   # onnx输入名字
output_name = "495"  # onnx输出名字
input_shape = [1, 3, 224, 224]
output_dir = "output/xh2"
device = "cpu"  # 设备类型
quant_type = "w8a8h1_sefp"  # 目前量化类型为w8a8h1_sefp
quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
quant_config = create_quant_config(quant_scheme)
hmonnx_model_path = os.path.join(output_dir, f"resnet50_xh2_{quant_type}.onnx")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
random_data = torch.randn(input_shape, dtype=torch.float32)

# 量化以及HMONNX导出
convert_onnx_to_hmonnx(
    model_path,
    [random_data],
    device_type=DeviceType.XH2a,
    out_hmonnx_file=hmonnx_model_path,
    quant_config=quant_config,
    input_names=[input_name],
    output_names=[output_name],
)
# 生成芯片所需格式模型
golden_dir = os.path.join(output_dir, "hmquant", "golden")
if not os.path.exists(golden_dir):
    os.makedirs(golden_dir)
else:
    shutil.rmtree(golden_dir)
session = HMONNXGoldenInference(hmonnx_model_path)
session.to(device)
session.save_golden = True
session.golden_dir = golden_dir
session.step = 0
# to float16
session(random_data.half().to(device))  #
