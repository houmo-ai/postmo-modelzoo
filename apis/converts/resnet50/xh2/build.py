import os
import tcim   # 编译器
import tcim_lite  # runtime
import numpy as np
import logging
logging.basicConfig(level="INFO")

output_dir = "output/xh2"
quant_type = "w8a8h1_sefp"
input_name = "input.1"   # onnx输入名字
output_name = "495"  # onnx输出名字
ncore = 1
opt_level = "O2"
hmonnx_model_path = os.path.join(output_dir, f"resnet50_xh2_{quant_type}.onnx")
hmmodel_name = f"resnet50_xh2_1batch_{ncore}core_{opt_level}"
work_dir = os.path.join(output_dir, "tcim")
golden_dir = os.path.join(output_dir, "hmquant", "golden")
enable_build = True   # 已经编译后可选择是否编译

if enable_build:
    # 编译
    tcim.build_from_hmonnx(
        hmonnx_model_path,
        output_name=hmmodel_name,
        ncore=ncore,
        opt_level=opt_level,
        target="xh2",
        batch=1,
        legacy=True,
        output_dir=output_dir,
        work_dir=work_dir,
    )
# 量化golden和编译输出验证
hmmodel_path = os.path.join(output_dir, f"{hmmodel_name}.hmm")
assert os.path.isfile(hmmodel_path)
module = tcim_lite.runtime.load(hmmodel_path)
num = module.get_num_inputs()
for idx in range(num):
    name = module.get_input_name(idx)
    info = module.get_input_info(name)
    shape = list(info.shape)
    dtype = np.dtype(info.dtype).name
    fmt = info.format.name
    print(f"[xh2] input[{idx}], name = {name}, shape = {shape}, dtype = {dtype}, format = {fmt}")
    
num = module.get_num_outputs()
for idx in range(num):
    name = module.get_output_name(idx)
    info = module.get_output_info(name)
    shape = list(info.shape)
    dtype = np.dtype(info.dtype).name
    fmt = info.format.name
    print(f"[xh2] output[{idx}], name = {name}, shape = {shape}, dtype = {dtype}, format = {fmt}")

# 加载量化产生的golden数据
golden_input_path = os.path.join(golden_dir, "step_0", f"hmquant_resnet50_xh2_{quant_type}_{input_name}_input.npy") 
assert os.path.isfile(golden_input_path)
golden_input_data = np.load(golden_input_path)
golden_output_path = os.path.join(golden_dir, "step_0", f"hmquant_resnet50_xh2_{quant_type}_{output_name}_output.npy")
assert os.path.isfile(golden_output_path)
golden_output_data = np.load(golden_output_path)
# 设置输入
module.set_input(input_name, golden_input_data)
# 推理
module.run()
module.sync()
# 获取输出
xh2_output_data = module.get_output(output_name)
# 计算余弦距离
v0 = golden_output_data.flatten().astype(np.float64)
v1 = xh2_output_data.numpy().flatten().astype(np.float64)
print(f"xh2 vs hmquant: {v0.dot(v1) / np.maximum(np.linalg.norm(v0) * np.linalg.norm(v1), np.finfo(np.float32).eps):.6f}")
