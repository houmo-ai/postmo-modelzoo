# HMATC

HouMo AI 模型辅助工具，用于模型量化、编译、验证和性能测试。

## 安装

```bash
# 安装
python setup.py install
# 打包
python setup.py bdist_wheel
# 开发模式
python setup.py develop
```

## 使用

### 量化

```bash
# 从配置文件量化
hmatc quant -c config.yml
# 生成默认配置文件（不包含预处理相关配置）
hmatc gen --onnx model.onnx --output config.yml
```

### 编译

```bash
# 从配置文件编译
hmatc build -c config.yml
# 指定并行任务数加快编译（默认为CPU物理核心数）
hmatc build -c config.yml -j 8
# 仅从hmonnx文件编译
hmatc build --hmonnx your_hmonnx_path
```

### 比较

```bash
# ONNX、hmquant(量化后)、HMM(编译后)三端精度比较
hmatc compare -c config.yml --data_path data.npz
# Golden数据验证
hmatc check -c config.yml
# 逐算子Golden验证
hmatc check -c config.yml --layers
# 直接指定HMM和Golden文件验证
hmatc check --hmm your_hmm_path --golden golden.npz
```

### Golden生成

```bash
# 生成Golden数据（不指定data_path则使用随机数据）
hmatc golden --hmonnx your_hmonnx_path --output golden.npz --data_path data.npz
# 逐算子生成Golden（会生成debug.onnx，需重新编译后再check）
hmatc golden --hmonnx your_hmonnx_path --output golden.npz --data_path data.npz --layers
hmatc build --hmonnx your_hmonnx_path
```

### 性能测试

```bash
hmatc perf -c config.yml -wn 10 -sn 1000 -tn 4
```

参数说明：
- `-wn`: warmup次数
- `-sn`: sample次数
- `-tn`: thread数量

### 模型演示

需实现模型推理模块，参考 `modelzoo` 示例。

```bash
hmatc demo -c config.yml         # 芯片推理
hmatc demo -c config.yml --onnx  # ONNX推理
```

### 模型评估

需实现数据集模块，参考 `modelzoo` 示例。

```bash
hmatc eval -c config.yml         # 芯片评估
hmatc eval -c config.yml --onnx  # ONNX评估
```

## 配置文件

```yaml
# 模型信息
model:
  name:                    # 模型名称，用于命名编译后模型
  save_dir:                # [必填] 输出目录
  model_path:              # [必填] ONNX模型路径

  inputs:
    input_name:            # 输入名称，需与ONNX一致
      shape: [1, 3, 640, 640]     # [必填] 输入形状 [N, C, H, W]
      data_format: RGB            # [可选] 图像格式: RGB/BGR/GRAY，非图像填null，默认null

      # 图像预处理参数（data_format非null时必填）
      mean: [0.0, 0.0, 0.0]       # 预处理均值
      std: [255.0, 255.0, 255.0]  # 预处理标准差
      resize_type: 1              # resize类型: 0-独立resize, 1-等比例resize
      padding_mode: 1             # padding模式: 0-左上角, 1-中心（等比例resize必填）
      padding_values: [114, 114, 114]  # padding数值（等比例resize必填）

      # Resizer配置（可选，启用芯片硬件resize）
      resizer:
        toYUV_format: YUV420SP    # YUV格式: YUV400/YUV420SP/YUV422SP/YUV444SP

        # Resizer输入尺寸 [H, W]
        # - 表示量化编译后模型的实际输入尺寸
        # - 缺省默认为原模型输入高宽
        # - 尺寸限制: H <= 4096, W <= 1024, 必须为偶数
        resizer_input_size: [640, 640]

        # Resizer模式
        # 1 - DYNAMIC_V2: 全动态（10参数: crop + resize + padding），灵活性最高
        # 2 - DYNAMIC_V1: 半动态（4参数: crop仅动态），resize/padding量化时固定
        # 3 - STATIC: 全静态，无运行时灵活性
        # XH2三种模式性能接近，按灵活性需求选择
        resizer_mode: 3

        # [仅STATIC模式] 静态裁剪区域 [y, x, h, w]
        # - 默认: [0, 0, resizer_input_h, resizer_input_w]
        # - 必须为偶数
        # - 缩放限制: crop -> 模型输入在 [1/32, 16] 范围内
        # - 动态模式下配置会报错
        resizer_crop: [0, 0, 640, 640]

  # [可选] 模型实现模块（demo/eval功能需填写）
  model_impl_module:        # 模块文件名（与yml同级目录）
  model_impl_cls:           # 类名

# 量化配置
quant:
  calib_data:               # [可选] 校准数据目录，留空使用随机数据

# 编译配置
build:
  ncore: 1                  # [可选] IPU核数: 1或2
  opt_level: 2              # [可选] 优化等级: 0/1/2
  batch: 1                  # [可选] 编译batch（最终batch = 模型batch * 编译batch）
  roi_num: 1                # [可选] ROI数量，仅动态resizer模式有效，需batch=1
  parallel_jobs: 4          # [可选] 并行编译任务数，默认CPU物理核心数

# 演示配置
demo:
  data_dir:                 # [必填] 图片或npz数据目录
  num: 0                    # [可选] 演示数量，0表示全部

# 评估配置
eval:
  data_dir:                 # [必填] 数据集目录
  num: 0                    # [可选] 评估数量，0表示全部
  dataset_module:           # [必填] 数据集模块（与yml同级目录）
  dataset_cls:              # [必填] 数据集类名
```

## 数据格式

校准数据、Golden数据、比较数据均为预处理后的NPZ格式：

```python
import numpy as np
# 单输入
data = {'input_name': np.ndarray}
# 多输入
data = {'input_a': np.ndarray, 'input_b': np.ndarray}
np.savez_compressed('data.npz', **data)
```

## Resizer说明

Resizer利用芯片硬件实现 crop -> resize -> padding 流程，适用于输入分辨率变化的场景。

### 模式对比

| 模式 | 参数 | 灵活性 | 适用场景 |
|------|------|--------|----------|
| STATIC (3) | 无 | 固定分辨率 | 输入分辨率确定 |
| DYNAMIC_V1 (2) | 4 (crop) | crop可变 | resize/padding固定 |
| DYNAMIC_V2 (1) | 10 | 全可变 | 所有参数运行时可调 |

### ROI_NUM说明

`roi_num > 1` 时，单张图片可输出多个ROI结果：
- 仅动态resizer模式有效
- 需要 `model_input_batch * build_batch == 1`
- 适用场景：检测模型单图多框