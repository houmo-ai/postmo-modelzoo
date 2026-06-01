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
hmatc check --hmm your_hmm_path --golden golden_dir
```

### Golden生成

```bash
# 生成Golden数据（不指定data_path则使用随机数据）
hmatc golden --hmonnx your_hmonnx_path --output golden_dir --data_path data.npz
# 逐算子生成Golden（会生成debug.onnx，需重新编译后再check）
hmatc golden --hmonnx your_hmonnx_path --output golden_dir --data_path data.npz --layers
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
      resize_type: 1              # resize类型: 0-直接resize, 1-等比例resize+padding, 2-固定高度等比例宽度+右侧padding(OCR识别)
      padding_mode: 1             # padding模式: 0-左上角, 1-中心（仅resize_type=1必填，resize_type=2固定右侧padding_mode=0）
      padding_values: [114, 114, 114]  # padding数值（resize_type=1/2必填）

      # Resizer配置（可选，启用芯片硬件resize）
      resizer:
        toYUV_format: YUV420SP    # YUV格式: YUV400/YUV420SP/YUV422SP/YUV444SP

        # Resizer输入尺寸 [H, W]
        # - 表示量化编译后模型的实际输入尺寸
        # - 缺省默认为原模型输入高宽
        # - 尺寸限制: H <= 4096, W <= 4096, 必须为偶数
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
  quant_type: w8a8h1_sefp       # [可选] 量化类型，支持混合位宽如 w8w16a8a16_sefp
  calib_data:                   # [可选] 校准数据目录，留空使用随机数据

  # [可选] 混合精度搜索配置（自动选择高精度层）
  # 注意：mix_search 与 resizer 配置互斥，两者不能同时使用
  mix_search:
    topk: 0.10                  # 选择混合高位宽的比例 (0-1)
    weight_bits:                # 权重混合候选位宽
      - 8
      - 16
    act_bits:                   # 激活混合候选位宽
      - 8
      - 16
    policy: topk                # 挑选策略: topk(前K%层)/threshold(误差阈值)
    task: cv                    # 任务类型: cv/cv_cls/llm
    metric: l1                  # 敏感度计算: l1(绝对差异)/sqnr(信噪比)/kl(KL散度)
    key_name: loss              # 输出属性名称（如logits或loss）

# 编译配置
build:
  ncore: 1                  # [可选] IPU核数: 1或2
  opt_level: 2              # [可选] 优化等级: 0/1/2
  batch: 1                  # [可选] 编译batch（最终batch = 模型batch * 编译batch）
  roi_num: 1                # [可选] ROI数量，仅动态resizer模式有效，需batch=1
  parallel_jobs: 4          # [可选] 并行编译任务数，默认CPU物理核心数
  cpp_backend: v1           # [可选] 算子实现的版本，可选v1, v2

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

## 量化格式

### quant_type 格式

量化类型字符串格式：`[w{bit}]*[a{bit}]*[h{h_flag}][n{nshare}][_{fp_mode}]`

| 部分 | 说明 | 可选性 | 示例 |
|------|------|--------|------|
| `w{bit}` | 权重位宽，可多次出现表示混合位宽 | 可选 | `w8`, `w8w4`, `w8w4w16` |
| `a{bit}` | 激活位宽，可多次出现表示混合位宽 | 可选 | `a8`, `a8a16` |
| `h{h_flag}` | hidden bit 标志 (0 或 1) | 可选 | `h0`, `h1` |
| `n{nshare}` | nshare 参数 (整数) | 可选 | `n64`, `n128` |
| `_{fp_mode}` | 浮点模式 (sefp/ssfp) | 可选 | `_sefp`, `_ssfp` |

解析示例：
```
"w8a8h1_sefp"       -> bit_w=[8], bit_a=[8], h_flag=1, fp_mode='sefp'
"w8w4a8a16h0_ssfp"  -> bit_w=[8,4], bit_a=[8,16], h_flag=0, fp_mode='ssfp'
"w8a8n64"           -> bit_w=[8], bit_a=[8], nshare=64
"w8a8h1n64_sefp"    -> bit_w=[8], bit_a=[8], h_flag=1, nshare=64, fp_mode='sefp'
```

### Scale 格式

后摩 float16 数据的量化表示方式：尾数(int) + scale

| 格式 | Scale 定义 | 特点 |
|------|-----------|------|
| **sefp** | scale = 2^(-n)，n为正整数 | 计算高效，适合CV模型 |
| **ssfp** | scale = float16 | 精度更高，适合LLM模型 |

### 常用量化规格

| 规格 | 含义 | 适用场景 |
|------|------|----------|
| `w8a8h1_sefp` | 权重8bit，激活8bit，hidden=1 | CV模型默认配置 |
| `w4a8h0_ssfp` | 权重4bit，激活8bit，hidden=0 | LLM模型推荐配置 |
| `w8a16h1_sefp` | 权重8bit，激活16bit | 高精度CV模型 |
| `w8w16a8a16_sefp` | 混合位宽 (配合 mix_search) | 需要精度敏感层分析的场景 |

### mix_search 混合精度搜索

当 quant_type 包含多个位宽选项时（如 `w8w16a8a16`），配合 `mix_search` 配置可自动选择高精度层：

```yaml
quant:
  quant_type: w8w16a8a16_sefp  # 混合位宽候选
  mix_search:
    topk: 0.1                  # 选择前10%敏感层使用高精度
    weight_bits: [8, 16]       # 权重候选位宽
    act_bits: [8, 16]          # 激活候选位宽
    policy: topk               # 选择策略: topk/threshold
    task: cv_cls               # 任务类型: cv/cv_cls/llm
    metric: l1                 # 敏感度度量: l1/sqnr/kl
    key_name: loss             # 输出属性名称
```

### task 参数说明

`task` 决定了 mix_search 计算敏感度时使用的 loss 函数：

| task | Loss 函数 | 适用场景 | 输出格式 |
|------|----------|---------|---------|
| **llm** | `cross_entropy` | 大语言模型 | dict/3D tensor (bs, seq_len, vocab_size) |
| **cv_cls** | `cross_entropy` | CV 分类任务 (OCR识别、图像分类) | tensor/dict/list |
| **cv** | `mse_loss` | 通用 CV 任务 (检测、分割等回归任务) | tensor/dict/list |

**注意：** `mix_search` 与 `resizer` 配置互斥，因为 mix_search 运行原始 ONNX 做敏感度分析需要 float32 输入，而 resizer 需要 YUV uint8 输入。

## Resizer说明

Resizer利用芯片硬件实现 crop -> resize -> padding 流程，适用于输入分辨率变化的场景。

### 模式对比

| 模式 | 参数 | 灵活性 | 适用场景 |
|------|------|--------|----------|
| STATIC (3) | 无 | 固定分辨率 | 输入分辨率确定 |
| DYNAMIC_V1 (2) | 4 (crop) | crop可变 | 暂不支持 |
| DYNAMIC_V2 (1) | 10 | 全可变 | crop/resize/padding均可运行时调整 |

### 尺寸限制

**通用限制：**
- W方向：max 4096，>2048 时需32对齐，≤2048 时需2对齐
- H方向：max 4096，需2对齐
- crop：4参数(y, x, h, w)，全部2对齐
- pad：4参数(top, left, bottom, right)，全部2对齐
- pad 可支持任意规格，不限制仅上下或左右单方向

**静态模式 (STATIC) 限制：**
- 缩放倍数范围：[1/32, 16]

**动态模式 (DYNAMIC_V2) 限制：**
- 放大倍数最大16
- H方向缩小倍数最大32
- W方向缩小倍数最大8（YUV444最大4）

**其他限制：**
- 不支持 one image multi roi（roi_num 必须为1）