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

### 大模型评测

大模型评测沿用 `hmatc eval` 命令，但不使用 `-c/--config`。模型适配实现内置于 HMATC，由 `--model-name` 选择模型族，`--model-size` 选择具体规格，`--model` 指向模型产物根目录。数据集编排和指标计算仍由 EvalScope 完成。

```bash
hmatc eval \
  --model-name gemma4 \
  --model-size e2b \
  --model /path/to/gemma4-e2b \
  --backend hmm \
  --dataset cmmlu \
  --limit 10 \
  --model-args devices=0 \
  --model-args max_new_tokens=2048
```

参数说明：
- `--model-name`：HMATC 内置模型族名称；当前支持 `gemma4`。
- `--model-size`：模型规格；Gemma4 支持 `26b-a4b`、`31b`、`e2b` 和 `e4b`。
- `--model`：模型产物根目录，传给 EvalScope 的 `TaskConfig.model`。
- `--backend`：后端类型，可选 `auto`、`raw`、`hmonnx` 或 `hmm`。`auto` 根据产物自动检测；Gemma4 当前仅实现 HMM 执行，`raw` 和 `hmonnx` 会返回明确的未实现错误。
- `--dataset`：一个或多个 EvalScope 数据集名称或路径。
- `--limit`：最多评测样本数，`0` 表示全量。
- `--output`：评测输出目录，默认 `./outputs`。
- `--model-args KEY=VALUE`：传给内置模型适配器的额外参数，可重复指定，例如 `devices=0,1`、`max_new_tokens=64` 或 `enable_thinking=true`；不能覆盖保留参数 `model_size` 和 `backend`。

Gemma4 默认从产物根目录读取 `gemma4-<size>_prefill.hmm`、`gemma4-<size>_decode.hmm`、`hmquant/quant_embedding.pt` 和 `hmquant/hf_config/`。`e2b`、`e4b` 还需要 `hmquant/per_layer_input_embedding.pt`。多设备执行使用 `.hmms` 产物。可通过 `prefill_path`、`decode_path`、`embedding_path`、`tokenizer_dir` 和 `PLE_path` 等 `--model-args` 覆盖默认路径。

## 大模型(LM) 配置文件

```yaml
version: 2
target: xh2
save_dir: 

# 模型信息
model:
  model_name: gemma4   # 必选，描述模型系列
  model_size: e2b      # 必选，描述模型参数规模
  model_dir:           # 必选，描述模型路径
  model_type:          # 可选，描述模型类型 是否已被量化，支持raw、quantized，缺省则自动判断
  modelscope_repo: ["google/gemma-4-E2B-it"]   # 可选，描述模型来源

# 量化信息
quant:
  # method缺省时默认为gptq；gptq/autoround精确选择对应的内置workflow
  # method: null时不执行新的GPTQModel量化，仅执行HMQuant导出：
  #   仅在相同speculative_decode和attention profile内，优先选择显式null注册，
  #   否则按gptq、autoround顺序选择已有workflow
  method: gptq
  speculative_decode: none  # 可选，默认none；支持none、mtp、dflash
  attention: default  # 可选，默认default；支持default、flash_attention、page_attention
  # 可选，仅用于所选workflow显式声明的投机模型路径；用户提供的路径必须是已存在目录
  # speculative_model:
  #   draft_model_dir: /path/to/draft-or-assistant-model
  #   target_model_dir: /path/to/target-model
  bits: 4  # 可选，仅覆写所选workflow的GPTQModel量化bit数；缺省时使用workflow原值，暂支持4、5、6、7、8
  prefill_chunk_length: 256  # 可选，描述导出的prefill模型的输入序列长度，默认256
  context_length: 2048  # 可选，描述导出llm的上下文长度，默认2048
  
# 编译信息
build:
  # 顶层字段作为所有已发现组件的默认配置；组件级显式值 > 顶层显式值 > HMATC默认值
  flash_attention: 2  # 可选，默认2
  llm_opt: true  # 可选，默认true
  enable_common_subgraph: false  # 可选，默认false
  ncore: 2  # 可选，默认2
  ndevice: 1  # 可选，默认1
  cpp_backend: v2  # 可选，默认v2
  all_logits: false  # 可选，默认false
  batch: 1  # 可选，默认1；decode组件必须为1
  device_kernel_split: 1  # 可选，默认1
  prefill_chunk_length: 320  # 可选，默认256；仅对prefill生效，具体取值约束由编译器校验
  context_length: null  # 可选，默认2048；仅对prefill和decode生效；null表示不修改原图context length

  # HMATC会扫描hmquant/下包含HMONNX的直接子目录，默认编译所有发现的组件
  # components只配置需要局部覆盖或跳过编译的组件；组件名必须与hmquant/的直接子目录名一致
  # type可选，支持hmonnx、prefill、decode，通常由每个HMONNX自动识别：
  #   无KV Cache为hmonnx；有KV Cache且唯一静态rank-3输入的序列长度>1为prefill，等于1为decode
  #   有KV Cache但无法确定唯一静态序列长度时，必须显式配置type: prefill或type: decode
  #   显式type与图中可明确识别出的类型冲突时会报错
  components:
    prefill:
      prefill_chunk_length: 320  # type可省略，由HMONNX自动识别
      # context_length: null  # 显式null表示该LLM组件不修改原图context length
    decode:
      batch: 1  # decode batch必须等于1；type通常可省略
      # context_length: 131072  # 组件级值优先于顶层值
    visual:
      enable_build: true  # 仅支持组件级配置，默认true；false时跳过编译
      enable_common_subgraph: true
      batch: 1
      ncore: 1

  # 顶层或组件级context_length配置为null时，仅对LLM的prefill/decode组件生效，传递None给编译器以保留原图context length；非LLM组件忽略该字段
  # HMATC不预先比较prefill_chunk_length与context_length，相关约束由编译器校验
  # 滑窗prefill当前不支持修改prefill_chunk_length：HMATC会warning并传None，
  # effective_build.yaml中该组件的prefill_chunk_length记录为null
  # 本次发现、继承、覆盖、自动识别及跳过后的实际配置保存到<save_dir>/<target>/effective_build.yaml
  # enable_build=false的组件也会记录在其中，其hmm为null
```

`speculative_decode` 和 `attention` 用于选择完整的内置 workflow，只支持当前模型规格实际注册的稀疏组合。枚举值合法但组合未注册时会直接报错，不会退回其他投机模式或 attention profile。

投机模型路径采用统一字段，但不同 profile 支持的字段不同：

- Gemma4 MTP 使用 `draft_model_dir` 和 `target_model_dir`，分别覆写 assistant 和 target Hugging Face 模型目录；
- Qwen3.5/Qwen3.6 DFlash 只使用 `draft_model_dir`，不接受 `target_model_dir`；
- Qwen3.5/Qwen3.6 MTP workflow 当前没有外部模型路径字段，因此不接受 `speculative_model`；
- workflow 中对应路径为 `null` 或空字符串时必须配置；workflow 已带非空路径时可沿用，也可用已存在的本地目录覆写。

Gemma4 MTP + page attention 示例：

```yaml
quant:
  method: gptq
  speculative_decode: mtp
  attention: page_attention
  speculative_model:
    draft_model_dir: /models/gemma-4-assistant
    target_model_dir: /models/gemma-4-target
```

Qwen3.5 DFlash 示例：

```yaml
quant:
  method: gptq
  speculative_decode: dflash
  attention: default
  speculative_model:
    draft_model_dir: /models/Qwen3.5-9B-DFlash
```

## ONNX 配置文件

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

  # [可选] DataLoader实现模块（多输入原始数据、非图像原始数据、复杂前处理时填写）
  dataloader_module:         # 模块文件名（与yml同级目录）
  dataloader_cls:            # 类名

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

# 演示配置
demo:
  data_dir:                 # [必填] 图片或npz数据目录
  num: 0                    # [可选] 演示数量，0表示全部

# 评估配置
eval:
  data_dir:                 # [必填] 数据集目录
  num: 0                    # [可选] 评估数量，0表示全部
  dataset_module:           # [必填] Eval Dataset模块，支持相对config目录或当前目录
  dataset_cls:              # [必填] Eval Dataset类名
```

## 数据格式

hmatc 内置三类 DataLoader：

- 单输入图像模型：读取图片并根据 `model.inputs` 中的 `data_format`、`mean`、`std`、`resize_type` 等配置完成前处理。
- `.npz` 输入：作为已预处理模型输入容器，支持单输入、多输入和非图像输入；key 必须与 ONNX 输入名一致。
- 随机输入：`quant.calib_data` 为空时用于快速量化，支持任意输入。

多输入原始数据、非图像原始数据、复杂前处理或复杂 demo/eval 元信息，请在 `model` 下配置自定义 DataLoader：

```yaml
model:
  dataloader_module: my_dataloader
  dataloader_cls: MyDataLoader
```

接口约定：

```python
class MyDataLoader:
    def __init__(self, data_dir, model_cfg=None, inputs_cfg=None, stage=None, num=0):
        ...

    def __len__(self):
        ...

    def __getitem__(self, index):
        return {
            "inputs": {
                "input_name": array,
            },
            "meta": {},
        }
```

`data_dir` 来自当前阶段配置：`quant.calib_data`、`demo.data_dir` 或 `eval.data_dir`；`num=0` 表示不截断。

### Eval Dataset 与 DataLoader 分工

`eval` 保留独立的 Dataset 配置，用于明确评估使用的数据集：

```yaml
eval:
  data_dir: ./coco2017
  num: 0
  dataset_module: dataset
  dataset_cls: Dataset
```

- `Dataset` 负责数据集加载、标注解析、样本列表、切片和数据集元信息。
- `DataLoader` 负责读取样本数据、前处理、Resizer 输入生成，并返回模型可直接推理的 `inputs` / `hmonnx_inputs` / `meta`。
- `dataset_module` 支持 `.py` 文件名或不带后缀的模块名；相对路径优先相对配置文件所在目录解析，其次相对当前运行目录解析。
- `model.dataloader_module` / `model.dataloader_cls` 仍用于多输入、非图像或复杂前处理等模型输入处理场景。

推荐 Dataset 接口：

```python
class Dataset:
    def __init__(self, data_dir=None, num=0):
        ...

    def __len__(self):
        ...

    def __getitem__(self, index):
        return {
            "path": "image.jpg",
            "image_id": 1,
            "label": 0,
        }
```

校准数据、Golden数据、比较数据的 NPZ 格式：

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