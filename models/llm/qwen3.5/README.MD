# Qwen3.5 / Qwen3.6

本目录提供 Qwen3.5 和 Qwen3.6 多模态模型在后摩 XH2 平台上的下载、量化、编译和演示脚本。当前脚本默认读取同目录下的 `config.yaml`，默认模型为 `qwen3.6` + `35b-a3b`。

## 支持的模型

- `qwen3.5`: `0.8b`, `2b`, `4b`, `9b`
- `qwen3.6`: `27b`, `35b-a3b`

对应 ModelScope 模型：

- Qwen3.5-0.8B: https://modelscope.cn/models/Qwen/Qwen3.5-0.8B
- Qwen3.5-2B: https://modelscope.cn/models/Qwen/Qwen3.5-2B
- Qwen3.5-4B: https://modelscope.cn/models/Qwen/Qwen3.5-4B
- Qwen3.5-9B: https://modelscope.cn/models/Qwen/Qwen3.5-9B
- Qwen3.6-27B: https://modelscope.cn/models/Qwen/Qwen3.6-27B
- Qwen3.6-35B-A3B: https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B

预训练模型在运行时下载，仓库本身不包含这些权重。

## 输出约定

编译与演示脚本统一使用 `${model_name}-${model_size}` 作为文件名前缀。

标准流程输出：

- `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `output/${HOUMO_TARGET}/${model_name}-${model_size}_prefill.hmm`
- `output/${HOUMO_TARGET}/${model_name}-${model_size}_decode.hmm`
- `output/${HOUMO_TARGET}/${model_name}-${model_size}_visual_${max_size_w}x${max_size_h}x${max_size_t}.hmm`

MTP 流程额外输出：

- `output/${HOUMO_TARGET}/${model_name}-${model_size}_prefill_mtp.hmm`
- `output/${HOUMO_TARGET}/${model_name}-${model_size}_decode_mtp.hmm`

`demo.py` 与 `demo_mtp.py` 默认从 `config.yaml` 中读取 `modelscope_repo`，并使用其最后一段作为 `tokenizer_dir` 默认值；若配置缺失，则回退到 `MODEL_NAME-MODEL_SIZE` 大写目录名。

## 快速开始

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
pip3 install -r ../../../hmodel/gptqmodel/requirements.txt
```

### 2. 下载模型

下载原始模型：

```bash
python3 get_model.py --type raw --model_name qwen3.5 --model_size 9b
python3 get_model.py --type raw --model_name qwen3.6 --model_size 35b-a3b
```

仅下载预编译模型用于演示：

```bash
python3 get_model.py --type hmm --model_name qwen3.6 --model_size 35b-a3b
```

### 3. 量化

量化需要 GPU 环境。推荐先通过 `get_model.py --type raw` 下载原始模型，再执行：

```bash
python3 ptq.py --model ./Qwen3.6-35B-A3B
```

如果当前目录下已经存在与配置匹配的模型目录，也可以直接运行 `python3 ptq.py`，脚本会按 `config.yaml` 推导默认模型目录名。

量化输出位于 `output/${HOUMO_TARGET}/hmquant`。

### 4. 编译

标准编译：

```bash
python3 build.py --model_name qwen3.6 --model_size 35b-a3b
```

MTP 编译：

```bash
python3 build_mtp.py --model_name qwen3.6 --model_size 35b-a3b
```

### 5. 演示

标准演示：

```bash
python3 demo.py --model_name qwen3.6 --model_size 35b-a3b
```

MTP 演示：

```bash
python3 demo_mtp.py --model_name qwen3.6 --model_size 35b-a3b
```

脚本默认会拼接如下路径：

- `demo.py`: `prefill.hmm`, `decode.hmm`, `visual_...hmm`
- `demo_mtp.py`: `prefill.hmm`, `prefill_mtp.hmm`, `decode_mtp.hmm`, `decode.hmm`

多设备场景下，demo 脚本会自动把默认模型后缀从 `.hmm` 切换为 `.hmms`。

## test.sh

`test.sh` 用于串起下载、量化、编译、演示和 MTP 演示。默认行为是执行 `demo`，默认模型是 `qwen3.6-35b-a3b`。

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

脚本参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -name, --model_name     Model name. Choices: qwen3.5, qwen3.6.
  -size, --model_size     Model size.
  --mtp                   Enable mtp mode.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

`test.sh` 会校验模型组合，只接受：

- `qwen3.5`: `0.8b`, `2b`, `4b`, `9b`
- `qwen3.6`: `27b`, `35b-a3b`

示例：

```bash
# 指定模型为 Qwen3.5-9B
bash test.sh --model_name qwen3.5 --model_size 9b

# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant --model_name qwen3.6 --model_size 35b-a3b

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build --model_name qwen3.6 --model_size 35b-a3b

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo --model_name qwen3.6 --model_size 35b-a3b

# 完整流程
bash test.sh -s all --model_name qwen3.6 --model_size 35b-a3b

# MTP 演示或编译
bash test.sh --mtp --model_name qwen3.6 --model_size 35b-a3b

# 跳过下载，直接使用当前目录已有模型
bash test.sh -s demo --skip_download --model_name qwen3.6 --model_size 35b-a3b
```

标准 demo 流程结束后，脚本还会执行：

```bash
python3 ${HOUMO_EXAMPLES_PATH}/tools/llm_perf/convert_embed.py --path output/${HOUMO_TARGET}/hmquant/quant_embedding.pt
llm_perf --model_name ${model_name}-${model_size} --devices 0 \
  --input 256,1024,2048 --output 256,256,256 --loop 1 --batch 1 \
  --prefill output/${HOUMO_TARGET}/${model_name}-${model_size}_prefill.hmm \
  --decode output/${HOUMO_TARGET}/${model_name}-${model_size}_decode.hmm \
  --visual output/${HOUMO_TARGET}/${model_name}-${model_size}_visual_896x896x2.hmm \
  --embedding output/${HOUMO_TARGET}/hmquant/quant_embedding.bin
```

## 关键参数说明

### get_model.py

- `--config`: 配置文件路径，默认当前目录下 `config.yaml`
- `--model_name`: 模型名称，默认取 `config.yaml`
- `--model_size`: 模型尺寸，默认取 `config.yaml`
- `--type`: `raw` 或 `hmm`
- `--source_type`: 下载源，支持 `jfrog` 和 `modelscope`
- `--mtp`: 下载 MTP 预编译模型

### ptq.py

- `--model`: 原始 HF 模型目录；默认根据 `modelscope_repo` 推导目录名
- `--model-name`: 导出使用的模型名前缀，默认取配置
- `--model-size`: 模型尺寸，默认取配置
- `--context-length`: 从配置继承的上下文长度
- `--max-pe-length`: 默认跟随 `context-length`
- `--input-sequence-length`: 默认取配置中的 `prefill_length`
- `--skip-export-vision` / `--skip-export-llm`: 选择性跳过导出

### build.py / build_mtp.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--model_dir`: 默认 `output/${HOUMO_TARGET}/hmquant`
- `--output_dir`: 默认 `output/${HOUMO_TARGET}`
- `--stage`: `build`, `test`, `all`
- `--monitor_interval`: 内存监控采样周期，默认 `1.0`
- `build.py --enable_xh2_stable_output`: 开启 stable output 编译选项

### demo.py / demo_mtp.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--tokenizer_dir`: 默认根据 `modelscope_repo` 推导
- `--embedding_path`: 默认 `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `--prefill_path`, `--decode_path`, `--vision_path`: 默认按 `${model_name}-${model_size}` 自动拼接
- `demo_mtp.py --prefill_mtp_path`, `--decode_mtp_path`, `--decode_verify_path`: 默认按 `${model_name}-${model_size}` 自动拼接

## 数据与免责声明

本项目引用 wikitext 数据集仅用于研究和评估目的，数据集许可为 CC BY-NC 4.0，来源为 https://www.modelscope.cn/datasets/modelscope/wikitext。该数据集不作为仓库内容分发，使用时需遵守其原始许可。

您明确了解并同意，本文档中引用的软件、数据和模型由第三方提供并维护。使用这些第三方资源时，应遵守各自的许可协议、使用条款和隐私政策，相关风险由使用者自行承担。
