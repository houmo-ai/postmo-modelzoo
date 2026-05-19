# Qwen3-30B-A3B

本目录提供 Qwen3-30B-A3B 在 XH2 平台上的下载、量化、编译、演示与一键测试脚本。当前默认配置来自同目录下的 `config.yaml`。

## 支持模型

- `model_name`: `qwen3`
- `model_size`: `30b-a3b`

对应 ModelScope 模型：

- Qwen3-30B-A3B: https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B

预训练模型在运行时下载，仓库本身不包含这些权重。

## 配置与默认值

`config.yaml` 是默认值的单一来源，当前约定如下：

- 默认模型：`qwen3` + `30b-a3b`
- 默认 `context_length=32k`
- 默认 `prefill_length=256`
- 默认输出前缀统一使用 `${model_name}-${model_size}`
- tokenizer / 原始模型目录默认从 `modelscope_repo` 最后一段推导，即 `Qwen3-30B-A3B`

## 输出约定

标准输出：

- `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `output/${HOUMO_TARGET}/${model_name}-${model_size}_prefill.hmm`
- `output/${HOUMO_TARGET}/${model_name}-${model_size}_decode.hmm`

当 `ndevice > 1` 时，demo 默认会把 `prefill` 和 `decode` 的后缀切换为 `.hmms`。

## 快速开始

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 2. 下载模型

下载原始模型：

```bash
python3 get_model.py --type raw --model_name qwen3 --model_size 30b-a3b
```

仅下载预编译模型用于演示：

```bash
python3 get_model.py --type hmm --model_name qwen3 --model_size 30b-a3b
```

### 3. 量化

量化需要 GPU 环境。推荐先下载原始模型，再执行：

```bash
python3 ptq.py --model-name qwen3 --model-size 30b-a3b
```

若需要启用 `gptqmodel`：

```bash
python3 ptq.py --model-name qwen3 --model-size 30b-a3b --gptqmodel
```

量化结果位于 `output/${HOUMO_TARGET}/hmquant`。

### 4. 编译

```bash
python3 build.py --model_name qwen3 --model_size 30b-a3b
```

编译输出：

- `qwen3-30b-a3b_prefill.hmm`
- `qwen3-30b-a3b_decode.hmm`

### 5. 演示

```bash
python3 demo.py --model_name qwen3 --model_size 30b-a3b
```

若未显式传 `tokenizer_dir`、`prefill_path`、`decode_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导。

## test.sh

`test.sh` 用于串起下载、量化、编译、演示与 perf。默认执行 `demo`，默认模型为 `qwen3-30b-a3b`。

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
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
bash test.sh -s quant
bash test.sh -s build
bash test.sh -s demo
bash test.sh -s all
bash test.sh -s demo --skip_download
```

标准 demo 流程结束后，如果环境中存在 `llm_perf`，脚本还会自动执行：

```bash
python3 ${HOUMO_EXAMPLES_PATH}/tools/llm_perf/convert_embed.py --path output/${HOUMO_TARGET}/hmquant/quant_embedding.pt
llm_perf --model_name qwen3-30b-a3b ...
```

## 关键参数说明

### get_model.py

- `--config`: 配置文件路径，默认当前目录下 `config.yaml`
- `--model_name`: 默认 `qwen3`
- `--model_size`: 默认 `30b-a3b`
- `--type`: `raw` 或 `hmm`
- `--context_length`: 默认取配置中的 `32k`

### ptq.py

- `--config`: 配置文件路径
- `--model`: 原始 HF 模型目录；默认根据 `modelscope_repo` 推导目录名
- `--model-name`: 默认取配置
- `--model-size`: 默认取配置
- `--context-length`: 默认按配置中的 `32k` 解析
- `--input-sequence-length`: 默认跟随配置中的 `prefill_length`
- `--quant-type`: 默认跟随配置

### build.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--model_dir`: 默认 `output/${HOUMO_TARGET}/hmquant`
- `--output_dir`: 默认 `output/${HOUMO_TARGET}`
- `--context_length`: 默认按配置中的 `32k` 解析

### demo.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--tokenizer_dir`: 默认根据 `modelscope_repo` 推导
- `--embedding_path`: 默认 `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `--prefill_path`, `--decode_path`: 默认按 `${model_name}-${model_size}` 自动拼接

## 数据与免责声明

本项目引用 wikitext 数据集仅用于研究和评估目的，数据集许可为 CC BY-NC 4.0，来源为 https://www.modelscope.cn/datasets/modelscope/wikitext。该数据集不作为仓库内容分发，使用时需遵守其原始许可。

本文档中引用的软件、数据和模型由第三方提供并维护。使用这些第三方资源时，应遵守各自的许可协议、使用条款和隐私政策，相关风险由使用者自行承担。
