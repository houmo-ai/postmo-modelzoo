# Qwen3

本目录提供 Qwen3 在 XH2 平台上的下载、量化、编译、演示与一键测试脚本。当前默认配置来自同目录下的 `config.yaml`，默认模型为 `qwen3` + `8b`。

## 支持模型

- `0.6b`
- `1.7b`
- `8b`
- `14b`

对应 ModelScope 模型：

- Qwen3-0.6B: https://modelscope.cn/models/Qwen/Qwen3-0.6B
- Qwen3-1.7B: https://modelscope.cn/models/Qwen/Qwen3-1.7B
- Qwen3-8B: https://modelscope.cn/models/Qwen/Qwen3-8B
- Qwen3-14B: https://modelscope.cn/models/Qwen/Qwen3-14B

预训练模型在运行时下载，仓库本身不包含这些权重。

## 配置与默认值

`config.yaml` 是默认值的单一来源，当前约定如下：

- 默认模型：`qwen3` + `8b`
- `0.6b` / `1.7b` / `8b` 默认 `context_length=32k`
- `14b` 默认 `context_length=16k`
- 默认输出前缀统一使用 `${model_name}-${model_size}`

脚本默认会从 `modelscope_repo` 取最后一段作为 tokenizer 或模型目录默认值，例如 `qwen/qwen3-8b` -> `qwen3-8b`。

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
python3 get_model.py --type raw --model_name qwen3 --model_size 0.6b
python3 get_model.py --type raw --model_name qwen3 --model_size 8b
```

仅下载预编译模型用于演示：

```bash
python3 get_model.py --type hmm --model_name qwen3 --model_size 8b
```

### 3. 量化

量化需要 GPU 环境。推荐先下载原始模型，再执行：

```bash
python3 ptq.py --model_name qwen3 --model_size 8b
```

如需指定校准集：

```bash
python3 ptq.py --model_name qwen3 --model_size 8b --calib_data ~/your_data.jsonl
```

量化结果位于 `output/${HOUMO_TARGET}/hmquant`。

### 4. 编译

```bash
python3 build.py --model_name qwen3 --model_size 8b
```

编译输出：

- `qwen3-8b_prefill.hmm`
- `qwen3-8b_decode.hmm`

### 5. 演示

```bash
python3 demo.py --model_name qwen3 --model_size 8b
```

若未显式传 `tokenizer_dir`、`prefill_path`、`decode_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导。

## test.sh

`test.sh` 用于串起下载、量化、编译、演示与 perf。默认执行 `demo`，默认模型为 `qwen3-8b`。

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
  -size, --model_size     Model size. Choices: 0.6b, 1.7b, 8b, 14b.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
bash test.sh --model_size 0.6b
bash test.sh -s quant --model_size 8b
bash test.sh -s build --model_size 8b
bash test.sh -s demo --model_size 8b
bash test.sh -s all --model_size 8b
bash test.sh -s demo --skip_download --model_size 8b
```

标准 demo 流程结束后，如果环境中存在 `llm_perf`，脚本还会自动执行：

```bash
python3 ${HOUMO_EXAMPLES_PATH}/tools/llm_perf/convert_embed.py --path output/${HOUMO_TARGET}/hmquant/quant_embedding.pt
llm_perf --model_name qwen3-8b ...
```

## 关键参数说明

### get_model.py

- `--config`: 配置文件路径，默认当前目录下 `config.yaml`
- `--model_name`: 模型名称，默认 `qwen3`
- `--model_size`: 模型尺寸，默认 `8b`
- `--type`: `raw` 或 `hmm`
- `--download_dir`: 下载目录
- `--extract_dir`: 解压目录

### ptq.py

- `--config`: 配置文件路径
- `--model`: 原始 HF 模型目录；默认根据 `modelscope_repo` 推导目录名
- `--model_name`: 模型名称，默认取配置
- `--model_size`: 模型尺寸，默认取配置
- `--out_dir`: 输出目录，默认 `output/${HOUMO_TARGET}`
- `--context_length`: 默认 `2048`
- `--input_sequence_length`: 默认跟随配置中的 `prefill_length`

### build.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--model_dir`: 默认 `output/${HOUMO_TARGET}/hmquant`
- `--output_dir`: 默认 `output/${HOUMO_TARGET}`
- `--context_length`: 默认按配置中的 `32k` / `16k` 解析
- `--enable_xh2_stable_output`: 开启 stable output 编译选项
- `--monitor_interval`: 内存监控采样周期

### demo.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--tokenizer_dir`: 默认根据 `modelscope_repo` 推导
- `--embedding_path`: 默认 `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `--prefill_path`, `--decode_path`: 默认按 `${model_name}-${model_size}` 自动拼接

## 数据与免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
