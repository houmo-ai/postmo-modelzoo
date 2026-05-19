# DeepSeek R1 Qwen3 8B

本目录提供 `deepseek-8b` 在 XH2 平台上的下载、量化、编译、演示与一键测试脚本。当前默认配置来自同目录下的 `config.yaml`。

## 支持模型

- `model_name`: `deepseek`
- `model_size`: `8b`

对应 ModelScope 模型：

- DeepSeek-R1-0528-Qwen3-8B: https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

预训练模型在运行时下载，仓库本身不包含这些权重。

## 配置与默认值

`config.yaml` 是默认值的单一来源，当前约定如下：

- 默认模型：`deepseek` + `8b`
- 默认 `context_length=32k`
- 默认 `prefill_length=256`
- 默认输出前缀统一使用 `${model_name}-${model_size}`
- tokenizer 默认从 `modelscope_repo` 最后一段推导，即 `DeepSeek-R1-0528-Qwen3-8B`

## 输出约定

量化输出：

- `output/${HOUMO_TARGET}/hmquant/prefill`
- `output/${HOUMO_TARGET}/hmquant/decoder`
- `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`

编译输出：

- `output/${HOUMO_TARGET}/${model_name}-${model_size}_prefill.hmm`
- `output/${HOUMO_TARGET}/${model_name}-${model_size}_decode.hmm`

当 `ndevice > 1` 时，单 batch demo 默认会把 `prefill` 和 `decode` 的后缀切换为 `.hmms`。

## 快速开始

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 2. 下载模型

下载原始模型：

```bash
python3 get_model.py --type raw --model_name deepseek --model_size 8b
```

仅下载预编译模型用于演示：

```bash
python3 get_model.py --type hmm --model_name deepseek --model_size 8b
```

### 3. 量化

`ptq.py` 需要 GPU 环境。默认原始模型目录会根据 `config.yaml` 中的 `modelscope_repo` 自动推导。

```bash
python3 ptq.py --model-name deepseek --model-size 8b
```

如果要使用自定义校准数据集，可直接传 `--calib_data`：

```bash
python3 ptq.py --model-name deepseek --model-size 8b --calib_data /path/to/custom.jsonl
```

### 4. 编译

```bash
python3 build.py --model_name deepseek --model_size 8b
```

### 5. 演示

单 batch：

```bash
python3 demo.py --model_name deepseek --model_size 8b
```

多 batch：

```bash
python3 demo_multibatch.py --model_name deepseek --model_size 8b --forbid_flush
```

如果未显式传 `tokenizer_dir`、`prefill_path`、`decode_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导。

### 6. test.sh

`test.sh` 默认执行 `demo`，会先下载预编译模型。

```bash
bash test.sh
bash test.sh -s quant
bash test.sh -s build
bash test.sh -s all
bash test.sh -b 2
bash test.sh -s demo --skip_download
```

标准单 batch demo 结束后，如果环境中存在 `llm_perf`，脚本会自动执行显式参数形式的性能测试，不再依赖 `llm_perf -c config.yaml`。

## 关键参数说明

### get_model.py

- `--config`: 配置文件路径，默认当前目录下 `config.yaml`
- `--model_name`: 默认 `deepseek`
- `--model_size`: 默认 `8b`
- `--type`: `raw` 或 `hmm`
- `--context_length`: 默认取配置中的 `32k`

### ptq.py

- `--config`: 配置文件路径
- `--model-name`, `--model-size`: 默认取配置
- `--model`: 默认根据 `modelscope_repo` 推导
- `--context-length`: 默认按配置中的 `32k` 解析
- `--input-sequence-length`: 默认取配置中的 `256`
- `--quant-type`: 默认取配置中的 `w4a8h0_ssfp`

### build.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--model_dir`: 默认 `output/${HOUMO_TARGET}/hmquant`
- `--output_dir`: 默认 `output/${HOUMO_TARGET}`
- `--context_length`: 默认按配置中的 `32k` 解析
- `--prefill_length`: 默认取配置中的 `256`

### demo.py / demo_multibatch.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--tokenizer_dir`: 默认根据 `modelscope_repo` 推导
- `--embedding_path`: 默认 `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `--prefill_path`, `--decode_path`: 默认按 `${model_name}-${model_size}` 自动拼接

## 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
