# Qwen2.5

本目录提供 `qwen2.5-7b` 在 XH2 平台上的下载、量化、编译、演示与一键测试脚本。当前默认配置来自同目录下的 `config.yaml`。

## 支持模型

- `model_name`: `qwen2.5`
- `model_size`: `7b`

对应 ModelScope 模型：

- Qwen2.5-7B-Instruct: https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct

预训练模型在运行时下载，仓库本身不包含这些权重。

## 配置与默认值

`config.yaml` 是默认值的单一来源，当前约定如下：

- 默认模型：`qwen2.5` + `7b`
- 默认 `context_length=8k`
- 默认 `prefill_length=256`
- 默认输出前缀统一使用 `${model_name}-${model_size}`
- tokenizer 默认从 `modelscope_repo` 最后一段推导，即 `Qwen2.5-7B-Instruct`

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
python3 get_model.py --type raw --model_name qwen2.5 --model_size 7b
```

仅下载预编译模型用于演示：

```bash
python3 get_model.py --type hmm --model_name qwen2.5 --model_size 7b
```

### 3. 量化

`ptq.py` 需要 GPU 环境。默认原始模型目录会根据 `config.yaml` 中的 `modelscope_repo` 自动推导。

```bash
python3 ptq.py --model-name qwen2.5 --model-size 7b
```

如果要使用自定义校准数据集，可直接传 `--calib_data`：

```bash
python3 ptq.py --model-name qwen2.5 --model-size 7b --calib_data /path/to/custom.jsonl
```

### 4. 编译

```bash
python3 build.py --model_name qwen2.5 --model_size 7b
```

### 5. 演示

单 batch：

```bash
python3 demo.py --model_name qwen2.5 --model_size 7b
```

多 batch：

```bash
python3 demo_multibatch.py --model_name qwen2.5 --model_size 7b --forbid_flush
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
- `--model_name`: 默认 `qwen2.5`
- `--model_size`: 默认 `7b`
- `--type`: `raw` 或 `hmm`
- `--context_length`: 默认取配置中的 `8k`

### ptq.py

- `--config`: 配置文件路径
- `--model-name`, `--model-size`: 默认取配置
- `--model`: 默认根据 `modelscope_repo` 推导
- `--context-length`: 默认按配置中的 `8k` 解析
- `--input-sequence-length`: 默认取配置中的 `256`
- `--quant-type`: 默认取配置中的 `w4a8h0_ssfp`

### build.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--model_dir`: 默认 `output/${HOUMO_TARGET}/hmquant`
- `--output_dir`: 默认 `output/${HOUMO_TARGET}`
- `--context_length`: 默认按配置中的 `8k` 解析
- `--prefill_length`: 默认取配置中的 `256`

### demo.py / demo_multibatch.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--tokenizer_dir`: 默认根据 `modelscope_repo` 推导
- `--embedding_path`: 默认 `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `--prefill_path`, `--decode_path`: 默认按 `${model_name}-${model_size}` 自动拼接

## 参考结果

### 演示结果

```bash
User question:
请介绍一下存算一体技术的优势
Model response:
存算一体技术，即存储计算一体化技术，是一种将数据存储和数据处理功能集成在一起的新型计算架构。这种技术的优势主要包括以下几个方面：

1存储和计算资源的高度集成，可以减少数据传输带来的延迟和能耗，提高计算效率。在处理大规模数据时，存算一体技术可以显著提高处理速度，减少数据传输带来的能耗，提高能效比。

2. 降低能耗：存算一体技术可以减少数据在存储器和处理器之间传输的次数，从而降低能耗。这对于需要长时间运行的设备尤为重要，如数据中心、超级计算机等。

3. 支持更复杂的数据处理任务：存算一体技术可以支持更复杂的数据处理任务，如机器学习、人工智能等，这些任务通常需要大量的数据处理和存储。

通过将存储和计算功能集成在一起，存算一体技术可以更有效地处理这些任务。

4. 适应性更强：存算一体技术可以根据不同的应用场景和需求，灵活地调整存储和计算资源的比例，以满足不同的性能要求。这对于需要灵活调整计算资源的应用场景尤为重要，如边缘计算、物联网等。

存算一体技术是未来计算技术发展的重要方向之一，具有广泛的应用前景。
```

### 性能结果

```bash
====================================================================================================
                     Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
Batch Size:      1
Input Length per Sample:     26 tokens
Output Length per Sample:    251 tokens
Prefill Model Load Time: 1585.67ms
Decode Model Load Time:  181.21ms
Prefill Stage Performance:
Total Time:  118.39ms | Speed:  219.61 tokens/s
Tokenization Time:   16.52ms
embedding Time:    3.23ms
API SetInput Time:   2.24ms
API Inference Time: 95.04ms | Prefill Speed:  273.56 tokens/s
API GetOutput Time:  0.22ms
Decode Stage Performance:
Total Time: 11146.76ms | Speed:   22.43 tokens/s
Tokenization Time:   25.84ms
Embedding Time:  204.11ms
API SetInput Time:   0.22ms
API Inference Time: 42.71ms | Decode Speed:   23.41 tokens/s
API GetOutput Time:  0.21ms
Overall Performance Metrics:
TTFT (Time To First Token):  118.39 ms
TPOT (Time Per Output Token): 44.59 ms/token
E2E Latency (End-to-End):     11.30 seconds
E2E TPS (Throughput):         22.21 tokens/s
====================================================================================================
```

## 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
