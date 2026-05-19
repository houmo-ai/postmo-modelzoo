# CoPaw-Flash

本目录提供 `copaw-flash-9b` 在 XH2 平台上的下载、量化、编译与图文演示脚本。当前默认配置来自同目录下的 `config.yaml`。

## 支持模型

- `model_name`: `copaw-flash`
- `model_size`: `9b`

对应 ModelScope 模型：

- CoPaw-Flash-9B: https://modelscope.cn/models/AgentScope/CoPaw-Flash-9B

预训练模型在运行时下载，仓库本身不包含这些权重。

## 配置与默认值

`config.yaml` 是默认值的单一来源，当前约定如下：

- 默认模型：`copaw-flash` + `9b`
- 默认 `context_length=8k`
- 默认 `prefill_length=256`
- 默认视觉尺寸：`448x448x2`
- 默认输出前缀统一使用 `${model_name}-${model_size}`
- tokenizer / 原始模型目录默认从 `modelscope_repo` 最后一段推导，即 `CoPaw-Flash-9B`

## 输出约定

量化输出：

- `output/${HOUMO_TARGET}/hmquant/prefill`
- `output/${HOUMO_TARGET}/hmquant/decode`
- `output/${HOUMO_TARGET}/hmquant/visual`
- `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`

编译输出：

- `output/${HOUMO_TARGET}/${model_name}-${model_size}_prefill.hmm`
- `output/${HOUMO_TARGET}/${model_name}-${model_size}_decode.hmm`
- `output/${HOUMO_TARGET}/${model_name}-${model_size}_visual_${max_size_w}x${max_size_h}x${max_size_t}.hmm`

当 `ndevice > 1` 时，单 batch demo 默认会把 `prefill` 和 `decode` 的后缀切换为 `.hmms`；视觉模型仍保持 `.hmm`。

## 快速开始

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 2. 下载模型

下载原始模型：

```bash
python3 get_model.py --type raw --model_name copaw-flash --model_size 9b
```

仅下载预编译模型用于演示：

```bash
python3 get_model.py --type hmm --model_name copaw-flash --model_size 9b
```

如需从 ModelScope 下载，可附加：

```bash
python3 get_model.py --type raw --source_type modelscope --model_name copaw-flash --model_size 9b
```

### 3. 量化

`ptq.py` 需要 GPU 环境。默认原始模型目录会根据 `config.yaml` 中的 `modelscope_repo` 自动推导。

```bash
python3 ptq.py --model-name copaw-flash --model-size 9b --gptqmodel
```

如果要降低导出规模或使用自定义校准文本，可显式传参：

```bash
python3 ptq.py --model-name copaw-flash --model-size 9b --gptqmodel \
  --context-length 2048 \
  --input-sequence-length 256 \
  --calib_data /path/to/custom.jsonl
```

### 4. 编译

```bash
python3 build.py --model_name copaw-flash --model_size 9b
```

如果需要显式调整上下文长度或 FlashAttention：

```bash
python3 build.py --model_name copaw-flash --model_size 9b --context_length 8192 --prefill_length 256 --flash_attention 2 1
```

### 5. 演示

默认单次 demo 会使用 `${HOUMO_PIC_PATH}/beach.jpeg` 作为输入图片；若环境变量未设置，则回退到仓库内 `data/pic/beach.jpeg`。

```bash
python3 demo.py --model_name copaw-flash --model_size 9b
```

指定图片：

```bash
python3 demo.py --model_name copaw-flash --model_size 9b --image_path ../../../data/pic/beach.jpeg
```

交互模式：

```bash
python3 demo.py --model_name copaw-flash --model_size 9b --it
```

如未显式传 `tokenizer_dir`、`prefill_path`、`decode_path`、`vision_path`，脚本会按 `config.yaml` 和 `${model_name}-${model_size}` 自动推导。

### 6. test.sh

`test.sh` 默认执行 `demo`，会先下载预编译模型。

```bash
bash test.sh
bash test.sh -s quant
bash test.sh -s build
bash test.sh -s all
bash test.sh -s demo --skip_download
```

当前目录没有单独的 `llm_perf` 流程；性能信息由 `demo.py` 内部统计输出。

## 关键参数说明

### get_model.py

- `--config`: 配置文件路径，默认当前目录下 `config.yaml`
- `--model_name`: 默认 `copaw-flash`
- `--model_size`: 默认 `9b`
- `--type`: `raw` 或 `hmm`
- `--context_length`: 默认取配置中的 `8k`

### ptq.py

- `--config`: 配置文件路径
- `--model-name`, `--model-size`: 默认取配置
- `--model`: 默认根据 `modelscope_repo` 推导
- `--context-length`: 默认按配置中的 `8k` 解析
- `--input-sequence-length`: 默认取配置中的 `256`
- `--quant-type`: 默认取配置中的 `w8a8h0_ssfp`
- `--gptqmodel`: 启用 GPTQ 量化分支

### build.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--model_dir`: 默认 `output/${HOUMO_TARGET}/hmquant`
- `--output_dir`: 默认 `output/${HOUMO_TARGET}`
- `--context_length`: 默认按配置中的 `8k` 解析
- `--prefill_length`: 默认取配置中的 `256`

### demo.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--tokenizer_dir`: 默认根据 `modelscope_repo` 推导
- `--embedding_path`: 默认 `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `--prefill_path`, `--decode_path`: 默认按 `${model_name}-${model_size}` 自动拼接
- `--vision_path`: 默认按 `${model_name}-${model_size}_visual_${max_size_w}x${max_size_h}x${max_size_t}.hmm` 自动拼接
- `--image_path`: 默认取 beach 示例图

## 代码版权说明

本目录中的以下文件包含基于第三方开源实现的适配代码，后续如对外开源、分发或单独复用，需保留原始版权声明、修改说明及 Apache License 2.0 许可信息：

- `image_processing_copawflash.py`
- `vision_process_copawflash.py`

## 参考结果

### 演示结果

以下为单张图片问答的参考输出：

```bash
question:
分析这张图片中主要内容。
response:
这张图片呈现了一个宁静而富有情感的场景，主要内容如下：

- **主体人物与动物**：画面中有一位女性和一只大型犬（看起来像金毛寻回犬或类似品种）。女性留着长发，身穿格子衬衫和深色裤子，坐在沙地上。狗狗则戴着蓝色项圈，正抬头望着女性，姿态温顺。

- **互动关系**：女性似乎正在与狗狗互动——可能是在喂食、抚摸或轻声说话，营造出一种亲密、信任的氛围。

- **环境背景**：场景位于一片开阔的沙滩上，远处是模糊的海平线和柔和的天空，整体光线偏暗，色调沉静，可能是在黄昏或阴天拍摄，增强了画面的静谧感。

- **整体氛围**：图像传达出人与自然、人与动物之间和谐共处的温馨意境，带有一种孤独却不寂寞的诗意美感。

总结：这是一张表现人与宠物在自然环境中共享宁静时刻的照片，强调陪伴、信任与平和的情感联结。
```

### 性能结果

以下为同一次运行的性能统计摘要：

```bash
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:316 - ====================================================================================================
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:317 -                     Model Inference Performance Summary Report
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:318 - ====================================================================================================
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:321 - Configuration Details:
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:322 -   Batch Size:      1
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:323 -   Input Length per Sample:    216 tokens
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:324 -   Output Length per Sample:    203 tokens
2026-04-16 04:55:27.828 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:326 -   Number of Images:      1 images
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:329 -   Prefill Model Load Time: 43709.12ms
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:331 -   Decode Model Load Time: 1104.04ms
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:333 -   Vision Model Load Time: 7247.46ms
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:338 - Vision Stage Performance:
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:339 -   Total Time:  262.85ms | Speed:    3.80 images/s
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:340 -   Preprocessing Time: 107.21ms | Speed:    9.33 images/s
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:341 -   API SetInput Time:   3.77ms
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:342 -   API Inference Time: 149.75ms | Speed:    6.68 images/s
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:343 -   API GetOutput Time:  1.34ms
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:346 - Prefill Stage Performance:
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:347 -   Total Time:  465.60ms | Speed:  463.92 tokens/s
2026-04-16 04:55:27.829 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:349 -   Tokenization Time:   18.43ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:352 -   Embedding Time:    7.40ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:353 -   API SetInput Time:  11.27ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:354 -   API Inference Time: 156.47ms | Prefill Speed: 1380.47 tokens/s
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:355 -   API GetOutput Time:  0.64ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:358 - Decode Stage Performance:
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:359 -   Total Time: 12759.69ms | Speed:   15.83 tokens/s
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:361 -   Tokenization Time: 2436.97ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:364 -   Embedding Time:   27.95ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:365 -   API SetInput Time:   1.83ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:366 -   API Inference Time: 48.63ms | Decode Speed:   20.57 tokens/s
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:367 -   API GetOutput Time:  0.60ms
2026-04-16 04:55:27.830 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:370 - Overall Performance Metrics:
2026-04-16 04:55:27.831 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:371 -   TTFT (Time To First Token):  728.45 ms
2026-04-16 04:55:27.831 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:372 -   TPOT (Time Per Output Token): 63.17 ms/token
2026-04-16 04:55:27.831 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:373 -   E2E Latency (End-to-End):     13.92 seconds
2026-04-16 04:55:27.831 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:374 -   E2E TPS (Throughput):         14.58 tokens/s
2026-04-16 04:55:27.831 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:376 - ====================================================================================================
```

### 3.3 备注

- 当前 demo 已内置视觉、Prefill、Decode 各阶段性能统计；
- 若仅验证功能，建议优先下载预编译模型直接运行；
- 若需重新量化，请优先确认 GPU 显存、主机空闲内存以及 CUDA 版本满足要求。

## 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
