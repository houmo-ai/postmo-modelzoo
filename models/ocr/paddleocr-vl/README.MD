# PaddleOCR-VL

本目录提供 `PaddleOCR-VL` 在 XH2 平台上的下载、量化、编译与图文演示脚本。当前默认配置来自同目录下的 `config.yaml`。

## 支持模型

- `model_name`: `PaddleOCR-VL`
- `model_size`: `0.9b`

对应 ModelScope 模型：

- PadlleOCR-VL: https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL

预训练模型在运行时下载，仓库本身不包含这些权重。

## 配置与默认值

`config.yaml` 是默认值的单一来源，当前约定如下：

- 默认模型：`paddleocr-vl` + `0.9b`
- 默认 `context_length=4k`
- 默认 `prefill_length=256`
- 默认视觉尺寸：`448x448`
- 默认输出前缀统一使用 `${model_name}-${model_size}`
- tokenizer / 原始模型目录默认从 `modelscope_repo` 最后一段推导，即 `PaddleOCR-VL`

## 输出约定

量化输出：

- `output/${HOUMO_TARGET}/hmquant/prefill`
- `output/${HOUMO_TARGET}/hmquant/decode`
- `output/${HOUMO_TARGET}/hmquant/visual`
- `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`

编译输出：

- `output/${HOUMO_TARGET}/${model_name}-${model_size}_prefill.hmm`
- `output/${HOUMO_TARGET}/${model_name}-${model_size}_decode.hmm`
- `output/${HOUMO_TARGET}/${model_name}-${model_size}_visual_${max_size_w}x${max_size_h}.hmm`

当 `ndevice > 1` 时，单 batch demo 默认会把 `prefill` 和 `decode` 的后缀切换为 `.hmms`；视觉模型仍保持 `.hmm`。

## 快速开始

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 2. 下载模型

下载原始模型：

```bash
python3 get_model.py --type raw --model_name paddleocr-vl --model_size 0.9b
```

仅下载预编译模型用于演示：

```bash
python3 get_model.py --type hmm --model_name paddleocr-vl --model_size 0.9b
```

如需从 ModelScope 下载，可附加：

```bash
python3 get_model.py --type raw --source_type modelscope --model_name paddleocr-vl --model_size 0.9b
```

### 3. 量化

`ptq.py` 需要 GPU 环境。默认原始模型目录会根据 `config.yaml` 中的 `modelscope_repo` 自动推导。

```bash
python3 ptq.py --model-name paddleocr-vl --model-size 0.9b
```

如果要降低导出规模或使用自定义校准文本，可显式传参：

```bash
python3 ptq.py --model-name paddleocr-vl --model-size 0.9b --gptqmodel \
  --context-length 4096 \
  --input-sequence-length 256 \
```

### 4. 编译

```bash
python3 build.py --model_name paddleocr-vl --model_size 0.9b
```

如果需要显式调整上下文长度或 FlashAttention：

```bash
python3 build.py --model_name paddleocr-vl --model_size 0.9b --context_length 4096 --prefill_length 256 --flash_attention 2 1
```

### 5. 演示

默认单次 demo 会使用 `${HOUMO_PIC_PATH}/ocr.jpeg` 作为输入图片；若环境变量未设置，则回退到仓库内 `data/pic/ocr.jpeg`。

```bash
python3 demo.py --model_name paddleocr-vl --model_size 0.9b
```

指定图片：

```bash
python3 demo.py --model_name paddleocr-vl --model_size 0.9b --image_path ../../../data/pic/ocr.jpeg
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
- `--model_name`: 默认 `paddleocr-vl`
- `--model_size`: 默认 `0.9b`
- `--type`: `raw` 或 `hmm`
- `--context_length`: 默认取配置中的 `4k`

### ptq.py

- `--config`: 配置文件路径
- `--model-name`, `--model-size`: 默认取配置
- `--model`: 默认根据 `modelscope_repo` 推导
- `--context-length`: 默认按配置中的 `4k` 解析
- `--input-sequence-length`: 默认取配置中的 `256`
- `--quant-type`: 默认取配置中的 `w8a8h0_ssfp`

### build.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--model_dir`: 默认 `output/${HOUMO_TARGET}/hmquant`
- `--output_dir`: 默认 `output/${HOUMO_TARGET}`
- `--context_length`: 默认按配置中的 `4k` 解析
- `--prefill_length`: 默认取配置中的 `256`

### demo.py

- `--config`: 配置文件路径
- `--model_name`, `--model_size`: 默认取配置
- `--tokenizer_dir`: 默认根据 `modelscope_repo` 推导
- `--embedding_path`: 默认 `output/${HOUMO_TARGET}/hmquant/quant_embedding.pt`
- `--prefill_path`, `--decode_path`: 默认按 `${model_name}-${model_size}` 自动拼接
- `--vision_path`: 默认按 `${model_name}-${model_size}_visual_${max_size_w}x${max_size_h}.hmm` 自动拼接
- `--image_path`: 默认取 ocr 示例图

## 代码版权说明

本目录中的以下文件包含基于第三方开源实现的适配代码，后续如对外开源、分发或单独复用，需保留原始版权声明、修改说明及 Apache License 2.0 许可信息：

- `image_processing.py`
- `processing_paddleocr_vl.py`

## 参考结果

### 演示结果

以下为单张图片问答的参考输出：

```bash
2026-05-21 08:49:51.919 | SUCCESS  | __main__:chat:885 - question:
OCR:
2026-05-21 08:49:51.919 | INFO     | __main__:chat:887 - images: ['/mnt/data/nan.xu/project/xh2_modelzoo/tmp_imodelzoo/imodelzoo/data/pic/ocr.jpeg']
2026-05-21 08:49:52.013 | INFO     | __main__:chat:931 - processor image_grid_thw: tensor([[ 1, 32, 32]])
2026-05-21 08:49:52.013 | INFO     | __main__:chat:932 - processor image token count: 256
2026-05-21 08:49:52.143 | INFO     | __main__:chat:938 - image_embeds shape: torch.Size([256, 1024])
2026-05-21 08:49:52.182 | SUCCESS  | __main__:chat:1018 - response:
后摩智能
M50芯片
```

### 性能结果

以下为同一次运行的性能统计摘要：

```bash
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:316 - ====================================================================================================
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:317 -                     Model Inference Performance Summary Report
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:318 - ====================================================================================================
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:321 - Configuration Details:
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:322 -   Batch Size:      1
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:323 -   Input Length per Sample:    269 tokens
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:324 -   Output Length per Sample:     10 tokens
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:326 -   Number of Images:      1 images
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:329 -   Prefill Model Load Time: 3484.83ms
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:331 -   Decode Model Load Time:  118.31ms
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:333 -   Vision Model Load Time: 7358.59ms
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:338 - Vision Stage Performance:
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:339 -   Total Time:  218.48ms | Speed:    4.58 images/s
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:340 -   Preprocessing Time: 87.50ms | Speed:   11.43 images/s
2026-05-21 08:49:52.284 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:341 -   API SetInput Time:   0.57ms
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:342 -   API Inference Time: 127.06ms | Speed:    7.87 images/s
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:343 -   API GetOutput Time:  0.58ms
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:346 - Prefill Stage Performance:
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:347 -   Total Time:  262.86ms | Speed: 1023.37 tokens/s
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:349 -   Tokenization Time:    5.36ms
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:352 -   Embedding Time:    2.65ms
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:353 -   API SetInput Time:   8.34ms
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:354 -   API Inference Time: 24.74ms | Prefill Speed: 10873.40 tokens/s
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:355 -   API GetOutput Time:  0.15ms
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:358 - Decode Stage Performance:
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:359 -   Total Time:   99.29ms | Speed:   90.64 tokens/s
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:361 -   Tokenization Time:   29.07ms
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:364 -   Embedding Time:    0.85ms
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:365 -   API SetInput Time:   1.06ms
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:366 -   API Inference Time:  6.44ms | Decode Speed:  155.25 tokens/s
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:367 -   API GetOutput Time:  0.28ms
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:370 - Overall Performance Metrics:
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:371 -   TTFT (Time To First Token):  481.34 ms
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:372 -   TPOT (Time Per Output Token): 11.03 ms/token
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:373 -   E2E Latency (End-to-End):      0.44 seconds
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:374 -   E2E TPS (Throughput):         22.77 tokens/s
2026-05-21 08:49:52.285 | SUCCESS  | hmatc.utils.perf_infomations:show_summary:376 - ====================================================================================================
```

### 3.3 备注

- 当前 demo 已内置视觉、Prefill、Decode 各阶段性能统计；
- 若仅验证功能，建议优先下载预编译模型直接运行；
- 若需重新量化，请优先确认 GPU 显存（24G空闲）、主机空闲内存（40G空闲）以及 CUDA 版本满足要求；
- 由于模型中存在大量reshape，如需修改context_length则需重新执行ptq.py和build.py。

## 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
