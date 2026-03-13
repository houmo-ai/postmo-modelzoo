# CosyVoice3

本示例展示如何将 **Fun-CosyVoice3-0.5B-2512** 模型进行量化与编译，并在后摩鸿途 **XH2** 设备上进行推理演示。

## 预训练模型

本项目使用的预训练模型为：

- 模型名称: Fun-CosyVoice3-0.5B-2512
- 上游项目/模型仓: FunAudioLLM/Fun-CosyVoice3-0.5B-2512（ModelScope）

预训练模型在运行时下载，工程发布 **不包含** 该模型文件。

[TOC]

## 1 模型说明

`cosyvoice3` 目录包含以下流程脚本：

- `get_model.py`: 下载原始模型文件（`raw`）或已编译模型（`hmm`）
- `ptq.py`: PTQ 量化（生成 HMONNX/权重/embedding 等产物）
- `build.py`: 使用 tcim 将 HMONNX 编译为 `.hmm`
- `demo.py`: 在 XH2 上执行端到端 cosyvoice3 推理示例
- `test.sh`: 一键下载/量化/编译/演示

## 2 快速开始

说明：

- 仅支持 `HOUMO_TARGET=xh2`
- **量化需要 GPU 环境**（`test.sh -t compile` 会检查 `nvidia-smi`）

### 2.1 获取模型

下载 **已编译模型**（推荐，仅跑演示）：

```bash
cd models/tts/cosyvoice3
python3 get_model.py --type hmm
```

下载 **原始模型文件**（用于自行量化/编译）：

```bash
cd models/tts/cosyvoice3
python3 get_model.py --type raw
```

常用参数：

```bash
python3 get_model.py -h
```

### 2.2 量化（PTQ）

在已准备好工具链环境后，执行：

```bash
cd models/tts/cosyvoice3
python3 ptq.py
```

默认产物目录为 `output/$HOUMO_TARGET/hmquant`，其中会包含多子目录与文件（示例）：

```text
output/xh2/hmquant
|-- campplus/
|-- speech_tokenizer/
|-- llm_prefill/
|-- llm_decode/
|-- llm_decoder/
|-- flow_encoder/
|-- flow_spk/
|-- flow_decoder/
|-- hift/
`-- quant_embedding.pt
`-- flow_input_embedding.pt
`-- llm_sos_embedding.pt
`-- llm_task_id_embedding.pt
`-- llm_speech_embedding.pt
```

### 2.3 编译（HM-ONNX → HMM）

将量化产物编译为芯片可运行的 `.hmm`：

```bash
cd models/tts/cosyvoice3
python3 build.py
```

编译产物默认位于 `output/$HOUMO_TARGET/`, `demo.py` 默认会从该目录加载以下文件（可通过参数覆盖）：

- `cosyvoice3_campplus.hmm`
- `cosyvoice3_speech_tokenizer.hmm`
- `cosyvoice3_llm_qwen2_prefill.hmm`
- `cosyvoice3_llm_qwen2_decode.hmm`
- `cosyvoice3_llm_decoder.hmm`
- `cosyvoice3_flow_spk.hmm`
- `cosyvoice3_flow_encoder.hmm`
- `cosyvoice3_flow_decoder.hmm`
- `cosyvoice3_hift_part1.hmm`
- `cosyvoice3_hift_part2.hmm`

说明：本仓库的 `build.py` 内部包含多个子模块的编译入口；如仅跑演示，优先使用 `get_model.py --type hmm` 获取已编译模型。

### 2.4 演示（cosyvoice3 推理）

执行推理示例（会下载 tokenizer 等必要资源，结果音频写入 `--output_dir` 目录， 默认为 `./results`）：

```bash
cd models/tts/cosyvoice3
python3 demo.py
```

更多参数：

```bash
python3 demo.py -h
```

### 2.5 一键执行

一键下载并运行演示（使用预编译模型）：

```bash
cd models/tts/cosyvoice3
bash test.sh
```

一键执行“下载原始模型 → 量化 → 编译 → 演示”（需要 GPU 和工具链依赖）：

```bash
cd models/tts/cosyvoice3
bash test.sh -t compile
```

## 3 参考结果

### 3.1 演示结果

```bash
2026-03-13 09:52:43.109 | INFO     | __main__:inference_zero_shot:1952 - Start inference zero shot, zero_shot_spk_id: , speed: 1.0, text_frontend: True
  0%|                       | 0/1 [00:00<?, ?it/s]
2026-03-13 09:52:43.331 | INFO     | __main__:inference_zero_shot:1980 - synthesis text 下面为您朗诵一段绕口令，希望您[j][ǐ]予好评，朗诵开始:八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。
2026-03-13 09:53:17.101 | INFO     | __main__:tts:1828 - tts, uuid: 617dc7a9-1ec2-11f1-9ab6-4dff60635bb1, release resources.
2026-03-13 09:53:17.101 | SUCCESS  | __main__:inference_zero_shot:1982 - generate tts speech successfully.
100%|████████████████████████| 1/1 [00:34<00:00, 34.02s/it]
  0%|                        | 0/1 [00:00<?, ?it/s]
2026-03-13 09:53:17.340 | INFO     | __main__:inference_cross_lingual:2004 - synthesis text You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]
2026-03-13 09:53:49.480 | INFO     | __main__:tts:1828 - tts, uuid: 75c30f6b-1ec2-11f1-b204-4dff60635bb1, release resources.
2026-03-13 09:53:49.480 | SUCCESS  | __main__:inference_cross_lingual:2006 - Generate tts speech successfully.
100%|████████████████████████| 1/1 [00:32<00:00, 32.38s/it]
  0%|                        | 0/1 [00:00<?, ?it/s]
2026-03-13 09:53:49.703 | INFO     | __main__:inference_instruct2:2028 - synthesis text 好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。
2026-03-13 09:54:20.955 | INFO     | __main__:tts:1828 - tts, uuid: 890d3c59-1ec2-11f1-abc6-4dff60635bb1, release resources.
2026-03-13 09:54:20.956 | SUCCESS  | __main__:inference_instruct2:2030 - generate tts speech successfully.
100%|████████████████████████| 1/1 [00:31<00:00, 31.47s/it]
```

### 3.2 性能结果

#### 性能评估指标

性能指标说明（每条报告对应一次 `tts()`）：

一次 `tts()`：指对 **一个待合成的文本分段（utterance）** 发起一次完整端到端合成调用，包含 **LLM 生成 speech token** 与 **`token2wav()` 生成波形** 两阶段。长文本会先被 `text_normalize(..., split=True)` 切分为多个分段，因此可能对应多次 `tts()` 与多条性能报告。

- **LLM Total Cost**：LLM 阶段总耗时（ms），对应 speech token 生成阶段累计时间。
- **LLM Prefill Speed**：LLM prefill 阶段速度（tokens/s）。token 数按 LLM 的输入序列长度统计。
- **TTFT (Time to First Token)**：从进入 LLM 推理到生成首个 speech token 的时间（ms）。
- **TPOT (Time Per Output Token)**：LLM decode 阶段生成 token 的速度（tokens/s）。
- **TTS Total Cost**：`token2wav()` 阶段耗时（ms），不包含 LLM 阶段。
- **TTS Real-Time Factor (RTF)**：当前计算方式为 `RTF = E2E Latency / 音频时长`。
- **TTS Generate Speed**：相对实时倍速，等价于 `1 / RTF`（例如 `0.42 x real-time`）。
- **E2E Latency (End-to-End Latency)**：端到端时延（秒），用于表示一次 `tts()` 调用从开始执行到返回最终音频的整体耗时。

#### M50 2core
```bash
2026-03-13 10:10:54.364 | INFO     | __main__:print_perf_summary:1925 -
[Perf #0]
LLM Total Cost 4830.223 ms
LLM Prefill Speed: 2998.75 tokens/s
TTFT (Time to First Token): 63.516 ms
TPOT (Time Per Output Token): 85.59 tokens/s
TTS Total Cost: 29395.319 ms
TTS Real-Time Factor(RTF): 2.222735
TTS Generate Speed: 0.45 x real-time
E2E Latency (End-to-End Latency): 34.230 s
2026-03-13 10:10:54.364 | INFO     | __main__:print_perf_summary:1925 -
[Perf #1]
LLM Total Cost 2773.767 ms
LLM Prefill Speed: 884.67 tokens/s
TTFT (Time to First Token): 58.831 ms
TPOT (Time Per Output Token): 91.72 tokens/s
TTS Total Cost: 29388.707 ms
TTS Real-Time Factor(RTF): 3.216384
TTS Generate Speed: 0.31 x real-time
E2E Latency (End-to-End Latency): 32.164 s
2026-03-13 10:10:54.364 | INFO     | __main__:print_perf_summary:1925 -
[Perf #2]
LLM Total Cost 1599.579 ms
LLM Prefill Speed: 838.54 tokens/s
TTFT (Time to First Token): 59.643 ms
TPOT (Time Per Output Token): 85.07 tokens/s
TTS Total Cost: 29399.501 ms
TTS Real-Time Factor(RTF): 5.871297
TTS Generate Speed: 0.17 x real-time
E2E Latency (End-to-End Latency): 31.000 s
```

## 4 免责声明

您明确了解并同意，上游模型/数据由第三方提供并负责维护。文档中出现的任何第三方名称、商标、标识、产品或服务并不构成明示或暗示的背书、担保或推荐行为。使用任何第三方软件、数据或模型应遵守其各自的使用条款、许可协议与隐私政策；相关风险由您自行承担。
