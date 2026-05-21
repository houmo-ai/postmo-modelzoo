# Qwen3-ASR

本示例展示如何把Qwen3-ASR模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于：

- 模型名称: Qwen3-ASR-0.6B / Qwen3-ASR-1.7B / Qwen3-ForcedAligner-0.6B
- 来源: [https://modelscope.cn/models/Qwen](https://modelscope.cn/models/Qwen)
- 许可: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型，本例只适用于xh2。

## 2 快速开始

目前Qwen3-ASR模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

注：由于本示例依赖的transformer版本高于python3.9最高支持的4.33.0版本，因此无法在python3.10以下版本（如ubuntu20.04）运行，如遇到找不到transformer库问题请自行安装python3.10以上版本。python3.10及以上版本运行示例可以执行如下指令安装依赖。
```bash
pip3 install -r requirements.txt
```

python3.9 对于requirements.txt中的qwen_asr需要从[https://github.com/QwenLM/Qwen3-ASR.git](https://github.com/QwenLM/Qwen3-ASR.git)下载源码，将qwen_asr文件夹放到当前目录下，并修改qwen_asr/core/transformers_backend/modeling_qwen3_asr.py的17行和1329行，改动后的代码如下：
``` python
# 17 行
from typing import Callable, Optional, Union, List
# 1329 行
eos_token_id: Union[int, List[int]] = [151645, 151643],
```
此外需要卸载scikit-learn，安装nagisa。

### 2.2 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
# Qwen3-ASR-0.6B (默认)
python3 get_model.py --type hmm

# Qwen3-ASR-1.7B
python3 get_model.py --type hmm --model_size 1.7b

# Qwen3-ForcedAligner-0.6B
python3 get_model.py --type hmm --model_name qwen3-forcealigner
```

如果需要自行编译模型，可以下载原始模式，下载方式如下：

```bash
# Qwen3-ASR-0.6B (默认)
python3 get_model.py --type raw

# Qwen3-ASR-1.7B
python3 get_model.py --type raw --model_size 1.7b

# Qwen3-ForcedAligner-0.6B
python3 get_model.py --type raw --model_name qwen3-forcealigner
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上，空闲内存需要达到100GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu20.04-py39-x86.64 /bin/bash
```

进入docker后source环境变量，并进入`models/asr/qwen3-asr`目录，执行ptq.py脚本。默认的量化方式为w8a8_sefp。

```bash
source env.sh
cd models/asr/qwen3-asr

# Qwen3-ASR-0.6B (默认)
python3 ptq.py

# Qwen3-ASR-1.7B
python3 ptq.py --model_size 1.7b --model Qwen3-ASR-1.7B

# Qwen3-ForcedAligner-0.6B
python3 ptq.py --model_name qwen3-forcealigner --model_size 0.6b --model Qwen3-ForcedAligner-0.6B
```

`config.yaml` 中的 `context_length` 使用字符串形式维护，例如 `2k`、`1k`、`0.5k`。`build.py` 会统一通过公共解析函数把它转换为实际整数。

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
# Qwen3-ASR-0.6B / Qwen3-ASR-1.7B (默认)
python3 build.py

# Qwen3-ASR-1.7B
python3 build.py --model_name qwen3-asr --model_size 1.7b --context_length 2k

# Qwen3-ForcedAligner-0.6B
python3 build.py --model_name qwen3-forcealigner --model_size 0.6b --context_length 2k
```

编译好的模型存放在当前目录，包括3个文件：
- qwen3-asr-0.6b_encode.hmm
- qwen3-asr-0.6b_decode.hmm
- qwen3-asr-0.6b_prefill.hmm

或（Qwen3-ForcedAligner-0.6B）：
- qwen3-forcealigner-0.6b_encode.hmm
- qwen3-forcealigner-0.6b_prefill.hmm

### 2.5 演示

Qwen3-ASR模型使用python API进行演示。

```bash
# Qwen3-ASR-0.6B (默认)
python3 demo_asr.py

# Qwen3-ASR-1.7B
python3 demo_asr.py --model_name qwen3-asr --model_size 1.7b

# Qwen3-ForcedAligner-0.6B
python3 demo_forcealigner.py --model_name qwen3-forcealigner --model_size 0.6b
```

两个 demo 都会默认从 `config.yaml` 读取 `model_name`、`model_size`、`ndevice`，并自动推导 `processor_dir` 与 HMM 路径；只有在需要覆盖默认值时才需要手工传参。

推理完成将输出性能结果。

### 2.6 一键评估

以上步骤可以通过test.sh脚本执行。默认执行demo步骤，默认模型为 qwen3-asr 0.6b。

```bash
# Qwen3-ASR-0.6B，默认执行 demo
bash test.sh

# Qwen3-ASR-1.7B，执行 demo
bash test.sh -name qwen3-asr -size 1.7b

# Qwen3-ForcedAligner-0.6B，执行 demo
bash test.sh -name qwen3-forcealigner -size 0.6b

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all -name qwen3-asr -size 1.7b
```

test.sh支持以下参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -size, --model_size     Model size.
  -name, --model_name     Model name.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

其中支持的模型组合为：

- `qwen3-asr` + `0.6b`
- `qwen3-asr` + `1.7b`
- `qwen3-forcealigner` + `0.6b`

示例：

```bash
# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant -name qwen3-asr -size 1.7b

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build -name qwen3-forcealigner -size 0.6b

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo -name qwen3-asr -size 1.7b

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -s all -name qwen3-forcealigner -size 0.6b

# 跳过模型下载，使用当前目录下已有的预编译模型进行演示
bash test.sh -s demo --skip_download -name qwen3-asr -size 0.6b
```

## 3 参考结果

### 3.1 演示结果

#### Qwen3-ASR-0.6B:
```bash
Device 0 -> Group 0, Chip 0
2026-05-20 13:05:57.637 | INFO     | __main__:__init__:204 - encode model loaded
2026-05-20 13:05:58.203 | INFO     | __main__:__init__:207 - prefill model loaded
2026-05-20 13:05:58.258 | INFO     | __main__:__init__:219 - decode model loaded
/hmdd/imodelzoo/models/asr/qwen3-asr/demo_asr.py:261: DeprecationWarning: get_input() deprecated: Use get_dev_input instead.
  cache = self.prefill.get_input(self.prefill.get_input_name(i))
每一天都要快乐哦。
2026-05-20 13:05:59.421 | SUCCESS  | __main__:run:580 - ============================================================
2026-05-20 13:05:59.421 | SUCCESS  | __main__:run:581 - Performance Statistics:
2026-05-20 13:05:59.422 | SUCCESS  | __main__:run:582 - ============================================================
2026-05-20 13:05:59.422 | SUCCESS  | __main__:run:583 - Audio duration: 2431.94 ms (2.43 s)
2026-05-20 13:05:59.422 | SUCCESS  | __main__:run:586 - Audio loading time: 444.255 ms
2026-05-20 13:05:59.422 | SUCCESS  | __main__:run:587 - Input preparation time: 7.453 ms
2026-05-20 13:05:59.422 | SUCCESS  | __main__:run:588 - Loop count: 1
2026-05-20 13:05:59.422 | SUCCESS  | __main__:run:589 - ------------------------------------------------------------
2026-05-20 13:05:59.422 | SUCCESS  | __main__:run:590 - Total Encode time: 11.730 ms
2026-05-20 13:05:59.423 | SUCCESS  | __main__:run:593 -   - Encode prep time: 0.110 ms
2026-05-20 13:05:59.423 | SUCCESS  | __main__:run:594 -   - Encode infer time: 11.400 ms
2026-05-20 13:05:59.423 | SUCCESS  | __main__:run:597 -   - Encode output time: 0.221 ms
2026-05-20 13:05:59.423 | SUCCESS  | __main__:run:600 - ------------------------------------------------------------
2026-05-20 13:05:59.423 | SUCCESS  | __main__:run:601 - Total Prefill time: 34.710 ms
2026-05-20 13:05:59.423 | SUCCESS  | __main__:run:604 -   - Prefill prep time: 0.593 ms
2026-05-20 13:05:59.423 | SUCCESS  | __main__:run:607 -   - Prefill infer time: 33.735 ms
2026-05-20 13:05:59.423 | SUCCESS  | __main__:run:610 -   - Prefill output time: 0.382 ms
2026-05-20 13:05:59.424 | SUCCESS  | __main__:run:613 - ------------------------------------------------------------
2026-05-20 13:05:59.424 | SUCCESS  | __main__:run:614 - Output 8 tokens
2026-05-20 13:05:59.424 | SUCCESS  | __main__:run:615 - Total Decode time(infer + embedding): 111.532 ms
2026-05-20 13:05:59.424 | SUCCESS  | __main__:run:618 -   - Decode infer total time: 84.596 ms
2026-05-20 13:05:59.424 | SUCCESS  | __main__:run:621 -   - Decode infer avg time: 10.575 ms
2026-05-20 13:05:59.424 | SUCCESS  | __main__:run:623 - Decode Speed: 71.73 tokens/s
2026-05-20 13:05:59.424 | SUCCESS  | __main__:run:627 - TPOT (Time Per Output Token): 13.942 ms/token
2026-05-20 13:05:59.424 | SUCCESS  | __main__:run:630 - ------------------------------------------------------------
2026-05-20 13:05:59.424 | SUCCESS  | __main__:run:631 - E2E Latency (End-to-End Latency): 0.168 seconds
2026-05-20 13:05:59.425 | SUCCESS  | __main__:run:633 - E2E TPS (End-to-End Tokens Per Second): 47.66 tokens/s
2026-05-20 13:05:59.425 | SUCCESS  | __main__:run:636 - RTF (Real Time Factor): 0.07
2026-05-20 13:05:59.425 | SUCCESS  | __main__:run:637 - ============================================================
```


#### Qwen3-ASR-1.7B:
```bash
Device 0 -> Group 0, Chip 0
2026-05-20 13:10:05.884 | INFO     | __main__:__init__:204 - encode model loaded
2026-05-20 13:10:13.767 | INFO     | __main__:__init__:207 - prefill model loaded
2026-05-20 13:10:13.952 | INFO     | __main__:__init__:219 - decode model loaded
/hmdd/imodelzoo/models/asr/qwen3-asr/demo_asr.py:261: DeprecationWarning: get_input() deprecated: Use get_dev_input instead.
  cache = self.prefill.get_input(self.prefill.get_input_name(i))
每一天都要快乐哦！
2026-05-20 13:10:15.262 | SUCCESS  | __main__:run:580 - ============================================================
2026-05-20 13:10:15.262 | SUCCESS  | __main__:run:581 - Performance Statistics:
2026-05-20 13:10:15.262 | SUCCESS  | __main__:run:582 - ============================================================
2026-05-20 13:10:15.262 | SUCCESS  | __main__:run:583 - Audio duration: 2431.94 ms (2.43 s)
2026-05-20 13:10:15.263 | SUCCESS  | __main__:run:586 - Audio loading time: 442.815 ms
2026-05-20 13:10:15.263 | SUCCESS  | __main__:run:587 - Input preparation time: 7.133 ms
2026-05-20 13:10:15.263 | SUCCESS  | __main__:run:588 - Loop count: 1
2026-05-20 13:10:15.263 | SUCCESS  | __main__:run:589 - ------------------------------------------------------------
2026-05-20 13:10:15.263 | SUCCESS  | __main__:run:590 - Total Encode time: 15.533 ms
2026-05-20 13:10:15.263 | SUCCESS  | __main__:run:593 -   - Encode prep time: 0.113 ms
2026-05-20 13:10:15.263 | SUCCESS  | __main__:run:594 -   - Encode infer time: 15.007 ms
2026-05-20 13:10:15.263 | SUCCESS  | __main__:run:597 -   - Encode output time: 0.413 ms
2026-05-20 13:10:15.263 | SUCCESS  | __main__:run:600 - ------------------------------------------------------------
2026-05-20 13:10:15.264 | SUCCESS  | __main__:run:601 - Total Prefill time: 46.440 ms
2026-05-20 13:10:15.264 | SUCCESS  | __main__:run:604 -   - Prefill prep time: 0.893 ms
2026-05-20 13:10:15.264 | SUCCESS  | __main__:run:607 -   - Prefill infer time: 45.078 ms
2026-05-20 13:10:15.264 | SUCCESS  | __main__:run:610 -   - Prefill output time: 0.470 ms
2026-05-20 13:10:15.264 | SUCCESS  | __main__:run:613 - ------------------------------------------------------------
2026-05-20 13:10:15.264 | SUCCESS  | __main__:run:614 - Output 8 tokens
2026-05-20 13:10:15.264 | SUCCESS  | __main__:run:615 - Total Decode time(infer + embedding): 180.232 ms
2026-05-20 13:10:15.264 | SUCCESS  | __main__:run:618 -   - Decode infer total time: 154.039 ms
2026-05-20 13:10:15.265 | SUCCESS  | __main__:run:621 -   - Decode infer avg time: 19.255 ms
2026-05-20 13:10:15.265 | SUCCESS  | __main__:run:623 - Decode Speed: 44.39 tokens/s
2026-05-20 13:10:15.265 | SUCCESS  | __main__:run:627 - TPOT (Time Per Output Token): 22.529 ms/token
2026-05-20 13:10:15.265 | SUCCESS  | __main__:run:630 - ------------------------------------------------------------
2026-05-20 13:10:15.265 | SUCCESS  | __main__:run:631 - E2E Latency (End-to-End Latency): 0.253 seconds
2026-05-20 13:10:15.265 | SUCCESS  | __main__:run:633 - E2E TPS (End-to-End Tokens Per Second): 31.66 tokens/s
2026-05-20 13:10:15.265 | SUCCESS  | __main__:run:636 - RTF (Real Time Factor): 0.10
2026-05-20 13:10:15.265 | SUCCESS  | __main__:run:637 - ============================================================
```

#### Qwen3-ForcedAligner-0.6B:
```bash
026-03-19 10:42:55.227 | INFO     | xhquant.lib:_load_hsum_impl:30 - using CPU hsum extension
[2026-03-19 10:42:59.460] [tid 6459] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-03-19 10:42:59.460] [tid 6459] [warning] Empty backend name, use Xh2HalBackend instead.
2026-03-19 10:43:00.029 | INFO     | __main__:__init__:103 - encode model loaded
2026-03-19 10:43:00.277 | INFO     | __main__:__init__:106 - prefill model loaded
WARNING: input_ids len=133 != output_ids len=411
2026-03-19 10:43:01.345 | SUCCESS  | __main__:run:344 -
输出:
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - HE          0.240   0.320
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - BEGAN       0.320   0.640
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - A           0.640   0.640
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - CONFUSED    0.640   1.200
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - COMPLAINT   1.200   1.600
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - AGAINST     1.600   1.920
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - THE         1.920   2.000
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - WIZARD      2.000   2.400
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - WHO         2.720   2.720
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - HAD         2.720   2.720
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - VANISHED    2.880   3.280
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - BEHIND      3.280   3.600
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - THE         3.600   3.680
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - CURTAIN     3.680   4.000
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - ON          4.000   4.160
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - THE         4.160   4.240
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:346 - LEFT        4.240   4.640
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:351 - ============================================================
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:352 - Performance Statistics:
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:353 - ============================================================
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:354 - Audio duration: 4905.00 ms (4.91 s)
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:355 - Audio loading time: 306.576 ms
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:356 - Input preparation time: 7.952 ms
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:357 - Encode time: 30.818 ms
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:358 - Prefill time: 125.748 ms
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:359 - E2E Latency (End-to-End Latency): 0.487 seconds
2026-03-19 10:43:01.346 | SUCCESS  | __main__:run:360 - ============================================================
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。