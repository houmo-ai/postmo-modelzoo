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
python3 get_model.py --type hmm --model_name qwen3_forcealigner
```

如果需要自行编译模型，可以下载原始模式，下载方式如下：

```bash
# Qwen3-ASR-0.6B (默认)
python3 get_model.py --type raw

# Qwen3-ASR-1.7B
python3 get_model.py --type raw --model_size 1.7b

# Qwen3-ForcedAligner-0.6B
python3 get_model.py --type raw --model_name qwen3_forcealigner
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
python3 ptq.py --model Qwen3-ASR-1.7B

# Qwen3-ForcedAligner-0.6B
python3 ptq.py --model Qwen3-ForcedAligner-0.6B --model_name qwen3_forcealigner
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
# Qwen3-ASR-0.6B / Qwen3-ASR-1.7B (默认)
python3 build.py

# Qwen3-ForcedAligner-0.6B
python3 build.py --model_name qwen3_forcealigner
```

编译好的模型存放在当前目录，包括3个文件：
- qwen3_asr_encode.hmm
- qwen3_asr_decode.hmm
- qwen3_asr_prefill.hmm

或（Qwen3-ForcedAligner-0.6B）：
- qwen3_forcealigner_encode.hmm
- qwen3_forcealigner_prefill.hmm

### 2.5 演示

Qwen3-ASR模型使用python API进行演示。

```bash
# Qwen3-ASR-0.6B (默认)
python3 demo_asr.py

# Qwen3-ASR-1.7B
python3 demo_asr.py --processor_dir Qwen3-ASR-1.7B

# Qwen3-ForcedAligner-0.6B
python3 demo_forcealigner.py
```

推理完成将输出性能结果。

### 2.6 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
# Qwen3-ASR-0.6B (默认)
bash test.sh

# Qwen3-ASR-1.7B
bash test.sh -m qwen3-asr-1.7b

# Qwen3-ForcedAligner-0.6B
bash test.sh -m qwen3-forcealigner-0.6b
```

test.sh支持以下参数：

```bash
Usage: test.sh [options]
  -s, --step         execution step, default is all, support: all, demo, build.
  -t, --model_type   The method for getting the compiled model, default is precompiled, support: precompiled, compile.
  -m, --model        model name, default is qwen3-asr-0.6b, support: qwen3-asr-0.6b, qwen3-asr-1.7b, qwen3-forcealigner-0.6b.
  -h, --help         help information
```

示例：

```bash
# 仅运行demo（使用预编译模型）
bash test.sh -s demo -m qwen3-asr-1.7b

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -t compile -m qwen3-forcealigner-0.6b
```

## 3 参考结果

### 3.1 演示结果

#### Qwen3-ASR-0.6B:
```bash
2026-03-19 11:05:11.167] [tid 622] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-03-19 11:05:11.167] [tid 622] [warning] Empty backend name, use Xh2HalBackend instead.
2026-03-19 11:05:11.340 | INFO     | __main__:__init__:113 - encode model loaded
2026-03-19 11:05:11.652 | INFO     | __main__:__init__:116 - prefill model loaded
2026-03-19 11:05:11.670 | INFO     | __main__:__init__:128 - decode model loaded
/hmdd/imodelzoo/models/asr/qwen3-asr/demo_asr.py:148: DeprecationWarning: get_input() deprecated: Use get_dev_input instead.
  cache = self.prefill.get_input(self.prefill.get_input_name(i))
后摩芯片算法部的小伙伴，你们好啊！我是多拉A梦，你们好厉害啊！好像你们拥有装满各种神奇道具的口袋，你们真棒！新的一年，祝你们身体健康，事业顺利，发大财。
2026-03-19 11:05:13.353 | SUCCESS  | __main__:run:453 - ============================================================
2026-03-19 11:05:13.353 | SUCCESS  | __main__:run:454 - Performance Statistics:
2026-03-19 11:05:13.353 | SUCCESS  | __main__:run:455 - ============================================================
2026-03-19 11:05:13.353 | SUCCESS  | __main__:run:456 - Audio duration: 14450.00 ms (14.45 s)
2026-03-19 11:05:13.353 | SUCCESS  | __main__:run:457 - Audio loading time: 397.648 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:458 - Input preparation time: 7.443 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:459 - Loop count: 1
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:460 - ------------------------------------------------------------
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:461 - Total Encode time: 47.164 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:462 -   - Encode prep time: 0.132 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:463 -   - Encode infer time: 46.355 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:464 -   - Encode output time: 0.677 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:465 - ------------------------------------------------------------
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:466 - Total Prefill time: 96.535 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:467 -   - Prefill prep time: 42.983 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:468 -   - Prefill infer time: 53.152 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:469 -   - Prefill output time: 0.400 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:470 - ------------------------------------------------------------
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:471 - Output 54 tokens
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:472 - Total Decode time(infer + embedding): 674.818 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:473 -   - Decode infer total time: 573.417 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:474 -   - Decode infer avg time: 10.619 ms
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:476 - Decode Speed: 80.02 tokens/s
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:478 - TPOT (Time Per Output Token): 12.497 ms/token
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:479 - ------------------------------------------------------------
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:480 - E2E Latency (End-to-End Latency): 1.225 seconds
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:482 - E2E TPS (End-to-End Tokens Per Second): 44.07 tokens/s
2026-03-19 11:05:13.354 | SUCCESS  | __main__:run:483 - RTF (Real Time Factor): 0.08
2026-03-19 11:05:13.355 | SUCCESS  | __main__:run:484 - ============================================================
```


#### Qwen3-ASR-1.7B:
```bash
[2026-03-19 11:02:00.782] [tid 248] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-03-19 11:02:00.782] [tid 248] [warning] Empty backend name, use Xh2HalBackend instead.
2026-03-19 11:02:00.993 | INFO     | __main__:__init__:113 - encode model loaded
2026-03-19 11:02:01.669 | INFO     | __main__:__init__:116 - prefill model loaded
2026-03-19 11:02:01.734 | INFO     | __main__:__init__:128 - decode model loaded
/hmdd/imodelzoo/models/asr/qwen3-asr/demo_asr.py:148: DeprecationWarning: get_input() deprecated: Use get_dev_input instead.
  cache = self.prefill.get_input(self.prefill.get_input_name(i))
后模芯片算法部的小伙伴，你们好啊！我是哆啦A梦，你们好厉害啊！好像你们拥有装满各种神奇道具的口袋，你们真棒！新的一年，祝你们身体健康，事业顺利，发大财！
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:454 - ============================================================
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:455 - Performance Statistics:
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:456 - ============================================================
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:457 - Audio duration: 14450.00 ms (14.45 s)
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:458 - Audio loading time: 356.167 ms
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:459 - Input preparation time: 8.329 ms
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:460 - Loop count: 1
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:461 - ------------------------------------------------------------
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:462 - Total Encode time: 31.466 ms
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:463 -   - Encode prep time: 0.143 ms
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:464 -   - Encode infer time: 30.612 ms
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:465 -   - Encode output time: 0.711 ms
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:466 - ------------------------------------------------------------
2026-03-19 11:02:04.025 | SUCCESS  | __main__:run:467 - Total Prefill time: 106.676 ms
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:468 -   - Prefill prep time: 37.037 ms
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:469 -   - Prefill infer time: 69.128 ms
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:470 -   - Prefill output time: 0.510 ms
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:471 - ------------------------------------------------------------
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:472 - Output 54 tokens
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:473 - Total Decode time(infer + embedding): 1220.985 ms
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:474 -   - Decode infer total time: 1069.999 ms
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:475 -   - Decode infer avg time: 19.815 ms
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:477 - Decode Speed: 44.23 tokens/s
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:479 - TPOT (Time Per Output Token): 22.611 ms/token
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:480 - ------------------------------------------------------------
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:481 - E2E Latency (End-to-End Latency): 1.726 seconds
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:483 - E2E TPS (End-to-End Tokens Per Second): 31.29 tokens/s
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:484 - RTF (Real Time Factor): 0.12
2026-03-19 11:02:04.026 | SUCCESS  | __main__:run:485 - ============================================================
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