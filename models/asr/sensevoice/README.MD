# SenseVoiceSmall

本示例展示如何把SenseVoiceSmall模型量化和编译，部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于：

- 模型名称: SenseVoiceSmall
- 来源: [https://www.modelscope.cn/models/iic/SenseVoiceSmall](https://www.modelscope.cn/models/iic/SenseVoiceSmall)
- 许可: [MIT License](https://github.com/modelscope/FunASR/blob/main/LICENSE)

预训练模型在运行时下载，工程发布**不包含**该模型，本例只适用于xh2。

## 2 快速开始

目前SenseVoiceSmall模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 获取模型和数据

通过get_model脚本下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果仅查看演示结果，可选择下载芯片模型，下载后的模型存放在当前目录（注意相同模型会覆盖，请自行保存）：

```bash
python3 get_model.py --type hmm
```
如果需要自行编译模型，可以下载原始模式，下载方式如下：
```bash
python3 get_model.py --type raw
```

### 2.2 量化

量化需要使用GPU，并在宿主机安装与CUDA12.2兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID，显存需要达到40GB以上，空闲内存需要达到100GB以上。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu20.04-py39-x86.64 /bin/bash
```

进入docker后source环境变量，并进入`models/asr/sensevoice`目录，执行ptq.py脚本。默认的量化方式为w8a8_sefp。

```bash
source env.sh
cd models/asr/sensevoice
python3 ptq.py
```

### 2.3 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型存放在当前目录的output/xh2目录下，包括1个文件：
sensevoice_small.hmm

### 2.4 演示

SenseVoiceSmall模型使用python API进行演示

2. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py
```

### 2.5 一键评估

以上步骤可以通过test.sh脚本一键执行：

```bash
bash test.sh
```

## 3 参考结果

### 3.1 演示结果

```bash
Loading tokens from SenseVoiceSmall/tokens.json
[2026-01-30 07:01:26.709] [tid 440] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-01-30 07:01:26.709] [tid 440] [warning] Empty backend name, use Xh2HalBackend instead.
Loading model from output/xh2/sensevoice_small.hmm
2026-01-30 07:01:34.333 | INFO     | __main__:run_inference:296 - ==================== Process SenseVoiceSmall/example/zh.mp3 ====================
2026-01-30 07:01:35.329 | INFO     | __main__:run_inference:349 - Clean Text: 开饭时间早上九点至下午五点
2026-01-30 07:01:35.330 | INFO     | __main__:run_inference:350 - ==================== Process SenseVoiceSmall/example/zh.mp3 take time 94.55108642578125 ms ====================
2026-01-30 07:01:35.330 | INFO     | __main__:run_inference:296 - ==================== Process SenseVoiceSmall/example/en.mp3 ====================
2026-01-30 07:01:35.436 | INFO     | __main__:run_inference:349 - Clean Text: the tribal chieftain called for the boy and presented him with fifty pieces of gold
2026-01-30 07:01:35.437 | INFO     | __main__:run_inference:350 - ==================== Process SenseVoiceSmall/example/en.mp3 take time 94.29383277893066 ms ====================
2026-01-30 07:01:35.437 | INFO     | __main__:run_inference:296 - ==================== Process SenseVoiceSmall/example/yue.mp3 ====================
2026-01-30 07:01:35.543 | INFO     | __main__:run_inference:349 - Clean Text: 呢几个字都表达唔到我想讲嘅意思
2026-01-30 07:01:35.543 | INFO     | __main__:run_inference:350 - ==================== Process SenseVoiceSmall/example/yue.mp3 take time 95.37458419799805 ms ====================
2026-01-30 07:01:35.543 | INFO     | __main__:run_inference:296 - ==================== Process SenseVoiceSmall/example/ja.mp3 ====================
2026-01-30 07:01:35.646 | INFO     | __main__:run_inference:349 - Clean Text: うちの中学は弁当制で持っていきない場合は50円の学校販売のパンを買う
2026-01-30 07:01:35.646 | INFO     | __main__:run_inference:350 - ==================== Process SenseVoiceSmall/example/ja.mp3 take time 87.16630935668945 ms ====================
2026-01-30 07:01:35.646 | INFO     | __main__:run_inference:296 - ==================== Process SenseVoiceSmall/example/ko.mp3 ====================
2026-01-30 07:01:35.739 | INFO     | __main__:run_inference:349 - Clean Text: 조금만 생각을 하면서 살면 훨씬 편할 거야
2026-01-30 07:01:35.739 | INFO     | __main__:run_inference:350 - ==================== Process SenseVoiceSmall/example/ko.mp3 take time 81.19869232177734 ms ====================
```

### 3.2 性能结果

```bash
2026-01-30 07:01:35.330 | INFO     | __main__:run_inference:350 - ==================== Process SenseVoiceSmall/example/zh.mp3 take time 94.55108642578125 ms ====================
2026-01-30 07:01:35.437 | INFO     | __main__:run_inference:350 - ==================== Process SenseVoiceSmall/example/en.mp3 take time 94.29383277893066 ms ====================
2026-01-30 07:01:35.543 | INFO     | __main__:run_inference:350 - ==================== Process SenseVoiceSmall/example/yue.mp3 take time 95.37458419799805 ms ====================
2026-01-30 07:01:35.646 | INFO     | __main__:run_inference:350 - ==================== Process SenseVoiceSmall/example/ja.mp3 take time 87.16630935668945 ms ====================
2026-01-30 07:01:35.739 | INFO     | __main__:run_inference:350 - ==================== Process SenseVoiceSmall/example/ko.mp3 take time 81.19869232177734 ms ====================
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
