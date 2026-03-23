# GLM-ASR

本示例展示如何把 GLM-ASR-Nano-2512 模型下载、量化、编译，并部署到后摩鸿途芯片设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于：

- 模型名称: GLM-ASR-Nano-2512
- 来源: https://modelscope.cn/models/ZhipuAI/GLM-ASR-Nano-2512
- 参考仓库: https://github.com/zai-org/GLM-ASR

预训练模型在运行时下载，工程发布不包含该模型，本例只适用于 xh2。

## 2 快速开始

目前 GLM-ASR 模型提供 python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 获取模型和数据

通过 get_model.py 下载模型，可通过参数选择下载原始模型或芯片模型。

如果仅查看演示结果，可直接下载预编译模型：

```bash
python3 get_model.py --type hmm
```

下载后的原始模型默认保存在当前目录下的 glm-asr-nano-2512：

```bash
python3 get_model.py --type raw
```

演示脚本默认读取仓库内的测试音频：

```bash
../../../data/audio/audio.mp3
```

### 2.2 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.2 兼容的 CUDA 和驱动。启动 docker 时将 GPU 映射到 docker 内，并指定可用的 GPU ID。参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:vx.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 models/asr/glm-asr 目录执行 ptq.py：

```bash
source env.sh
cd models/asr/glm-asr
python3 ptq.py --model glm-asr-nano-2512
```

注意：当前 ptq.py 默认引用 config/llm/glm_asr_decode_xh2a.py。如果目录下没有该配置文件，需要在执行前补齐对应配置，或通过 --config 显式传入可用配置文件。

### 2.3 编译

将量化模型编译为在芯片上运行的模型。执行脚本：

```bash
python3 build.py
```

编译好的模型默认输出到 output/xh2 目录，包括 3 个文件：

- glm-asr_encode.hmm
- glm-asr_decode.hmm
- glm-asr_prefill.hmm

### 2.4 演示

如果使用 get_model.py 下载的预编译模型，可直接执行：

```bash
python3 demo.py
```

如果使用 build.py 自行编译得到的模型，建议显式传入模型路径：

```bash
python3 demo.py \
  --encoder_path output/xh2/glm-asr_encode.hmm \
  --prefill_path output/xh2/glm-asr_prefill.hmm \
  --decode_path output/xh2/glm-asr_decode.hmm \
  --audio ../../../data/audio/audio.mp3
```

### 2.5 一键评估

以上步骤可以通过 test.sh 一键执行：

```bash
bash test.sh
```

test.sh 默认使用预编译模型，适合快速验证 demo。

如果要走下载原始模型、量化、编译流程，可执行：

```bash
bash test.sh -t compile
```

注意：compile 模式依赖有效的量化配置文件和可用的 embedding 权重文件。若这两项未准备好，建议先分别执行下载、量化、编译命令排查环境。

## 3 输出说明

本 README 生成时未在当前环境实际执行模型推理，因此这里不附带固定参考数值。运行 demo.py 后，终端会输出两类信息：

- 识别文本结果

```bash
后膜芯片算法部的小伙伴，你们好啊，我是哆啦A梦，你们好厉害啊，好像你们拥有装满各种神奇道具的口袋，你们真棒。新的一年，祝你们身体健康，事业顺利，发大财。
```

- 性能统计，包括 Encoder Time、Prefill Cost、Decode Cost、TTFT、TPOT 和 E2E Latency

```bash
Encoder Time: 581.081 ms
Prefill Cost: 61.433 ms
Decode Cost: 755.838 ms
Output 48 tokens, Decode Speed: 63.51 tokens/s
TTFT (Time to First Token): 65.472 ms
TPOT (Time Per Output Token): 15.747 ms/token
## 4 免责声明

本示例依赖的模型与相关资源由第三方提供并维护。使用这些模型、数据或其他外部资源时，请遵循对应的许可证、使用条款和隐私政策，并自行评估由此带来的合规性与使用风险。