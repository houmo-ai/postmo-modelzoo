# Qwen3.5 / Qwen3.6

本示例展示如何把 Qwen3.5 和 Qwen3.6 模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: Qwen3.5-0.8B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.5-0.8B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3.5-2B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.5-2B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3.5-4B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.5-4B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3.5-9B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.5-9B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3.5-122B-A10B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.5-122B-A10B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3.6-27B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.6-27B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

- 模型名称: Qwen3.6-35B-A3B
- 来源: https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。

当前目录使用 `config.yaml` 管理 `get_model.py`、`ptq.py`、`build.py`、`demo.py` 和 `demo_mtp.py` 的模型默认值。每个 `model_name` / `model_size` 节点通过 `workflow_config` 指定 `configs_merak` 中的量化导出配置。当前默认模型为：

- `model_name=qwen3.6`
- `model_size=35b-a3b`
- `context_length=256k`
- `prefill_length=256`
- `ndevice=1`

支持的模型组合为：

- `qwen3.5`: `0.8b`、`2b`、`4b`、`9b`、`122b-a10b`
- `qwen3.6`: `27b`、`35b-a3b`

标准流程编译产物命名统一为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`
- `${model_name}-${model_size}_visual_${max_size_w}x${max_size_h}x${max_size_t}.hmm`

MTP 流程额外产物命名为：

- `${model_name}-${model_size}_prefill_mtp.hmm`
- `${model_name}-${model_size}_decode_mtp.hmm`

`demo.py` 与 `demo_mtp.py` 默认从 `config.yaml` 中读取 `modelscope_repo`，并使用其最后一段作为 `tokenizer_dir` 默认值。

### 用于校准的数据集

本项目引用 wikitext 数据集仅用于**研究和评估目的**。

- 数据集许可: CC BY-NC 4.0
- 来源: https://www.modelscope.cn/datasets/modelscope/wikitext

⚠️ 重要提示:

- 本数据集**不作为本仓库的一部分进行分发**。
- 本数据集**不受本项目 Apache License 2.0 许可的约束**。
- 任何对数据集的使用都必须遵守其原始许可 (CC BY-NC 4.0)，包括**非商业用途限制**。
- 用户有责任确保在使用本数据集时遵守相关规定。

## 2 快速开始

目前 Qwen3.5 / Qwen3.6 模型仅提供 python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 python 依赖，建议使用 venv。

```bash
pip3 install -r requirements.txt
pip3 install -r ../../../hmodel/gptqmodel/requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，可通过参数选择下载原始模型或芯片模型。

```bash
# 下载原始模型用于量化
python3 get_model.py --type raw --model_name qwen3.5 --model_size 9b
python3 get_model.py --type raw --model_name qwen3.6 --model_size 35b-a3b

# 下载预编译模型用于演示
python3 get_model.py --type hmm --model_name qwen3.6 --model_size 35b-a3b
```

若需要 MTP 预编译模型，可增加 `--mtp`。

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时需要将 GPU 映射到容器内，例如：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

注意：qwen3.5-122b-a10b 模型的量化需要单个 GPU 显存不少于 80 G

进入 docker 后，在仓库根目录执行 `source env.sh`，再进入当前目录运行 `ptq.py`。

```bash
source env.sh
cd models/llm/qwen3.5
```

推荐先通过 `get_model.py --type raw` 下载原始模型，再通过模型名称和尺寸执行量化：

```bash
python3 ptq.py \
  --model_name qwen3.6 \
  --model_size 35b-a3b
```

常用可选参数：

```bash
# 显式指定原始模型目录
python3 ptq.py \
  --model_name qwen3.6 \
  --model_size 35b-a3b \
  --model_dir /path/to/Qwen3.6-35B-A3B

# 覆盖 workflow yaml 中的量化 bit
python3 ptq.py \
  --model_name qwen3.6 \
  --model_size 35b-a3b \
  --bits 8

# 覆盖视觉导出分辨率
python3 ptq.py \
  --model_name qwen3.6 \
  --model_size 35b-a3b \
  --max_size_w 896 \
  --max_size_h 896

# 自动选择对应的 MTP workflow
python3 ptq.py \
  --model_name qwen3.6 \
  --model_size 35b-a3b \
  --mtp

# 显式覆盖 workflow 配置，例如使用 DFlash workflow
python3 ptq.py \
  --model_name qwen3.6 \
  --model_size 35b-a3b \
  --quant_config_path /path/to/configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_dflash.yaml
```

主要入参如下：

- `--config_path`：模型索引配置，默认为当前目录的 `config.yaml`。
- `--model_name`、`--model_size`：选择 `config.yaml` 中的模型节点。
- `--model_dir`：原始模型目录；不传时根据 `modelscope_repo` 自动生成。
- `--quant_config_path`：显式 workflow yaml；不传时根据 `workflow_config` 自动生成。
- `--mtp`：使用自动生成的 `_mtp.yaml` workflow。
- `--output_dir`：最终 hmquant 产物输出目录，默认 `./output/${HOUMO_TARGET}/hmquant`。
- `--work_dir`：quant/export 中间输出目录，默认 `./work_dirs`。
- `--device`：quant、export、golden 和 quick test 使用的设备，默认 `cuda`。
- `--overwrite`：export 前删除已有的 export 输出目录。
- `--export-from-quanted-model`：输入已经量化的 HF 模型时跳过量化阶段。
- `--bits`：覆盖 workflow 中的 `quant.bits`。
- `--max_size_w`、`--max_size_h`：覆盖视觉模型导出尺寸。
- `--dump-golden`：导出后生成 golden 数据。
- `--quick-test`：导出后运行快速 HMONNX 测试。

`config.yaml` 当前为全部支持的模型组合配置了标准 `workflow_config`。仓库中另外提供
以下 MTP workflow：

- `qwen3.5/9b`
- `qwen3.6/27b`
- `qwen3.6/35b-a3b`

对没有对应 `_mtp.yaml` 文件的模型使用 `--mtp` 时，`ptq.py` 会在最终工作流文件检查
阶段报 `Workflow config not found`。DFlash、GPTQ 或其他变体可通过
`--quant_config_path` 显式指定对应的 workflow yaml。

量化导出输出位于 `output/${HOUMO_TARGET}/hmquant`。

### 2.4 编译

标准编译：

```bash
python3 build.py --model_name qwen3.6 --model_size 35b-a3b
```

MTP 编译：

```bash
python3 build.py --model_name qwen3.6 --model_size 35b-a3b --mtp
```

标准输出名称为：

- `${model_name}-${model_size}_prefill.hmm`
- `${model_name}-${model_size}_decode.hmm`
- `${model_name}-${model_size}_visual_${max_size_w}x${max_size_h}x${max_size_t}.hmm`

视觉模型文件名中的尺寸来自 `config.yaml` 或量化导出的实际 visual bucket。例如大多数模型使用 `visual_896x896x2.hmm`，`qwen3.5/122b-a10b` 默认使用 `visual_448x448x2.hmm`。MTP 编译还会生成：

- `${model_name}-${model_size}_prefill_mtp.hmm`
- `${model_name}-${model_size}_decode_mtp.hmm`

### 2.5 演示

标准演示：

```bash
python3 demo.py --model_name qwen3.6 --model_size 35b-a3b
```

MTP 演示：

```bash
python3 demo_mtp.py --model_name qwen3.6 --model_size 35b-a3b
```

多设备场景下，demo 脚本会自动把 `prefill` / `decode` 默认模型后缀从 `.hmm` 切换为 `.hmms`，`visual` 仍保持 `.hmm`。

## 3 一键评估

以上步骤可以通过 `test.sh` 脚本执行。脚本默认使用：

- `MODEL_NAME=qwen3.6`
- `MODEL_SIZE=35b-a3b`
- `NDEVICE=1`
- `MTP=false`

默认执行 `demo` 步骤：

```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + workflow 量化导出 + 编译 + 演示
bash test.sh -s all
```

运行前请确保 `HOUMO_EXAMPLES_PATH` 指向当前仓库根目录。

`test.sh` 支持以下常用参数：

```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -name, --model_name     Model name. Choices: qwen3.5, qwen3.6.
  -size, --model_size     Model size.
  --ndevice               Number of devices. Default: 1.
  --mtp                   Enable mtp mode.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

示例：

```bash
# 指定模型为 Qwen3.5-9B
bash test.sh --model_name qwen3.5 --model_size 9b

# 仅运行 quant，会先下载原始模型，再调用 ptq.py 量化导出
bash test.sh -s quant --model_name qwen3.6 --model_size 35b-a3b

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build --model_name qwen3.6 --model_size 35b-a3b

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo --model_name qwen3.6 --model_size 35b-a3b

# 完整流程
bash test.sh -s all --model_name qwen3.6 --model_size 35b-a3b

# 验证 Qwen3.5-0.8B 完整流程
bash test.sh -s all --model_name qwen3.5 --model_size 0.8b

# MTP 完整流程
bash test.sh -s all --mtp --model_name qwen3.6 --model_size 35b-a3b

# 跳过下载，直接使用当前目录已有模型
bash test.sh -s demo --skip_download --model_name qwen3.6 --model_size 35b-a3b
```

当前脚本行为说明：

- `quant`：下载原始模型并调用 `ptq.py` 执行量化导出；启用 `--mtp` 时会把 `--mtp` 同步传给 `ptq.py`，由 `ptq.py` 自动选择 `_mtp.yaml` workflow。
- `build`：默认调用 `build.py`；当 `--mtp` 启用时调用 `build.py` 并配置参数 `--mtp`。
- `demo`：下载预编译模型后执行 `demo.py`；当 `--mtp` 启用时执行 `demo_mtp.py`。
- 非 MTP `demo`：先执行 Python demo，再生成 C++ demo 使用的 embedding 文件并编译 C++ demo；单设备时执行 C++ demo，环境中存在 `llm_perf` 时继续执行性能测试。
- C++ demo 和 `llm_perf` 会按以下顺序选择第一个存在的 visual HMM：`visual_896x896x2.hmm`、`visual_448x448x2.hmm`、`visual.hmm`。如果同时存在多个文件，优先使用排序更靠前的文件；如果三个文件都不存在，脚本会报错退出。

## 4 参考结果

### 4.1 演示结果

#### Qwen3.6-35B-A3B 演示结果

```bash
prefill model loaded
decode model loaded
vision model loaded
question: 描述这些图片
images: ['/home/workspace/imodelzoo/tests/apis_tests/../../data/pic/beach.jpeg']
processor image_grid_thw: tensor([[ 1, 56, 56]])
processor image token count: 784
Running vision model...
image_embeds shape: torch.Size([784, 2048])
response:
这张图片展现了一幅温馨、宁静又充满活力的海滩场景，主要人物是一位年轻女性和一只金毛犬。
以下是详细描述：
🌅 **环境与氛围**：
- 时间似乎是日落或日出时分，阳光从画面右侧斜射过来，在沙滩和海面上洒下温暖的金色光芒。
- 背景是平静的海洋，海浪轻轻拍打着岸边，远处海天相接，视野开阔。
- 整体色调柔和温暖，带有梦幻般的逆光效果，营造出一种治愈、放松、人与自然和谐共处的氛围。

🐶 **狗狗**：
- 一只体型健壮的金毛寻回犬（或类似品种），毛色呈浅金色，在阳光下显得柔软发亮。
- 它戴着一条彩色图案的胸背带（可能是牵引绳的一部分），正坐着，前爪抬起，与女子击掌（“high five”），表情专注而友好。
- 尾巴自然垂在沙地上，姿态放松但配合度高，显示出良好的训练和亲密关系。

👩 **女子**：
- 年轻女性，长发披肩，面带灿烂笑容，侧脸对着镜头，眼神温柔地注视着狗狗。
- 她穿着格子衬衫（蓝白或灰白相间）和深色裤子，赤脚坐在沙滩上，姿态随意自然。
- 她伸出右手与狗狗击掌，动作轻盈，充满互动乐趣。

🏖️ **细节**：
- 沙滩上有清晰的脚印和凹陷，说明有人活动过。
- 狗狗旁边放着一个红色的小玩具（可能是绳结玩具），暗示他们刚刚玩耍过。
- 光线在人物和动物身上形成柔和的轮廓光，增强了画面的艺术感和情感温度。

✅ **总结**：
这是一张充满生活气息和情感温度的照片，捕捉了人与宠物之间纯真快乐的瞬间。它不仅展现了美丽的自然风光，更传递了陪伴、信任与幸福的主题，适合用于表达“治愈”、“陪伴”、“自由”或“简单快乐”的概念。

——  

====================================================================================================
                     Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
  Batch Size:      1
  Input Length per Sample:    809 tokens
  Output Length per Sample:    433 tokens
  Number of Images:      1 images
  Prefill Model Load Time: 7380.25ms
  Decode Model Load Time:  206.93ms
  Vision Model Load Time: 1118.98ms
Vision Stage Performance:
  Total Time: 1278.11ms | Speed:    0.78 images/s
  Preprocessing Time: 81.88ms | Speed:   12.21 images/s
  API SetInput Time:  11.24ms
  API Inference Time: 1171.61ms | Speed:    0.85 images/s
  API GetOutput Time:  2.23ms
Prefill Stage Performance:
  Total Time: 2588.12ms | Speed:  312.58 tokens/s
  Tokenization Time:   15.26ms
  Embedding Time:   12.26ms
  API SetInput Time:  23.54ms
  API Inference Time: 1245.24ms | Prefill Speed:  649.67 tokens/s
  API GetOutput Time:  0.78ms
Decode Stage Performance:
  Total Time: 20242.60ms | Speed:   21.34 tokens/s
  Tokenization Time: 4578.06ms
  Embedding Time:   78.09ms
  API SetInput Time:   1.83ms
  API Inference Time: 33.72ms | Decode Speed:   29.66 tokens/s
  API GetOutput Time:  0.66ms
Overall Performance Metrics:
  TTFT (Time To First Token): 3866.24 ms
  TPOT (Time Per Output Token): 46.86 ms/token
  E2E Latency (End-to-End):     22.98 seconds
  E2E TPS (Throughput):         18.84 tokens/s
====================================================================================================
```

#### Qwen3.5-9B MTP 演示结果

```bash
prefill and prefill_mtp models loaded
decode_verify and decode_mtp models loaded
question:
请介绍一下存算一体技术的优势
response:
**存算一体**（Processing-in-Memory, PIM）是一种突破性的计算机架构技术，旨在解决传统冯·诺依曼架构中存在的核心瓶颈。

在传统的冯·诺依曼架构中，处理器（CPU/GPU）负责计算，而内存（DRAM/SSD）负责存储数据。两者通过总线进行通信。这种分离导致了著名的"**存储墙**"（Memory Wall）问题。存算一体技术通过将计算单元直接嵌入到存储单元中，让数据在存储位置直接进行计算，从而彻底改变了这一局面。

以下是存算一体技术的主要优势：

### 1. 突破“存储墙”，大幅提升能效比
这是存算一体最核心的优势。
*   **消除数据搬运能耗**：在传统架构中，移动数据（从内存到 CPU 再写回内存）所消耗的电能往往占据了总能耗的 60%~80%。存算一体让数据“原地计算”，几乎消除了长距离的数据传输能耗。
*   **降低功耗**：由于大幅减少了数据移动，存算一体芯片的功耗显著低于传统芯片，特别是在处理大量数据读取和写入的场景下。这使得它在**边缘计算**、**物联网**（IoT）和**移动设备**等对电池寿命敏感的场景中具有巨大潜力。

### 2. 显著降低延迟，提高计算速度
*   **减少通信延迟**：数据无需经过复杂的片外总线或片上缓存层级，直接在存储阵列内部完成运算，极大地缩短了数据访问和处理的等待时间。
*   **缓解带宽瓶颈**：传统架构中，内存带宽往往是限制系统整体速度的瓶颈。存算一体通过并行处理多个存储单元的数据，有效利用了存储带宽，使得系统整体吞吐量大幅提升。

### 3. 天然适合人工智能（AI）和大数据负载
现代 AI 应用（如深度学习、大语言模型）具有“数据密集”和“计算密集”的特点，且往往涉及大量的矩阵乘法运算。
*   **匹配 AI 负载特征**：AI 训练和推理过程中，数据在内存和处理器之间频繁往返。存算一体架构完美契合这种模式，能够高效处理大规模矩阵运算。
*   **支持稀疏计算**：许多存算一体架构（特别是基于 SRAM 或 ReRAM 的）能够利用存储单元的非线性特性，直接进行稀疏矩阵运算，这对于优化后的神经网络模型非常高效。

### 4. 简化系统架构，降低设计复杂度
*   **减少互联复杂度**：传统高性能计算需要极其复杂的片上网络（NoC）和片外互联来连接成千上万个核心和内存模块。存算一体将计算和存储集成在一起，简化了芯片内部的布线设计。
*   **降低封装成本**：由于减少了对外部高速内存接口（如 HBM 或 GDDR）的依赖，芯片封装和系统集成的成本有望降低。

### 5. 克服摩尔定律放缓带来的挑战
随着晶体管尺寸逼近物理极限，单纯依靠缩小晶体管来提升 CPU 频率和性能（摩尔定律）已难以为继，且功耗急剧上升。存算一体提供了一种**架构层面的创新路径**，在不依赖晶体管微缩的情况下，通过改变数据流动方式来继续提升性能并控制功耗。

---

### 总结与展望

| 特性 | 传统冯·诺依曼架构 | 存算一体架构 (PIM) |
| :--- | :--- | :--- |
| **数据流向** | 内存 $\leftrightarrow$ 处理器 (长距离) | 存储单元内部直接计算 (零距离) |
| **主要瓶颈** | 内存带宽与延迟 | 计算单元密度与互连 |
| **能耗分布** | 大部分能耗用于数据搬运 | 大部分能耗用于计算本身 |
| **适用场景** | 通用计算、逻辑控制 | AI 推理、大数据分析、边缘计算 |

**潜在挑战**：
尽管优势明显，存算一体目前也面临一些挑战，例如：
*   **编程模型复杂**：传统的软件（如 C/C++）难以直接利用 PIM 特性，需要新的编程模型和编译器支持。
*   **数据一致性**：在多线程或多核环境下，保证数据的一致性（Cache Coherence）比传统架构更复杂。
*   **制造良率**：新型存储材料（如 ReRAM, MRAM）与逻辑电路的集成工艺尚需成熟。

总体而言，存算一体被视为后摩尔时代解决算力与能耗矛盾的关键技术之一，特别是在人工智能和大数据爆发的当下，其应用前景非常广阔。

[SpecDecode] mode=mtp rounds=399 total_tokens=970 draft_tokens=1596 accepted=572 avg_accepted_per_round=1.43 acceptance_rate=35.84% mtp_prefill_tokens=29 drafts_per_round=4 elapsed_s=53.7641

====================================================================================================
                     Model Inference Performance Summary Report
====================================================================================================
Configuration Details:
  Batch Size:      1
  Input Length per Sample:     30 tokens
  Output Length per Sample:    970 tokens
  Prefill Model Load Time: 9817.03ms
  Decode Model Load Time:  435.12ms
Prefill Stage Performance:
  Total Time:  326.92ms | Speed:   91.77 tokens/s
  Tokenization Time:   18.20ms
  Embedding Time:    2.60ms
  API SetInput Time:   9.26ms
  API Inference Time: 192.97ms | Prefill Speed:  155.47 tokens/s
  API GetOutput Time: 76.16ms
Decode Stage Performance:
  Total Time: 53761.95ms | Speed:   18.02 tokens/s
  Tokenization Time:  981.17ms
  Embedding Time:  257.02ms
  API SetInput Time:   2.79ms
  API Inference Time: 40.65ms | Decode Speed:   24.60 tokens/s
  API GetOutput Time:  7.11ms
Overall Performance Metrics:
  TTFT (Time To First Token):  326.92 ms
  TPOT (Time Per Output Token): 55.48 ms/token
  E2E Latency (End-to-End):     54.13 seconds
  E2E TPS (Throughput):         17.92 tokens/s
====================================================================================================
```

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
