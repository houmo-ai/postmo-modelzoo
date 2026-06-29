---
name: large-model-readme-generation
description: 用于生成、修改、重构和检查 imodelzoo 大模型示例 README，统一结构、命名、章节顺序、示例命令和免责声明格式。适用于 LLM、VLM、ASR、TTS、Embedding、Reranker、OCR、Omni 等大模型示例目录。
---

# 大模型示例 README 生成规范

## 适用范围

当你需要**生成、修改、重构或检查大模型示例 README** 时，使用本 skill：

- `models/llm/**/README.MD`
- `models/vlm/**/README.MD`
- `models/asr/**/README.MD`
- `models/tts/**/README.MD`
- `models/embedding/**/README.MD`
- `models/reranker/**/README.MD`
- `models/ocr/glm-ocr/README.MD`
- `models/ocr/paddleocr-vl/README.MD`
- `models/omni/**/README.MD`
- 与 README 描述强相关的 `get_model.py`、`ptq.py`、`build.py`、`demo.py`、`test.sh`

不适用于以下非大模型示例目录：

- `models/backbone/**`
- `models/detection/**`
- `models/segmentation/**`
- `models/autodrive/**`
- `models/ocr/lprnet/**`
- `models/ocr/PPOCRv3/**`
- 其他以传统 CV / 通用推理模型为主的目录

典型触发词：

- "为大模型示例生成 README"
- "参考其他 README 调整格式"
- "同步 README 和脚本真实行为"
- "补齐 README 的标准章节"
- "统一免责声明/ModelScope 链接格式"
- "检查 README 是否缺少性能结果章节"

## 目标

确保大模型示例（LLM、VLM、ASR、TTS、Embedding、Reranker、OCR、Omni 等）的 README 在结构、命名、内容上保持一致，并满足以下约束：

1. README 结构清晰，章节顺序与仓库主流模型保持一致。
2. README 中的默认模型、默认路径、输出文件名与脚本真实行为一致。
3. README 中的命令示例、`test.sh` 参数说明、性能结果章节完整且可维护。
4. ModelScope 链接、免责声明、平台限制说明等关键文案风格统一。

## 典型场景

- 新增大模型示例时，参考现有格式生成 README
- 为已有大模型示例补齐 README 的标准章节
- 根据脚本真实行为生成或重写 README
- 修改现有 README，使其与脚本真实行为保持一致
- 重构现有 README 的章节结构、命名或示例命令
- 检查 README 是否缺少章节、参数说明、结果章节或免责声明

## 开始前先读

优先阅读以下文件：

1. 目标目录下的 `README.MD` 或 `README.md`
2. 目标目录下的 `get_model.py`
3. 目标目录下的 `ptq.py`（若存在）
4. 目标目录下的 `build.py`
5. 目标目录下的 `demo.py`
6. 目标目录下的 `test.sh`

若需要选择参考模板，再补读同类型模型中格式较完整的 README。

## README 标准结构（最新）

推荐 README 收敛到以下结构（基于 qwen3-embedding 的优化）：

````
# Model-Name                  ← 标题：优先使用通用模型名称，避免带过细规格后缀

本示例展示如何把...模型量化和编译，部署到后摩芯片的设备上。

[TOC]                         ← 注意 TOC 位置：紧跟在标题和简介之后

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: ...
- 来源: https://www.modelscope.cn/...  ← 注意使用带 www 的链接
- 许可: Apache License 2.0 (...)

预训练模型在运行时下载，工程发布**不包含**该模型。

本例只适用于xh2。              ← 所有模型都需要这一句

### 默认配置

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：
- `model_name=...`
- `model_size=...`
- `context_length=...`
- ...

如果当前示例包含多个子模型，可在默认配置后补充子模型说明：
- `sub_model_a`: ...
- `sub_model_b`: ...

### 资源要求

**后摩设备运行要求：**
- `模型规格A`: 单卡12GB可运行
- `模型规格B`: 需要在后摩 2chips x 30GB 以上的卡才能运行

后摩设备运行要求必须按模型规格列出。若当前仓库、脚本、配置或用户输入中没有对应规格的明确运行硬件要求，默认写“单卡12GB可运行”。

**量化硬件要求：**
- `模型规格A`: 需要 GPU，显存最小 xxGB
- `模型规格B`: 暂无数据

量化硬件要求必须按模型规格列出。若当前仓库、脚本、配置或用户输入中没有对应规格的明确数据，写“暂无数据”，不要推测。若默认量化流程不需要 GPU，可写“不需要 GPU”或按实际脚本说明。

**编译时 host 内存要求：**
- `模型规格A`: 需要空闲 host 内存最小 xxGB
- `模型规格B`: 暂无数据

编译时 host 内存要求必须按模型规格列出。若当前仓库、脚本、配置或用户输入中没有对应规格的明确数据，写“暂无数据”，不要推测。

### 用于校准的数据集          ← 仅使用 wikitext 等校准数据集的模型需要此子章节

本项目引用 wikitext 数据集仅用于**研究和评估目的**。
- 数据集许可: CC BY-NC 4.0
- 来源: https://www.modelscope.cn/datasets/modelscope/wikitext

⚠️ 重要提示:
- 本数据集**不作为本仓库的一部分进行分发**。
- 本数据集**不受本项目 Apache License 2.0 许可的约束**。
- 任何对数据集的使用都必须遵守其原始许可 (CC BY-NC 4.0)，包括**非商业用途限制**。
- 用户有责任确保在使用本数据集时遵守相关规定。

## 2 快速开始

目前...模型仅提供python脚本编译方式。芯片模型只能在linux环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv
```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 get_model.py 下载模型，可通过参数选择下载原始模型、量化模型或芯片模型。

如果需要自己量化，可选择下载原始模型...：
```bash
python3 get_model.py --type raw
```

如果仅查看演示结果，可选择下载芯片模型...：
```bash
python3 get_model.py --type hmm
```

### 2.3 量化

量化需要使用GPU，并在宿主机安装与CUDA12.8兼容的CUDA和驱动。启动docker时将GPU映射到docker内，并指定可用的GPU ID...参考命令：
```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入 `models/xxx/yyy` 目录，执行 `ptq.py` 脚本...：
```bash
source env.sh
cd models/xxx/yyy
python3 ptq.py
```

硬规则：只要 README 中存在 `### 2.3 量化`，并且该目录的默认量化流程依赖 GPU / `ptq.py`，就必须包含以上三项信息，缺一不可：

1. GPU + CUDA 12.8 前提说明
2. 完整的 `docker run -it --gpus all ...` 命令
3. 进入 docker 后执行 `source env.sh`，并 `cd` 到当前模型目录再运行 `ptq.py` 的说明与示例命令

[可选] 量化完成的模型存放在`output/$HOUMO_TARGET/hmquant`目录：
```bash
.
|-- prefill
|-- decoder
|-- quant_embedding.pt
```

### 2.4 编译

将量化模型编译为在芯片上运行的模型。默认输出名称为 `xxx.hmm`，执行脚本：
```bash
python3 build.py
```

默认输出名称分别为： ← 需要列出所有输出文件
- `xxx.hmm`
- `yyy.hmm`

### 2.5 演示

xxx 模型使用 python API 进行演示。

1. `demo.py` 默认从 `config.yaml` 推导 `tokenizer_dir`，并默认读取：
- `output/${HOUMO_TARGET}/xxx.hmm`
- ...

2. 执行脚本，推理完成将输出性能结果。
```bash
python3 demo.py
```

## 3 一键评估              ← 注意：一键评估现在是第3章，不是 2.6

以上步骤可以通过 test.sh 脚本执行。默认执行 demo 步骤：
```bash
# 默认执行 demo，会先下载预编译模型
bash test.sh

# 完整流程：下载原始模型 + 量化 + 编译 + 演示
bash test.sh -s all
```

test.sh 支持以下参数：
```bash
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
                          Supports comma-separated values or repeated flags...
  -name, --model_name     Model name. Default: xxx.
  -size, --model_size     Model size. Default: ...
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

注意：README 中的 `test.sh` 参数说明必须根据**当前模型目录下 test.sh 实际使用/消费的参数**生成，只列出该脚本真实支持并在本模型流程中有意义的参数。不要把 `models/test_common.sh` 或其他共享解析器里存在、但当前模型脚本没有用到的通用参数一并列出来。

示例：
```bash
# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant

... [其他示例保持一致]
```

## 4 参考结果              ← 注意：参考结果现在是第4章

### 4.1 演示结果

```bash
... [实际的演示输出]
```

### 4.2 性能结果          ← 默认建议保留；若 4.1 已包含完整性能数据，可省略单独的 4.2

```bash
... [实际的性能数据，例如 qwen3-embedding 中的]
Total Input: 89 tokens, Prefill Cost 108.958 ms
Prefill Speed: 816.83 tokens/s
E2E Latency (End-to-End Latency): 0.430 seconds
```

如果是**重新创建** README，且当前仓库内没有可核对的真实演示输出或性能数据，则**不要虚构** 4.1 / 4.2 内容。此时仍需保留相应章节，并明确写明“暂无结果”或“暂无可提供的参考性能数据，待后续实测补充”等说明。

## 5 免责声明            ← 注意：免责声明现在是第5章

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护...
[完整的免责声明文本保持一致]
```

## 新旧结构对比

### 旧结构（已废弃）
```
# 标题
简介
## 预训练模型
## 用于校准的数据集
[TOC]
## 1 模型说明
## 2 快速开始
### 2.1 环境准备
### 2.2 获取模型和数据
### 2.3 量化
### 2.4 编译
### 2.5 演示
### 2.6 一键评估
## 3 参考结果
## 4 免责声明
```

### 新结构（推荐）
```
# 标题
简介
[TOC]                          ← TOC 位置提前
## 1 模型说明                  ← 整合预训练模型信息
### 默认配置                  ← config.yaml 默认值说明
### 资源要求                  ← 后摩运行、量化、编译资源要求
### 用于校准的数据集          ← 数据集作为子章节（如需要）
## 2 快速开始
### 2.1 环境准备
### 2.2 获取模型和数据
### 2.3 量化
### 2.4 编译
### 2.5 演示
## 3 一键评估                  ← 提升为一级章节
## 4 参考结果
### 4.1 演示结果
### 4.2 性能结果
## 5 免责声明
```

## 核心检查项

### 文件名（强规则）
- [ ] README 文档的名称必须为 `README.MD`（大写 MD）

### 标题与命名
- [ ] 标题使用清晰且通用的模型名称（如 `Qwen2.5`、`Qwen3-Embedding`、`GLM-4.5V`），不使用笼统描述
- [ ] 标题大小写格式一致（单词首字母大写）
- [ ] 标题与开头描述中的模型名称保持一致
- [ ] 标题默认尽量不要带 `7B`、`Instruct`、`Chat`、上下文长度等过细规格；除非该目录同时包含多个同系列变体，必须依靠这些信息区分

### 模型说明
- [ ] 第1章模型说明中每个模型必须包含三项：`模型名称`、`来源`、`许可`（缺一不可）
- [ ] 如果示例中包含多个模型，在模型说明章节中需要将各个模型的信息分开写，不要放到一行
- [ ] 许可信息以**官方仓库的 LICENSE 文件为准**（如 GitHub 上的 LICENSE），不要仅凭印象或 ModelScope 页面上的标签写许可名称
- [ ] 许可行后括号内附上 LICENSE 文件链接，方便核对，例如：`- 许可: Apache License 2.0 (https://github.com/zai-org/GLM-ASR/blob/main/LICENSE)`
- [ ] 第1章模型说明中包含“### 默认配置”；列出 `config.yaml` 作为默认值单一真值来源，并列出当前默认模型的关键配置
- [ ] 第1章模型说明中包含“### 资源要求”
- [ ] “### 资源要求”中包含“后摩设备运行要求”；按模型规格逐项列出，缺少明确数据的规格默认写“单卡12GB可运行”
- [ ] “### 资源要求”中包含“量化硬件要求”；按模型规格逐项列出，缺少明确数据的规格写“暂无数据”
- [ ] “### 资源要求”中包含“编译时 host 内存要求”；按模型规格逐项列出，缺少明确数据的规格写“暂无数据”

### TOC 位置
- [ ] `[TOC]` 在标题和简介之后，第1章之前

### 链接格式
- [ ] 所有 ModelScope 链接使用 `https://www.modelscope.cn/...` 格式（带 www）
- [ ] 链接使用方括号包裹的 markdown 格式：`[链接文字](URL)`

### 章节结构（新）
- [ ] 第1章"模型说明"中包含预训练模型信息
- [ ] "默认配置"作为第1章的子章节出现
- [ ] "资源要求"作为第1章的子章节出现
- [ ] "用于校准的数据集"作为第1章的子章节出现（仅在需要时）
- [ ] 章节编号：1 模型说明、2 快速开始、3 一键评估、4 参考结果、5 免责声明
- [ ] 子章节编号正确（2.1、2.2...；4.1、4.2）

### 内容一致性
- [ ] 第1章包含"本例只适用于xh2。"
- [ ] 2.2 获取模型的描述风格统一
- [ ] 2.3 量化有完整的 docker 命令（需要 GPU 的模型）
- [ ] 2.3 量化同时包含 GPU/CUDA 前提、`docker run` 命令、以及进入 docker 后 `source env.sh` + `cd models/...` 的示例（需要 GPU 的模型）
- [ ] 2.3 量化的 docker 镜像名统一为 `harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64`（不要使用旧的 `vx.y.z-ubuntu24.04-x86.64`、`vx.y.z-ubuntu20.04-py39-x86.64`、`x86_64` 等变体）
- [ ] 2.4 编译有输出文件名列表
- [ ] 有 4.1 演示结果 和 4.2 性能结果 两个子章节（如果 4.1 演示结果中已包含完整的性能数据，则不需要单独的 4.2 章节）
- [ ] 性能结果用真实数据，而非 `xxx` 占位符；若为新建 README 且暂无实测数据，保留章节并明确标注“暂无结果”，不要编造
- [ ] `test.sh` 参数说明仅包含当前模型 `test.sh` 实际使用的参数，不要把共享脚本中的未使用通用参数写进 README
- [ ] 如果有多个模型，在 4.1 演示结果中用 `####` 子子章节分别展示
- [ ] 免责声明文本完全一致

### 代码块
- [ ] 所有命令使用 `bash` 标注
- [ ] 代码块缩进正确

## 操作步骤

### 1. 选择参考模板
推荐选择同类型模型中最完整的 README 作为参考模板：
- Embedding 模型：`models/embedding/qwen3-embedding/README.MD`（推荐，结构最新）
- LLM 模型：`models/llm/qwen3/README.MD`
- VLM 模型：`models/vlm/qwen3-vl/README.MD`
- ASR 模型：`models/asr/qwen3-asr/README.MD`
- TTS 模型：`models/tts/cosyvoice3/README.MD`
- Reranker 模型：`models/reranker/qwen3-reranker/README.MD`
- OCR 模型：`models/ocr/glm-ocr/README.MD`
- Omni 模型：`models/omni/minicpmo/README.MD`

### 2. 检查现有内容
```bash
# 读取需要修改的 README
cat models/xxx/yyy/README.MD

# 对比与参考模板的差异
diff -u models/embedding/qwen3-embedding/README.MD models/xxx/yyy/README.MD
```

### 3. 确定是否需要数据集章节
检查 ptq.py 和 get_model.py：
```bash
grep -i "wikitext\|calib\|dataset\|hf-dataset\|hf_dataset\|load_dataset\|librispeech\|openslr\|huggingface" models/xxx/yyy/*.py
```
- 如果使用校准数据集 → 需要"### 用于校准的数据集"子章节
- 如果使用随机数据等 → 不需要此子章节

数据集来源不仅限于 wikitext，常见的还包括：
- 文本类：wikitext（LLM/Embedding 常用）
- 音频类：openslr/librispeech_asr（ASR 模型常用，许可 CC BY 4.0）
- 图像类：coco、imagenet 等
- 其他通过 HuggingFace `load_dataset` 或 `--hf-dataset` 参数引入的数据集

⚠️ 只要 `ptq.py` 的**默认参数**会触发下载/使用第三方数据集（即使是 HuggingFace 接口动态拉取，未打包进仓库），就必须添加"### 用于校准的数据集"子章节并声明许可。判断依据是脚本默认行为，而不是数据是否在仓库内分发。

不同数据集的许可参考：
- wikitext: CC BY-NC 4.0（注意非商业限制）
- LibriSpeech (openslr/librispeech_asr): CC BY 4.0
- 其他数据集需查证原始来源

### 4. 逐步修改（按新结构）
1. 调整 `[TOC]` 位置到标题和简介之后
2. 修改标题为准确且尽量通用的模型名称；默认不要带 `7B`、`Instruct` 等过细规格，除非目录中存在多个必须区分的同系列模型
3. 将"预训练模型"内容整合到"## 1 模型说明"中
4. 将"用于校准的数据集"改为"### 用于校准的数据集"子章节（如需要）
5. 将"一键评估"从 2.6 提升为"## 3 一键评估"
6. 重新编号后续章节：参考结果→第4章，免责声明→第5章
7. 重新编号子章节：4.1、4.2
8. 检查/统一 ModelScope 链接格式（加 www）
9. 确保"本例只适用于xh2。"在第1章
10. 检查/补充“### 默认配置”：明确当前目录使用 `config.yaml` 作为默认值单一真值来源，并列出当前默认模型关键配置；如示例包含多个子模型，可在此处补充子模型说明。
11. 检查/补充“### 资源要求”：将资源要求与默认配置分开，避免混在同一段落中。
12. 检查/补充“后摩设备运行要求”：优先使用用户明确给出的数据；其次从同目录 README、配置、脚本或可信参考中读取；缺少明确数据时默认填“单卡12GB可运行”。
13. 检查/补充“量化硬件要求”：优先使用用户明确给出的数据；其次从同目录 README、配置、脚本或可信参考中读取；缺少明确数据时填“暂无数据”，不要推测。若默认量化流程不需要 GPU，按实际脚本说明。
14. 检查/补充“编译时 host 内存要求”：优先使用用户明确给出的数据；其次从同目录 README、配置、脚本或可信参考中读取；缺少明确数据时填“暂无数据”，不要推测。已知规则：`20b` 需要空闲 host 内存最小 80GB，`120b` 需要空闲 host 内存最小 350GB。
15. 检查/补充 2.3 量化的 docker 说明
16. 检查/补充 2.4 编译的输出文件名
17. 检查/更新 4.2 性能结果（用真实数据）
18. 验证免责声明一致

### 5. 验证
```bash
# 阅读修改后的 README，检查流畅性
cat models/xxx/yyy/README.MD

# 与同类型其他 README 并排比较
ls -la models/xxx/*/README.MD
```

## 最小验证

完成 README 修改后，至少确认以下几点：

1. README 中引用的脚本、参数名、默认值能在对应脚本中找到。
2. README 中列出的输出文件名与 `build.py` 或 `test.sh` 真实产物一致。
3. README 中 `test.sh` 参数列表只覆盖当前模型脚本实际消费的参数，而不是共享解析器中的参数超集。
4. README 中的 ModelScope 链接、免责声明和章节顺序没有偏离仓库主流格式（新结构）。
5. 章节编号正确：1-模型说明、2-快速开始、3-一键评估、4-参考结果、5-免责声明。

## 注意事项

### 模型名称准确性
- 优先使用官方/实际的模型主名称，并与当前目录默认模型保持一致（如 `Qwen2.5`、`Qwen3-Embedding`、`GLM-4.5V`）
- 标题和正文的模型名称保持一致
- README 标题默认应尽量收敛为通用模型名，不要机械照搬完整仓库名、权重规格名或 `-Instruct` / `-Chat` / `-Preview` 后缀
- 如果一个目录内同时覆盖多个必须区分的变体，再在标题中保留必要的规格信息
- 如果有多个模型，标题中同时列出（如 `BGE-M3 和 BGE-Reranker-V2-M3`、`Qwen3-ASR 和 Qwen3-ForcedAligner`）

### 多个模型的情况处理
- 如果一个目录包含多个模型（如 qwen3-asr 包含 0.6B/1.7B/forcealigner）：
  - 在"## 1 模型说明"中，每个模型的信息分开列出（模型名称、来源、许可各占一组）
  - 在"### 2.2 获取模型和数据"中，分别说明每个模型的下载命令
  - 在"### 2.3 量化"中，分别说明每个模型的量化命令
  - 在"### 2.4 编译"中，按模型分组列出输出文件名（如"对于 xxx：- ... - ..."）
  - 在"### 2.5 演示"中，分别说明每个演示脚本的用法
  - 在"## 3 一键评估"中，列出支持的模型组合
  - 在"### 4.1 演示结果"中，用 `####` 子子章节分别展示每个模型的演示输出
  - 子子章节标题格式：`#### 模型名`（不需要冒号结尾）

### 演示结果内容要求
- 演示结果应使用真实、完整的输出
- 包含所有相关的日志信息（如设备信息、处理步骤、性能数据等）
- 保持实际运行时的输出格式和顺序
- 如果是**重新创建** README，且当前没有可验证的真实输出，则不要根据经验、其它模型 README 或模型常识拼接一段“看起来合理”的输出；应保留章节并明确写明“暂无结果，待后续实测补充”

### 特殊环境要求的处理
- 如果模型有特殊的环境要求（如 python 版本限制、需要修改源码、需要额外安装/卸载特定包），在"### 2.1 环境准备"中详细说明
- 提供具体的操作步骤和代码片段，如修改源码的行数和内容

### 目录章节逻辑顺序
- 多个模型来源说明时，"本例只适用于xh2。"的位置要合理，避免打断逻辑
- 第1章推荐顺序：模型信息→预训练模型不随工程发布说明→xh2说明→默认配置→资源要求→数据集（如需要）
- 默认配置和资源要求应使用独立子章节，避免把 `config.yaml` 默认值、后摩设备运行要求、量化 GPU 要求和编译 host 内存要求混在同一段落中

### 平台特定说明
- 如果某个模型只适用于特定平台，在第1章明确说明
- 当前多数模型适用于 xh2

### 输出文件名
- 确保 2.4 编译章节列出的文件名与实际 build.py 的输出一致
- 检查 build.py 中的模型命名逻辑

### 性能结果数据
- 如果是**重新创建** README 且暂时无法获取真实数据，保留 4.2 章节并明确写明“暂无可提供的参考性能数据，待后续实测补充”，不要填写推测值、其它模型数据或测试配置中的非实测占位内容
- 真实数据格式可参考 qwen3-embedding 的 4.2 章节

### 不同模型类型的适配
- **LLM**: 通常有 prefill/decoder 等输出，量化需要 wikitext
- **VLM**: 可能有 visual 相关的额外输出和步骤
- **TTS**: 可能有不同的输入输出格式
- **Embedding**: 通常只有 embedding/prefill，部分有 reranker
- 根据不同模型类型调整具体内容描述，但保持整体结构一致

### 性能结果章节的灵活处理
- 如果 4.1 演示结果中已经包含了完整的性能数据（如 qwen3-asr 模型那样，演示输出直接带有详细的性能统计），则不需要单独的 4.2 性能结果章节
- 如果演示结果只是展示功能输出，而性能数据是单独整理的，则保留 4.1 和 4.2 两个子章节
````
