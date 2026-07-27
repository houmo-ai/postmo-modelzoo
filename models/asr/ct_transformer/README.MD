# CT-Transformer

本示例展示如何把 CT-Transformer 标点恢复模型量化和编译，部署到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope 的预训练模型：

- 模型名称: CT-Transformer（标点恢复/Punctuation Restoration）
- 来源: [ModelScope iic/punc_ct-transformer_cn-en-common-vocab471067-large](https://www.modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large)
- 许可: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

原始预训练模型用于导出和量化，工程发布**不包含**原始权重。

本例只适用于xh2。

### 默认配置

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=ct_transformer`
- `model_size=large`
- `ncore=2`
- `ndevice=1`
- `quant_type=w8a8_sefp`
- `HOUMO_TARGET=xh2`（未设置时由脚本默认使用 `xh2`）

HMM 的输入为固定 batch=1 的文本序列，当前编译产物的静态输入长度通常为
`fixed_length=220`。`demo.py` 会从 HMM 输入 shape 读取该值，而不是根据输入文本重新
计算。

### 资源要求

**后摩设备运行要求：**

- CT-Transformer large：单卡12GB可运行

**量化硬件要求：**

- CT-Transformer large：需要 GPU

**编译时 host 内存要求：**

- CT-Transformer large：需要空闲 host 内存最小 10 GB

## 2 快速开始

目前 CT-Transformer 模型仅提供 Python 脚本编译方式。芯片模型只能在 Linux 环境下量化和编译。

### 2.1 环境准备

安装示例运行所需的 Python 依赖，建议使用 venv：

```bash
pip3 install -r requirements.txt
```

`demo.py` 需要能够导入 `funasr.models.ct_transformer.utils`，并使用 `jieba`（仅当本地提供用户词典时加载）。

### 2.2 获取模型和数据

通过 `get_model.py` 获取原始模型资源或预编译芯片模型。

如果仅查看演示结果，可选择下载预编译 HMM：

```bash
python3 get_model.py --type hmm
```

请准备以下本地资源：

```text
ct_transformer/
├── tokens.json                         # 必需
├── jieba_usr_dict                      # 可选，也支持 .txt / seg_dict
├── config.yaml 或 configuration.json    # 推荐，用于读取标点配置
└── example/punc_example.txt            # 默认输入
output/xh2/
└── ct_transformer.hmm
```

原始模型下载命令如下：

```bash
python3 get_model.py --type raw
```

### 2.3 量化

量化需要使用 GPU，并在宿主机安装与 CUDA 12.8 兼容的 CUDA 和驱动。启动 docker 时将 GPU 映射到 docker 内，参考命令：

```bash
docker run -it --gpus all --pid=host -w /hmdd -v $PWD:/hmdd --shm-size 64g harbor.houmo.ai/toolchain/release:Dadao-xh2-x.y.z-ubuntu24.04-x86.64 /bin/bash
```

进入 docker 后 source 环境变量，并进入当前模型目录执行 `ptq.py`：

```bash
source env.sh
cd models/asr/ct_transformer
python3 ptq.py --model_name ct_transformer --model_size large
```

默认量化方式从 `config.yaml` 读取，也可以通过 `--quant_type` 指定，例如：

```bash
python3 ptq.py --model_name ct_transformer --model_size large --quant_type w8a8_sefp
```

### 2.4 编译

将量化模型编译为在芯片上运行的 HMM：

```bash
python3 build.py --model_name ct_transformer --model_size large --stage build
```

默认输出目录为 `output/xh2/`，输出文件为：

- `output/xh2/ct_transformer.hmm`

### 2.5 演示

CT-Transformer 使用 Python API 进行离线演示。默认值来自 `config.yaml`，默认从 `ct_transformer/` 读取前处理 metadata，并从 `output/xh2/` 查找 HMM：

```bash
python3 demo.py --model_name ct_transformer --model_size large
```

也可以显式指定 metadata、HMM 和输入文件：

```bash
python3 demo.py \
  --model_dir ./ct_transformer \
  --model_path ./output/xh2/ct_transformer.hmm \
  --input ./ct_transformer/example/punc_example.txt
```

当前 `demo.py` 支持的参数如下：

- `--config`：配置文件路径，默认 `config.yaml`
- `--model_dir`：本地前处理 metadata 目录，默认 `./ct_transformer`
- `--model_name`：模型名称，默认从 config 读取
- `--model_size`：模型大小，默认从 config 读取
- `--input`：输入文本文件，默认 `ct_transformer/example/punc_example.txt`
- `--split_size`：每个 mini sentence 的 token 数量，默认 `20`
- `--hmm_dir`：HMM 所在目录，默认 `output/xh2/`，也会查找一级子目录
- `--model_path`：直接指定 HMM 路径，优先于 `--hmm_dir`

输入和输出约定：

| 角色 | 名称 | 形状 | 类型 |
|------|------|------|------|
| Input | `text` | `[1, fixed_length]` | `int32` |
| Input | `text_lengths` | `[1]` | `int32` |
| Output | `logits` | `[1, fixed_length, num_puncs]` | `float32` |

每个 mini sentence（含缓存后的 token）不能超过 HMM 的 `fixed_length`。demo 会对输入
文本分词、切分、补齐后调用 HMM，并输出恢复标点后的文本和性能摘要。

## 3 一键评估

以上步骤可以通过 `test.sh` 执行。脚本默认执行 demo 步骤：

```bash
# 默认执行 demo，会下载预编译 HMM 和执行 demo 所需资源
bash test.sh

# 仅量化（会先下载原始模型）
bash test.sh -s quant

# 仅编译
bash test.sh -s build

# 量化 + 编译 + 演示
bash test.sh -s all
```

`test.sh` 当前流程支持的参数如下：

```text
Usage: test.sh [options]
  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all.
  -name, --model_name     Model name. Default: ct_transformer.
  -size, --model_size     Model size. Default: large.
  -h, --help              Show this help message.
```

示例：

```bash
# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant --model_name ct_transformer --model_size large

# 仅运行 demo，跳过下载
bash test.sh -s demo --skip_download --model_name ct_transformer --model_size large
```

## 4 参考结果

### 4.1 演示结果

测试输入为 `ct_transformer/example/punc_example.txt`，包含 3 段文本。一次实测的推理结果如下（文本内容取决于输入文件和本地 metadata）：

```bash
Inference results: [{'text': ' 1跨境河流是养育沿岸人民的生命之源。长期以来，为帮助下游地区防灾减灾，中方技术人员在上游地区极为恶劣的自然条件下克服巨大困难，甚至冒着生命危险，向印方提供汛期水文资料，处理紧急事件。中方重视印方在跨境河流问题上的关切，愿意进一步完善双方联合工作机制。凡是中方能做的，我们都会去做，而且会做得更好。我请印度朋友们放心，中国在上游的任何开发利用，都会经过科学规划和论证，兼顾上下游的利益。'}, {'text': ' 2从存储上来说，仅仅是全景图片，它就会是图片的四倍的容量。然后全景的视频会是普通视频八倍的这个存储的容要求，而三d的模型会是图片的十倍。这都对我们今天运行在的云计算的平台，存储的平台提出了更高的要求。'}, {'text': ' 3那今天的会就到这里吧，happy new year,明年见。'}]
```

### 4.2 性能结果

```bash
====================================================================================================
                    Model Inference Performance Summary Report
====================================================================================================
Performance Details:
  Load Model+Tokenizer   : 19445.77 ms
  Total Inference (3  seqs) :   76.61 ms
  Average Inference/seq  :   25.54 ms
====================================================================================================
```

其中：

- `Load Model+Tokenizer` 包含 FunASR 工具首次导入、本地 metadata 读取、jieba 初始化和 HMM 加载。
- `Total Inference` 只统计所有输入文本的推理时间，不包含加载阶段。
- `Average Inference/seq` 等于 `Total Inference` 除以输入文本段数。

耗时会受到 CPU/NPU 状态、文件系统、Python 环境以及缓存状态影响，README 不固定给出单次实测毫秒值。

## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

## 附录一：demo 推理流程

本附录按照当前 `demo.py` 的执行顺序说明一次完整推理。

### A.1 整体流程

```text
命令行参数
    │
    ├─ 读取 config.yaml 默认模型配置
    ├─ 加载本地 metadata（tokens.json、用户词典、标点配置）
    ├─ 解析并加载 HMM
    ├─ 从 HMM 输入 shape 获取 fixed_length
    ├─ 读取输入文本并过滤空行
    └─ 对每段文本执行：分词 → token ID → 分块 → padding → HMM 推理
                                      │
                                      └─ logits → argmax → 标点后处理 → 输出文本
```

### A.2 解析参数和默认路径

程序入口为 `main(get_args())`。`get_args()` 首先解析命令行参数，然后从 `config.yaml` 读取 `default_model_name` 和 `default_model_size`，仅在用户没有显式传参时使用这些默认值。
默认路径如下：

- metadata 目录：`<demo.py 所在目录>/ct_transformer`
- HMM 目录：`<demo.py 所在目录>/output/<HOUMO_TARGET>`，`HOUMO_TARGET` 默认为 `xh2`
- 输入文件：`<demo.py 所在目录>/ct_transformer/example/punc_example.txt`

如果指定了 `--model_path`，它优先于 `--hmm_dir`。否则程序先查找 `<hmm_dir>/<model_name>.hmm`，再查找 HMM 目录下一级子目录中的同名文件。

### A.3 加载本地前处理 metadata

`_LocalPreprocessor` 只读取推理所需的本地信息：

1. `tokens.json`：构造 `_LocalTokenizer`，支持 token 列表、token-to-id 字典和
   id-to-token 字典三种常见格式，并确定未知词对应的 ID。
2. 用户词典：按以下顺序查找 `jieba_usr_dict`、`jieba_usr_dict.txt` 和 `seg_dict`。
   找到后加载 jieba 用户词典；未找到时使用 FunASR 分词工具的默认行为。
3. 标点配置：从 `punc_list.json`、`config.yaml` 或 `configuration.json` 读取
   `punc_list`。
4. 句末标点 ID：从同类 metadata 读取 `sentence_end_id`。缺少配置时，代码使用内置
   fallback 值，因此建议交付与模型匹配的配置文件。

在这一步缺少 `tokens.json`、metadata 目录不存在或依赖无法导入时，demo 会记录错误并返回非零退出码。原始 `model.pt` 不参与此过程。

### A.4 加载 HMM 并确定静态输入长度

程序通过 `tcim_lite.runtime.load()` 加载 `.hmm` 文件，并从 HMM 的第一个输入读取：

- 输入名称和数据类型
- 第二维序列长度 `fixed_length`

`fixed_length` 是编译时确定的静态长度。例如当前模型通常返回 `220`。
demo 不会根据某一段输入文本动态调整该值，而是把每个 mini sentence 补齐到该长度；如果实际 token 数量超过该长度，则直接报错，提示需要减小 `--split_size` 或调整编译模型的输入长度。

### A.5 读取文本并切分 mini sentence

`load_text_items()` 读取 `--input` 指定的 UTF-8 文本文件，逐行去除首尾空白并忽略空行。每个非空行作为一条独立推理序列，最终用于计算 `Total Inference` 和 `Average Inference/seq`。

对于每条文本，`tokenize_to_chunks()` 执行以下操作：

1. 调用 FunASR CT-Transformer 工具的 `split_words()` 分词。
2. 使用本地 tokenizer 将 token 转换为整数 ID，未知 token 使用 `<unk>`、`<UNK>` 或
   `[UNK]` 对应的 ID；若词表没有这些标记，则使用代码中的 fallback ID。
3. 按 `--split_size` 将 token 和 token ID 同步切分成多个 mini sentence。默认每块为 20 个 token。

token 和 token ID 必须保持一一对应，否则模型输出的标点位置将无法映射回原始文本。

### A.6 单个 mini sentence 的 HMM 推理

`run_punc_inference_hmm()` 对每个 mini sentence 循环执行：

1. 将上一块保留的 `cache_sent` 和当前 mini sentence 拼接，形成当前 HMM 输入。
2. 将 token ID 转成 `int32` 数组，并通过 `pad_text_ids()` 在右侧补零到
   `[1, fixed_length]`。
3. 设置 `text` 输入；如果 HMM 有第二个输入，则设置模型要求的长度输入为固定的
   `fixed_length`。代码同时保留实际有效 token 长度 `text_lengths`，用于推理后截取
   padding 区域之外的 logits。
4. 调用 `module.run()` 执行推理，再调用 `module.sync()` 等待设备计算完成。
5. 读取第一个输出 `logits`，其形状为 `[1, fixed_length, num_puncs]`，只保留有效 token
   长度对应的部分，并在标点类别维度执行 `argmax`。

### A.7 跨分块缓存和标点后处理

文本过长时，一个 mini sentence 的结尾可能无法独立判断句子边界，因此代码会保留一段 `cache_sent` 和 `cache_sent_id`，并在下一块推理时拼接到当前输入前面：

- 优先从后向前查找中文句号 `。` 或问号 `？` 作为安全切分点。
- 如果缓存长度超过 `200`，且没有找到句号/问号，则退化到最近的中文逗号 `，`。
- 选定切分点后，已确定的部分加入输出，未确定的尾部继续进入下一轮缓存。

模型输出的标点类别会经过以下后处理：

- `_` 表示不添加标点。
- 中文 `，`、`。`、`？` 在英文 token 后转换为 `,`、`.`、`?`。
- 英文 token 在句首或中英文相邻位置按代码规则补充空格，并在句首进行大小写处理。
- 最后一块文本如果没有句末标点，会根据末字符类型补充中文句号 `。` 或英文句号 `.`。

### A.8 性能统计边界

程序使用两段计时：

- 加载计时从首次导入 FunASR CT-Transformer 工具开始，到本地 metadata 和 HMM 加载完成
  结束，对应 `Load Model+Tokenizer`。
- 推理计时在每条输入文本调用 `run_punc_inference_hmm()` 前后分别记录，并累加所有文本
  耗时，对应 `Total Inference`；它不包含加载阶段。

最后用输入文本条数除 `Total Inference` 得到 `Average Inference/seq`。
因此首次运行的 Python 导入、jieba 初始化和文件缓存可能显著影响加载时间，但不会被重复计入每条文本的推理时间。
