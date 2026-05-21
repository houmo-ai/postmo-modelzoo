# Bge embedder和reranker

本示例展示如何把bge系列模型量化和编译，部署到后摩M50芯片的设备上。

## 特别事项

本项目使用从 ModelScope 下载的配置文件：

- 模型名称: [bge-m3, bge-reranker-m3-v2]
- 来源: [https://modelscope.cn/models/BAAI/bge-m3, https://modelscope.cn/models/BAAI/bge-reranker-v2-m3]
- 许可: Apache License 2.0 （https://www.apache.org/licenses/LICENSE-2.0）

所需配置文件在运行时下载，工程发布**不包含**该下载包。

[TOC]

## 1 模型说明

本例使用的 bge-reranker 实现来源于[https://modelscope.cn/BAAI/bge-reranker-v2-m3](https://modelscope.cn/BAAI/bge-reranker-v2-m3)。
本例使用的 bge-m3 实现来源于[https://modelscope.cn/BAAI/bge-m3](https://modelscope.cn/BAAI/bge-m3)。

当前目录使用 `config.yaml` 作为默认值单一真值来源。当前默认模型为：

- `model_name=bge`
- `model_size=0.5b`
- `context_length=0.5k`（即 512）
- `batch=10`
- `ndevice=1`

## 2 快速开始

bge系列模型提供在x86平台下提供量化、编译和部署范例，其它平台提供编译后模型直接部署demo。

### 2.1 环境准备

安装示例运行所需的python依赖，建议使用venv

```bash
pip3 install -r requirements.txt
```

### 2.2 获取模型和数据

通过 `get_model.py` 下载模型，如果不需要量化和编译，则只需下载编译好的 hmm 模型。

```bash
python3 get_model.py --type hmm
```

也可以直接下载以上所有模型和数据集（注意相同模型会覆盖，请自行保存）,量化和编译所需的原始onnx浮点模型：

```bash
python3 get_model.py --type raw
```

### 2.3 量化

当前 `ptq.py` 默认读取 `./onnx` 下的 `bge-m3` 与 `bge-reranker-v2-m3` ONNX 子目录，量化精度默认来自 `config.yaml`，当前默认值为 `w8a8_sefp`。

```bash
# 仅量化
python3 ptq.py
```

### 2.4 编译

```bash
# 量化+编译
python3 build.py
```

默认输出名称分别为：

- `bge-m3-0.5b.hmm`
- `bge-reranker-v2-m3-0.5b.hmm`

### 2.5 演示

bge模型使用python API进行演示

1. `demo.py` 默认从 `config.yaml` 推导 tokenizer 目录，并默认读取：

  - `output/${HOUMO_TARGET}/bge-m3-0.5b.hmm`
  - `output/${HOUMO_TARGET}/bge-reranker-v2-m3-0.5b.hmm`

2. 执行脚本，推理完成将输出性能结果。

```bash
python3 demo.py
```

### 2.6 一键评估

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
                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build.
  -name, --model_name     Model name. Default: bge.
  -size, --model_size     Model size. Default: 0.5b.
  --skip_download         Skip model download steps.
  -h, --help              Show this help message.
```

当前脚本会把 `model_name` 和 `model_size` 透传给 `get_model.py`、`build.py`、`demo.py`。默认值分别是 `bge` 和 `0.5b`。

示例：

```bash
# 仅运行 quant，会先下载原始模型再量化
bash test.sh -s quant

# 仅运行 build，要求当前目录下已经有量化输出
bash test.sh -s build

# 仅运行 demo，会先下载预编译模型
bash test.sh -s demo

# 完整流程（下载原始模型、量化、编译、演示）
bash test.sh -s all

# 显式指定模型名和模型规格（当前默认就是 bge 0.5b）
bash test.sh -s demo -name bge -size 0.5b

# 组合执行多个步骤
bash test.sh -s quant,build

# 跳过模型下载，使用当前目录下已有的预编译模型进行演示
bash test.sh -s demo --skip_download
```

## 3 参考结果

### 3.1 演示结果

```bash
对corpus编码，查询符合query的资料：
    corpus = [
        "北京有很多著名的牛肉面馆。",
        "上海是中国的金融中心。",
        "故宫是北京著名的旅游景点。",
        "兰州牛肉面是非常有名的中国美食。",
        "深圳以科技产业闻名。",
        "北京海淀区有一家西北牛肉面馆非常好吃。"
    ]

    query = "北京哪里有好吃的牛肉面"
结果:
==== 粗召回结果 (bge-m3) ====
北京海淀区有一家西北牛肉面馆非常好吃。 score:0.971890
上海是中国的金融中心。 score:0.523505
故宫是北京著名的旅游景点。 score:0.503346

==== 精排结果 (bge-reranker-v2-m3) ====
doc: 北京海淀区有一家西北牛肉面馆非常好吃。  →  score: -1.9336
doc: 故宫是北京著名的旅游景点。  →  score: -4.0898
doc: 上海是中国的金融中心。  →  score: -5.6172
```

### 3.2 性能结果

```bash
bge-m3资料库编码耗时：441.012144ms
bge-m3查询值编码耗时: 439.535618ms
bge-reranker-v2-m3精排编码耗时: 430.264235ms
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
