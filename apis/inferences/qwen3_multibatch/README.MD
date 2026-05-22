# Qwen3 MultiBatch

本示例展示如何在后摩芯片设备上运行 qwen3-8b-16k 4batch 多 batch 推理。

[TOC]

## 1 模型说明

本例使用的模型实现来源于 ModelScope：

- 模型名称: Qwen3-8B
- 来源: https://modelscope.cn/models/Qwen/Qwen3-8B
- 许可: Apache License 2.0

预训练模型在运行时下载，工程发布不包含该模型。

目前本示例仅支持 xh2。

## 2 快速开始

1. 设置环境变量：

```bash
cd houmo-example-xh2
source env.sh
```

2. 进入示例目录：

```bash
cd apis/inferences/qwen3_multibatch
```

3. 直接执行示例：

```bash
bash run.sh
```

run.sh 会自动完成以下步骤：

- 下载 qwen3-8b-16k 4batch 预编译模型
- 检查推理所需文件是否齐全
- 执行 demo_multibatch.py

## 3 手动执行

如需分步执行，可使用以下命令：

```bash
python3 get_model.py
```

```bash
python3 demo_multibatch.py --forbid_flush
```

说明：

- demo_multibatch.py 会从模型文件中自动读取 batch 大小。
- demo_multibatch.py 默认使用 qwen3-8b tokenizer。
- 如果 tokenizer 不在默认目录，可通过 --tokenizer_dir 手动指定。

## 4 参数说明

### 4.1 get_model.py

| Args | Default Value | Description |
| --- | --- | --- |
| --download_dir | . | 下载目录 |
| --extract_dir | None | 解压目录 |
| --source_type | jfrog | 下载源，支持 jfrog / modelscope |

说明：该脚本固定下载 qwen3-8b-16k 4batch 预编译模型，默认下载到当前目录。

### 4.2 demo_multibatch.py

| Args | Default Value | Description |
| --- | --- | --- |
| --tokenizer_dir | qwen3-8b | tokenizer 路径 |
| --embedding_path | output/{target}/hmquant/quant_embedding.pt | embedding 权重路径 |
| --prefill_path | output/{target}/qwen3-8b_prefill.hmm | prefill 模型路径 |
| --decode_path | output/{target}/qwen3-8b_decode.hmm | decode 模型路径 |
| --forbid_flush | False | 禁止终端刷屏展示 |

### 4.3 run.sh

run.sh 不接收参数，直接执行下载并运行 qwen3-8b-16k 4batch 示例。

## 5 注意事项

1. 请确保 HOUMO_TARGET 已设置为 xh2。
2. 本目录仅保留 qwen3-8b-16k 4batch 示例。
3. 如果需要自行量化和编译，请参考 models/llm/qwen3 目录下对应流程。

## 6 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据，应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。