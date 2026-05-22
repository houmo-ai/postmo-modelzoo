# Qwen3 Pipeline(Tensor & Pipeline Parallel)

本示例展示如何使用流水线并行和张量并行方式部署编译后的 Qwen3 模型到后摩芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于：

-   模型名称: Qwen3-8B
-   来源: [https://modelscope.cn/models/Qwen/Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B)
-   许可: [Apache License 2.0](https://modelscope.cn/models/Qwen/Qwen3-8B/file/view/master/LICENSE)

预训练模型在运行时下载，工程发布**不包含**该模型。
目前芯片模型只能在 linux 环境下量化和编译，请参考 houmo-examples/models/llm/qwen3 中的量化和编译。

## 2 快速开始

目前 Qwen3 Pipeline 模型仅提供 python 脚本编译方式。

1. 设置环境变量：

```bash
cd houmo-example-xh2
source env.sh
```

2. 获取模型文件，提供2chip 2 core模型，4chip模型需要用户自行量化编译，执行命令：

```bash
# 下载 4chip 2 core 模型
python3 get_model.py
```

如果需要其他规格，请参考 houmo-examples 自行编译或通过其他途径获取。

3. Qwen3 Pipeline 模型使用 python API 进行演示，提供两种方式的 demo：

-   `Pipeline Parallel`: 执行流水线并行推理
-   `Tensor Parallel`: 执行张量并行推理

```bash
# Prefill + Decode Pipeline Parallel demo 执行方式 (适用于4芯片配置)
python3 prefill_pp_decode_pp.py

# Prefill Pipeline Parallel + Tensor Parallel demo 执行方式 (适用于混合并行配置)
python3 prefill_pp_decode_tp.py
```

## 3 并行策略说明

### 3.1 流水线并行 (Pipeline Parallel)

在 `prefill_pp_decode_pp.py` 中实现了流水线并行策略：

- Prefill阶段在多个芯片上进行流水线分割
- Decode阶段在特定芯片上执行
- 利用队列机制在不同阶段间传递数据
- 支持多芯片协同工作以加速推理过程

### 3.2 张量并行 (Tensor Parallel)

在 `prefill_pp_decode_tp.py` 中实现了张量并行策略：

- Prefill阶段采用流水线并行方式执行
- Decode阶段采用张量并行方式执行
- 多个芯片共同参与单次计算，分担计算负载
- 通过共享KV缓存实现高效通信

## 4 参数说明

运行脚本支持以下参数：

| Args                  | Default Value                              | Description              |
| --------------------- | ------------------------------------------ | ------------------------ |
| --tokenizer_dir       | qwen3-8b                                   | Tokenizer path           |
| --embedding_path      | output/{target}/hmquant/quant_embedding.pt | Embedding weight path    |
| --model_path          | output/{target}                            | the path of hmm model    |
| --ndevice             | 2                                          | device nums of M50       |
| --input_tokens        | 30720                                      | the num of input tokens  |
| --repetition_penalty  | 1.0                                        | repetition_penalty       |
| --topk                | None                                       | Top-K                    |
| --topp                | 1.0                                        | Top-P                    |
| --temperature         | 1.0                                        | Temperature              |

## 5 性能结果

### 5.1 prefill_pp_decode_tp.py

```bash
2026-04-10 07:59:45.899] [tid 3920] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
2026-04-10 07:59:46.308 | INFO     | __main__:main:707 - MultiChip pipeline parallel run demo Start!
[2026-04-10 07:59:46.326] [tid 3920] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-04-10 07:59:46.326] [tid 3920] [warning] Empty backend name, use Xh2HalBackend instead.
[2026-04-10 07:59:46.332] [tid 3920] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-04-10 07:59:46.332] [tid 3920] [warning] Empty backend name, use Xh2HalBackend instead.
2026-04-10 07:59:47.318 | INFO     | __main__:prefill_producer:343 - Prefill Stage0 load model successfully
2026-04-10 07:59:47.677 | INFO     | __main__:prefill_consumer:481 - Prefill Stage1 load model successfully
2026-04-10 07:59:48.009 | SUCCESS  | __main__:prefill_producer:365 - User question:
Hello, '1' *30694
2026-04-10 07:59:48.047 | INFO     | __main__:prefill_producer:384 - Prefill input tokens length: 30720
2026-04-10 07:59:48.535 | INFO     | __main__:decode_runner:551 - Decode Model Run on devices [0, 1] load model successfully
2026-04-10 08:00:10.705 | INFO     | __main__:prefill_producer:462 - [Prefill Stage 0] Process Finished, time: 18.41 s.
2026-04-10 08:00:12.299 | INFO     | __main__:prefill_consumer:534 - [Prefill Stage 1] Process Finished, time: 18.73 s.
2026-04-10 08:00:12.301 | INFO     | __main__:decode_runner:594 - [Result] total Run Prefill 120 times, total_time : 19.94 s, output id = [2132]
2026-04-10 08:00:19.775 | SUCCESS  | __main__:decode_runner:614 - Model response:
It seems like you've sent a long string of repeated numbers. Are you trying to test something or is there a specific question or task you'd like help with? Let me know how I can assist you!
2026-04-10 08:00:22.446 | INFO     | __main__:decode_runner:692 - Decode Model Run on devices [0, 1], decode time 2.37 s.
2026-04-10 08:00:22.446 | INFO     | __main__:decode_runner:695 - Decode Model average time 56.37 ms/token
2026-04-10 08:00:22.521 | INFO     | __main__:main:763 - MultiChip pipeline parallel run demo finished!
```

### 5.2 prefill_pp_decode_pp.py

```bash
2026-03-05 14:25:34.347 | INFO     | __main__:main:690 - MultiChip pipeline parallel run demo Start!
[2026-03-05 14:25:34.354] [tid 1591] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-03-05 14:25:34.354] [tid 1591] [warning] Empty backend name, use Xh2HalBackend instead.
[2026-03-05 14:25:34.357] [tid 1592] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-03-05 14:25:34.357] [tid 1592] [warning] Empty backend name, use Xh2HalBackend instead.
[2026-03-05 14:25:34.360] [tid 1595] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-03-05 14:25:34.360] [tid 1595] [warning] Empty backend name, use Xh2HalBackend instead.
[2026-03-05 14:25:34.363] [tid 1597] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-03-05 14:25:34.363] [tid 1597] [warning] Empty backend name, use Xh2HalBackend instead.
2026-03-05 14:25:35.331 | INFO     | __main__:consumer:465 - Prefill Stage1 load model successfully
2026-03-05 14:25:35.332 | INFO     | __main__:consumer:465 - Prefill Stage2 load model successfully
2026-03-05 14:25:35.338 | INFO     | __main__:producer:305 - Prefill Stage0 load model successfully
2026-03-05 14:25:35.365 | INFO     | __main__:consumer:478 - Decode Stage1 load model successfully
2026-03-05 14:25:35.366 | INFO     | __main__:consumer:478 - Decode Stage2 load model successfully
2026-03-05 14:25:35.372 | INFO     | __main__:producer:323 - Decode Stage0 load model successfully
2026-03-05 14:25:35.413 | INFO     | __main__:consumer:465 - Prefill Stage3 load model successfully
2026-03-05 14:25:35.452 | INFO     | __main__:consumer:478 - Decode Stage3 load model successfully
2026-03-05 14:25:36.016 | SUCCESS  | __main__:producer:332 - User question:
Hello, '1' *30694
2026-03-05 14:25:36.051 | INFO     | __main__:producer:351 - Prefill input tokens length: 30720
2026-03-05 14:25:47.972 | INFO     | __main__:producer:413 - [Prefill Stage 0] Process Finished, time: 10.33 s.
2026-03-05 14:25:48.119 | INFO     | __main__:consumer:513 - [Prefill Stage 1] Process Finished, time: 10.32 s.
2026-03-05 14:25:48.263 | INFO     | __main__:consumer:513 - [Prefill Stage 2] Process Finished, time: 10.35 s.
2026-03-05 14:25:48.409 | INFO     | __main__:consumer:513 - [Prefill Stage 3] Process Finished, time: 10.70 s.
2026-03-05 14:25:48.431 | INFO     | __main__:result_shower:591 - [Result] total Run Prefill 120 times, total_time : 12.35 s, output id = [2132]
2026-03-05 14:25:48.597 | SUCCESS  | __main__:result_shower:602 - Model response:
It seems like you've sent a long string of ones and zeros. Are you trying to send a message or test something? Let me know how I can assist you!
2026-03-05 14:25:52.755 | INFO     | __main__:result_shower:683 - [Result] total Run Decode 34 times, total_time : 4.14 s
2026-03-05 14:25:52.755 | INFO     | __main__:result_shower:686 - Decode Model average time 121.83 ms/token
2026-03-05 14:25:52.769 | INFO     | __main__:producer:455 - [Decode Stage 0] Process Finished, time: 0.80 s.
2026-03-05 14:25:52.770 | INFO     | __main__:consumer:557 - [Decode Stage 1] Process Finished, time: 0.81 s.
2026-03-05 14:25:52.770 | INFO     | __main__:consumer:557 - [Decode Stage 2] Process Finished, time: 0.81 s.
2026-03-05 14:25:52.770 | INFO     | __main__:consumer:557 - [Decode Stage 3] Process Finished, time: 0.90 s.
2026-03-05 14:25:52.851 | INFO     | __main__:main:729 - MultiChip pipeline parallel run demo finished!
```

### 5.3 性能比较

| Input    | 1chip prefill | 4chips pp prefill  | 4chips tp prefill  |
| -------- | ------------- | ------------------ | ------------------ |
| 16k      | 15.66s        | 5.16s (3.03x)      | 8.09s (1.94x)      |
| 30k      | 41.56s        | 12.23s (3.39x)     | 18.23s (2.28x)     |

## 6 使用场景

### 6.1 流水线并行适用场景

- 当模型层间可以有效分割时
- 内存受限环境中，每块芯片只需要存储部分模型权重
- 层间计算负载相对均衡的情况

### 6.2 张量并行适用场景

- 当单层内部的计算密集操作可以并行化时
- 通信开销相对较低的芯片间连接
- 需要高吞吐量的推理任务

## 7 注意事项

1. 确保正确设置 `HOUMO_TARGET` 环境变量为 `xh2`
2. 根据实际硬件配置调整 `--ndevice` 参数
3. 确保有足够的内存支持多芯片并行推理
4. 在流水线并行模式下，队列大小可能影响性能，可通过调整 `maxsize` 参数优化

## 8 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。