# llm_perf 性能测试工具

本代码用于在后摩鸿途系列芯片（XH2）的设备上测试llm模型和vllm模型的单/多batch模型推理性能。当前支持linux、windows11和android native环境（使用adb运行）。

## 目录

- [工具说明](#1工具说明)
  - [文件及功能支持度说明](#11-文件及功能支持度说明)
  - [流程说明](#12-流程说明)
  - [参数说明](#13-参数说明)
  - [用法说明](#14-用法说明)
- [快速开始](#2快速开始)
  - [环境准备](#21-环境准备)
  - [一键运行](#22-一键运行)
- [免责声明](#3免责声明)

## 1.工具说明

### 1.1 文件及功能支持度说明

文件目录如下：
| 文件名称                | 说明                                                |
| ----------------------- | --------------------------------------------------- |
| llm_perf.cc             | 主入口，拉起大语言模型推理流程                      |
| HmEmbedding.cc          | embedding类实现                                     |
| HmllmInfer.cc           | 主文件，实现llm模型单batch具体推理流程              |
| HmvllmInfer.cc          | 主文件，实现vllm模型单batch具体推理流程             |
| HmllmInferMultiBatch.cc | 主文件，实现llm多batch具体推理流程，仅支持xh2       |
| convert_embed.py        | 将embedding的pt模型转换为bin文件方便工具读取权重    |
| build_linux.sh          | Linux平台llm_perf编译脚本                           |
| build_ndk.sh            | Android平台llm_perf编译脚本                         |
| build_win.bat           | Win11平台llm_perf编译脚本                           |

功能支持度说明：
  - 具体支持的模型参考models目录下llm和vllm模型支持的平台。
  - 支持的芯片型号为：xh2。
  - 支持的大道软件版本要求如下：xh2≥v0.6.0 。

### 1.2 流程说明

程序按以下流程进行：

1. 解析用户设置的perf参数
2. 依据参数判断模型类别，创建模型推理Module
3. 准备模型输入输出数据和输入输出内存
4. 根据设定的次数执行输入数据->推理->获取结果循环，统计单次推理的性能和平均性能
5. 输出性能评估结果指标参考如下表：

| 指标                                    | 说明                                                                |
| --------------------------------------- | ------------------------------------------------------------------- |
| Prefill Time                            | Prefill模型纯推理耗时，单位（ms）                                   |
| Decode Time                             | Decode模型纯推理耗时，单位（ms）                                    |
| Vision Time                             | Vision模型纯推理耗时，单位（ms）                                    |
| Prefill Speed                           | Prefill模型推理速度，单位（tokens/s）                               |
| Decode Speed                            | Decode模型推理速度，单位（tokens/s）                                |
| TTFT (Time to First Token)              | 输出第一个token的耗时，包含tokenizer+embedding+infer时间，单位（ms）|
| TPOT (Time Per Output Token)            | Decode模型每秒输出的token数，单位（tokens/s）                       |
| E2E Latency (End-to-End Latency)        | 端到端总时间，单位（s）                                             |
| E2E TPS (End-to-End Tokens Per Second)  | 端到端每秒输出的token数，单位（tokens/s）                           |
| Embedding Time                          | Embedding操作总耗时，单位（ms）                                     |

### 1.3 参数说明
1. 命令行llm_perf以指定参数启动
- --prefill     : 指定prefill模型路径
- --decode      : 指定decode模型路径
- --visual      : 指定visual模型路径，仅vllm需要指定，不指定时默认模型为llm模型。
- --embedding   : 指定embedding模型路径
- --input       : 指定prefill阶段输入token数量
- --stop        : 指定docode阶段输出多少token后停止
- --ndevices    : 指定模型使用的device数量，不指定默认为1
- --loop        : 指定循环执行单个模型次数，不指定默认为1
- --batch       : 指定模型的batch数，不指定则--batch的值为1。多batch模型必须指定，会校验从模型读取到的batch和用户指定的值是否一致。目前多batch仅支持xh2.
- --no_warm_up  : 默认执行warm_up，配置后会跳过warm_up流程
- --help, -h    : 打印帮助信息
2. 命令行llm_perf以配置文件启动
- --config, -c  : 使用配置文件启动perf程序，需要提前准备好配置文件，配置文件模板参考1.4节。


### 1.4 用法说明

#### 1.4.1 embedding模型准换
提供了convert_embed.py对quant_embedding.pt进行准换，转换结果为quant_embedding.bin，可以方便llm_perf工具embedding时读取权重。
使用方式如下：python convert_embed.py --path <embedding_weight_path> --type <llm/vllm>  。

#### 1.4.2 指令启动

手动输入参数启动模型测试，仅支持单个模型的loop测试，参考如下：
``` shell
# 禁用warm_up
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --stop 2048 --ndevices 2 --loop 2 --batch 1 --no_warm_up
# 开启warm_up
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --stop 2048 --ndevices 2 --loop 2 --batch 1
```

#### 1.4.3 配置文件启动

用户可以使用如下任一指令指定配置文件：
``` shell
./llm_perf -c config.json 
./llm_perf --config config.json
```
配置文件启动等效于多次指令启动，配置文件的参考模板perf_config.json，详细内容示例如下：
``` json
{
  "Streams":[
    {
      "ModelName" : "deepseek_8b_256_4k_b1_1chip_2cores_v0.5.0",
      "prefill" : "deepseek/deepseek_prefill.hmm",
      "decode" : "deepseek/deepseek_decode.hmm",
      "embedding" : "deepseek/hmquant/quant_embedding.bin",
      "input" : 256,
      "stop" : 100,
      "ndevices" : 1,
      "loop" : 2,
      "batch" : 1,
      "no_warm_up" : ""
    },
    {
      "ModelName": "qwen2.5_7b_256_8k_b1_1chip_2cores_v0.5.0",
      "prefill": "qwen2.5/qwen2.5_prefill.hmm",
      "decode": "qwen2.5/qwen2.5_decode.hmm",
      "embedding": "qwen2.5/hmquant/quant_embedding.bin",
      "input": 256,
      "stop": 100,
      "ndevices": 1,
      "loop": 2,
      "batch" : 1
    },
    {
      "ModelName": "qwen3_8b_256_8k_b4_1chip_2cores_v0.5.0",
      "prefill": "qwen3/qwen3_prefill.hmm",
      "decode": "qwen3/qwen3_decode.hmm",
      "embedding": "qwen3/hmquant/quant_embedding.bin",
      "input": 256,
      "stop": 100,
      "ndevices": 1,
      "loop": 1,
      "batch" : 4
    },
    {
      "ModelName": "qwen3_14b_256_8k_b1_1chip_2cores_v0.5.0",
      "prefill": "qwen3-14b/qwen3_prefill.hmm",
      "decode": "qwen3-14b/qwen3_decode.hmm",
      "embedding": "qwen3-14b/hmquant/quant_embedding.bin",
      "input": 256,
      "stop": 100,
      "ndevices": 1,
      "loop": 1
    },
    {
      "ModelName": "qwen3_30b-a3b",
      "prefill": "qwen3-30b-a3b/qwen3_prefill.hmm",
      "decode": "qwen3-30b-a3b/qwen3_decode.hmm",
      "embedding": "qwen3-30b-a3b/hmquant/quant_embedding.bin",
      "input": 256,
      "stop": 100,
      "ndevices": 1,
      "loop": 1
    },
    {
      "ModelName": "qwen2.5-vl",
      "prefill": "qwen2.5-vl/qwen2.5-vl_prefill.hmm",
      "decode": "qwen2.5-vl/qwen2.5-vl_decode.hmm",
      "visual": "qwen2.5-vl/qwen2.5-vl_visual.hmm",
      "embedding": "qwen2.5-vl/hmquant/quant_embedding.bin",
      "input": 256,
      "stop": 100,
      "ndevices": 1,
      "loop": 1
    },
    {
      "ModelName": "qwen3-vl",
      "prefill": "qwen3-vl/qwen3-vl_prefill.hmm",
      "decode": "qwen3-vl/qwen3-vl_decode.hmm",
      "visual": "qwen3-vl/qwen3-vl_visual.hmm",
      "embedding": "qwen3-vl/hmquant/quant_embedding.bin",
      "input": 256,
      "stop": 100,
      "ndevices": 1,
      "loop": 1
    }
  ]
}
```
上述模板等效于同时执行多次指令启动，用户可以根据实际需求自行配置。

## 2.快速开始

### 2.1 环境准备

进入houmo-examples根目录，先检查 `env.sh` 里的环境变量，并且执行以下命令：

```bash
source env.sh
```

如果需要使用其他平台编译链进行交叉编译需要自行下载交叉编译链并设置TCIM_RUNTIME_PATH到目标平台runtime库目录, 以android平台为例：

下载官方NDK[https://developer.android.google.cn/ndk/downloads/index.html?hl=ro]解压到toolchains目录下（也可通过NDK_PATH自由指定路径）。

本例在android-ndk-r28c版本上验证通过，用户的环境如果不一致请自行修改适配。

设置完成后进入程序目录编译：

如果是Linux版本请执行：

```bash
cd tools/llm_perf/
./build_linux.sh
```

编译生成的llm_perf可执行文件在tools/bin目录下。

如果是编译android版本请执行：

```bash
#需要修改脚本第11行的Android版本tcim库在当前系统的路径
./build_ndk.sh
```

编译生成的llm_perf可执行文件在tools/android目录下。

如果是windows11版本请执行:

``` bat
rem 需要结合用户目录更改三四行的环境变量实际路径
./build_win.bat
```

编译生成的llm_perf可执行文件在tools/bin目录下，在win11运行llm_perf.exe需要注意以下事项：
1. 设置环境变量: set HOUMO_TARGET=xh2 和 set TCIM_BACKEND=Xh2HalBackend。
2. C++样例依赖libhal_xh2a.dll和tcim_runtime_lite.dll可以设置环境变量或和llm_perf.exe放在同一目录。

### 2.2 一键运行
``` shell
#单次启动perf
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --stop 2048 --ndevices 2 --loop 2 --batch 1
#多次启动perf
./llm_perf -c config.json 
./llm_perf --config config.json
```

结果示例：
```bash
#############################################Start of Task 1, All Task:3, ModelName:"qwen3_8b_256_8k_b4_1chip_2cores_v0.5.0".#############################################
========================= Perf Settings =========================
prefill path : qwen3/qwen3_prefill.hmm
decode path : qwen3/qwen3_decode.hmm
embedding path : qwen3/hmquant/quant_embedding.bin
input token len : 256
stop token len : 100
ndevices : 1
loop : 1
batch : 4
warm_up : enable
=================================================================
Multi batch Use Devices 0
[2025-12-01 15:19:43.720] [tid 4735] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2025-12-01 15:19:43.720] [tid 4735] [warning] Empty backend name, use Xh2HalBackend instead.

==============================(v)LLM Perf WarmUp: input 256, output 256==============================
Decode: 100% |**************************************************| 256/256
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                                      1595.45 ms
Decode Time                                       35946.62 ms
Prefill Speed                                     641.82 tokens/s
Decode Speed                                      28.49 tokens/s
TTFT (Time to First Token)                        1609.99 ms
TPOT (Time Per Output Token)                      35.10 ms/token
E2E Latency (End-to-End Latency)                  39.00 seconds
E2E TPS (End-to-End Tokens Per Second)            26.28 tokens/s
Embedding Time                                    2.05 ms
--------------------------------------------------------------
==================================================================================

==============================(v)LLM Perf Loop Progress: 1/1==============================
Decode: 100% |**************************************************| 100/100
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                                      1595.81 ms
Decode Time                                       14018.04 ms
Prefill Speed                                     641.68 tokens/s
Decode Speed                                      28.53 tokens/s
TTFT (Time to First Token)                        1610.51 ms
TPOT (Time Per Output Token)                      35.05 ms/token
E2E Latency (End-to-End Latency)                  16.18 seconds
E2E TPS (End-to-End Tokens Per Second)            24.78 tokens/s
Embedding Time                                    0.75 ms
--------------------------------------------------------------
==================================================================================
============================== (v)LLM Perf Avarage Information ==============================

-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                                      1595.81 ms
Decode Time                                       14018.04 ms
Prefill Speed                                     641.68 tokens/s
Decode Speed                                      28.53 tokens/s
TTFT (Time to First Token)                        1610.51 ms
TPOT (Time Per Output Token)                      35.05 ms/token
E2E Latency (End-to-End Latency)                  16.18 seconds
E2E TPS (End-to-End Tokens Per Second)            24.78 tokens/s
Embedding Time                                    0.75 ms
--------------------------------------------------------------
==========================================================================================
############################################# End of Task 1, All Task:3,  ModelName:"qwen3_8b_256_8k_b4_1chip_2cores_v0.5.0".#############################################


#############################################Start of Task 2, All Task:3, ModelName:"qwen2.5-vl".#############################################
========================= Perf Settings =========================
prefill path : qwen2.5-vl/qwen2.5-vl_prefill.hmm
decode path : qwen2.5-vl/qwen2.5-vl_decode.hmm
visual path : qwen2.5-vl/qwen2.5-vl_visual.hmm
embedding path : qwen2.5-vl/hmquant/quant_embedding.bin
input token len : 256
stop token len : 100
ndevices : 1
loop : 1
batch : 1
warm_up : enable
=================================================================
Use Devices 0
[2025-12-01 15:20:44.887] [tid 4735] [warning] Empty backend name, use Xh2HalBackend instead.

==============================(v)LLM Perf WarmUp: input 256, output 256==============================
Decode: 100% | **************************************************| 256/256
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                                      372.78 ms
Decode Time                                       15165.83 ms
Vision Time                                       357.58 ms
Prefill Speed                                     686.73 tokens/s
Decode Speed                                      16.88 tokens/s
TTFT (Time to First Token)                        376.93 ms
TPOT (Time Per Output Token)                      59.24 ms/token
E2E Latency (End-to-End Latency)                  16.18 seconds
E2E TPS (End-to-End Tokens Per Second)            15.88 tokens/s
Embedding Time                                    0.73 ms
--------------------------------------------------------------
==================================================================================

==============================(v)LLM Perf Loop Progress: 1/1==============================
Decode: 100% | **************************************************| 100/100
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                                      372.37 ms
Decode Time                                       5925.43 ms
Vision Time                                       358.21 ms
Prefill Speed                                     687.49 tokens/s
Decode Speed                                      16.88 tokens/s
TTFT (Time to First Token)                        376.42 ms
TPOT (Time Per Output Token)                      59.25 ms/token
E2E Latency (End-to-End Latency)                  6.55 seconds
E2E TPS (End-to-End Tokens Per Second)            15.41 tokens/s
Embedding Time                                    0.66 ms
--------------------------------------------------------------
==================================================================================
============================== (v)LLM Perf Avarage Information ==============================

-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                                      372.37 ms
Decode Time                                       5925.43 ms
Prefill Speed                                     687.49 tokens/s
Decode Speed                                      16.88 tokens/s
TTFT (Time to First Token)                        376.42 ms
TPOT (Time Per Output Token)                      59.25 ms/token
E2E Latency (End-to-End Latency)                  6.55 seconds
E2E TPS (End-to-End Tokens Per Second)            15.41 tokens/s
Embedding Time                                    0.66 ms
--------------------------------------------------------------
==========================================================================================
############################################# End of Task 2, All Task:3,  ModelName:"qwen2.5-vl".#############################################


#############################################Start of Task 3, All Task:3, ModelName:"qwen3-vl".#############################################
========================= Perf Settings =========================
prefill path : qwen3-vl/qwen3-vl_prefill.hmm
decode path : qwen3-vl/qwen3-vl_decode.hmm
visual path : qwen3-vl/qwen3-vl_visual.hmm
embedding path : qwen3-vl/hmquant/quant_embedding.bin
input token len : 256
stop token len : 100
ndevices : 1
loop : 1
batch : 1
warm_up : enable
=================================================================
Use Devices 0
[2025-12-01 15:21:11.118] [tid 4735] [warning] Empty backend name, use Xh2HalBackend instead.

==============================(v)LLM Perf WarmUp: input 256, output 256==============================
Decode: 100% | **************************************************| 256/256
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                                      357.66 ms
Decode Time                                       18310.78 ms
Vision Time                                       156.22 ms
Prefill Speed                                     715.76 tokens/s
Decode Speed                                      13.98 tokens/s
TTFT (Time to First Token)                        365.99 ms
TPOT (Time Per Output Token)                      71.53 ms/token
E2E Latency (End-to-End Latency)                  19.46 seconds
E2E TPS (End-to-End Tokens Per Second)            13.21 tokens/s
Embedding Time                                    0.79 ms
--------------------------------------------------------------
==================================================================================

==============================(v)LLM Perf Loop Progress: 1/1==============================
Decode: 100% | **************************************************| 100/100
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                                      358.03 ms
Decode Time                                       7156.79 ms
Vision Time                                       156.32 ms
Prefill Speed                                     715.03 tokens/s
Decode Speed                                      13.97 tokens/s
TTFT (Time to First Token)                        366.38 ms
TPOT (Time Per Output Token)                      71.57 ms/token
E2E Latency (End-to-End Latency)                  7.84 seconds
E2E TPS (End-to-End Tokens Per Second)            12.89 tokens/s
Embedding Time                                    0.71 ms
--------------------------------------------------------------
==================================================================================
============================== (v)LLM Perf Avarage Information ==============================

-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                                      358.03 ms
Decode Time                                       7156.79 ms
Prefill Speed                                     715.03 tokens/s
Decode Speed                                      13.97 tokens/s
TTFT (Time to First Token)                        366.38 ms
TPOT (Time Per Output Token)                      71.57 ms/token
E2E Latency (End-to-End Latency)                  7.84 seconds
E2E TPS (End-to-End Tokens Per Second)            12.89 tokens/s
Embedding Time                                    0.71 ms
--------------------------------------------------------------
==========================================================================================
############################################# End of Task 3, All Task:3,  ModelName:"qwen3-vl".#############################################
```

## 3.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
