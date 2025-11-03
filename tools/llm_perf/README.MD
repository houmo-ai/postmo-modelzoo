# llm_perf 性能测试工具

本代码用于在后摩鸿途系列芯片的设备上测试大语言模型单batch模型推理性能。当前支持linux和android native环境（使用adb运行）。

## 目录

- [文件说明](#1文件说明)
- [流程说明](#2流程说明)
- [快速开始](#3快速开始)
  - [环境准备](#31-环境准备)
  - [一键运行](#32-一键运行)
- [免责声明](#4免责声明)

## 1.工具说明

### 1.1 文件说明

| 文件名        | 说明                                     |
| -------------- | ---------------------------------------- |
| llm_perf.cc    | 主入口，拉起大语言模型推理流程             |
| HmEmbedding.cc | embedding类实现             |
| HmllmInfer.cc | 主文件，实现大语言模型具体推理流程             |
| convert_embed.py | 将embedding的pt模型转换为bin文件方便工具读取权重             |

### 1.2 流程说明

程序按以下流程进行：

1. 解析输入参数
2. 创建推理Module
3. 准备输入数据和输入输出内存
4. 根据设定的次数执行输入数据->推理->获取结果循环，统计单次推理的性能和平均性能

### 1.3 参数说明

- --prefill : 指定prefill模型路径
- --decode : 指定decode模型路径
- --embedding : 指定embedding模型路径
- --input : 指定输入token数量
- --stop : 指定输出token数量
- --ndevices : 指定模型使用的device数量
- --loop : 指定循环执行单个模型次数
- --help, -h : 打印帮助信息
- --config, -c : 使用配置文件启动perf程序


### 1.4 用法说明

#### 1.4.1 指令启动

手动输入参数启动模型测试，仅支持单个模型的loop测试，参考如下：
``` shell
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --stop 2048 --ndevices 2 --loop 2
```

#### 1.4.2 配置文件启动

用户可以使用如下任一指令指定配置文件：
``` shell
./llm_perf -c config.json 
./llm_perf --config config.json
```
配置文件启动等效于多次指令启动，配置文件的参考模板如下：
``` json
{
  "Streams":[
    {
      "ModelName" : "deepseek_8b_256_4k_b1_1chip_2cores_v0.5.0",
      "prefill" : "deepseek/deepseek__prefill.hmm",
      "decode" : "deepseek/deepseek_decode.hmm",
      "embedding" : "hmquant/quant_embedding.bin",
      "input" : 100,
      "stop" : 100,
      "ndevices" : 1,
      "loop" : 2
    },
    {
      "ModelName": "qwen2.5_7b_256_8k_b1_1chip_2cores_v0.5.0",
      "prefill": "qwen2.5/qwen2.5_prefill.hmm",
      "decode": "qwen2.5/qwen2.5_decode.hmm",
      "embedding": "hmquant/quant_embedding.bin",
      "input": 200,
      "stop": 200,
      "ndevices": 1,
      "loop": 2
    },
    {
      "ModelName": "qwen3_8b_256_2k_b1_1chip_2cores_v0.5.0",
      "prefill": "qwen3/qwen3_prefill.hmm",
      "decode": "qwen3/qwen3_decode.hmm",
      "embedding": "qwen3/hmquant/quant_embedding.bin",
      "input": 300,
      "stop": 400,
      "ndevices": 1,
      "loop": 1
    },
    {
      "ModelName": "qwen3_14b_256_8k_b1_1chip_2cores_v0.5.0",
      "prefill": "qwen3-14b/qwen3_prefill.hmm",
      "decode": "qwen3-14b/qwen3_decode.hmm",
      "embedding": "qwen3-14b/hmquant/quant_embedding.bin",
      "input": 400,
      "stop": 400,
      "ndevices": 1,
      "loop": 1
    }
  ]
}
```
上述模板等效于同时执行四次指令启动，用户可以根据实际需求自行配置。

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

```bash
cd tools/llm_perf/
./build_linux.sh
```

编译生成llm_perf可执行文件在tools/bin目录下。

如果是编译android版本请执行：

```bash
./build_ndk.sh
```

编译生成llm_perf可执行文件在tools/android目录下。

### 2.2 一键运行
``` shell
#单次启动perf
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --stop 2048 --ndevices 2 --loop 2
#多次启动perf
./llm_perf -c config.json 
./llm_perf --config config.json
```

结果：

```bash
#############################################Start of Task 1, All Task:4, ModelName:"deepseek_8b_256_4k_b1_1chip_2cores_v0.5.0".#############################################
========================= Perf Settings =========================
prefill path : "deepseek/deepseek_prefill.hmm"
decode path : "deepseek/deepseek_decode.hmm"
embedding path : "deepseek/hmquant/quant_embedding.bin"
input token len : 100
stop token len : 100
ndevices : 1
loop : 2
=================================================================
Use Devices 0
[2025-11-03 10:41:46.683] [tid 1461] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2025-11-03 10:41:46.683] [tid 1461] [warning] Empty backend name, use Xh2HalBackend instead.

==============================LLM-Perf Progress: 1/2==============================
Decode: 100% |**************************************************| 100/100
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  319.16 ms
Decode Time                   7248.38 ms
Prefill Speed                 313.32 tokens/s
Decode Speed                  13.80 tokens/s
TTFT                          319.16 ms
TPOT                          72.48 ms/token
E2E Latency                   7.57 seconds
E2E TPS                       13.21 tokens/s
Embedding Time                0.16 ms
--------------------------------------------------------------
==================================================================================

==============================LLM-Perf Progress: 2/2==============================
Decode: 100% |**************************************************| 100/100
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  319.65 ms
Decode Time                   7247.73 ms
Prefill Speed                 312.84 tokens/s
Decode Speed                  13.80 tokens/s
TTFT                          319.65 ms
TPOT                          72.48 ms/token
E2E Latency                   7.57 seconds
E2E TPS                       13.21 tokens/s
Embedding Time                0.26 ms
--------------------------------------------------------------
==================================================================================
============================== LLM-Perf Avarage Information ==============================

-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  319.41 ms
Decode Time                   7248.05 ms
Prefill Speed                 313.08 tokens/s
Decode Speed                  13.80 tokens/s
TTFT                          319.41 ms
TPOT                          72.48 ms/token
E2E Latency                   7.57 seconds
E2E TPS                       13.21 tokens/s
Embedding Time                0.21 ms
--------------------------------------------------------------
==========================================================================================
############################################# End of Task 1, All Task:4,  ModelName:"deepseek_8b_256_4k_b1_1chip_2cores_v0.5.0".#############################################


#############################################Start of Task 2, All Task:4, ModelName:"qwen2.5_7b_256_8k_b1_1chip_2cores_v0.5.0".#############################################
========================= Perf Settings =========================
prefill path : "qwen2.5/qwen2.5_prefill.hmm"
decode path : "qwen2.5/qwen2.5_decode.hmm"
embedding path : "qwen2.5/hmquant/quant_embedding.bin"
input token len : 200
stop token len : 200
ndevices : 1
loop : 2
=================================================================
Use Devices 0
[2025-11-03 10:42:07.086] [tid 1461] [warning] Empty backend name, use Xh2HalBackend instead.

==============================LLM-Perf Progress: 1/2==============================
Decode: 100% |**************************************************| 200/200
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  332.12 ms
Decode Time                   12840.67 ms
Prefill Speed                 602.20 tokens/s
Decode Speed                  15.58 tokens/s
TTFT                          332.12 ms
TPOT                          64.20 ms/token
E2E Latency                   13.17 seconds
E2E TPS                       15.18 tokens/s
Embedding Time                0.31 ms
--------------------------------------------------------------
==================================================================================

==============================LLM-Perf Progress: 2/2==============================
Decode: 100% |**************************************************| 200/200
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  332.83 ms
Decode Time                   12847.48 ms
Prefill Speed                 600.90 tokens/s
Decode Speed                  15.57 tokens/s
TTFT                          332.83 ms
TPOT                          64.24 ms/token
E2E Latency                   13.18 seconds
E2E TPS                       15.17 tokens/s
Embedding Time                0.39 ms
--------------------------------------------------------------
==================================================================================
============================== LLM-Perf Avarage Information ==============================

-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  332.47 ms
Decode Time                   12844.07 ms
Prefill Speed                 601.55 tokens/s
Decode Speed                  15.57 tokens/s
TTFT                          332.47 ms
TPOT                          64.22 ms/token
E2E Latency                   13.18 seconds
E2E TPS                       15.18 tokens/s
Embedding Time                0.35 ms
--------------------------------------------------------------
==========================================================================================
############################################# End of Task 2, All Task:4,  ModelName:"qwen2.5_7b_256_8k_b1_1chip_2cores_v0.5.0".#############################################


#############################################Start of Task 3, All Task:4, ModelName:"qwen3_8b_256_2k_b1_1chip_2cores_v0.5.0".#############################################
========================= Perf Settings =========================
prefill path : "qwen3/qwen3_prefill.hmm"
decode path : "qwen3/qwen3_decode.hmm"
embedding path : "qwen3/hmquant/quant_embedding.bin"
input token len : 300
stop token len : 400
ndevices : 1
loop : 1
=================================================================
Use Devices 0
[2025-11-03 10:42:36.278] [tid 1461] [warning] Empty backend name, use Xh2HalBackend instead.

==============================LLM-Perf Progress: 1/1==============================
Decode: 100% |**************************************************| 400/400
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  471.97 ms
Decode Time                   27159.36 ms
Prefill Speed                 635.64 tokens/s
Decode Speed                  14.73 tokens/s
TTFT                          471.97 ms
TPOT                          67.90 ms/token
E2E Latency                   27.63 seconds
E2E TPS                       14.48 tokens/s
Embedding Time                0.60 ms
--------------------------------------------------------------
==================================================================================
============================== LLM-Perf Avarage Information ==============================

-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  471.97 ms
Decode Time                   27159.36 ms
Prefill Speed                 635.64 tokens/s
Decode Speed                  14.73 tokens/s
TTFT                          471.97 ms
TPOT                          67.90 ms/token
E2E Latency                   27.63 seconds
E2E TPS                       14.48 tokens/s
Embedding Time                0.60 ms
--------------------------------------------------------------
==========================================================================================
############################################# End of Task 3, All Task:4,  ModelName:"qwen3_8b_256_2k_b1_1chip_2cores_v0.5.0".#############################################


#############################################Start of Task 4, All Task:4, ModelName:"qwen3_14b_256_8k_b1_1chip_2cores_v0.5.0".#############################################
========================= Perf Settings =========================
prefill path : "qwen3-14b/qwen3_prefill.hmm"
decode path : "qwen3-14b/qwen3_decode.hmm"
embedding path : "qwen3-14b/hmquant/quant_embedding.bin"
input token len : 400
stop token len : 400
ndevices : 1
loop : 1
=================================================================
Use Devices 0
[2025-11-03 10:43:06.820] [tid 1461] [warning] Empty backend name, use Xh2HalBackend instead.

==============================LLM-Perf Progress: 1/1==============================
Decode: 100% |**************************************************| 400/400
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  1183.00 ms
Decode Time                   44927.45 ms
Prefill Speed                 338.12 tokens/s
Decode Speed                  8.90 tokens/s
TTFT                          1183.00 ms
TPOT                          112.32 ms/token
E2E Latency                   46.11 seconds
E2E TPS                       8.67 tokens/s
Embedding Time                0.86 ms
--------------------------------------------------------------
==================================================================================
============================== LLM-Perf Avarage Information ==============================

-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  1183.00 ms
Decode Time                   44927.45 ms
Prefill Speed                 338.12 tokens/s
Decode Speed                  8.90 tokens/s
TTFT                          1183.00 ms
TPOT                          112.32 ms/token
E2E Latency                   46.11 seconds
E2E TPS                       8.67 tokens/s
Embedding Time                0.86 ms
--------------------------------------------------------------
==========================================================================================
############################################# End of Task 4, All Task:4,  ModelName:"qwen3_14b_256_8k_b1_1chip_2cores_v0.5.0".#############################################
```

## 3.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
