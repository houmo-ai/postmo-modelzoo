# llm_perf 性能测试工具

本代码用于在后摩鸿途系列芯片的设备上测试大语言模型单/多batch模型推理性能。当前支持linux、windows11和android native环境（使用adb运行）。

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
| HmllmInfer.cc | 主文件，实现大语言模型单batch具体推理流程             |
| HmllmInferMultiBatch.cc | 主文件，实现大语言模型多batch具体推理流程，仅支持xh2            |
| convert_embed.py | 将embedding的pt模型转换为bin文件方便工具读取权重             |
| build_linux.sh | Linux平台llm_perf编译脚本             |
| build_ndk.sh | Android平台llm_perf编译脚本           |
| build_win.bat | Win11平台llm_perf编译脚本             |

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
- --batch : 指定模型的batch数，不指定则--batch的值为1。多batch模型必须指定，会校验从模型读取到的batch和用户指定的值是否一致。目前多batch仅支持xh2.
- --no_warm_up : 配置后跳过warm_up流程
- --help, -h : 打印帮助信息
- --config, -c : 使用配置文件启动perf程序


### 1.4 用法说明

#### 1.4.1 指令启动

手动输入参数启动模型测试，仅支持单个模型的loop测试，参考如下：
``` shell
# 禁用warm_up
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --stop 2048 --ndevices 2 --loop 2 --batch 1 --no_warm_up
# 开启warm_up
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --stop 2048 --ndevices 2 --loop 2 --batch 1
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
      "loop" : 2,
      "batch" : 1,
      "no_warm_up" : ""
    },
    {
      "ModelName": "qwen2.5_7b_256_8k_b1_1chip_2cores_v0.5.0",
      "prefill": "qwen2.5/qwen2.5_prefill.hmm",
      "decode": "qwen2.5/qwen2.5_decode.hmm",
      "embedding": "hmquant/quant_embedding.bin",
      "input": 200,
      "stop": 200,
      "ndevices": 1,
      "loop": 2,
      "batch" : 1
    },
    {
      "ModelName": "qwen3_8b_256_8k_b4_1chip_2cores_v0.5.0",
      "prefill": "qwen3/qwen3_prefill.hmm",
      "decode": "qwen3/qwen3_decode.hmm",
      "embedding": "qwen3/hmquant/quant_embedding.bin",
      "input": 300,
      "stop": 400,
      "ndevices": 1,
      "loop": 1,
      "batch" : 4
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
batch : 1
=================================================================
Use Devices 0
[2025-11-11 08:55:16.153] [tid 32419] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2025-11-11 08:55:16.153] [tid 32419] [warning] Empty backend name, use Xh2HalBackend instead.

==============================LLM Perf Progress: 1/2==============================
Decode: 100% |**************************************************| 100/100
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  314.85 ms
Decode Time                   6718.96 ms
Prefill Speed                 317.61 tokens/s
Decode Speed                  14.88 tokens/s
TTFT                          314.85 ms
TPOT                          67.19 ms/token
E2E Latency                   7.03 seconds
E2E TPS                       14.22 tokens/s
Embedding Time                0.13 ms
--------------------------------------------------------------
==================================================================================

==============================LLM Perf Progress: 2/2==============================
Decode: 100% |**************************************************| 100/100
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  315.40 ms
Decode Time                   6714.97 ms
Prefill Speed                 317.06 tokens/s
Decode Speed                  14.89 tokens/s
TTFT                          315.40 ms
TPOT                          67.15 ms/token
E2E Latency                   7.03 seconds
E2E TPS                       14.22 tokens/s
Embedding Time                0.42 ms
--------------------------------------------------------------
==================================================================================
============================== LLM Perf Avarage Information ==============================

-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  315.12 ms
Decode Time                   6716.97 ms
Prefill Speed                 317.34 tokens/s
Decode Speed                  14.89 tokens/s
TTFT                          315.12 ms
TPOT                          67.17 ms/token
E2E Latency                   7.03 seconds
E2E TPS                       14.22 tokens/s
Embedding Time                0.28 ms
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
batch : 1
=================================================================
Use Devices 0
[2025-11-11 08:55:35.170] [tid 32419] [warning] Empty backend name, use Xh2HalBackend instead.

==============================LLM Perf Progress: 1/2==============================
Decode: 100% |**************************************************| 200/200
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  327.89 ms
Decode Time                   11784.94 ms
Prefill Speed                 609.95 tokens/s
Decode Speed                  16.97 tokens/s
TTFT                          327.89 ms
TPOT                          58.92 ms/token
E2E Latency                   12.11 seconds
E2E TPS                       16.51 tokens/s
Embedding Time                0.20 ms
--------------------------------------------------------------
==================================================================================

==============================LLM Perf Progress: 2/2==============================
Decode: 100% |**************************************************| 200/200
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  327.97 ms
Decode Time                   14180.40 ms
Prefill Speed                 609.81 tokens/s
Decode Speed                  14.10 tokens/s
TTFT                          327.97 ms
TPOT                          70.90 ms/token
E2E Latency                   14.51 seconds
E2E TPS                       13.79 tokens/s
Embedding Time                0.54 ms
--------------------------------------------------------------
==================================================================================
============================== LLM Perf Avarage Information ==============================

-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  327.93 ms
Decode Time                   12982.67 ms
Prefill Speed                 609.88 tokens/s
Decode Speed                  15.41 tokens/s
TTFT                          327.93 ms
TPOT                          64.91 ms/token
E2E Latency                   13.31 seconds
E2E TPS                       15.03 tokens/s
Embedding Time                0.37 ms
--------------------------------------------------------------
==========================================================================================
############################################# End of Task 2, All Task:4,  ModelName:"qwen2.5_7b_256_8k_b1_1chip_2cores_v0.5.0".#############################################


#############################################Start of Task 3, All Task:4, ModelName:"qwen3_8b_256_8k_b4_1chip_2cores_v0.5.0".#############################################
========================= Perf Settings =========================
prefill path : "qwen3/qwen3_prefill.hmm"
decode path : "qwen3/qwen3_decode.hmm"
embedding path : "qwen3/hmquant/quant_embedding.bin"
input token len : 300
stop token len : 400
ndevices : 1
loop : 1
batch : 4
=================================================================
Multi batch Use Devices 0
[2025-11-11 08:56:04.428] [tid 32419] [warning] Empty backend name, use Xh2HalBackend instead.

==============================LLM Perf Progress: 1/1==============================
Decode: 100% |**************************************************| 400/400
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  3214.78 ms
Decode Time                   58139.14 ms
Prefill Speed                 373.28 tokens/s
Decode Speed                  27.52 tokens/s
TTFT                          3214.78 ms
TPOT                          36.34 ms/token
E2E Latency                   61.38 seconds
E2E TPS                       26.07 tokens/s
Embedding Time                2.72 ms
--------------------------------------------------------------
==================================================================================
============================== LLM Perf Avarage Information ==============================

-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  3214.78 ms
Decode Time                   58139.14 ms
Prefill Speed                 373.28 tokens/s
Decode Speed                  27.52 tokens/s
TTFT                          3214.78 ms
TPOT                          36.34 ms/token
E2E Latency                   61.38 seconds
E2E TPS                       26.07 tokens/s
Embedding Time                2.72 ms
--------------------------------------------------------------
==========================================================================================
############################################# End of Task 3, All Task:4,  ModelName:"qwen3_8b_256_8k_b4_1chip_2cores_v0.5.0".#############################################


#############################################Start of Task 4, All Task:4, ModelName:"qwen3_14b_256_8k_b1_1chip_2cores_v0.5.0".#############################################
========================= Perf Settings =========================
prefill path : "qwen3-14b/qwen3_prefill.hmm"
decode path : "qwen3-14b/qwen3_decode.hmm"
embedding path : "qwen3-14b/hmquant/quant_embedding.bin"
input token len : 400
stop token len : 400
ndevices : 1
loop : 1
batch : 1
=================================================================
Use Devices 0
[2025-11-11 08:57:09.311] [tid 32419] [warning] Empty backend name, use Xh2HalBackend instead.

==============================LLM Perf Progress: 1/1==============================
Decode: 100% |**************************************************| 400/400
-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  1176.22 ms
Decode Time                   42903.35 ms
Prefill Speed                 340.07 tokens/s
Decode Speed                  9.32 tokens/s
TTFT                          1176.22 ms
TPOT                          107.26 ms/token
E2E Latency                   44.08 seconds
E2E TPS                       9.07 tokens/s
Embedding Time                0.97 ms
--------------------------------------------------------------
==================================================================================
============================== LLM Perf Avarage Information ==============================

-------------------  Performance Summary  --------------------
Metric                        Value
--------------------------------------------------------------
Prefill Time                  1176.22 ms
Decode Time                   42903.35 ms
Prefill Speed                 340.07 tokens/s
Decode Speed                  9.32 tokens/s
TTFT                          1176.22 ms
TPOT                          107.26 ms/token
E2E Latency                   44.08 seconds
E2E TPS                       9.07 tokens/s
Embedding Time                0.97 ms
--------------------------------------------------------------
==========================================================================================
############################################# End of Task 4, All Task:4,  ModelName:"qwen3_14b_256_8k_b1_1chip_2cores_v0.5.0".#############################################
```

## 3.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
