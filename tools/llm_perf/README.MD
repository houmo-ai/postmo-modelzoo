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
| device_monitor          | 监控设备温度、功率、IPU频率的类实现                 |
| perf_tracker            | 性能采集类功能实现                                  |
| perf_dumper             | 性能数据落盘及性能数据可视化类功能实现              |
| host_monitor            | 主机侧内存监控类实现                                |
| build_linux.sh          | Linux平台llm_perf编译脚本                           |
| build_ndk.sh            | Android平台llm_perf编译脚本                         |
| build_win.bat           | Win11平台llm_perf编译脚本                           |

功能支持度说明：
  - 具体支持的模型参考models目录下llm和vllm模型支持的平台。
  - 支持的芯片型号为：xh2。
  - 支持的大道软件版本要求如下：xh2≥v0.6.0 。
  - 支持了device的温度、功率、IPU频率监控，记录保存在./device_logs/device_x.log。
  - 温度监控带告警功能，当温度大于80℃时，程序会打印警告信息，温度大于100℃时，程序会强制退出。
  - 主机侧的内存监控仅支持linux平台。
  - **修改配置文件启动方式为yaml文件，对之前版本的json文件启动兼容。**

### 1.2 流程说明

程序按以下流程进行：

1. 解析用户设置的perf参数
2. 依据参数判断模型类别，创建模型推理Module
3. 准备模型输入输出数据和输入输出内存
4. 根据设定的次数执行输入数据->推理->获取结果循环，统计单次推理的性能和平均性能
5. 输出性能评估结果指标参考如下多表：
- 表一：Prefill/Decode模型指标
  | Prefill/Decode Stage 指标               | 说明                                                                |
  | --------------------------------------- | ------------------------------------------------------------------- |
  | Prefill/Decode Total Time               | Prefill/Decode阶段耗时，单位（ms）                                  |
  | Prefill/Decode Total Speed              | Prefill/Decode阶段速度，单位（tokens/s）                            |
  | Tokenization Time                       | Tokenization耗时，单位（ms）                                        |
  | Embedding Time                          | Embedding操作总耗时，单位（ms）                                     |
  | API SetInput Time                       | tcim runtime setinput API耗时，单位（ms）                           |
  | API Inference Time                      | tcim runtime run+sync API耗时，单位（ms）                           |
  | API Inference Speed                     | tcim runtime run+sync API速度，单位（tokens/s）                     |
  | API GetOutput Time                      | tcim runtime GetOutput API耗时，单位（ms）                          |

- 表二：Vision模型指标
  | Vision Stage 指标                       | 说明                                                                |
  | --------------------------------------- | ------------------------------------------------------------------- |
  | Vision Total Time                       | Vision阶段耗时，单位（ms）                                          |
  | Vision Total Speed                      | Vision阶段速度，单位（tokens/s）                                    |
  | Preprocessing Time                      | Preprocessing耗时，单位（ms）                                       |
  | API SetInput Time                       | tcim runtime setinput API耗时，单位（ms）                           |
  | API Inference Time                      | tcim runtime run+sync API耗时，单位（ms）                           |
  | API Inference Speed                     | tcim runtime run+sync API速度，单位（tokens/s）                     |
  | API GetOutput Time                      | tcim runtime GetOutput API耗时，单位（ms）                          |

- 表三：总体性能指标
  | Overall 指标                            | 说明                                                                |
  | --------------------------------------- | ------------------------------------------------------------------- |
  | TTFT (Time to First Token)              | 输出第一个token的耗时，单位（ms）                                   |
  | TPOT (Time Per Output Token)            | Decode模型每秒输出的token数，单位（tokens/s）                       |
  | E2E Latency (End-to-End Latency)        | 端到端总时间，单位（s）                                             |
  | E2E TPS (End-to-End Tokens Per Second)  | 端到端每秒输出的token数，单位（tokens/s）                           |

相关指标可以对照表一、表二、表三的指标含义。

### 1.3 参数说明
1. 命令行llm_perf以指定参数启动
- --prefill         : 指定prefill模型路径，支持相对/绝对路径。
- --decode          : 指定decode模型路径，支持相对/绝对路径。
- --visual          : 指定visual模型路径，，支持相对/绝对路径，仅vllm需要指定，不指定时默认模型为llm模型。
- --embedding       : 指定embedding模型路径，需要转换成bin文件，支持相对/绝对路径。
- --input           : 指定prefill阶段输入token数量。
- --output          : 指定docode阶段输出多少token后停止。
- --ndevices        : 指定模型使用的device数量，不指定默认为1。
- --loop            : 指定循环执行单个模型次数，不指定默认为1。
- --batch           : 指定模型的batch数，不指定则--batch的值为1。多batch模型必须指定，会校验从模型读取到的batch和用户指定的值是否一致。目前多batch仅支持xh2.
- --no_warm_up      : 默认执行warm_up，配置后会跳过warm_up流程。
- --warm_up_output  : 设置warm_up阶段decode模型输出的token数量，不指定则--warm_up_output的默认值为--output。
- --LazyMode        : 默认关闭LazyMode，配置后会打开LazyMode，开启LazyMode后，模型加载实际会变长，内存占用会变小。
- --interval        : 配置DeviceMonitor(温度、IPU频率、功率)、HostMonitor(physical、virtual memory)监控线程的采样间隔时间，单位秒(ms)，默认为500ms，范围100-60000。
- --skip_perf       : 默认执行性能评估，配置后会跳过性能评估。
- --dump_file       : 默认关闭dump性能数据到yaml文件，配置后会将性能数据保存到yaml文件中。
- --help, -h        : 打印帮助信息
2. 命令行llm_perf以配置文件启动
- --config, -c      : 使用配置文件启动perf程序，需要提前准备好配置文件，配置文件模板参考1.4节。
3. 目前支持通过DEBUG_DEV_INPUT宏开启SetInput输入的Debug模式，须在CMakeLists.txt添加如下编译选项：target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_DEV_INPUT)。


### 1.4 用法说明

#### 1.4.1 embedding模型准换
提供了convert_embed.py对quant_embedding.pt进行准换，转换结果为quant_embedding.bin，可以方便llm_perf工具embedding时读取权重。
使用方式如下：python convert_embed.py --path <embedding_weight_path> --type <llm/vllm>  。

#### 1.4.2 指令启动

手动输入参数启动模型测试，仅支持单个模型的loop测试，参考如下：
``` shell
# 禁用warm_up
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --output 2048 --ndevices 2 --loop 2 --batch 1 --no_warm_up
# 开启warm_up
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --output 2048 --ndevices 2 --loop 2 --batch 1 --warn_up_output 10
# 使能dump_file
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --output 2048 --ndevices 2 --loop 2 --batch 1 --dump_file model.yaml
```

#### 1.4.3 配置文件启动

用户可以使用如下任一指令指定配置文件：
``` shell
./llm_perf -c config.yaml
./llm_perf --config config.yaml
```
配置文件启动等效于多次指令启动，配置文件的参考模板perf_config.yaml，详细内容示例如下：
``` yaml
# Configuration for multiple LLM/VLLM performance tests
Streams:
  # Configuration for a single model test
  - ModelName: deepseek_8b_256_4k_b1_1chip_2cores_v0.5.0
    prefill: deepseek/deepseek_prefill.hmm                 # Path to prefill model file
    decode: deepseek/deepseek_decode.hmm
    embedding: deepseek/hmquant/quant_embedding.bin
    input: 256                                             # Number of input tokens for prefill stage
    output: 100                                              # Number of output tokens to generate in decode stage
    ndevices: 1                                            # Number of devices to use for inference
    loop: 2                                                # Number of times to repeat the test
    batch: 1                                               # Batch size for the model
    no_warm_up: ''                                         # If present (any value), skip warm-up phase
  - ModelName: qwen2.5_7b_256_8k_b1_1chip_2cores_v0.5.0
    prefill: qwen2.5/qwen2.5_prefill.hmm
    decode: qwen2.5/qwen2.5_decode.hmm
    embedding: qwen2.5/hmquant/quant_embedding.bin
    input: 256
    output: 100
    ndevices: 1
    loop: 2
    batch: 1
  - ModelName: qwen3_8b_256_8k_b4_1chip_2cores_v0.5.0
    prefill: qwen3/qwen3_prefill.hmm
    decode: qwen3/qwen3_decode.hmm
    embedding: qwen3/hmquant/quant_embedding.bin
    input: 256
    output: 100
    ndevices: 1
    loop: 1
    batch: 4
  - ModelName: qwen3_14b_256_8k_b1_1chip_2cores_v0.5.0
    prefill: qwen3-14b/qwen3_prefill.hmm
    decode: qwen3-14b/qwen3_decode.hmm
    embedding: qwen3-14b/hmquant/quant_embedding.bin
    input: 256
    output: 100
    ndevices: 1
    loop: 1
  - ModelName: qwen3_30b-a3b
    prefill: qwen3-30b-a3b/qwen3_prefill.hmm
    decode: qwen3-30b-a3b/qwen3_decode.hmm
    embedding: qwen3-30b-a3b/hmquant/quant_embedding.bin
    input: 256
    output: 100
    ndevices: 1
    loop: 1
  - ModelName: qwen2.5-vl
    prefill: qwen2.5-vl/qwen2.5-vl_prefill.hmm
    decode: qwen2.5-vl/qwen2.5-vl_decode.hmm
    visual: qwen2.5-vl/qwen2.5-vl_visual.hmm              # Path to visual processing model file (VLLM only)
    embedding: qwen2.5-vl/hmquant/quant_embedding.bin
    input: 256
    output: 100
    ndevices: 1
    loop: 1
  - ModelName: qwen3-vl
    prefill: qwen3-vl/qwen3-vl_prefill.hmm
    decode: qwen3-vl/qwen3-vl_decode.hmm
    visual: qwen3-vl/qwen3-vl_visual.hmm
    embedding: qwen3-vl/hmquant/quant_embedding.bin
    input: 256
    output: 100
    ndevices: 1
    loop: 1

# Global output file for all performance results
# Note: This has higher priority than individual dump_file settings in Streams
dump_file: perf_result.json
```
**dump_file最外层的优先级高于Streams下的dump_file**
- **如果外层dump_file不存在，则使用Streams下的dump_file。**
- **外层dump_file存在时，则使用外层dump_file，忽略Streams下的dump_file。**
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
# 1. 需要下载Android NDK，并设置NDK_PATH。
# 2. 需要下载Android版本houmo-drv-xh2，设置HOUMO_SDK_PATH为当前Android版本路径。
# 3. 需要下载Android版本runtime-sdk，设置TCIM_RUNTIME_PATH为当前Android版本路径。
# 示例如下：
# export NDK_PATH={the_path_of_android}/android-ndk-r28c
# export HOUMO_SDK_PATH={the_path_of_android}/houmo-drv-xh2
# export TCIM_RUNTIME_PATH={the_path_of_android}/houmo-tcim-runtime-xh2
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
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --output 2048 --ndevices 2 --loop 2 --batch 1
#多次启动perf
./llm_perf -c config.yaml
./llm_perf --config config.yaml
```

结果示例：
./llm_perf --prefill qwen3-8b/qwen3_prefill.hmm --decode qwen3-8b/qwen3_decode.hmm --embedding qwen3-8b/hmquant/quant_embedding.bin  --input 256 --output 200 --warm_up_output 100 --loop 2
```bash
[HostMonitorImpl] Monitoring thread has started, interval: 500ms
[2026-02-27 11:57:16.966] [tid 49674] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-02-27 11:57:17.316] [info] DEVICE-0 Logger initialization complete, log file: ./device_logs/device_0.log
========================= Perf Settings =========================
prefill path : qwen3-8b/qwen3_prefill.hmm
decode path : qwen3-8b/qwen3_decode.hmm
embedding path : qwen3-8b/hmquant/quant_embedding.bin
input token len : 256
stop token len : 200
ndevices : 1
loop : 2
batch : 1
warm_up : enable
LazyMode : disable
skip_perf : disable
=================================================================
Use Devices 0
[2026-02-27 11:57:17.383] [tid 49674] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-02-27 11:57:17.383] [tid 49674] [warning] Empty backend name, use Xh2HalBackend instead.
[2026-02-27 11:57:17.436] [info] DEVICE-1 Logger initialization complete, log file: ./device_logs/device_1.log

==============================(v)LLM Perf WarmUp: input 256, output 100==============================
 Device temperature: 54.51 °C
Decode: 100% |**************************************************| 100/100
==================================================================================

==============================(v)LLM Perf Loop Progress: 1/2==============================
 Device temperature: 63.498 °C
Decode: 100% |**************************************************| 200/200
==================================================================================

==============================(v)LLM Perf Loop Progress: 2/2==============================
 Device temperature: 67.753 °C
Decode: 100% |**************************************************| 200/200
==================================================================================

==================================================================================
                    Model Inference Performance Summary Report
----------------------------------------------------------------------------------
                            Configuration Details
----------------------------------------------------------------------------------
  Batch Size:      1
  Input Length per Sample:    256 tokens
  Output Length per Sample:    200 tokens
----------------------------------------------------------------------------------
                            Inference Performance
----------------------------------------------------------------------------------
  Prefill Time: 128.15 ms/token | Speed: 1997.71 tokens/s
  Decode  Time: 9964.82 ms/token | Speed: 20.07 tokens/s
  Vision  Time: 0.00 ms/image | Speed: 0.00 images/s
----------------------------------------------------------------------------------
                            Overall Performance
----------------------------------------------------------------------------------
  Prefill Model Load Time: 6993.45ms
  Decode Model Load Time:  274.73ms
  Vision Model Load Time:    0.00ms
  TTFT (Time To First Token):  138.52 ms
  TPOT (Time Per Output Token): 56.55 ms/token
  E2E Latency (End-to-End):     11.46 seconds
  E2E TPS (Throughput):         17.46 tokens/s
----------------------------------------------------------------------------------
                            Memory Usage (Max Values)
----------------------------------------------------------------------------------
  Max Physical Memory: 4.01 GB
  Max Virtual Memory: 4.29 GB
----------------------------------------------------------------------------------
                            Device Stats (Max Values)
----------------------------------------------------------------------------------
Device 1:
Temperature    |  49.20°C(Min)      |  54.51°C(Max)      |  52.83°C(Avg)      |
Power          |  0.00 W(Min)       |  0.00 W(Max)       |  0.00 W(Avg)       |
IPU Freq       |  1400.00 Mhz(Min)  |  1400.00 Mhz(Max)  |  1400.00 Mhz(Avg)  |
Mem Info       |  24448 MB(Total)   |  0 MB(Used)        |  24448 MB(Avail)   |
Device 0:
Temperature    |  48.62°C(Min)      |  54.22°C(Max)      |  52.80°C(Avg)      |
Power          |  12.60 W(Min)      |  15.18 W(Max)      |  14.81 W(Avg)      |
IPU Freq       |  1400.00 Mhz(Min)  |  1400.00 Mhz(Max)  |  1400.00 Mhz(Avg)  |
Mem Info       |  24448 MB(Total)   |  6240 MB(Used)     |  18208 MB(Avail)   |
----------------------------------------------------------------------------------
                            Prefill Stage Performance
----------------------------------------------------------------------------------
  Total Time: 138.52 ms | Speed: 1848.08 tokens/s
  Tokenization Time: Skipped (No operation)
  Embedding Time: 3.74   ms
  API SetInput Time: 1.01  ms
  API Inference Time: 128.15ms | Speed: 1997.71 tokens/s
  API GetOutput Time: 0.00 ms
----------------------------------------------------------------------------------
                            Decode Stage Performance
----------------------------------------------------------------------------------
  Total Time: 11310.32ms | Speed: 17.68   tokens/s
  Tokenization Time: Skipped (No operation)
  Embedding Time: 0.33   ms
  API SetInput Time: 0.39  ms/token
  API Inference Time: 49.82ms/token | Speed: 20.07   tokens/s
  API GetOutput Time: 0.00 ms/token
==================================================================================
Every Loop Performance Result has been written to perf_dumper.log
```

## 3.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
