# llm_perf 性能测试工具

本代码用于在后摩系列芯片（XH2）的设备上测试llm模型和vlm模型的单/多batch模型推理性能。当前支持linux、windows11和android native环境（使用adb运行）。

## 目录

- [llm\_perf 性能测试工具](#llm_perf-性能测试工具)
  - [目录](#目录)
  - [1.工具说明](#1工具说明)
    - [1.1 文件及功能支持度说明](#11-文件及功能支持度说明)
    - [1.2 流程说明](#12-流程说明)
    - [1.3 参数说明](#13-参数说明)
    - [1.4 用法说明](#14-用法说明)
      - [1.4.1 embedding模型准换](#141-embedding模型准换)
      - [1.4.2 指令启动](#142-指令启动)
      - [1.4.3 配置文件启动](#143-配置文件启动)
  - [2.快速开始](#2快速开始)
    - [2.1 环境准备](#21-环境准备)
    - [2.2 一键运行](#22-一键运行)
  - [3.免责声明](#3免责声明)

## 变更日志
- [20260713]
  - ASR模拟性能测试入参由 `--audio_len` 改为 `--chunk`，默认1；程序加载模型后根据 `encoder_window` 自行计算 `audio_len`
  - `--token_per_second` 默认值统一为3
- [20260707]
  - Windows平台支持ASR模拟性能测试，`ENABLE_ASR` 默认开启
  - Windows编译复用 `houmo-llm-engine` 产物，`build_win.bat` 会自动触发 `houmo-llm-engine` Windows构建
  - Windows编译支持多核心并行构建，默认使用 `%NUMBER_OF_PROCESSORS%`
  - Windows运行时依赖统一安装到 `$HOUMO_EXAMPLES_PATH/tools/common/lib`，请确保该目录已加入 `PATH`
  - CMake平台配置复用 `tools/common/houmo-llm-engine/cmake/platforms`，便于后续按平台维护
- [20260626]
  - ASR性能测试改为模拟数据模式（PerfAsr），不再依赖真实音频、Tokenizer、Embedding
  - 新增--audio_len/--token_per_second参数，用于模拟指定音频时长的ASR性能测试
  - 移除--audio/--tokenizer/--embedding等ASR真实推理参数
  - 移除Whisper ASR性能测试支持, 目前仅支持llm类型的asr模型(glm-asr和qwen3-asr)
  - 新增Decode进度条显示
  - 自动推断GLM-ASR/Qwen3-ASR模型差异（encoder输入dtype、input_lengths等），零model_type分支
- [20260615]
  - 新增ASR(Automatic Speech Recognition)语音识别功能支持
  - 新增--encode/--prefill/--decode/--tokenizer/--audio等ASR相关参数
  - 新增ENABLE_ASR编译选项
  - 支持Whisper、GlmAsr、Qwen3Asr三种ASR模型
- [20260424]
  - 修改--ndevices参数为--devices，由指定设备数量改为指定设备ID
  - --devices不设置时优先从HOUMO_VISIBLE_DEVICES读取，环境变量未设置则默认--devices 0
  - 对于--devices入参数量大于等于2时，增加模型名称必须为hmms校验和对应设备ID的CTC校验。

## 1.工具说明

### 1.1 文件及功能支持度说明

文件目录如下：
| 文件名称                | 说明                                                |
| ----------------------- | --------------------------------------------------- |
| llm_perf.cc             | 主入口，拉起大语言模型推理流程                      |
| HmEmbedding.cc          | embedding类实现                                     |
| HmllmInfer.cc           | 主文件，实现llm模型单batch具体推理流程              |
| HmvllmInfer.cc          | 主文件，实现vlm模型单batch具体推理流程             |
| HmllmInferMultiBatch.cc | 主文件，实现llm多batch具体推理流程，仅支持xh2       |
| HmAsrInfer.cc           | 主文件，实现ASR模拟性能测试（PerfAsr）流程          |
| convert_embed.py        | 将embedding的pt模型转换为bin文件方便工具读取权重    |
| device_monitor          | 监控设备温度、功率、IPU频率的类实现                 |
| perf_tracker            | 性能采集类功能实现                                  |
| perf_dumper             | 性能数据落盘及性能数据可视化类功能实现              |
| host_monitor            | 主机侧内存监控类实现                                |
| build_linux.sh          | Linux平台llm_perf编译脚本                           |
| build_ndk.sh            | Android平台llm_perf编译脚本                         |
| build_win.bat           | Win11平台llm_perf编译脚本                           |

功能支持度说明：
  - 具体支持的模型参考models目录下llm和vlm模型支持的平台。
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
- 表一：Prefill模型指标
  | Prefill Stage 指标                      | 说明                                                                |
  | --------------------------------------- | ------------------------------------------------------------------- |
  | Prefill Total Time                      | Prefill阶段总耗时，单位（ms）                                       |
  | Prefill Total Speed                     | Prefill阶段总速度，单位（tokens/s）                                 |
  | Tokenization total Time                 | Tokenization耗时，单位（ms）                                        |
  | Embedding total Time                    | Embedding操作总耗时，单位（ms）                                     |
  | API SetInput  total Time                | tcim runtime setinput API总耗时，单位（ms）                           |
  | API Inference total Time                | tcim runtime run+sync API总耗时，单位（ms）                           |
  | API Inference Speed                     | tcim runtime run+sync API速度，单位（tokens/s）                     |
  | API GetOutput total Time                | tcim runtime GetOutput API总耗时，单位（ms）                          |

- 表二：Decode模型指标
  | Decode Stage 指标                       | 说明                                                                |
  | --------------------------------------- | ------------------------------------------------------------------- |
  | Decode Total Time                       | Decode阶段耗时，单位（ms）                                          |
  | Decode Total Speed                      | Decode阶段速度，单位（tokens/s）                                    |
  | Tokenization total Time                 | Tokenization耗时，单位（ms）                                        |
  | Embedding total Time                    | Embedding操作总耗时，单位（ms）                                     |
  | API SetInput  avg Time                  | tcim runtime setinput API平均耗时，单位（ms/token）                 |
  | API Inference avg Time                  | tcim runtime run+sync API平均耗时，单位（ms/token）                 |
  | API Inference avg Speed                 | tcim runtime run+sync API速度，单位（tokens/s）                     |
  | API GetOutput avg Time                  | tcim runtime GetOutput API平均耗时，单位（ms/token）                |

- 表三：Vision模型指标
  | Vision Stage 指标                       | 说明                                                                |
  | --------------------------------------- | ------------------------------------------------------------------- |
  | Vision Total Time                       | Vision阶段耗时，单位（ms）                                          |
  | Vision Total Speed                      | Vision阶段速度，单位（tokens/s）                                    |
  | Preprocessing Time                      | Preprocessing耗时，单位（ms）                                       |
  | API SetInput  total Time                | tcim runtime setinput API总耗时，单位（ms）                         |
  | API Inference total Time                | tcim runtime run+sync API总耗时，单位（ms）                         |
  | API Inference Speed                     | tcim runtime run+sync API速度，单位（images/s）                     |
  | API GetOutput total Time                | tcim runtime GetOutput API总耗时，单位（ms）                        |

- 表四：总体性能指标
  | Overall 指标                            | 说明                                                                |
  | --------------------------------------- | ------------------------------------------------------------------- |
  | TTFT (Time to First Token)              | 输出第一个token的耗时，单位（ms）                                   |
  | TPOT (Time Per Output Token)            | Decode模型每秒输出的token数，单位（tokens/s）                       |
  | E2E Latency (End-to-End Latency)        | 端到端总时间，单位（s）                                             |
  | E2E TPS (End-to-End Tokens Per Second)  | 端到端每秒输出的token数，单位（tokens/s）                           |

- 表五：ASR模拟性能测试指标（需要ENABLE_ASR编译选项开启）
  | ASR Stage 指标                          | 说明                                                                |
  | --------------------------------------- | ------------------------------------------------------------------- |
  | encode preprocess                       | encode预处理耗时，单位（ms）                                        |
  | encode inference                        | encode推理耗时，单位（ms）                                          |
  | encode postprocess                      | encode后处理耗时，单位（ms）                                        |
  | prefill preprocess                      | Prefill预处理耗时，单位（ms）                                       |
  | prefill inference                       | Prefill推理耗时，单位（ms）                                         |
  | prefill postprocess                     | Prefill后处理耗时，单位（ms）                                       |
  | decode preprocess                       | Decode预处理耗时，单位（ms），括号内为调用次数和平均耗时             |
  | decode inference                        | Decode推理耗时，单位（ms），括号内为调用次数和平均耗时               |
  | decode postprocess                      | Decode后处理耗时，单位（ms），括号内为调用次数和平均耗时             |
  | transcribe total                        | ASR转录总耗时，单位（ms）                                           |
  | E2E                                     | 端到端总耗时，单位（ms）                                            |
  | TTFT                                    | 首个token耗时，单位（ms）                                           |
  | Prefill TPS                             | Prefill阶段吞吐量，单位（tokens/s）                                 |
  | Decode TPS                              | Decode阶段吞吐量，单位（tokens/s）                                  |
  | Overall TPS                             | 整体吞吐量，单位（tokens/s）                                        |
  | Avg Decode Latency                      | 平均Decode延迟，单位（ms/token）                                    |
  | Simulated Audio                         | 模拟音频时长，单位（s）                                             |
  | Chunks                                  | 分块数量（由 `--chunk` 指定，audio_len=chunk×encoder_window×0.01s） |
  | Output Tokens                           | 输出token总数（=Σ每chunk的decode轮数）                              |
  | Overall RTF                             | 整体实时因子（Real-Time Factor），小于1表示快于实时                  |
  | Inference RTF                           | 推理实时因子                                                        |
  | Encode Per Chunk                        | 每个chunk的encode阶段平均耗时，单位（s）                            |
  | Prefill Per Chunk                       | 每个chunk的prefill阶段平均耗时，单位（s）                           |
  | Decode Per Chunk                        | 每个chunk的decode阶段平均耗时，单位（s）                            |

相关指标可以对照表一、表二、表三、表四、表五的指标含义。

### 1.3 参数说明
1. 命令行llm_perf以指定参数启动
- --prefill         : 指定prefill模型路径，支持相对/绝对路径。
- --decode          : 指定decode模型路径，支持相对/绝对路径。
- --visual          : 指定visual模型路径，，支持相对/绝对路径，仅vlm需要指定，不指定时默认模型为llm模型。
- --embedding       : 指定embedding模型路径，需要转换成bin文件，支持相对/绝对路径。
- --input           : 指定prefill阶段输入token数量，支持单值或逗号分隔的多组参数，例如`256,512,1024`。
- --output          : 指定decode阶段输出token数量，支持单值或逗号分隔的多组参数，组数必须与--input一致。
- --devices         : 指定模型使用的device_id，不指定时优先从HOUMO_VISIBLE_DEVICES环境变量读取，如果环境变量未设置默认为0，支持单值或逗号分隔的多组参数，例如:0 或者0,1。
- --loop            : 指定循环执行单个模型次数，不指定默认为1。
- --batch           : 指定模型的batch数，不指定则--batch的值为1。多batch模型必须指定，会校验从模型读取到的batch和用户指定的值是否一致。目前多batch仅支持xh2.
- --no_warm_up      : 默认执行warm_up，配置后会跳过warm_up流程。
- --warm_up_input   : 配置warm_up的输入数据长度，不指定时默认使用第一组input的值。
- --warm_up_output  : 设置warm_up阶段decode模型输出的token数量，不指定时默认使用第一组output的值。
- --LazyMode        : 默认关闭LazyMode，配置后会打开LazyMode，开启LazyMode后，模型加载实际会变长，内存占用会变小。
- --interval        : 配置DeviceMonitor(温度、IPU频率、功率)、HostMonitor(physical、virtual memory)监控线程的采样间隔时间，单位毫秒(ms)，默认为500ms，范围100-60000。
- --model_name      : 可选，指定模型名称，用于dump yaml和日志展示。
- --skip_perf       : 默认执行性能评估，配置后会跳过性能评估。
- --dump_file       : 默认关闭dump性能数据到yaml文件，配置后会将性能数据保存到yaml文件中。
- --help, -h        : 打印帮助信息

2. ASR模拟性能测试相关参数（需要ENABLE_ASR编译选项开启）
- --encode           : 指定ASR编码器模型路径，支持相对/绝对路径。设置此参数将触发ASR性能测试流程。
- --prefill          : ASR模式下指定prefill模型路径。
- --decode           : ASR模式下指定decode模型路径。
- --chunk            : 模拟encoder分块数，默认1。程序加载模型后根据encoder_window计算audio_len（audio_len=chunk×encoder_window×0.01s）。
- --token_per_second : 模拟每秒decode token数，默认3。每个chunk的decode轮数=该chunk音频时长×token_per_second。
- --devices          : 指定模型使用的device_id，不指定时优先从HOUMO_VISIBLE_DEVICES环境变量读取，如果环境变量未设置默认为0。
- --loop             : 指定循环执行次数，不指定默认为1。
- --no_warm_up       : 默认执行warm_up，配置后会跳过warm_up流程。
- --interval         : 配置DeviceMonitor监控线程的采样间隔时间，单位毫秒(ms)，默认为500ms。
- --model_name       : 可选，指定模型名称，用于日志展示。

说明：ASR性能测试使用模拟数据（随机mel特征+随机embedding），无需真实音频文件、Tokenizer和Embedding。自动支持GLM-ASR和Qwen3-ASR模型，根据模型元数据动态推断差异（encoder输入dtype、是否有input_lengths等），无需手动指定模型类型。

> **关于 `token_per_second` 参数的说明**：该参数指的是**文本文字 token**（语音转文字后，ASR 输出文字 token），而不是音频声学 token（语音编码）。
>
> **人类正常说话语速基准：**
> - 中文日常闲聊：180~220 字/分钟 = 3~3.6 汉字/秒
> - 中文正式演讲：240~280 字/分钟 = 4~4.7 汉字/秒
> - 英文日常：130~160 单词/分钟 ≈ 2~2.7 词/秒
>
> **换算成文本 token（通用分词规则）：**
> - 中文：1 token ≈ 1.5 个汉字 → 闲聊 ≈ 2 token/s，演讲快语速 ≈ 3.1 token/s
> - 英文：1 token ≈ 0.75 单词 → 日常英文 ≈ 3.6 token/s
>
> **综合结论（含自然停顿）：**
> - 正常普通人对话（含停顿换气）：2～3 文本 token/秒
> - 无停顿快速朗读：3～4 token/s

3. 命令行llm_perf以配置文件启动
- --config, -c      : 使用配置文件启动perf程序，需要提前准备好配置文件，配置文件模板参考1.4节。
4. 目前支持通过DEBUG_DEV_INPUT宏开启SetInput输入的Debug模式，须在CMakeLists.txt添加如下编译选项：target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_DEV_INPUT)。
5. ASR功能编译说明：ASR功能由CMake选项`ENABLE_ASR`控制，当前Linux、Windows、Android平台默认开启；如需关闭可在配置CMake时添加`-DENABLE_ASR=OFF`。


### 1.4 用法说明

#### 1.4.1 embedding模型准换
提供了convert_embed.py对quant_embedding.pt进行准换，转换结果为quant_embedding.bin，可以方便llm_perf工具embedding时读取权重。
使用方式如下：python convert_embed.py --path <embedding_weight_path>。

#### 1.4.2 指令启动

手动输入参数启动模型测试，仅支持单个模型的loop测试，参考如下：
``` shell
# --devices 指定使用0,1运行2chip模型
./llm_perf --prefill prefill_model.hmms --decode decode_model.hmms --embedding quant_embedding.bin --input 100 --output 2048 --devices 0,1 --loop 2 --batch 1 --no_warm_up
# 环境变量HOUMO_VISIBLE_DEVICES指定使用device 1
HOUMO_VISIBLE_DEVICES=1 ./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --output 2048 --loop 2 --batch 1 --no_warm_up
# 禁用warm_up
./llm_perf --prefill prefill_model.hmms --decode decode_model.hmms --embedding quant_embedding.bin --input 100 --output 2048 --devices 0,1 --loop 2 --batch 1 --no_warm_up
# 开启warm_up
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --output 2048 --devices 0 --loop 2 --batch 1 --warm_up_output 10
# 使能dump_file
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --output 2048 --devices 0 --loop 2 --batch 1 --dump_file model.yaml
# 多组input/output，按组依次测试并分别dump到yaml文件中
./llm_perf --model_name qwen3_8b --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 256,512 --output 100,200 --loop 2 --dump_file model.yaml
```

ASR模拟性能测试使用示例（需要ENABLE_ASR编译选项开启）：
``` shell
# GLM-ASR模型性能测试（默认1个chunk，audio_len由encoder_window推导）
./llm_perf --encode glm_encode.hmm --prefill glm_prefill.hmm --decode glm_decode.hmm --chunk 1 --token_per_second 3 --devices 0
# Qwen3-ASR模型性能测试（多个chunk模拟更长音频）
./llm_perf --encode qwen3-asr_encode.hmm --prefill qwen3-asr_prefill.hmm --decode qwen3-asr_decode.hmm --chunk 10 --token_per_second 4 --devices 0 --loop 3
# 跳过warm_up的短测
./llm_perf --encode glm_encode.hmm --prefill glm_prefill.hmm --decode glm_decode.hmm --chunk 1 --token_per_second 3 --devices 0 --no_warm_up
```

#### 1.4.3 配置文件启动

用户可以使用如下任一指令指定配置文件：
``` shell
./llm_perf -c config.yaml
./llm_perf --config config.yaml
```
配置文件启动等效于多次指令启动，配置文件的参考模板perf_config.yaml，详细内容示例如下：
``` yaml
# Configuration for multiple LLM/VLM performance tests
Streams:
  # Configuration for a single model test
  - ModelName: deepseek_8b_256_4k_b1_1chip_2cores_v0.5.0
    prefill: deepseek/deepseek_prefill.hmm                 # Path to prefill model file
    decode: deepseek/deepseek_decode.hmm
    embedding: deepseek/hmquant/quant_embedding.bin
    input: "256,512"                                       # Single value or comma-separated groups
    output: "100,200"                                     # Must match input group count
    devices: 1                                            # device ids to use for inference
    loop: 2                                                # Number of times to repeat each input/output group
    batch: 1                                               # Batch size for the model
    no_warm_up: ''                                         # If present (any value), skip warm-up phase
  - ModelName: qwen2.5_7b_256_8k_b1_1chip_2cores_v0.5.0
    prefill: qwen2.5/qwen2.5_prefill.hmm
    decode: qwen2.5/qwen2.5_decode.hmm
    embedding: qwen2.5/hmquant/quant_embedding.bin
    input: 256
    output: 100
    devices: 1
    loop: 2
    batch: 1
  - ModelName: qwen3_8b_256_8k_b4_1chip_2cores_v0.5.0
    prefill: qwen3/qwen3_prefill.hmm
    decode: qwen3/qwen3_decode.hmm
    embedding: qwen3/hmquant/quant_embedding.bin
    input: 256
    output: 100
    devices: 1
    loop: 1
    batch: 4
  - ModelName: qwen3_14b_256_8k_b1_1chip_2cores_v0.5.0
    prefill: qwen3-14b/qwen3_prefill.hmm
    decode: qwen3-14b/qwen3_decode.hmm
    embedding: qwen3-14b/hmquant/quant_embedding.bin
    input: 256
    output: 100
    devices: 1
    loop: 1
  - ModelName: qwen3_30b-a3b
    prefill: qwen3-30b-a3b/qwen3_prefill.hmm
    decode: qwen3-30b-a3b/qwen3_decode.hmm
    embedding: qwen3-30b-a3b/hmquant/quant_embedding.bin
    input: 256
    output: 100
    devices: 1
    loop: 1
  - ModelName: qwen2.5-vl
    prefill: qwen2.5-vl/qwen2.5-vl_prefill.hmm
    decode: qwen2.5-vl/qwen2.5-vl_decode.hmm
    visual: qwen2.5-vl/qwen2.5-vl_visual.hmm              # Path to visual processing model file (VLM only)
    embedding: qwen2.5-vl/hmquant/quant_embedding.bin
    input: 256
    output: 100
    devices: 1
    loop: 1
  - ModelName: qwen3-vl
    prefill: qwen3-vl/qwen3-vl_prefill.hmms
    decode: qwen3-vl/qwen3-vl_decode.hmms
    visual: qwen3-vl/qwen3-vl_visual.hmms
    embedding: qwen3-vl/hmquant/quant_embedding.bin
    input: 256
    output: 100
    devices: 0,1
    loop: 1
  # ASR模拟性能测试配置示例（requires ENABLE_ASR compilation option）
  - ModelName: glm_asr_perf
    encode: glm-asr/glm-asr_encode.hmm
    prefill: glm-asr/glm-asr_prefill.hmm
    decode: glm-asr/glm-asr_decode.hmm
    chunk: 1
    token_per_second: 3
    devices: "0"
    loop: 3
  - ModelName: qwen3_asr_perf
    encode: qwen3-asr/qwen3-asr_encode.hmm
    prefill: qwen3-asr/qwen3-asr_prefill.hmm
    decode: qwen3-asr/qwen3-asr_decode.hmm
    chunk: 10
    token_per_second: 4
    devices: "0"
    loop: 1

# Global output file for all performance results
# Note: This has higher priority than individual dump_file settings in Streams
dump_file: perf_result.yaml
```
**dump_file最外层的优先级高于Streams下的dump_file**
- **如果外层dump_file不存在，则使用Streams下的dump_file。**
- **外层dump_file存在时，则使用外层dump_file，忽略Streams下的dump_file。**
上述模板等效于同时执行多次指令启动，用户可以根据实际需求自行配置。
如果单个Streams条目内的input/output配置为多组参数，则会在单次模型加载和单次warm_up之后，按组顺序执行性能测试，并在yaml结果中保留每组测试结果。


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

如果是windows11版本请执行：

```bat
rem 首次构建会自动构建并安装 houmo-llm-engine Windows 产物到 tools/common/lib
build_win.bat
```

**编译生成的llm_perf可执行文件在tools/bin目录下，在win11运行llm_perf.exe前需要参考 `tools/win_envs` 正确设置Windows环境变量：**
1. 推荐在根目录执行 `env.bat --set`，该脚本会调用 `tools/win_envs/set_environs.py` 根据 `tools/win_envs/env.json` 配置环境变量。
2. 执行完成后需要重新打开cmd窗口，使新的环境变量生效。
3. 确认 `PATH` 中包含 `%HOUMO_EXAMPLES_PATH%\tools\common\lib`，该目录保存 `houmo_infer.dll`、`tokenizer_lib.dll`、`kaldi-native-fbank-core.dll`、`samplerate.dll` 等运行时依赖。
4. 确认 `HOUMO_TARGET=xh2`、`TCIM_BACKEND=Xh2HalBackend`、`TCIM_RUNTIME_PATH`、`HOUMO_SDK_PATH` 等变量已按 `tools/win_envs/env.json` 正确设置。
5. C++样例依赖 `libhal_xh2a.dll` 和 `tcim_runtime_lite.dll`，可通过 `env.bat --set` 加入PATH，或和 `llm_perf.exe` 放在同一目录。
6. Windows平台当前默认开启ASR模拟性能测试；如需关闭，可在CMake配置时添加 `-DENABLE_ASR=OFF`。

### 2.2 一键运行
``` shell
#单次启动perf
./llm_perf --prefill prefill_model.hmm --decode decode_model.hmm --embedding quant_embedding.bin --input 100 --output 2048 --devices 2 --loop 2 --batch 1
#多次启动perf
./llm_perf -c config.yaml
./llm_perf --config config.yaml
```

结果示例：
./llm_perf --prefill qwen3-8b/qwen3_prefill.hmm --decode qwen3-8b/qwen3_decode.hmm --embedding qwen3-8b/hmquant/quant_embedding.bin  --input 256 --output 200 --warm_up_output 100 --loop 2
```bash
[HostMonitorImpl] Monitoring thread has started, interval: 500ms
[2026-03-09 09:42:52.227] [tid 4651] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-03-09 09:42:52.352] [info] DEVICE-0 Logger initialization complete, log file: ./device_logs/device_0.log
========================= Perf Settings =========================
prefill path : qwen3-8b/qwen3_prefill.hmm
decode path : qwen3-8b/qwen3_decode.hmm
embedding path : qwen3-8b/hmquant/quant_embedding.bin
input token len : 256
stop token len : 200
devices : 1
loop : 2
batch : 1
warm_up : enable
LazyMode : disable
skip_perf : disable
=================================================================
Use Devices 0
[2026-03-09 09:42:52.392] [tid 4651] [warning] Using TCIM_BACKEND = Xh2HalBackend as default backend
[2026-03-09 09:42:52.392] [tid 4651] [warning] Empty backend name, use Xh2HalBackend instead.
[2026-03-09 09:42:52.539] [info] DEVICE-1 Logger initialization complete, log file: ./device_logs/device_1.log

==============================(v)LLM Perf WarmUp: input 256, output 100==============================
 Device temperature: 42.048 °C
Prefill: 100% |**************************************************|
Decode: 100% |**************************************************| 100/100
==================================================================================

==============================(v)LLM Perf Loop Progress: 1/2==============================
 Device temperature: 42.338 °C
Prefill: 100% |**************************************************|
Decode: 100% |**************************************************| 200/200
==================================================================================

==============================(v)LLM Perf Loop Progress: 2/2==============================
 Device temperature: 44.074 °C
Prefill: 100% |**************************************************|
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
  Prefill API Inference total Time: 113.08 ms | Speed: 2263.93 tokens/s
  Decode  API Inference total Time: 9870.75 ms | Speed: 20.26 tokens/s
  Vision  API Inference total Time: 0.00 ms | Speed: 0.00 images/s
----------------------------------------------------------------------------------
                            Overall Performance
----------------------------------------------------------------------------------
  Prefill Model Load Time: 95552.91ms
  Decode Model Load Time: 3530.92ms
  Vision Model Load Time:    0.00ms
  TTFT (Time To First Token):  114.49 ms
  TPOT (Time Per Output Token): 49.97 ms/token
  E2E Latency (End-to-End):     10.11 seconds
  E2E TPS (Throughput):         19.78 tokens/s
----------------------------------------------------------------------------------
                            Memory Usage (Max Values)
----------------------------------------------------------------------------------
  Physical Memory: 2.30 GB
  Virtual Memory: 2.53 GB
  Max Physical Memory: 3.99 GB
  Max Virtual Memory: 4.66 GB
----------------------------------------------------------------------------------
                            Device Stats (Max Values)
----------------------------------------------------------------------------------
Device 1:
Temperature    |  30.06°C(Min)      |  39.63°C(Max)      |  37.18°C(Avg)      |
Power          |  0.00 W(Min)       |  0.00 W(Max)       |  0.00 W(Avg)       |
IPU Freq       |  1400.00 Mhz(Min)  |  1400.00 Mhz(Max)  |  1400.00 Mhz(Avg)  |
Mem Info       |  24448 MB(Total)   |  0 MB(Used)        |  24448 MB(Avail)   |
Device 0:
Temperature    |  26.88°C(Min)      |  42.05°C(Max)      |  36.20°C(Avg)      |
Power          |  8.61 W(Min)       |  23.87 W(Max)      |  15.83 W(Avg)      |
IPU Freq       |  1400.00 Mhz(Min)  |  1400.00 Mhz(Max)  |  1400.00 Mhz(Avg)  |
Mem Info       |  24448 MB(Total)   |  8897 MB(Used)     |  15551 MB(Avail)   |
----------------------------------------------------------------------------------
                            Prefill Stage Performance
----------------------------------------------------------------------------------
  Total Time: 114.49 ms | Speed: 2236.00 tokens/s
  Tokenization total Time: Skipped (No operation)
  Embedding total Time: 0.32   ms
  API SetInput  total Time: 0.62  ms
  API Inference total Time: 113.08ms | Speed: 2263.93 tokens/s
  API GetOutput total Time: 0.12 ms
----------------------------------------------------------------------------------
                            Decode Stage Performance
----------------------------------------------------------------------------------
  Total Time: 9995.00ms | Speed: 20.01   tokens/s
  Tokenization total Time: Skipped (No operation)
  Embedding total Time: 0.03   ms
  API SetInput  avg Time: 0.20  ms/token
  API Inference avg Time: 49.35ms/token | Speed: 20.26   tokens/s
  API GetOutput avg Time: 0.15 ms/token
==================================================================================
Every Loop Performance Result has been written to perf_dumper.log
```

## 3.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
