# llm_perf 性能测试工具

本代码用于在后摩系列芯片（XH2）的设备上测试llm、vlm、asr和tts模型推理性能。当前支持linux、windows11和android native环境（使用adb运行）。

## 目录

- [llm\_perf 性能测试工具](#llm_perf-性能测试工具)
  - [目录](#目录)
  - [1.工具说明](#1工具说明)
    - [1.1 文件及功能支持度说明](#11-文件及功能支持度说明)
    - [1.2 流程说明](#12-流程说明)
    - [1.3 参数说明](#13-参数说明)
    - [1.4 用法说明](#14-用法说明)
      - [1.4.1 embedding模型转换](#141-embedding模型转换)
      - [1.4.2 指令启动](#142-指令启动)
      - [1.4.3 配置文件启动](#143-配置文件启动)
  - [2.快速开始](#2快速开始)
    - [2.1 环境准备](#21-环境准备)
    - [2.2 一键运行](#22-一键运行)
  - [3.免责声明](#3免责声明)

## 变更日志
- [20260804]
  - 新增Qwen3-TTS CustomVoice模拟性能测试，支持streaming fixed-frame完整推理链路
  - 新增TTS E2E、RTF、TTFA、阶段耗时和Codec Frames/s指标
  - 新增TTS命令行和yaml配置启动方式，支持可选WAV和性能结果落盘
  - 新增ENABLE_TTS编译选项，默认开启
- [20260713]
  - ASR模拟性能测试入参由 `--audio_len` 改为 `--chunk`，默认1；程序加载模型后根据 `encoder_window` 自行计算 `audio_len`
  - `--token_per_second` 默认值统一为3
- [20260707]
  - Windows平台支持ASR模拟性能测试，`ENABLE_ASR` 默认开启
  - Windows编译复用 `houmo-llm-engine` 产物，`build_win.bat` 会自动触发 `houmo-llm-engine` Windows构建
  - Windows编译支持多核心并行构建，默认使用 `%NUMBER_OF_PROCESSORS%`
  - Windows运行时依赖统一安装到 `$HOUMO_EXAMPLES_PATH/utils/lib`，请确保该目录已加入 `PATH`
  - CMake平台配置复用 `utils/cpp/houmo_engine/cmake/platforms`，便于后续按平台维护
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
| PerfTts.cc              | 主文件，实现Qwen3-TTS模拟性能测试流程               |
| convert_embed.py        | 将LLM/VLM/ASR单表embedding的pt模型转换为bin文件；暂不支持Qwen3-TTS |
| device_monitor          | 监控设备温度、功率、IPU频率的类实现                 |
| perf_tracker            | 性能采集类功能实现                                  |
| perf_dumper             | 性能数据落盘及性能数据可视化类功能实现              |
| host_monitor            | 主机侧内存监控类实现                                |
| build_linux.sh          | Linux平台llm_perf编译脚本                           |
| build_ndk.sh            | Android平台llm_perf编译脚本                         |
| build_win.bat           | Win11平台llm_perf编译脚本                           |
| build_OHOS.sh          | OHOS平台llm_perf交叉编译脚本                      |

功能支持度说明：
  - 具体支持的模型参考models目录下llm和vlm模型支持的平台。
  - 支持的芯片型号为：xh2。
  - 支持的大道软件版本要求如下：xh2≥v0.6.0 。
  - 支持了device的温度、功率、IPU频率监控，记录保存在./device_logs/device_x.log。
  - 温度监控带告警功能，当温度大于80℃时，程序会打印警告信息，温度大于100℃时，程序会强制退出。
  - 主机侧的内存监控仅支持linux平台。
  - **修改配置文件启动方式为yaml文件，对之前版本的json文件启动兼容。**
  - 支持Qwen3-TTS CustomVoice性能测试，TTS功能由CMake选项`ENABLE_TTS`控制，默认开启。
  - TTS首版本固定为streaming fixed-frame模式，固定Chinese、vivian、seed 42和单device，不处理真实文本，不支持用户设置mode、language、speaker或seed。
  - TTS使用随机正文token执行TextEmbedding、TextProjection、Talker、CodePredictor和StatefulDecoder完整链路；`token_per_second`默认3，`output_wav`可选。

Qwen3-TTS支持度如下：

| 项目 | 当前支持情况 |
| --- | --- |
| 模型类型 | 仅支持Qwen3-TTS CustomVoice；不支持Base voice clone |
| 模型规格 | 支持CustomVoice模型的相同HMM接口，已验证`0.6b-customvoice`；`1.7b-customvoice`需使用对应规格的完整HMM和embedding资产 |
| 推理模式 | 仅支持streaming fixed-frame，不支持oneshot和自然EOS停止 |
| 输入 | 不接收真实文本和tokenizer；根据`tts_audio_length * token_per_second`生成固定seed的随机正文token |
| 音频长度 | `ceil(tts_audio_length / 0.08)`得到目标codec frame数；实际nominal时长可能向上取整到0.08秒的整数倍 |
| Prompt条件 | language固定Chinese，speaker固定vivian，seed固定42 |
| 设备 | 仅支持单device；不支持TTS `.hmms`多device运行 |
| 平台 | 当前TTS perf已在Linux XH2验证；llm_perf的其他平台构建不等同于TTS运行已验证 |
| 模型链路 | TextEmbedding、TextProjection、Talker Prefill/Decode、CodePredictor Prefill/Decode、StatefulDecoder |
| 指标 | E2E、RTF、TTFA、Codec Frames/s及各阶段总耗时/调用次数/平均耗时 |
| 进度与日志 | Warmup和每个正式loop显示codec frame进度条；每个loop写入`perf_dumper.log`，终端和YAML展示Average结果 |
| WAV输出 | `--output_wav`可选，仅保存最后一个正式loop，写盘耗时不计入E2E和RTF |
| Embedding准备 | 不支持使用`tools/llm_perf/convert_embed.py`转换；必须使用Qwen3-TTS demo目录的`get_model.py`准备 |

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

- 表六：TTS模拟性能测试指标（需要ENABLE_TTS编译选项开启）
  | TTS指标                                 | 说明                                                                |
  | --------------------------------------- | ------------------------------------------------------------------- |
  | E2E                                     | 本轮请求开始到完整waveform在host侧可用的端到端耗时，单位（ms）；YAML字段为`e2e_latency` |
  | RTF                                     | 实时因子，`E2E_seconds / nominal_audio_length`，小于1表示快于实时    |
  | TTFA                                    | 本轮请求开始到第一个非空waveform chunk在host侧可用的耗时，单位（ms）|
  | Text Embedding                          | role、正文和TTS special token的embedding耗时，单位（ms）             |
  | Text Projection                         | TextProjection总耗时，包含调用次数和平均耗时，单位（ms）             |
  | Prompt Prepare                          | streaming prompt和codec prompt准备耗时，单位（ms）                   |
  | Talker Prefill                          | Talker Prefill耗时，包含调用次数和平均耗时，单位（ms）               |
  | Talker Decode                           | Talker Decode耗时，包含调用次数和平均耗时，单位（ms）                |
  | Talker Sampling                         | Talker采样耗时，包含调用次数和平均耗时，单位（ms）                   |
  | Codec Frame Prepare                     | codec frame组装耗时，单位（ms）                                     |
  | CodePredictor Prepare                   | CodePredictor逐帧准备耗时，单位（ms）                               |
  | CodePredictor Prefill                   | CodePredictor Prefill耗时，包含调用次数和平均耗时，单位（ms）        |
  | CodePredictor Decode                    | CodePredictor Decode耗时，包含调用次数和平均耗时，单位（ms）         |
  | CodePredictor Sampling                  | CodePredictor采样耗时，包含调用次数和平均耗时，单位（ms）            |
  | StatefulDecoder                         | StatefulDecoder耗时，包含调用次数和平均耗时，单位（ms）              |
  | Other                                   | E2E中未被上述阶段单独计量的耗时，单位（ms）                          |
  | Codec Frames/s                          | codec生成速度，单位（frames/s）                                     |

TTS使用以下固定换算和指标公式：

```text
body_text_tokens = max(1, ceil(tts_audio_length * token_per_second))
text_projection_tokens = body_text_tokens + 6
target_codec_frames = ceil(tts_audio_length / 0.08)
nominal_audio_length = target_codec_frames * 0.08
expected_audio_samples = target_codec_frames * 1920
decoder_chunks = ceil(target_codec_frames / 12)
RTF = E2E_seconds / nominal_audio_length
codec_generation_seconds = Talker Prefill + Talker Decode + Talker Sampling
                           + Codec Frame Prepare + CodePredictor Prepare
                           + CodePredictor Prefill + CodePredictor Decode
                           + CodePredictor Sampling
codec_frames_per_second = target_codec_frames / codec_generation_seconds
```

其中采样率固定为24000 Hz，每个codec frame固定1920个采样点，即0.08秒。E2E包含TextEmbedding、TextProjection、prompt准备、Talker、CodePredictor、StatefulDecoder和必要host处理，不包含模型/embedding加载、监控线程启动、WAV写盘和性能结果yaml写盘。TTFA包含首批codec frame生成和第一次StatefulDecoder调用。

相关指标可以对照表一、表二、表三、表四、表五、表六的指标含义。

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

3. TTS模拟性能测试相关参数（需要ENABLE_TTS编译选项开启）
- --text_projection        : 指定TextProjection模型路径，必填。
- --talker_prefill         : 指定Talker Prefill模型路径，必填。
- --talker_decode          : 指定Talker Decode模型路径，必填。
- --code_predictor_prefill : 指定CodePredictor Prefill模型路径，必填。
- --code_predictor_decode  : 指定CodePredictor Decode模型路径，必填。
- --stateful_decoder       : 指定StatefulDecoder模型路径，必填。
- --embedding              : 指定Talker codec token embedding文件，即`quant_embedding.bin`，必填。
- --code_embedding         : 指定CodePredictor 15组codebook embedding文件，即`quant_embedding_code_predictor.bin`，必填。
- --text_embedding         : 指定文本token embedding文件，即`text_embedding.bin`，必填。
- --tts_audio_length       : 请求音频时长，单位秒，必填，必须是有限且大于0的单值。设置此参数将触发TTS性能测试流程。
- --token_per_second       : 每秒模拟正文token数，只支持正整数，默认3。
- --devices                : 指定单个device_id，不指定时优先从HOUMO_VISIBLE_DEVICES读取，环境变量未设置时默认0。TTS不支持多device。
- --loop                   : 指定正式性能测试轮数，默认1，范围1-1000000。
- --no_warm_up             : 默认执行一次完整TTS warm_up，配置后跳过warm_up。
- --interval               : 配置DeviceMonitor和HostMonitor采样间隔，单位毫秒(ms)，默认500ms。
- --model_name             : 指定日志和性能结果中的模型名称，默认`qwen3-tts`。
- --output_wav             : 可选，仅保存最后一个成功正式loop的waveform，采样率固定24000 Hz，写盘耗时不计入E2E和RTF。
- --dump_file              : 可选，将TTS性能结果保存到yaml文件。

TTS固定条件：仅支持streaming fixed-frame模式；language固定Chinese；speaker固定vivian；seed固定42；StatefulDecoder chunk固定12 frames；每个codec frame固定0.08秒；`token_per_second`默认3；仅支持单device；`output_wav`可选。正文token由固定seed随机生成，不依赖真实文本或tokenizer。TTS不支持`--mode`、`--input`、`--output`、`--text`、`--tokenizer`、`--language`、`--speaker`、`--seed`、`--chunk_size`、`--max_new_tokens`、`--max_frames`、`--greedy`、`--temperature`、`--top_k`和`--top_p`。

TTS运行期间，Warmup和每个正式loop通过codec frame进度条显示当前生成进度。每个正式loop的E2E、RTF、TTFA、Codec Frames/s和阶段明细写入当前工作目录的`perf_dumper.log`；终端只展示所有正式loop的Average结果；`--dump_file`指定的YAML同样保存Average结果。

4. 命令行llm_perf以配置文件启动
- --config, -c      : 使用配置文件启动perf程序，需要提前准备好配置文件，配置文件模板参考1.4节。
5. 目前支持通过DEBUG_DEV_INPUT宏开启SetInput输入的Debug模式，须在CMakeLists.txt添加如下编译选项：target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_DEV_INPUT)。
6. ASR功能编译说明：ASR功能由CMake选项`ENABLE_ASR`控制，当前Linux、Windows、Android平台默认开启；如需关闭可在配置CMake时添加`-DENABLE_ASR=OFF`。
7. TTS功能编译说明：TTS功能由CMake选项`ENABLE_TTS`控制，默认开启；如需关闭可在配置CMake时添加`-DENABLE_TTS=OFF`。


### 1.4 用法说明

#### 1.4.1 embedding模型转换

对于普通LLM/VLM/ASR单表embedding，可以使用`tools/llm_perf/convert_embed.py`将一个`.pt`权重转换为同名`.bin`文件：

```bash
python convert_embed.py --path <embedding_weight_path>
```

当前llm_perf的ASR模拟性能测试直接使用随机输入，不要求通过命令行传入embedding；但ASR模型的其他推理或资产准备流程如果使用单表`.pt` embedding，仍可使用该脚本转换。

> **Qwen3-TTS限制：**`tools/llm_perf/convert_embed.py`暂时不支持Qwen3-TTS。Qwen3-TTS perf同时依赖三类不同结构的embedding，其中CodePredictor权重包含15组按codebook排列的embedding，不能按普通单表embedding方式转换。

Qwen3-TTS需要以下三个二进制文件：

| llm_perf参数 | 文件 | 内容 |
| --- | --- | --- |
| `--embedding` | `quant_embedding.bin` | Talker codec token embedding |
| `--code_embedding` | `quant_embedding_code_predictor.bin` | CodePredictor的15组codebook embedding |
| `--text_embedding` | `text_embedding.bin` | 文本token embedding |

请进入Qwen3-TTS demo目录，通过`get_model.py`下载预编译HMM并自动准备C++/llm_perf所需的embedding资产：

```bash
cd ${HOUMO_EXAMPLES_PATH}/models/tts/qwen3-tts

python3 get_model.py \
  --type hmm \
  --model_name qwen3-tts \
  --model_size 0.6b-customvoice
```

如需使用1.7B CustomVoice，将`--model_size`改为：

```bash
--model_size 1.7b-customvoice
```

对于CustomVoice，`get_model.py --type hmm`下载预编译模型后会调用demo目录的Qwen3-TTS专用转换逻辑，一次性生成三份`.bin`文件。默认产物位置为：

```text
models/tts/qwen3-tts/output/${HOUMO_TARGET}/
  qwen3-tts-<model_size>_text_projection.hmm
  qwen3-tts-<model_size>_talker_prefill.hmm
  qwen3-tts-<model_size>_talker_decode.hmm
  qwen3-tts-<model_size>_code_predictor_prefill.hmm
  qwen3-tts-<model_size>_code_predictor_decode.hmm
  qwen3-tts-<model_size>_stateful_decoder.hmm
  hmquant/
    quant_embedding.bin
    quant_embedding_code_predictor.bin
    text_embedding.bin
```

这些文件和llm_perf参数的对应关系如下：

| llm_perf参数 | 默认产物 |
| --- | --- |
| `--text_projection` | `output/${HOUMO_TARGET}/qwen3-tts-<model_size>_text_projection.hmm` |
| `--talker_prefill` | `output/${HOUMO_TARGET}/qwen3-tts-<model_size>_talker_prefill.hmm` |
| `--talker_decode` | `output/${HOUMO_TARGET}/qwen3-tts-<model_size>_talker_decode.hmm` |
| `--code_predictor_prefill` | `output/${HOUMO_TARGET}/qwen3-tts-<model_size>_code_predictor_prefill.hmm` |
| `--code_predictor_decode` | `output/${HOUMO_TARGET}/qwen3-tts-<model_size>_code_predictor_decode.hmm` |
| `--stateful_decoder` | `output/${HOUMO_TARGET}/qwen3-tts-<model_size>_stateful_decoder.hmm` |
| `--embedding` | `output/${HOUMO_TARGET}/hmquant/quant_embedding.bin` |
| `--code_embedding` | `output/${HOUMO_TARGET}/hmquant/quant_embedding_code_predictor.bin` |
| `--text_embedding` | `output/${HOUMO_TARGET}/hmquant/text_embedding.bin` |

TTS perf不使用SpeechTokenizer HMM、真实文本tokenizer或模型目录中的`tokenizer.json`。如果上述三个`.bin`文件缺失，不要使用通用`convert_embed.py`逐个转换，应重新执行Qwen3-TTS目录下的`get_model.py --type hmm`，确保CodePredictor embedding的15组权重顺序正确。

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

TTS模拟性能测试使用示例（需要ENABLE_TTS编译选项开启）：
``` shell
# 首次运行前先在models/tts/qwen3-tts目录执行get_model.py --type hmm，
# 准备六个HMM和quant_embedding.bin、quant_embedding_code_predictor.bin、text_embedding.bin。
# 使用默认token_per_second=3、默认warm_up和单device 0，不保存WAV
./llm_perf --model_name qwen3-tts --text_projection qwen3-tts/text_projection.hmm --talker_prefill qwen3-tts/talker_prefill.hmm --talker_decode qwen3-tts/talker_decode.hmm --code_predictor_prefill qwen3-tts/code_predictor_prefill.hmm --code_predictor_decode qwen3-tts/code_predictor_decode.hmm --stateful_decoder qwen3-tts/stateful_decoder.hmm --embedding qwen3-tts/quant_embedding.bin --code_embedding qwen3-tts/quant_embedding_code_predictor.bin --text_embedding qwen3-tts/text_embedding.bin --tts_audio_length 10.0
# 保存最后一个正式loop的WAV并落盘性能结果
./llm_perf --model_name qwen3-tts --text_projection qwen3-tts/text_projection.hmm --talker_prefill qwen3-tts/talker_prefill.hmm --talker_decode qwen3-tts/talker_decode.hmm --code_predictor_prefill qwen3-tts/code_predictor_prefill.hmm --code_predictor_decode qwen3-tts/code_predictor_decode.hmm --stateful_decoder qwen3-tts/stateful_decoder.hmm --embedding qwen3-tts/quant_embedding.bin --code_embedding qwen3-tts/quant_embedding_code_predictor.bin --text_embedding qwen3-tts/text_embedding.bin --tts_audio_length 10.0 --token_per_second 3 --devices 0 --loop 3 --output_wav qwen3-tts/output.wav --dump_file qwen3-tts/perf_result.yaml
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
  # Qwen3-TTS模拟性能测试配置示例（requires ENABLE_TTS compilation option）
  - ModelName: qwen3-tts
    text_projection: qwen3-tts/text_projection.hmm
    talker_prefill: qwen3-tts/talker_prefill.hmm
    talker_decode: qwen3-tts/talker_decode.hmm
    code_predictor_prefill: qwen3-tts/code_predictor_prefill.hmm
    code_predictor_decode: qwen3-tts/code_predictor_decode.hmm
    stateful_decoder: qwen3-tts/stateful_decoder.hmm
    embedding: qwen3-tts/quant_embedding.bin
    code_embedding: qwen3-tts/quant_embedding_code_predictor.bin
    text_embedding: qwen3-tts/text_embedding.bin
    tts_audio_length: 10.0
    token_per_second: 3
    devices: "0"
    loop: 3
    output_wav: qwen3-tts/output.wav

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

如果是OHOS版本请执行：

```bash
# 1. 需要下载 OpenHarmony SDK，并设置 OHOS_SDK。
# 2. 需要下载 OHOS 版本 houmo-sdk，设置 HOUMO_SDK_PATH 为当前 OHOS 版本路径。
# 3. 需要下载 OHOS 版本 runtime-sdk，设置 TCIM_RUNTIME_PATH 为当前 OHOS 版本路径。
# 示例如下：
# export OHOS_SDK={the_path_of_ohos}/release/6.0-Release/linux/native
# export HOUMO_TARGET=xh2
# export HOUMO_SDK_PATH={the_path_of_ohos}/houmo-sdk
# export TCIM_RUNTIME_PATH={the_path_of_ohos}/houmo
./build_OHOS.sh
```

编译生成的llm_perf可执行文件在tools/bin目录下。

如果是windows11版本请执行：

```bat
rem 首次构建会自动构建并安装 houmo-llm-engine Windows 产物到 utils/lib
build_win.bat
```

**编译生成的llm_perf可执行文件在tools/bin目录下，在win11运行llm_perf.exe前需要参考 `tools/win_envs` 正确设置Windows环境变量：**
1. 推荐在根目录执行 `env.bat --set`，该脚本会调用 `tools/win_envs/set_environs.py` 根据 `tools/win_envs/env.json` 配置环境变量。
2. 执行完成后需要重新打开cmd窗口，使新的环境变量生效。
3. 确认 `PATH` 中包含 `%HOUMO_EXAMPLES_PATH%\utils\lib`，该目录保存 `houmo_infer.dll`、`tokenizer_lib.dll`、`kaldi-native-fbank-core.dll`、`samplerate.dll` 等运行时依赖。
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
