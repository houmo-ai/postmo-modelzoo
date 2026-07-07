# Houmo Inference Framework

Houmo Inference Framework 是面向 Houmo NPU 的 C++ 推理框架，基于 TCIM Runtime 提供模型加载、上下文管理、推理执行、采样解码、Embedding 读取、多模态预处理和性能统计等通用能力。

本目录承载框架源码、公共头文件、模块实现、测试和开发文档。具体模型能力由各模型目录中的实现和构建配置决定，本文档只描述框架能力，不维护具体模型支持列表。

## 核心能力

- **统一基础类型**：`Token`、`ModelConfig`、`ModelInfo`、`SamplingParams`、`PerfStats` 等定义在 `include/base/houmo.h`。
- **模型抽象**：`LLMModel`、`VLMModel`、`ASRModel` 分别覆盖文本生成、视觉语言和语音识别模型的公共接口。
- **上下文抽象**：`Context` 管理单次推理状态、采样器、生成历史、性能统计；`ASRContext` 扩展转写流程和 ASR 专用性能指标。
- **模型工厂**：`ModelFactory<T>` 提供类型安全的静态注册和运行时创建能力，LLM/VLM/ASR 可分别注册到对应工厂实例。
- **通用模块**：Tokenizer、Embedding、Sampler、StreamingDecoder、ImageProcessor、AudioProcessor、PerfProfiler 等模块可被不同模型复用。
- **性能统计**：`PerfProfiler` 支持层级 stage、RAII scope、TTFT、TPS、平均延迟和树形/表格输出。
- **ASR 支持**：提供音频加载、重采样、单声道转换、PCM 分块、Mel 特征提取、转写模板方法和 RTF/TPS 指标计算。

## 目录结构

```text
├── include/
│   ├── base/           # 基础类型、配置、异常和 TCIM 工具
│   ├── core/           # LLM/VLM/ASR 模型基类、Context、Factory
│   └── modules/        # Tokenizer、Embedding、Sampler、Audio/Image、Profiler
├── src/
│   ├── core/           # 核心抽象实现
│   └── modules/        # 通用模块实现
├── tests/              # GTest 单元测试和测试数据
├── docs/               # API、Pipeline 和模型适配说明
├── cmake/
│   └── platforms/      # Windows/Linux/Android 平台专用 CMake 配置
├── CMakeLists.txt      # CMake 构建入口
├── build_linux.sh      # Linux 构建脚本
├── build_ndk.sh        # Android NDK 构建脚本
├── build_win.bat       # Windows 构建脚本
└── test.sh             # 测试脚本
```

## 主要组件

### 基础类型

`include/base/houmo.h` 定义框架公共数据结构：

- `ModelConfig`：运行设备、batch、lazy mode、prefill/decode/embedding/tokenizer/vision 路径和扩展参数。
- `ModelInfo`：模型类型、名称、batch、词表、hidden size、layer、context、prefill 长度和 logits 信息。
- `SamplingParams`：temperature、top-p、top-k、重复惩罚、停止 token、最大生成长度和 ASR language 选项。
- `PerfStats`：生成类推理的 prefill/decode/total/TTFT/TPS 和 token 数指标。

### LLM/VLM 抽象

`LLMModel` 保存模型配置、Tokenizer、Embedding、TCIM prefill/decode module、输入 tensor map 和模型元信息。子类负责具体加载流程和推理细节。

`VLMModel` 继承 `LLMModel`，增加 vision module、vision input map 和 `encode_image()` 接口，用于视觉语言模型扩展。

`Context` 是生成类模型的请求级状态对象，提供：

- `prefill()`、`decode()`、`generate()` 推理接口
- `set_keep_history()`、`reset()` 状态管理
- `set_sampler()` 采样器管理
- `perf_stats()` 和 `profiler()` 性能统计访问

### ASR 抽象

`ASRModel` 是语音识别模型基类，保存 ASR 配置和公共模型参数，并要求子类实现：

- `create_context()` 创建 ASR 上下文
- `sot_token_id()`、`lang_token_id()`、`transcribe_token_id()`、`notimestamps_token_id()`、`eos_token_ids()` 等转写 token 接口
- `supports_language_detection()` 语言检测能力声明

`ASRContext` 继承 `Context`，管理 ASR 请求级状态，提供：

- `Encode()` 音频特征编码
- `DetectLanguage()` 语言检测
- `BuildPrompt()` 构造转写 prompt
- `Transcribe()` 完整音频转写
- `set_language()` 设置语言
- `perf_info()` 获取 `ASRPerfInfo`

ASRContext 内部使用模板方法封装打点：子类只实现 `encode_*_impl`、`detect_lang_*_impl`、`prefill_*_impl`、`decode_*_impl` 钩子，基类自动记录 `transcribe.encode.*`、`transcribe.prefill.*`、`transcribe.decode.*` 等阶段耗时。

### 音频处理

`AudioProcessor` 提供 ASR 前处理：

1. `LoadAudio(path)`：读取音频，重采样到 16kHz，转单声道，归一化到 `[-1, 1]`。
2. `ChunkPCM(audio)`：按固定秒数切分 PCM，短块自动补零。
3. `ExtractFeatures(audio)`：计算 Mel Spectrogram，输出 FP16 特征。
4. `Process(path)`：一站式完成加载、切分和特征提取。

`AudioProcessorConfig` 控制 sample rate、mel bins、chunk 秒数、encoder window、FFT、hop length、window length、feature threads、feature mode 和 mel 频率范围。

### 性能统计

`PerfProfiler` 支持：

- `start(path)` / `stop(path)` 手动打点
- `scope(path)` RAII 自动打点
- `get_time_ms()`、`get_count()`、`get_avg_time_ms()` 查询耗时
- `set_input_tokens()`、`add_output_token()` 记录 token 数
- `record_ttft()`、`overall_tps()`、`decode_tps()` 等吞吐指标
- `print_summary()` 输出 Tree/Table/Compact 视图

## 构建

### 环境依赖

- C++17 编译器
- CMake >= 3.16
- TCIM Runtime (`tcim_lite`)
- OpenCV（图像处理）
- tokenizer.cpp 和 half.hpp
- 音频依赖：miniaudio、libsamplerate、kaldi-native-fbank
- GTest（单元测试）

需要设置 `HOUMO_EXAMPLES_PATH`，并确保 `TCIM_RUNTIME_PATH` 可由 `tcim_runtime.cmake` 找到。Windows平台建议参考根目录 [env.bat](../../../env.bat) 和 [tools/win_envs](../../win_envs/) 正确设置环境变量。

### Linux

```bash
./build_linux.sh
```

### Android

```bash
export NDK_PATH=/path/to/android-ndk
export TCIM_RUNTIME_PATH=/path/to/tcim_runtime
./build_ndk.sh
```

### Windows

Windows平台建议先在仓库根目录执行：

```bat
env.bat --set
```

该脚本会调用 [tools/win_envs/set_environs.py](../../win_envs/set_environs.py)，并根据 [tools/win_envs/env.json](../../win_envs/env.json) 设置 `HOUMO_EXAMPLES_PATH`、`TCIM_RUNTIME_PATH`、`HOUMO_SDK_PATH`、`OPENCV_PATH`、`PATH` 等变量。执行完成后需要重新打开cmd窗口，使环境变量生效。

确认环境变量生效后，在当前目录执行：

```bat
build_win.bat
```

Windows构建默认使用 `Release`，并使用 `%NUMBER_OF_PROCESSORS%` 进行并行编译。构建和安装产物会输出到 `$HOUMO_EXAMPLES_PATH/tools/common/lib`，该目录应已由 `env.bat --set` 加入 `PATH`，以便运行依赖 `houmo_infer.dll`、`tokenizer_lib.dll`、`kaldi-native-fbank-core.dll`、`samplerate.dll` 等动态库的工具。

### 手动 CMake

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

## 测试

```bash
./test.sh
```

或手动执行：

```bash
ctest --test-dir build --output-on-failure
```

Windows 使用 Visual Studio multi-config 生成器，手动执行 CTest 时需要指定配置：

```bat
ctest --test-dir build -C Release --output-on-failure
```

当前测试覆盖 Tokenizer、Embedding、Sampler、PerfProfiler、AudioProcessor 和 128 mel 音频特征流程。

## 开发文档

- [API Reference](docs/api_reference.md) - 核心 API、数据结构和模块接口
- [Inference Pipeline](docs/inference_pipeline.md) - LLM/VLM/ASR 推理流程和性能打点
- [New Model Adaptation Guide](docs/new_model_adaptation_guide.md) - 新模型适配规范

## License

Copyright (c) 2026 HOUMO AI. Licensed under the Apache License, Version 2.0.
