# Houmo Inference Framework

Houmo Inference Framework 是面向 Houmo NPU 的 C++17 推理基础库。当前目录提供模型抽象、请求上下文、TCIM Runtime 接入、Tokenizer、Embedding、采样、流式解码、图像/音频预处理和性能统计等公共能力；具体模型的加载、张量绑定和推理循环由各模型目录实现。

当前源码版本为 `0.1.0`，可通过 `houmo::version()` 或 `houmo::build_info()` 查询。

## 当前能力

- `LLMModel`：保存 LLM 公共配置、Tokenizer、Embedding、prefill/decode module 和输入 tensor map。
- `VLMModel`：在 `LLMModel` 基础上增加 vision module、vision input map 和 `encode_image()` 扩展点。
- `ASRModel` / `ASRContext`：定义 ASR 参数、转写接口，以及 encode、语言检测、prefill、decode 的统一性能打点模板。
- `Context`：保存请求级上下文长度、历史 token、Sampler 和性能统计；实际 `prefill()`、`decode()`、`generate()` 由模型子类覆盖。
- `ModelFactory<T>`：按基类类型维护独立注册表，支持静态注册和按名称或 `ModelSeries` 创建模型。
- `AudioProcessor`：使用 miniaudio 解码为目标采样率单声道 PCM，完成分块和 FP16 log-Mel 特征提取。
- `HmImageProcessor`：使用 stb_image 解码，支持等比缩放后右下补边或直接 resize，并可输出 `[C=3, T=2, H, W]` FP16 tensor。
- `PerfProfiler`：支持层级 stage、RAII 计时、TTFT、TPS 和 Tree/Table/Compact 输出。

## 目录结构

```text
├── include/
│   ├── base/           # 公共类型、版本接口和 TCIM 工具
│   ├── core/           # Context、LLM/VLM/ASR 基类和 ModelFactory
│   └── modules/        # Tokenizer、Embedding、Sampler、图像/音频、Profiler
├── src/
│   ├── core/           # 核心抽象实现
│   └── modules/        # 公共模块实现
├── docs/               # API、推理流程和模型适配文档
├── cmake/platforms/    # Linux、Android、Windows 平台配置
├── CMakeLists.txt
├── tcim_runtime.cmake
├── build_linux.sh
├── build_ndk.sh
├── build_win.bat
└── convert_embed.py
```

当前目录没有内置测试目标，也不包含具体模型实现。

## 依赖

- CMake >= 3.16.3
- C++17 编译器
- TCIM Runtime (`tcim_runtime_lite`)
- `$HOUMO_EXAMPLES_PATH/apis/common/tokenizer.cpp`
- `$HOUMO_EXAMPLES_PATH/apis/common/hpp/half/half.hpp`
- `$HOUMO_EXAMPLES_PATH/apis/common/hpp/stb/stb_image.h`
- `$HOUMO_EXAMPLES_PATH/apis/common/hpp/audio/miniaudio.h`
- Eigen 头文件，当前从 `$HOUMO_EXAMPLES_PATH/apis/common` 引入

必须设置 `HOUMO_EXAMPLES_PATH`。`TCIM_RUNTIME_PATH` 未设置时，CMake 会回退到：

```text
$DADAO_VENV/lib/python3.12/site-packages/tcim_lite
```

## 构建

### Linux

脚本只接受 Linux `x86_64` / `aarch64`，且当前仅支持 `HOUMO_TARGET=xh2`：

```bash
export HOUMO_EXAMPLES_PATH=/path/to/imodelzoo
export TCIM_RUNTIME_PATH=/path/to/tcim_lite
export HOUMO_TARGET=xh2
./build_linux.sh
```

Linux 平台的安装规则将 `houmo_infer` 和 `tokenizer_lib` 安装到：

```text
$HOUMO_EXAMPLES_PATH/tools/common/lib
```

`convert_embed.py` 安装到 CMake install prefix 的根目录。需要注意，平台文件为库目标设置了绝对安装目录，因此 `-DCMAKE_INSTALL_PREFIX` 不会改变库的安装位置。

### Android

Android 脚本要求 Linux `x86_64` 和 Android NDK r28c，固定使用 `arm64-v8a`、`android-35` 和 Ninja：

```bash
export HOUMO_EXAMPLES_PATH=/path/to/imodelzoo
export TCIM_RUNTIME_PATH=/path/to/android/tcim_lite
export NDK_PATH=/path/to/android-ndk-r28c
./build_ndk.sh release
```

也可传入 `debug` 和额外 CMake 参数：

```bash
./build_ndk.sh debug -DCMAKE_VERBOSE_MAKEFILE=ON
```

Android 库安装到 `$HOUMO_EXAMPLES_PATH/tools/common/android`，`convert_embed.py` 安装到当前目录下的 `android/` install prefix。

### Windows

先配置 `HOUMO_EXAMPLES_PATH`、`TCIM_RUNTIME_PATH` 及运行时 DLL 搜索路径，再执行：

```bat
build_win.bat
```

脚本使用 Visual Studio 2022 x64 和 `Release` 配置。DLL/LIB 安装到 `%HOUMO_EXAMPLES_PATH%\tools\common\lib`，`convert_embed.py` 安装到当前目录的 `bin\` install prefix。

### 手动 CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build
```

当前 `CMakeLists.txt` 不提供 `BUILD_TESTS` 选项或 CTest 目标。

## Embedding 转换

`convert_embed.py` 将 PyTorch `.pt` embedding 权重导出为同名 FP16/原始 dtype `.bin` 文件。当前脚本要求 `HOUMO_TARGET=xh2` 且依赖 PyTorch：

```bash
export HOUMO_TARGET=xh2
python convert_embed.py --path /path/to/embedding.pt
```

脚本可读取 tensor、包含 `weight` 的字典、仅含一个 tensor 的字典、常见 `*.weight` state dict，或带 `.weight` 属性的对象。BF16 权重会先转换为 FP16。

## 重要实现约束

- 基类不负责加载模型。模型子类需要创建 `DevManager`、`WeightManager`、TCIM modules、Tokenizer、Embedding 和输入 tensor map。
- `Context::generate()`、`ASRContext::Transcribe()` 是扩展接口，基础库没有默认推理循环。
- `SamplingParams` 中部分字段是预留配置。当前 `Sampler` 实际处理 repetition penalty、presence penalty、top-k、temperature 和 top-p，但最终使用 `argmax`，不是随机抽样；`frequency_penalty`、`min_p` 和 `greedy` 当前未参与实现。
- `HmImageProcessor::LoadAndProcess()` 加载失败时返回填充值为 `114` 的目标尺寸 RGB 图像，而不是抛出异常。
- `AudioProcessor::ChunkPCM()` 不补零；固定 encoder window 的补零或截断发生在 `ExtractFeatures()` 内部。
- 性能统计默认启用。编译时定义 `HOUOMO_ENABLE_PROFILING=0` 可切换到 no-op `PerfProfiler`。

## 文档

- [API Reference](docs/api_reference.md)
- [Inference Pipeline](docs/inference_pipeline.md)
- [New Model Adaptation Guide](docs/new_model_adaptation_guide.md)

## License

Copyright (c) 2026 HOUMO AI. Licensed under the Apache License, Version 2.0.
