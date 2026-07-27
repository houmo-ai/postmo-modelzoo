# GLM-ASR C++ 推理示例

本示例展示如何使用 C++ 在后摩系列芯片（XH2）上运行 GLM-ASR 自动语音识别模型推理。支持 Linux、Windows 11 和 Android Native 环境。

## 目录

- [1. 文件说明](#1-文件说明)
- [2. 流程说明](#2-流程说明)
- [3. 参数说明](#3-参数说明)
- [4. 快速开始](#4-快速开始)
- [5. 免责声明](#5-免责声明)

## 1. 文件说明

| 文件 | 说明 |
|------|------|
| `demo.cc` | 主程序入口，GLM-ASR 推理 demo |
| `src/glm_asr_model.cc` | GLM-ASR 模型实现 |
| `include/glm_asr_model.h` | GLM-ASR 模型头文件 |
| `hm_engine.cmake` | houmo-llm-engine 依赖配置模块（可复用） |
| `CMakeLists.txt` | CMake 编译配置 |
| `build_linux.sh` | Linux 编译脚本 |
| `build_win.bat` | Windows 11 编译脚本 |
| `build_ndk.sh` | Android 交叉编译脚本 |

功能支持：
- 芯片型号：xh2
- 大道软件版本要求：xh2 ≥ v1.0.0
- 模型：glm-asr-nano-2512
- 支持长音频分块处理（默认 30s）
- 仅支持 16kHz 采样率音频输入

## 2. 流程说明

1. 解析命令行参数，加载模型配置
2. 加载 Encoder / Prefill / Decode 三个 `.hmm` 模型
3. 加载 Embedding 和 Tokenizer
4. 加载音频文件，分块提取 Mel 频谱特征
5. Encoder 处理音频特征，生成 audio embedding
6. 构造 Prompt（融合 audio embedding 和 text token）
7. Prefill 处理 prompt 序列，得到第一个输出 token
8. Decode 循环生成 token，直到命中 eos token
9. StreamingDecoder 实时解码输出文本

性能指标：

| 指标 | 说明 |
|------|------|
| Audio Duration | 音频时长（秒） |
| Chunks | 分块数量 |
| Output Tokens | 输出 token 总数 |
| Overall RTF | 端到端实时率（<1 表示实时处理） |
| Inference RTF | 纯推理实时率 |
| Decode TPS | Decode 阶段速度（tokens/s） |
| Overall TPS | 端到端速度（tokens/s） |

## 3. 参数说明

| 参数 | 说明 |
|------|------|
| `--audio <path>` | 音频文件路径（.wav/.mp3 等），**必填** |
| `--encode <path>` | Encoder 模型路径（.hmm） |
| `--prefill <path>` | Prefill 模型路径（.hmm） |
| `--decode <path>` | Decode 模型路径（.hmm） |
| `--tokenizer <path>` | Tokenizer 目录路径 |
| `--embedding <path>` | Embedding 文件路径（.bin 或 .pt） |
| `-h, --help` | 打印帮助信息 |

用法示例：

```bash
./demo \
    --audio audio.wav \
    --encode ../output/xh2/glm-asr-nano-2512_encode.hmm \
    --prefill ../output/xh2/glm-asr-nano-2512_prefill.hmm \
    --decode ../output/xh2/glm-asr-nano-2512_decode.hmm \
    --tokenizer ../GLM-ASR-Nano-2512 \
    --embedding ../output/xh2/hmquant/quant_embedding.bin
```

## 4. 快速开始

### 4.1 环境准备

```bash
export HOUMO_TARGET=xh2
export HOUMO_EXAMPLES_PATH=/hmdd/imodelzoo
export TCIM_RUNTIME_PATH=/usr/local/houmo
export TCIM_BACKEND=Xh2HalBackend
```

Windows 环境建议参考仓库根目录的 `env.bat` 和 `tools/win_envs`：

```bat
cd /d "%HOUMO_EXAMPLES_PATH%"
env.bat --set
```

`env.bat --set` 会根据 `tools/win_envs/env.json` 设置 `HOUMO_EXAMPLES_PATH`、`TCIM_RUNTIME_PATH`、`HOUMO_SDK_PATH`、`OPENCV_PATH`、`PATH` 等变量。执行完成后需要重新打开 cmd 窗口，使环境变量生效。

请确认 `PATH` 中包含 `%HOUMO_EXAMPLES_PATH%\utils\lib`，该目录保存 `houmo_infer.dll`、`tokenizer_lib.dll`、`kaldi-native-fbank-core.dll`、`samplerate.dll` 等运行时依赖。

Android 交叉编译还需：

```bash
export NDK_PATH=/path/to/android-ndk-r28c
export HOUMO_SDK_PATH=/path/to/houmo-drv-xh2
```

### 4.2 编译

**Linux：**

```bash
cd /hmdd/imodelzoo/models/asr/glm-asr/cpp
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

或直接执行脚本：

```bash
bash build_linux.sh
```

**Android：**

```bash
bash build_ndk.sh
```

**Windows：**

```bat
build_win.bat
```

Windows 构建默认使用 `Release`，并使用 `%NUMBER_OF_PROCESSORS%` 并行编译。首次构建时，如果 `$HOUMO_EXAMPLES_PATH/utils/lib` 下缺少 `houmo_infer` 相关产物，`hm_engine.cmake` 会自动触发 `utils/cpp/houmo_engine/build_win.bat` 构建依赖。

编译产物位于 `build/Release/demo.exe`，安装产物位于 `../bin/demo.exe`。

**CMakeLists.txt 依赖说明：**

当前 CMakeLists.txt 通过 `hm_engine.cmake` 模块自动管理 `houmo_infer` 依赖。Linux/Android平台下，`hm_engine.cmake` 会在首次构建或源文件变更时自动调用 houmo-llm-engine 的 `build_linux.sh`/`build_ndk.sh` 编译依赖 so 库。Windows平台下，`hm_engine.cmake` 会复用 `$HOUMO_EXAMPLES_PATH/utils/lib` 中的 `houmo_infer.lib/.dll`、`tokenizer_lib.lib/.dll`；如缺少这些产物，会自动调用 houmo-llm-engine 的 `build_win.bat` 构建依赖。

`hm_engine.cmake` 也可被其他工程复用：

```cmake
include(path/to/hm_engine.cmake)
target_link_libraries(your_target PRIVATE hm_engine)
```

### 4.3 运行

确保模型文件已准备：
- `../output/xh2/glm-asr-nano-2512_encode.hmm`
- `../output/xh2/glm-asr-nano-2512_prefill.hmm`
- `../output/xh2/glm-asr-nano-2512_decode.hmm`
- `../GLM-ASR-Nano-2512/`（tokenizer 目录）
- `../output/xh2/hmquant/quant_embedding.bin`

```bash
./build/demo \
    --audio audio.wav \
    --encode ../output/xh2/glm-asr-nano-2512_encode.hmm \
    --prefill ../output/xh2/glm-asr-nano-2512_prefill.hmm \
    --decode ../output/xh2/glm-asr-nano-2512_decode.hmm \
    --tokenizer ../GLM-ASR-Nano-2512 \
    --embedding ../output/xh2/hmquant/quant_embedding.bin
```

## 5. 免责声明

本目录下的模型转换、量化 Python 脚本仅供技术研究使用。使用者在运行前应自行检查模型许可证合规性。使用本代码产生的任何结果或风险由使用者自行承担。
