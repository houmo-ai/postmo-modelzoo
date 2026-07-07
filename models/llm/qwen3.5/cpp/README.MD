# Qwen3.5 MLLM C++ 推理示例

本目录展示如何在 Linux、Windows 11 和 Android Native 环境中使用 C++ 调用当前目录 `include/` 和 `src/` 下的 Qwen3.5 MLLM 子类实现，并通过 `hm_engine.cmake` 复用 `houmo-llm-engine` 的基础框架能力。

示例支持：

- 文本生成：只传入 prefill/decode/embedding/tokenizer。
- 图文理解：额外传入 vision 模型和一个或多个图片路径。
- Token 级流式输出：通过 `StreamingDecoder` 将 token 流解码为文本。
- 性能统计：输出 `PerfProfiler` 汇总和 `PerfStats` 指标。

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo.cc` | Qwen3.5 MLLM 推理 demo 主程序 |
| `include/qwen35_mllm_model.h` | `Qwen35MLLMModel` / `Qwen35MLLMContext` 声明 |
| `src/qwen35_mllm_model.cc` | Qwen3.5 MLLM 子类实现和静态注册 |
| `hm_engine.cmake` | houmo-llm-engine 依赖配置模块 |
| `CMakeLists.txt` | demo 编译配置 |
| `build_linux.sh` | Linux 编译脚本 |
| `build_win.bat` | Windows 11 编译脚本 |
| `build_ndk.sh` | Android 交叉编译脚本 |

## 调用关系

`demo.cc` 直接包含本目录头文件：

```cpp
#include "qwen35_mllm_model.h"
```

核心调用流程：

1. 解析 `--prefill`、`--decode`、`--vision`、`--embedding`、`--tokenizer`、`--prompt`、`--image` 等参数。
2. 构造 `houmo::ModelConfig`。
3. 创建 `houmo::Qwen35MLLMModel model(config)`。
4. 调用 `model.create_context()` 并转换为 `houmo::Qwen35MLLMContext`。
5. 如传入图片，调用 `qwen_ctx->set_images(image_paths)`。
6. 使用 `model.tokenize(prompt, false, false)` 编码 prompt。
7. 调用 `qwen_ctx->generate(tokens, params, callback)` 流式生成。
8. 输出 profiler 和 `PerfStats` 指标。

`CMakeLists.txt` 会把本地实现编进 demo：

```cmake
add_executable(demo
  demo.cc
  src/qwen35_mllm_model.cc
)

target_include_directories(demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(demo PRIVATE hm_engine)
```

## 环境准备

Linux 构建前需要准备：

```bash
export HOUMO_TARGET=xh2
export HOUMO_EXAMPLES_PATH=/hmdd/imodelzoo
export TCIM_RUNTIME_PATH=/path/to/tcim_runtime
export TCIM_BACKEND=Xh2HalBackend
```

Windows 环境建议参考仓库根目录的 `env.bat` 和 `tools/win_envs`：

```bat
cd /d "%HOUMO_EXAMPLES_PATH%"
env.bat --set
```

`env.bat --set` 会根据 `tools/win_envs/env.json` 设置 `HOUMO_EXAMPLES_PATH`、`TCIM_RUNTIME_PATH`、`HOUMO_SDK_PATH`、`OPENCV_PATH`、`PATH` 等变量。执行完成后需要重新打开 cmd 窗口，使环境变量生效。

请确认 `PATH` 中包含 `%HOUMO_EXAMPLES_PATH%\tools\common\lib`，该目录保存 `houmo_infer.dll`、`tokenizer_lib.dll` 等运行时依赖。

Android 交叉编译还需要：

```bash
export NDK_PATH=/path/to/android-ndk-r28c
export HOUMO_SDK_PATH=/path/to/houmo-drv-xh2
```

## 编译

### Linux

```bash
bash build_linux.sh
```

或手动编译：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Android

```bash
bash build_ndk.sh
```

### Windows

```bat
build_win.bat
```

Windows 构建默认使用 `Release`，并使用 `%NUMBER_OF_PROCESSORS%` 并行编译。首次构建时，如果 `$HOUMO_EXAMPLES_PATH/tools/common/lib` 下缺少 `houmo_infer` 相关产物，`hm_engine.cmake` 会自动触发 `tools/common/houmo-llm-engine/build_win.bat` 构建依赖。

Linux 编译产物位于：

```text
build/demo
```

Windows 编译产物位于：

```text
build/Release/demo.exe
../bin/demo.exe
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `--prefill <path>` | Prefill `.hmm` 模型路径，必填 |
| `--decode <path>` | Decode `.hmm` 模型路径，必填 |
| `--vision <path>` | Vision `.hmm` 模型路径，图像输入时必填 |
| `--embedding <path>` | Embedding `.bin` 文件路径，必填 |
| `--tokenizer <path>` | Tokenizer JSON 或 tokenizer 目录路径，必填 |
| `--prompt <text>` | 输入 prompt，默认是中文自我介绍问题 |
| `--image <path>` | 图片路径，可重复传入多张图片 |
| `--max-tokens <num>` | 最大生成 token 数，默认 256 |
| `--device <id>` | 设备 ID，默认 0 |
| `-h, --help` | 打印帮助信息 |

## 文本生成示例

```bash
./build/demo \
  --prefill ../output/xh2/qwen3.5-2b_prefill.hmm \
  --decode ../output/xh2/qwen3.5-2b_decode.hmm \
  --embedding ../output/xh2/hmquant/quant_embedding.bin \
  --tokenizer ../tokenizer.json \
  --prompt "介绍一下端侧多模态模型。" \
  --max-tokens 256
```

## 图文理解示例

图像输入时需要额外传入 `--vision` 和至少一个 `--image`。demo 会自动在 prompt 前插入图片占位 token：

```text
<|vision_start|><|image_pad|><|vision_end|>
```

运行示例：

```bash
./build/demo \
  --prefill ../output/xh2/qwen3.5-2b_prefill.hmm \
  --decode ../output/xh2/qwen3.5-2b_decode.hmm \
  --vision ../output/xh2/qwen3.5-2b_visual_448x448x2.hmm \
  --embedding ../output/xh2/hmquant/quant_embedding.bin \
  --tokenizer ../tokenizer.json \
  --image ./test.jpg \
  --prompt "请描述这张图片。" \
  --max-tokens 256
```

多图输入可以重复传入 `--image`：

```bash
./build/demo \
  --prefill ../output/xh2/qwen3.5-2b_prefill.hmm \
  --decode ../output/xh2/qwen3.5-2b_decode.hmm \
  --vision ../output/xh2/qwen3.5-2b_visual_448x448x2.hmm \
  --embedding ../output/xh2/hmquant/quant_embedding.bin \
  --tokenizer ../tokenizer.json \
  --image ./a.jpg \
  --image ./b.jpg \
  --prompt "比较这两张图片的差异。"
```

## 输出指标

demo 推理结束后会输出：

- `qwen_ctx->profiler().print_summary()` 的层级耗时。
- `qwen_ctx->perf_stats()` 中的输入 token、输出 token、prefill 时间、decode 时间、总耗时、TTFT 和 TPS。

## 注意事项

- 文本生成不需要传入 `--vision`。
- 只要传入 `--image`，就必须同时传入有效的 `--vision`。
- `--tokenizer` 需要与模型编译时使用的 tokenizer 保持一致。
- `--embedding` 需要使用框架要求的量化 embedding 格式。
- `hm_engine.cmake` 会链接 `houmo_infer`、`tcim_runtime_lite`，并复用 `houmo-llm-engine` 的 include 和 lib 配置；Windows 平台下还会导入 `tokenizer_lib.lib/.dll`，并在依赖缺失时自动调用 `tools/common/houmo-llm-engine/build_win.bat` 构建。

## 免责声明

本目录下示例仅供技术研究和集成验证使用。使用者在运行前应自行检查模型许可证、输入数据和部署环境的合规性。
