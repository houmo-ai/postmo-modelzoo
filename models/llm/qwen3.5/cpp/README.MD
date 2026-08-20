# Qwen3.5 MLLM C++ 推理示例

本目录展示如何在 Linux、Windows 11 和 Android Native 环境中注册当前目录 `include/` 和 `src/` 下的 Qwen3.5 MLLM 子类实现，并通过 `ModelFactory` 和 `$HOUMO_EXAMPLES_PATH/cmake/houmo_engine.cmake` 复用 `houmo_engine` 的基础框架能力。

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
| `$HOUMO_EXAMPLES_PATH/cmake/houmo_engine.cmake` | houmo_engine 依赖配置模块 |
| `CMakeLists.txt` | demo 编译配置 |
| `build_linux.sh` | Linux 编译脚本 |
| `build_win.bat` | Windows 11 编译脚本 |
| `build_ndk.sh` | Android 交叉编译脚本 |

## 调用关系

`src/qwen35_mllm_model.cc` 使用 `REGISTER_MODEL` 在程序启动时注册模型：

```cpp
REGISTER_MODEL(LLMModel, qwen35_mllm, ModelSeries::kQwen35MLLM,
    [](const ModelConfig& c) { return std::make_unique<Qwen35MLLMModel>(c); },
    "Qwen3.5 multimodal MLLM");
```

核心调用流程：

1. 解析 `--config`、`--model_name`、`--model_size`、`--prefill`、`--decode`、`--visual`、`--embedding`、`--tokenizer`、`--prompt`、`--image` 等参数。
2. 构造 `houmo::ModelConfig`。
3. 调用 `ModelFactory<LLMModel>::Create(ModelSeries::kQwen35MLLM, config)`，通过注册表创建模型。
4. 通过返回的 `LLMModel` 基类指针调用 `model->create_context()`；图像输入时再转换为 `houmo::Qwen35MLLMContext`。
5. 如传入图片，调用 `qwen_ctx->set_images(image_paths)`。
6. 使用 `model->tokenize(prompt, false, false)` 编码 prompt。
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

target_link_libraries(demo PRIVATE houmo_engine)
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

请确认 `PATH` 中包含 `%HOUMO_EXAMPLES_PATH%\utils\lib`，该目录保存 `houmo_infer.dll`、`tokenizer_lib.dll` 等运行时依赖。

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

Windows 构建默认使用 `Release`，并使用 `%NUMBER_OF_PROCESSORS%` 并行编译。首次构建时，如果 `$HOUMO_EXAMPLES_PATH/utils/lib` 下缺少 `houmo_infer` 相关产物，`$HOUMO_EXAMPLES_PATH/cmake/houmo_engine.cmake` 会自动触发 `utils/cpp/houmo_engine/build_win.bat` 构建依赖。

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
| `--config <path>` | `config.yaml` 路径，默认当前模型目录下的 `config.yaml` |
| `--model_name <name>` | 模型名称，默认从 config 读取 |
| `--model_size <size>` | 模型尺寸，默认从 config 读取 |
| `--prefill <path>` | Prefill `.hmm` 模型路径，默认按模型名自动生成 |
| `--decode <path>` | Decode `.hmm` 模型路径，默认按模型名自动生成 |
| `--visual <path>` | Dynamic visual `.hmm` 文件或 gear 目录，默认使用 `visual.hmm` |
| `--embedding <path>` | Embedding `.bin` 文件路径，默认按模型名自动生成 |
| `--tokenizer <path>` | Tokenizer JSON 或 tokenizer 目录，默认按 `modelscope_repo` 自动生成 |
| `--prompt <text>` / `--question <text>` | 输入 prompt，默认 `描述这些图片` |
| `--system_prompt <text>` | 覆盖默认 system prompt |
| `--image <path>` / `--image_path <path>` | 图片路径，可重复传入多张图片；不传时使用默认 beach.jpeg |
| `--max-tokens <num>` / `--max-new-tokens <num>` | 最大生成 token 数 |
| `--device <id>` / `--ndevice <num>` | 设备 ID/数量别名，当前只支持 1 个设备 |
| `--batch <num>` | batch 大小，当前只支持 1 |
| `-h, --help` | 打印帮助信息 |

`--tokenizer` 传入 `tokenizer.json` 文件时，程序会自动使用其父目录加载完整
tokenizer 配置，并从同一目录读取 `preprocessor_config.json`；也可以直接传入模型目录。

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

也可以像 Python demo 一样只指定模型名称和尺寸：

```bash
./build/demo \
  --model_name qwen3.5 \
  --model_size 9b \
  --max-tokens 256
```

## Dynamic 图文理解

Qwen3.5 dynamic visual 使用模型目录内的 ImageProcess，不修改公共
`houmo_engine` ImageProcessor，也不引入 OpenCV、llama.cpp 或 ggml 依赖。
ImageProcess 从 tokenizer/model 目录下的 `preprocessor_config.json` 读取
`image_mean` 和 `image_std`。如果配置文件无法读取、JSON 解析失败、字段缺失或值
非法，则回退到 Transformers `Qwen2VLImageProcessor` 默认的 CLIP mean/std。
图片经过动态 resize、归一化和 merge-major patchify 后，送入 dynamic visual
HMM 的五个输入：`pixel_values`、`position_ids`、`position_weights`、
`rotary_position_ids` 和 `attention_mask`。

如果只提供一个最大 gear，可以传入 `visual.hmm`；如果提供多 gear，可以将
`--visual` 指向包含 `_m96`、`_m196`、`_m384`、`_m704`、`_m1536` 文件的目录。
程序会在启动时加载实际存在的 gear，并根据图片真实 image token 数选择最小可容纳
的 visual HMM。

推荐的多 gear 文件名为：

```text
output/xh2/qwen3.5-9b_visual_m96.hmm
output/xh2/qwen3.5-9b_visual_m196.hmm
output/xh2/qwen3.5-9b_visual_m384.hmm
output/xh2/qwen3.5-9b_visual_m704.hmm
output/xh2/qwen3.5-9b_visual_m1536.hmm
```

当前 C++ 实现还兼容 Qwen3.8 风格的图契约：tensor 名称匹配会忽略
`.hmcc.format` 后缀，recurrent state output 缺失时不会强制绑定 prefill output，
prefill/decode 的输入会按语义名称处理，而不是只依赖固定输入下标。

图像输入时需要额外传入 `--visual` 和至少一个 `--image`。demo 会自动在 prompt
前插入图片占位 token：

```text
<|vision_start|><|image_pad|><|vision_end|>
```

示例：

```bash
./build/demo \
  --prefill ../output/xh2/qwen3.5-9b_prefill.hmm \
  --decode ../output/xh2/qwen3.5-9b_decode.hmm \
  --visual ../output/xh2/qwen3.5-9b_visual.hmm \
  --embedding ../output/xh2/hmquant/quant_embedding.bin \
  --tokenizer ../Qwen3.5-9B \
  --image ./test.jpg \
  --prompt "请描述这张图片。" \
  --max-tokens 256
```

多 gear 目录示例：

```bash
./build/demo \
  --prefill ../output/xh2/qwen3.5-9b_prefill.hmm \
  --decode ../output/xh2/qwen3.5-9b_decode.hmm \
  --visual ../output/xh2/gears \
  --embedding ../output/xh2/hmquant/quant_embedding.bin \
  --tokenizer ../Qwen3.5-9B \
  --image ./test.jpg \
  --prompt "请描述这张图片。"
```

## 输出指标

demo 推理结束后会输出：

- `qwen_ctx->profiler().print_summary()` 的层级耗时。
- `qwen_ctx->perf_stats()` 中的输入 token、输出 token、prefill 时间、decode 时间、总耗时、TTFT 和 TPS。

## 注意事项

- 文本生成不需要传入 `--visual`。
- 只要传入 `--image`，就必须同时传入 dynamic visual `.hmm` 文件或 gear 目录。
- `--tokenizer` 需要与模型编译时使用的 tokenizer 保持一致。
- `--embedding` 需要使用框架要求的量化 embedding 格式。
- `$HOUMO_EXAMPLES_PATH/cmake/houmo_engine.cmake` 会链接 `houmo_infer`、`tcim_runtime_lite`，并复用 `houmo_engine` 的 include 和 lib 配置；Windows 平台下还会导入 `tokenizer_lib.lib/.dll`，并在依赖缺失时自动调用 `utils/cpp/houmo_engine/build_win.bat` 构建。

## 免责声明

本目录下示例仅供技术研究和集成验证使用。使用者在运行前应自行检查模型许可证、输入数据和部署环境的合规性。
