# Houmo Inference Framework

A pure C++ inference framework for LLM, VLM and ASR models, optimized for Houmo NPU using TCIM Runtime.

## Features

- **Multi-model support**: LLM (Qwen3, Qwen2.5, DeepSeek), VLM (Qwen3.5, Qwen3-VL), ASR (Whisper, Qwen3-ASR, GLM-ASR)
- **Streaming generation**: Token-level callback mode for real-time output
- **Multi-turn dialogue**: Context-level history management
- **Vision understanding**: VLM models support image input and understanding
- **Speech recognition**: ASR models with encode-decoder architecture and RTF metrics
- **Performance profiling**: Built-in hierarchical performance analyzer with RTF/throughput metrics
- **Template method pattern**: ASR profiling hooks — new models get profiling automatically
- **Factory pattern**: Runtime dynamic model instance creation

## Supported Models

| Model | Type | Description |
|-------|------|-------------|
| Qwen3-0.6B / 4B | LLM | Qwen3 pure-text models |
| Qwen3.5-0.8B / 2B / 4B | VLM | Qwen3.5 multimodal models |
| Qwen3-VL-4B / 8B | VLM | Qwen3-VL vision-language models |
| Qwen3-VL-MoE | VLM | Qwen3-VL MoE model |
| Qwen3.6-27B | VLM | Qwen3.6 multimodal model |
| Qwen2.5-7B | LLM | Qwen2.5 pure-text model |
| Qwen2.5-VL-7B | VLM | Qwen2.5 vision-language model |
| DeepSeek-8B | LLM | DeepSeek model |
| CoPaw-Flash-9B | VLM | CoPaw-Flash model |
| Whisper-large-v3-turbo | ASR | Whisper ASR with language detection |
| Qwen3-ASR-1.7B | ASR | Qwen3-ASR with per-loop encoding |
| GLM-ASR-nano | ASR | GLM-ASR with PCM chunking |

## Quick Start

### Prerequisites

- C++17 compiler
- CMake >= 3.16
- TCIM Runtime (NPU backend)
- OpenCV (optional, for image processing)

### Build

#### Linux 编译

```bash
cd tools/common/houmo-llm-engine/
./build_linux.sh
```

编译生成的可执行文件在 `bin/` 目录下。

#### Android 编译

需要先准备以下环境：

1. 下载 [Android NDK](https://developer.android.google.cn/ndk/downloads/index.html?hl=ro)（推荐 r28c 版本），解压到 toolchains 目录下
2. 下载 Android 版本 houmo-tcim-runtime-xh2

设置环境变量后执行编译：

```bash
# 设置 NDK 路径
export NDK_PATH=/path/to/android-ndk-r28c

# 设置 TCIM Runtime 路径（Android 版本）
export TCIM_RUNTIME_PATH=/path/to/houmo-tcim-runtime-xh2

# 执行编译
./build_ndk.sh
```

编译生成的可执行文件在 `android/` 目录下。

#### 手动 CMake 编译

```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

### Environment Setup

```bash
# Set model path
export HM_ENGINE_PATH=/path/to/houmo-llm-engine

# Set TCIM Runtime path
export TCIM_RUNTIME_PATH=$DADAO_VENV/lib/python3.12/site-packages/tcim_lite
```

### Run Inference

```bash
# LLM inference
./sample_infer --model qwen3_llm --prompt "介绍一下端侧AI"

# VLM image understanding
./sample_infer --model qwen3_vlm --image test.jpg --prompt "描述这张图片"

# Multi-turn dialogue
./sample_infer --model qwen3_llm --multi-turn --prompt "你好"

# Whisper ASR
./sample_whisper_asr --audio audio.mp3 \
  --encode models/whisper-large-v3-turbo/whisper_encode.hmm \
  --prefill models/whisper-large-v3-turbo/whisper_prefill.hmm \
  --decode models/whisper-large-v3-turbo/whisper_decode.hmm \
  --tokenizer tokenizers/whisper-large-v3-turbo/tokenizer.json

# Qwen3-ASR
./sample_qwen3_asr --audio audio.mp3 \
  --encode models/qwen3-asr-1.7b/qwen3-asr-1.7b_encode.hmm \
  --prefill models/qwen3-asr-1.7b/qwen3-asr-1.7b_prefill.hmm \
  --decode models/qwen3-asr-1.7b/qwen3-asr-1.7b_decode.hmm \
  --tokenizer tokenizers/qwen3-asr-1.7b/tokenizer.json \
  --embedding models/qwen3-asr-1.7b/hmquant/quant_embedding.bin

# GLM-ASR
./sample_glm_asr --audio audio.mp3 \
  --encode models/glm-asr-nano-2512/glm-asr-nano-2512_encode.hmm \
  --prefill models/glm-asr-nano-2512/glm-asr-nano-2512_prefill.hmm \
  --decode models/glm-asr-nano-2512/glm-asr-nano-2512_decode.hmm \
  --tokenizer tokenizers/glm-asr-nano-2512/tokenizer.json \
  --embedding models/glm-asr-nano-2512/hmquant/quant_embedding.bin

# Show model info
./sample_infer --model qwen3_llm --info
```

## Usage

### Basic Usage (C++ API)

```cpp
#include "core/model_factory.h"
#include "modules/streaming_decoder.h"

int main() {
    // 1. Configure model
    houmo::ModelConfig config;
    config.devices = {0};
    config.prefill = "models/qwen3-4b/qwen3-4b_prefill.hmm";
    config.decode = "models/qwen3-4b/qwen3-4b_decode.hmm";
    config.embedding = "models/qwen3-4b/quant_embedding.bin";
    config.tokenizer = "models/tokenizers/Qwen3-4B/tokenizer.json";

    // 2. Create model (factory pattern)
    auto model = houmo::ModelFactory::Create("qwen3_llm", config);
    auto ctx = model->create_context();

    // 3. Streaming generation
    auto tokens = model->tokenize("你好", false, false);
    houmo::SamplingParams params;
    params.max_tokens = 256;

    houmo::StreamingDecoder decoder(model->tokenizer());
    ctx->generate(tokens, params, [&](houmo::Token token) {
        std::cout << decoder.decode(token) << std::flush;
        return true;
    });

    return 0;
}
```

### Multi-turn Dialogue

```cpp
ctx->set_keep_history(true);

// Round 1
auto tokens1 = model->tokenize("1 + 1 = ?", false, false);
ctx->generate(tokens1, params, callback);

// Round 2 (auto-retains context)
auto tokens2 = model->tokenize("2 + 2 = ?", false, false);
ctx->generate(tokens2, params, callback);

// Reset
ctx->reset();
```

### VLM Image Understanding

```cpp
#include "models/qwen3_vlm_model.h"

auto ctx = model->create_context();

// Set image
auto* vlm_ctx = dynamic_cast<houmo::Qwen3VLMContext*>(ctx.get());
vlm_ctx->set_image("test.jpg");

// Generate
auto tokens = model->tokenize("描述这张图片", false, false);
ctx->generate(tokens, params, callback);
```

### ASR Speech Recognition

```cpp
#include "core/model_factory.h"
#include "models/whisper_model.h"

houmo::ModelConfig config;
config.devices = {0};
config.prefill = "models/whisper/whisper_prefill.hmm";
config.decode = "models/whisper/whisper_decode.hmm";
config.tokenizer = "tokenizers/whisper/tokenizer.json";
config.extra_params["encode"] = "models/whisper/whisper_encode.hmm";

auto model = houmo::ModelFactory<houmo::ASRModel>::Create(
    houmo::ModelSeries::kWhisperASR, config);
auto ctx = model->create_context();
auto* whisper_ctx = dynamic_cast<houmo::WhisperContext*>(ctx.get());
whisper_ctx->set_audio_processor(16000, 30, 30);

std::vector<houmo::Token> all_tokens;
houmo::SamplingParams params;
whisper_ctx->Transcribe("audio.mp3", params,
    [&all_tokens](houmo::Token token) {
        all_tokens.push_back(token);
        return true;
    });

std::cout << static_cast<houmo::WhisperModel*>(model.get())
             ->tokenizer()->decode(all_tokens);

// Performance output
ctx->profiler().print_summary();
const auto& info = ctx->perf_info();
std::cout << "Overall RTF: " << info.overall_rtf
          << ", Inference RTF: " << info.inference_rtf << "\n";
```

### Performance Profiling

```cpp
// Print formatted report
ctx->profiler().print_summary();

// Programmatic access
houmo::PerfStats stats = ctx->profiler().to_perf_stats();
std::cout << "Prefill: " << stats.prefill_time_ms << " ms\n";
std::cout << "TTFT: " << stats.ttft_ms << " ms\n";
std::cout << "TPS: " << stats.tps << " tokens/s\n";
```

## Architecture

```
┌───────────────────────────────────────────────┐
│              User Code Layer                   │
│  sample_infer.cc / sample_*_asr.cc / User App │
├───────────────────────────────────────────────┤
│              C++ API Layer                     │
│  LLMModel / VLMModel / ASRModel / Context     │
│  PerfProfiler (hierarchical performance)      │
├───────────────────────────────────────────────┤
│              Module Layer                      │
│  Tokenizer / Embedding / AudioProcessor        │
│  ImageProcessor / StreamingDecoder / Sampler  │
├───────────────────────────────────────────────┤
│              Backend Layer                     │
│           TCIM Runtime (NPU)                   │
└───────────────────────────────────────────────┘
```

### Class Hierarchy

```
LLMModel (base)
  ├── Qwen3LLMModel
  └── VLMModel (VLM base)
        ├── Qwen35MLLMModel
        └── Qwen3VLMModel

ASRModel (ASR base)
  ├── WhisperModel
  ├── Qwen3AsrModel
  └── GlmAsrModel

Context (base)
  ├── Qwen3Context (LLM)
  ├── Qwen35MLLMContext (VLM)
  ├── Qwen3VLMContext (VLM)
  └── ASRContext (ASR base — template method profiling)
        ├── WhisperContext
        ├── Qwen3AsrContext
        └── GlmAsrContext
```

## Project Structure

```
├── include/
│   ├── base/           # Core types (Token, ModelConfig, etc.)
│   ├── core/           # Base classes (LLMModel, VLMModel, Context)
│   ├── modules/        # Modules (Tokenizer, Embedding, Sampler, etc.)
│   └── models/         # Model implementations
├── src/                # Source files
├── tests/              # GTest unit tests
├── sample/             # Example programs
├── models/             # Model files (.hmm, .bin, .json)
└── 3rdparty/           # Third-party dependencies
```

## Model Files

Models use `.hmm` (Houmo Model) format:

| File | Description |
|------|-------------|
| `*_prefill.hmm` | Prefill model (processes prompt) |
| `*_decode.hmm` | Decode model (autoregressive generation) |
| `*_encode.hmm` | Encoder model (ASR audio encoding) |
| `embedding.bin` | Embedding weight table |
| `vision.hmm` | Vision encode (VLM only) |
| `tokenizer.json` | Tokenizer vocabulary |

## Testing

```bash
cd build
ctest --output-on-failure

# Run specific test
./qwen3_llm_test
./qwen3_vlm_test
./whisper_test
./perf_profiler_test
```

## Documentation

- [API Reference](docs/api_reference.md) - API signatures and usage
- [Inference Pipeline](docs/inference_pipeline.md) - Pipeline flow details
- [New Model Adaptation Guide](docs/new_model_adaptation_guide.md) - How to add new models
- [Whisper Implementation Plan](docs/whisper_implementation_plan.md) - Whisper ASR implementation

## License

Copyright (c) 2026 HOUMO AI. Licensed under the Apache License, Version 2.0.
