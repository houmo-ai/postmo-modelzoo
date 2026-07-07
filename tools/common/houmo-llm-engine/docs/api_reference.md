# Houmo Inference Framework - API 文档

> 版本: v0.2.0
> 日期: 2026-06-01

---

## 文件结构

```text
houmo-llm-engine/
├── include/
│   ├── base/
│   │   ├── houmo.h              # Token、配置、模型信息、采样参数、性能结构
│   │   └── tcim_utils.h         # TCIM 辅助工具
│   ├── core/
│   │   ├── context.h            # Context 请求级推理状态基类
│   │   ├── llm_model.h          # LLMModel 生成类模型基类
│   │   ├── vlm_model.h          # VLMModel 视觉语言模型基类
│   │   ├── asr_model.h          # ASRModel / ASRContext 语音识别基类
│   │   └── model_factory.h      # ModelFactory 模板工厂
│   └── modules/
│       ├── tokenizer.h          # HfTokenizer
│       ├── embedding.h          # Embedding
│       ├── sampler.h            # Sampler
│       ├── streaming_decoder.h  # StreamingDecoder
│       ├── image_processor.h    # HmImageProcessor
│       ├── audio_processor.h    # AudioProcessor
│       └── perf_profiler.h      # PerfProfiler
├── src/
│   ├── core/
│   │   ├── context.cc
│   │   ├── llm_model.cc
│   │   ├── vlm_model.cc
│   │   ├── asr_model.cc
│   │   ├── model_factory.cc
│   │   └── version.cc
│   └── modules/
│       ├── audio_processor.cc
│       ├── tokenizer.cc
│       ├── embedding.cc
│       ├── sampler.cc
│       ├── streaming_decoder.cc
│       ├── image_processor.cc
│       └── perf_profiler.cc
├── cmake/
│   └── platforms/
│       ├── windows.cmake        # Windows/MSVC 平台配置
│       ├── linux.cmake          # Linux 平台配置
│       └── android.cmake        # Android NDK 平台配置
├── tests/
│   ├── *_test.cc                # GTest 单元测试
│   └── data/                    # 测试图片、音频和 logits 数据
├── docs/                        # API、Pipeline 和模型适配文档
├── CMakeLists.txt               # CMake 构建入口
├── tcim_runtime.cmake           # TCIM Runtime 依赖配置
├── build_linux.sh               # Linux 构建脚本
├── build_ndk.sh                 # Android NDK 构建脚本
├── build_win.bat                # Windows / Visual Studio 构建脚本
├── test.sh                      # Linux 测试入口
├── get_3rdparty.py              # 第三方依赖准备脚本
└── convert_embed.py             # embedding 转换工具，安装到输出目录
```

---

## 基础数据结构

### Token 与异常

```cpp
namespace houmo {

using Token = int32_t;

constexpr Token TokenNull = -1;
constexpr Token TokenBos = -2;
constexpr Token TokenEos = -3;

class Exception : public std::runtime_error {
 public:
  explicit Exception(const std::string& msg);
};

}
```

### ModelType / ModelKind

```cpp
enum class ModelType {
  LLM,
  VLM,
  ASR,
  TTS,
};

enum class ModelKind {
  LLM,
  VLM,
  ASR,
  TTS,
};
```

### ModelConfig

模型配置承载运行参数、模型文件路径和扩展参数。

```cpp
struct ModelConfig {
  std::vector<int> devices = {0};
  int batch_size = 1;
  bool lazy_mode = false;

  std::string prefill_path;
  std::string decode_path;
  std::string embedding_path;
  std::string tokenizer_path;
  std::string vision_path;

  std::map<std::string, std::string> extra_params;
};
```

| 字段 | 说明 |
|------|------|
| `devices` | TCIM Runtime 使用的设备 ID 列表 |
| `batch_size` | 推理 batch 大小 |
| `lazy_mode` | 是否启用延迟加载 |
| `prefill_path` | Prefill `.hmm` 路径 |
| `decode_path` | Decode `.hmm` 路径 |
| `embedding_path` | Embedding 权重 `.bin` 路径 |
| `tokenizer_path` | Tokenizer JSON 路径，可按模型需要使用 |
| `vision_path` | Vision `.hmm` 路径，VLM 使用 |
| `extra_params` | 子类扩展参数，例如 ASR encode 路径、语言配置等 |

### ModelInfo

```cpp
struct ModelInfo {
  ModelType type;
  std::string model_name;
  int n_batch = 0;
  int n_vocab = 0;
  int n_embd = 0;
  int n_layer = 0;
  int n_ctx = 0;
  int prefill_length = 0;
  int kv_cache_layers = 0;
  int n_logits = 0;
};
```

### SamplingParams

```cpp
struct SamplingParams {
  float temperature = 1.0f;
  float top_p = 1.0f;
  int top_k = 1;
  float repetition_penalty = 1.0f;
  int penalty_last_n = 64;
  int max_tokens = 0;
  std::vector<Token> stop_tokens;
  float frequency_penalty = 0.0f;
  float presence_penalty = 1.5f;
  float min_p = 0.0f;
  bool greedy = false;

  bool add_bos = false;
  bool add_eos = false;

  std::string language = "auto";
};
```

`language` 是 ASR 专用选项：`"auto"` 表示尝试语言检测，或传入具体语言代码。

### PerfStats

生成类推理的通用性能指标。

```cpp
struct PerfStats {
  double prefill_time_ms = 0;
  double decode_time_ms = 0;
  double total_time_ms = 0;
  double ttft_ms = 0;
  double tpot_ms = 0;
  double tps = 0;
  double embedding_time_ms = 0;

  int n_input_tokens = 0;
  int n_output_tokens = 0;

  size_t cpu_memory_used = 0;
  size_t npu_memory_used = 0;
  size_t kv_cache_size = 0;
};
```

---

## ModelFactory

`ModelFactory<ModelT>` 是模板化模型工厂。每种模型基类拥有独立注册表，例如 `ModelFactory<LLMModel>` 和 `ModelFactory<ASRModel>` 互不混用。

```cpp
#include "core/model_factory.h"
```

### ModelSeries

`ModelSeries` 是框架内部的模型系列枚举，包含生成类和 ASR 类模型系列。实际可创建的模型取决于当前链接进最终程序的注册对象。

```cpp
enum class ModelSeries {
  kUnknown,
  kYourLLM,
  kYourVLM,
  kYourASR,
};
```

上面的 `kYour*` 只表示新增模型系列的占位写法；实际枚举值以当前源码和各模型目录注册为准。

### 字符串转换

```cpp
std::string ModelSeriesToString(ModelSeries series);
ModelSeries StringToModelSeries(const std::string& str);
```

### 工厂方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `Register(name, series, creator, description)` | `void` | 注册模型创建函数 |
| `Create(series, config)` | `std::unique_ptr<ModelT>` | 按模型系列创建实例 |
| `Create(name, config)` | `std::unique_ptr<ModelT>` | 按注册名创建实例 |
| `ListRegisteredTypes()` | `std::vector<std::string>` | 返回已注册名称 |
| `GetRegisteredModels()` | `std::vector<RegistryEntry>` | 返回注册详情 |
| `IsRegistered(series)` | `bool` | 判断系列是否已注册 |
| `IsRegistered(name)` | `bool` | 判断名称是否已注册 |

### 注册宏

```cpp
REGISTER_MODEL(BaseType, model_key, ModelSeries::kYourSeries,
               [](const ModelConfig& c) {
                 return std::make_unique<YourModel>(c);
               },
               "description");
```

注册宏使用静态对象在程序启动时注册模型。最终可执行文件链接静态注册对象时，需要确保目标文件不被链接器裁剪。

---

## LLMModel

`LLMModel` 是生成类模型基类，保存配置、Tokenizer、Embedding、TCIM prefill/decode module、输入 tensor map 和模型元信息。它不规定具体加载流程，子类负责加载 TCIM 模块和初始化输入。

```cpp
#include "core/llm_model.h"
```

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `type()` | `ModelType` | 默认返回 `ModelType::LLM` |
| `tokenize(text, add_bos, add_eos)` | `std::vector<Token>` | 通过 tokenizer 编码文本 |
| `token_to_str(token)` | `std::string` | 单 token 解码 |
| `tokens_to_str(tokens)` | `std::string` | token 序列解码 |
| `vocab_size()` | `int` | 词表大小 |
| `embedding_dim()` | `int` | Embedding 维度 |
| `max_ctx_available()` | `int` | 最大上下文长度 |
| `model_info()` | `ModelInfo` | 返回模型元信息 |
| `create_context(n_ctx)` | `std::unique_ptr<Context>` | 创建请求上下文，默认需子类覆盖 |
| `has_tokenizer()` | `bool` | tokenizer 是否已加载 |
| `bos_token_id()` | `Token` | BOS token |
| `eos_token_id()` | `Token` | EOS token |
| `tokenizer()` | `std::shared_ptr<HfTokenizer>` | 返回 tokenizer |
| `prefill_module()` | `std::shared_ptr<tcim::Module>` | 返回 prefill module |
| `decode_module()` | `std::shared_ptr<tcim::Module>` | 返回 decode module |
| `embedding()` | `std::shared_ptr<Embedding>` | 返回 embedding |
| `prefill_input_map()` | `std::map<std::string, tcim::Tensor>&` | Prefill 输入 tensor map |
| `decode_input_map()` | `std::map<std::string, tcim::Tensor>&` | Decode 输入 tensor map |
| `prefill_length()` | `int` | Prefill 长度 |
| `attn_idx_start()` | `int` | Attention 输入起始索引 |

---

## VLMModel

`VLMModel` 继承 `LLMModel`，增加 vision encode module 和视觉输入管理。

```cpp
#include "core/vlm_model.h"
```

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `type()` | `ModelType` | 返回 `ModelType::VLM` |
| `vision_module()` | `std::shared_ptr<tcim::Module>` | Vision encode module |
| `encode_image(image_data, width, height, channels)` | `std::vector<float16>` | 图像编码接口，默认由子类覆盖 |
| `create_context(n_ctx)` | `std::unique_ptr<Context>` | 创建 VLM 上下文，默认由子类覆盖 |
| `vision_input_map()` | `std::map<std::string, tcim::Tensor>&` | Vision 输入 tensor map |

---

## Context

`Context` 是生成类模型的请求级状态基类。

```cpp
#include "core/context.h"
```

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `prefill(tokens)` | `Token` | Prefill 阶段，默认返回 `TokenNull` |
| `decode(prev_token)` | `Token` | Decode 阶段，默认返回 `TokenNull` |
| `set_image(image_path)` | `void` | 单图输入接口，默认空实现 |
| `generate(prompt, params, callback)` | `void` | token 级流式生成，默认空实现 |
| `set_keep_history(keep)` | `void` | 设置是否保留多轮上下文 |
| `keep_history()` | `bool` | 返回历史保留状态 |
| `context_length()` | `int` | 当前上下文长度 |
| `reset()` | `void` | 重置上下文长度和已生成 token |
| `set_sampler(params)` | `void` | 创建采样器 |
| `sampler()` | `Sampler*` | 返回当前采样器 |
| `perf_stats()` | `PerfStats` | 返回性能统计 |
| `reset_perf_stats()` | `void` | 清空性能统计 |
| `profiler()` | `PerfProfiler&` | 返回层级性能统计器 |

---

## ASRModel / ASRContext

### ASRPerfInfo

ASR 专用性能指标，包含音频时长、RTF 和 decode TPS。

```cpp
struct ASRPerfInfo {
  float audio_load_time = 0.0f;
  float encode_time = 0.0f;
  float detect_lang_time = 0.0f;
  float prefill_time = 0.0f;
  float decode_time = 0.0f;
  float total_time = 0.0f;
  float ttft_time = 0.0f;
  int output_tokens = 0;
  int n_chunks = 0;
  float audio_duration = 0.0f;
  float overall_rtf = 0.0f;
  float inference_rtf = 0.0f;
  float decode_tps = 0.0f;
  float overall_tps = 0.0f;
};
```

### ASRModel

```cpp
#include "core/asr_model.h"
```

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `create_context(n_ctx)` | `std::unique_ptr<Context>` | 创建 ASR 上下文 |
| `sot_token_id()` | `Token` | Start-of-transcript token |
| `lang_token_id(language)` | `Token` | 语言 token |
| `transcribe_token_id()` | `Token` | 转写任务 token |
| `notimestamps_token_id()` | `Token` | 不输出时间戳 token |
| `eos_token_ids()` | `std::vector<Token>` | EOS token 集合 |
| `supports_language_detection()` | `bool` | 是否支持语言检测 |
| `n_mels()` | `int` | Mel bin 数 |
| `n_frames()` | `int` | Encoder 帧数 |
| `num_heads()` | `int` | Attention head 数 |
| `cache_max_len()` | `int` | KV cache 最大长度 |
| `num_decode_layers()` | `int` | Decoder 层数 |

### ASRContext

`ASRContext` 继承 `Context`，提供完整 ASR 转写接口和模板方法式性能打点。

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `Encode(mel_features, n_mels, n_frames)` | `std::vector<float16>` | Encoder 前向接口 |
| `DetectLanguage()` | `Token` | 语言检测 |
| `BuildPrompt(language_token)` | `std::vector<Token>` | 构造转写 prompt |
| `Transcribe(audio_path, params, callback)` | `void` | 从音频文件完整转写 |
| `set_language(language)` | `void` | 设置语言，支持 `auto` 或具体语言码 |
| `perf_info()` | `const ASRPerfInfo&` | 获取 ASR 性能统计 |
| `asr_model()` | `ASRModel*` | 返回 ASR model 指针 |

子类实现以下钩子，基类自动包裹 profiler scope：

| 钩子 | 对应阶段 |
|------|----------|
| `encode_preprocess_impl()` | `transcribe.encode.preprocess` |
| `encode_inference_impl()` | `transcribe.encode.inference` |
| `encode_postprocess_impl()` | `transcribe.encode.postprocess` |
| `detect_lang_preprocess_impl()` | `transcribe.detect_lang.preprocess` |
| `detect_lang_inference_impl()` | `transcribe.detect_lang.inference` |
| `detect_lang_postprocess_impl()` | `transcribe.detect_lang.postprocess` |
| `prefill_preprocess_impl()` | `transcribe.prefill.preprocess` |
| `prefill_inference_impl()` | `transcribe.prefill.inference` |
| `prefill_postprocess_impl()` | `transcribe.prefill.postprocess` |
| `decode_preprocess_impl()` | `transcribe.decode.preprocess` |
| `decode_inference_impl()` | `transcribe.decode.inference` |
| `decode_postprocess_impl()` | `transcribe.decode.postprocess` |

---

## AudioProcessor

`AudioProcessor` 为 ASR 模型提供音频前处理。

```cpp
#include "modules/audio_processor.h"
```

### 数据结构

```cpp
struct AudioData {
  std::vector<float> pcm;
  int sample_rate = 16000;
  float duration = 0.0f;
};

struct MelFeatures {
  std::vector<float16> data;
  int feature_dim = 0;
  int num_frames = 0;
  float duration = 0.0f;
};

enum class AudioFeatureMode {
  kCenterPad,
  kWhisper,
};

struct AudioProcessorConfig {
  int sample_rate = 16000;
  int n_mels = 80;
  int chunk_seconds = 30;
  int encoder_window_seconds = 30;
  int fft_size = 400;
  int hop_length = 160;
  int win_length = 400;
  int feature_threads = 4;
  AudioFeatureMode feature_mode = AudioFeatureMode::kCenterPad;
  float mel_fmin = 0.0f;
  float mel_fmax = 8000.0f;
};
```

### 方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `LoadAudio(path)` | `AudioData` | 读取 wav/mp3/flac 等音频，重采样到 16kHz，转单声道并归一化 |
| `ExtractFeatures(audio)` | `MelFeatures` | 计算 Mel Spectrogram，输出 FP16 特征 |
| `ChunkPCM(audio)` | `std::vector<AudioData>` | 按 `chunk_seconds` 切分 PCM，短块补零 |
| `Process(path)` | `std::vector<MelFeatures>` | 加载、切分、提取特征的一站式接口 |
| `feature_dim()` | `int` | 返回特征维度 |
| `sample_rate()` | `int` | 返回采样率 |
| `n_mels()` | `int` | 返回 Mel bin 数 |
| `config()` | `const AudioProcessorConfig&` | 返回配置 |

---

## 图像、Embedding、Tokenizer、采样和解码模块

### HmImageProcessor

`HmImageProcessor` 负责加载图像、resize/pad、归一化并输出模型所需像素数据。

```cpp
#include "modules/image_processor.h"
```

核心结构包括 `ImageDims` 和 `ProcessedImage`，核心接口为 `LoadAndProcess(image_path)`。

### Embedding

```cpp
#include "modules/embedding.h"
```

| 方法 | 说明 |
|------|------|
| `lookup(token)` | 单 token 查表，返回 embedding 指针 |
| `lookup(tokens, output)` | 批量查表写入输出 buffer |
| `vocab_size()` | 词表大小 |
| `hidden_dim()` | hidden 维度 |

### HfTokenizer

```cpp
#include "modules/tokenizer.h"
```

| 方法 | 说明 |
|------|------|
| `encode(text, add_bos, add_eos)` | 文本编码为 token |
| `decode(token)` | 单 token 解码 |
| `decode(tokens)` | token 序列解码 |
| `bos_token_id()` / `eos_token_id()` / `pad_token_id()` | 特殊 token |

### Sampler

```cpp
#include "modules/sampler.h"
```

`Sampler` 根据 `SamplingParams` 执行 greedy、top-k、top-p、min-p、temperature、frequency/presence penalty 和 repetition penalty 等采样逻辑。

### StreamingDecoder

```cpp
#include "modules/streaming_decoder.h"
```

`StreamingDecoder` 用滑动窗口处理 UTF-8 多字节字符，适合 token callback 流式输出。

---

## PerfProfiler

`PerfProfiler` 是层级性能统计器。

```cpp
#include "modules/perf_profiler.h"
```

| 方法 | 说明 |
|------|------|
| `start(path)` / `stop(path)` | 手动开始/结束阶段计时 |
| `scope(path)` | 返回 RAII 计时器 |
| `get_time_ms(path)` | 查询累计耗时 |
| `get_count(path)` | 查询阶段执行次数 |
| `get_avg_time_ms(path)` | 查询平均耗时 |
| `get_children(path)` | 查询子阶段 |
| `has_stage(path)` | 判断阶段是否存在 |
| `set_root_stage(stage)` | 设置根阶段，生成默认是 `generate`，ASR 可设为 `transcribe` |
| `set_input_tokens(n)` / `add_output_token()` | token 统计 |
| `record_ttft()` | 记录首 token 延迟 |
| `e2e_ms()` / `ttft_ms()` | 端到端和 TTFT 查询 |
| `prefill_tps()` / `decode_tps()` / `overall_tps()` | 吞吐指标 |
| `avg_decode_latency_ms()` | 平均 decode 延迟 |
| `to_perf_stats()` | 导出 `PerfStats` |
| `print_summary(format)` | 输出 Tree/Table/Compact 汇总 |
| `reset()` | 清空统计 |

---

## 构建与测试入口

`CMakeLists.txt` 构建共享库 `houmo_infer`，平台差异拆分在 `cmake/platforms/` 下：

| 文件 | 说明 |
|------|------|
| `cmake/platforms/windows.cmake` | MSVC 编译选项、Windows OpenCV 预编译库、DLL 安装开关 |
| `cmake/platforms/linux.cmake` | Linux 编译选项、OpenCV so 安装开关、pthread 配置 |
| `cmake/platforms/android.cmake` | Android NDK、目标 lib 路径和交叉编译配置 |

主要源码列表：

- Core sources：`version.cc`、`context.cc`、`vlm_model.cc`、`model_factory.cc`、`llm_model.cc`、`asr_model.cc`
- Module sources：`audio_processor.cc`、`tokenizer.cc`、`embedding.cc`、`sampler.cc`、`streaming_decoder.cc`、`image_processor.cc`、`perf_profiler.cc`
- 依赖：TCIM Runtime、tokenizer.cpp、OpenCV、kaldi-native-fbank、libsamplerate、GTest

构建/辅助入口：

| 文件 | 说明 |
|------|------|
| `build_linux.sh` | Linux 构建入口 |
| `build_ndk.sh` | Android NDK 构建入口 |
| `build_win.bat` | Windows / Visual Studio 构建入口，并准备音频 3rdparty 依赖 |
| `test.sh` | Linux 测试入口 |
| `get_3rdparty.py` | 准备第三方依赖 |
| `tcim_runtime.cmake` | TCIM Runtime include/lib 配置 |
| `convert_embed.py` | embedding 转换工具，随 install 输出到目标目录 |

安装规则会输出 `houmo_infer`、`tokenizer_lib`、音频依赖库、OpenCV 运行库（按平台开关）和 `convert_embed.py`。

当前 CTest 目标包括：

| 测试 | 覆盖内容 |
|------|----------|
| `TokenizerTest` | tokenizer 编解码 |
| `EmbeddingTest` | embedding 加载和查表 |
| `SamplerTest` | 采样策略 |
| `PerfProfilerTest` | 层级计时、TTFT、TPS、导出统计 |
| `AudioProcessorTest` | 音频加载、PCM 分块、Mel 特征 |
| `AudioProcessor128Test` | 128 mel 特征配置 |
