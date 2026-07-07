# 新模型适配指南

本文档说明如何基于 Houmo Inference Framework 适配新的生成类模型（LLM/VLM）或语音识别模型（ASR）。文档只描述框架要求和通用步骤，不维护具体模型支持列表。

---

## 1. 适配前先确认模型类型

| 模型类型 | Model 基类 | Context 基类 | 典型入口 | 说明 |
|----------|------------|--------------|----------|------|
| LLM | `LLMModel` | `Context` | `generate()` | 文本 token 自回归生成 |
| VLM | `VLMModel` | `Context` | `generate()` | 在 LLM 基础上增加视觉编码和图像 embedding 注入 |
| ASR | `ASRModel` | `ASRContext` | `Transcribe()` | 音频特征编码 + decoder 转写 |

选择规则：

- 纯文本生成模型继承 `LLMModel`。
- 视觉语言模型继承 `VLMModel`，不要直接继承 `LLMModel` 后自行复制 vision 成员。
- 语音识别模型继承 `ASRModel`，上下文继承 `ASRContext`，复用 ASR 模板方法打点。
- 不要在子类中重新声明基类已有状态，例如 `config_`、`tokenizer_`、`embedding_`、`prefill_module_`、`decode_module_`、`context_length_`、`profiler_`。

---

## 2. 通用文件组织

模型实现通常放在模型自己的 `cpp/` 目录中，框架基类位于当前工程。当前 `houmo-llm-engine` 目录结构如下：

```text
houmo-llm-engine/
├── include/
│   ├── base/              # 基础类型、配置、异常和 TCIM 工具
│   ├── core/              # LLM/VLM/ASR 基类、Context、Factory
│   └── modules/           # Tokenizer、Embedding、Sampler、Audio/Image、Profiler
├── src/
│   ├── core/              # 基类实现
│   └── modules/           # 通用模块实现
├── cmake/
│   └── platforms/         # Windows/Linux/Android 平台专用 CMake 配置
├── tests/                 # GTest 单元测试和测试数据
├── docs/                  # API、Pipeline 和模型适配说明
├── CMakeLists.txt         # CMake 构建入口
├── tcim_runtime.cmake     # TCIM Runtime 依赖配置
├── build_linux.sh         # Linux 构建脚本
├── build_ndk.sh           # Android NDK 构建脚本
├── build_win.bat          # Windows / Visual Studio 构建脚本
├── test.sh                # 测试入口
├── get_3rdparty.py        # 第三方依赖准备脚本
└── convert_embed.py       # embedding 转换工具
```

新模型的文件建议与模型目录放在一起：

```text
include/<model_name>_model.h
src/<model_name>_model.cc
tests/<model_name>_test.cc
```

如果所在模型目录已有固定组织方式，优先遵循该目录现有结构。

---

## 3. ModelConfig 使用规范

所有模型都通过 `ModelConfig` 传入运行参数和文件路径。

```cpp
houmo::ModelConfig config;
config.devices = {0};
config.batch_size = 1;
config.lazy_mode = false;
config.prefill_path = "path/to/prefill.hmm";
config.decode_path = "path/to/decode.hmm";
config.embedding_path = "path/to/embedding.bin";
config.tokenizer_path = "path/to/tokenizer.json";
config.vision_path = "path/to/vision.hmm";
config.extra_params["encode_path"] = "path/to/encode.hmm";
```

字段使用建议：

| 字段 | LLM | VLM | ASR | 说明 |
|------|-----|-----|-----|------|
| `devices` | 必需 | 必需 | 必需 | 不要在模型实现中硬编码设备号 |
| `prefill_path` | 必需 | 必需 | 通常必需 | decoder prefill 模型 |
| `decode_path` | 必需 | 必需 | 通常必需 | decoder decode 模型 |
| `embedding_path` | 按需 | 按需 | 按需 | token embedding 权重 |
| `tokenizer_path` | 按需 | 按需 | 按需 | tokenizer JSON |
| `vision_path` | 不使用 | 必需 | 不使用 | vision encoder 模型 |
| `extra_params` | 按需 | 按需 | 常用 | ASR encode 路径、语言等扩展参数 |

---

## 4. LLM 适配

### 4.1 头文件模板

```cpp
#pragma once

#include "core/context.h"
#include "core/llm_model.h"

namespace houmo {

class YourLLMContext : public Context {
 public:
  explicit YourLLMContext(LLMModel* model, int n_ctx);

  Token prefill(const std::vector<Token>& tokens) override;
  Token decode(Token prev_token) override;
  void generate(const std::vector<Token>& prompt,
                const SamplingParams& params,
                std::function<bool(Token)> callback) override;
  void reset() override;

 private:
  Token do_prefill_inference(const std::vector<Token>& tokens, Sampler* sampler);
  Token do_decode_inference(Token prev_token, Sampler* sampler);

  void prefill_preprocess_chunk(int chunk,
                                const std::vector<Token>& tokens,
                                int32_t seq_length,
                                int prefill_length);
  void prefill_inference_chunk();
  Token prefill_postprocess(Sampler* sampler, int32_t seq_length);

  void decode_preprocess(Token prev_token);
  void decode_inference();
  Token decode_postprocess(Sampler* sampler);
};

class YourLLMModel : public LLMModel {
 public:
  explicit YourLLMModel(const ModelConfig& config);

  std::unique_ptr<Context> create_context(int n_ctx = 0) override;
  void ClearKVCache();

 private:
  void load();
  void init_prefill_inputs();
  void init_decode_inputs();
};

}  // namespace houmo
```

### 4.2 load() 要求

`load()` 由模型子类实现，至少完成：

1. 使用 `config_.devices` 创建 `tcim::DevManager`。
2. 使用 `*dev_manager_` 创建 `tcim::Module::WeightManager`。
3. 加载 prefill module。
4. 加载 decode module。
5. 按模型需要共享或初始化 KV Cache。
6. 加载 embedding。
7. 加载 tokenizer。
8. 填充 `info_`、`prefill_length_`、`attn_idx_start_` 等基类成员。
9. 初始化 `prefill_input_map_` 和 `decode_input_map_`。

不要在子类中重新声明 `dev_manager_`、`weight_manager_`、`prefill_module_`、`decode_module_`。

### 4.3 generate() 要求

`generate()` 应包含完整性能统计：

```cpp
void YourLLMContext::generate(const std::vector<Token>& prompt,
                              const SamplingParams& params,
                              std::function<bool(Token)> callback) {
  profiler_.reset();
  profiler_.start("generate");
  profiler_.set_input_tokens(static_cast<int>(prompt.size()));

  set_sampler(params);

  Token token;
  {
    auto t = profiler_.scope("generate.prefill");
    token = prefill(prompt);
  }

  profiler_.record_ttft();

  if (!callback(token)) {
    profiler_.stop("generate");
    perf_stats_ = profiler_.to_perf_stats();
    return;
  }

  while (params.max_tokens <= 0 ||
         generated_ids_.size() < static_cast<size_t>(params.max_tokens)) {
    {
      auto t = profiler_.scope("generate.decode");
      token = decode(token);
    }
    profiler_.add_output_token();

    if (!callback(token)) break;
  }

  profiler_.stop("generate");
  perf_stats_ = profiler_.to_perf_stats();
}
```

模型应根据自身 EOS/BOS/stop token 和 context 上限补充停止条件。

---

## 5. VLM 适配

VLM 继承 `VLMModel`，复用 `LLMModel` 的 prefill/decode 基础成员，并增加视觉编码。

### 5.1 头文件差异

```cpp
#include "core/vlm_model.h"

class YourVLMModel : public VLMModel {
 public:
  explicit YourVLMModel(const ModelConfig& config);

  std::unique_ptr<Context> create_context(int n_ctx = 0) override;
  std::vector<float16> encode_image(const uint8_t* image_data,
                                    int width,
                                    int height,
                                    int channels) override;

 private:
  void load();
  void init_vision_inputs();
  void init_prefill_inputs();
  void init_decode_inputs();
};
```

### 5.2 VLM Context 额外状态

```cpp
class YourVLMContext : public Context {
 public:
  void set_image(const std::string& image_path) override;
  void set_images(const std::vector<std::string>& image_paths);

 private:
  void run_vision();
  void vision_preprocess(int image_idx);
  void vision_inference();
  void vision_postprocess(int image_idx);

  std::vector<std::string> image_paths_;
  std::vector<float16> flat_image_embeds_;
  bool use_vlm_ = false;
};
```

### 5.3 VLM Prefill 插入点

VLM 通常在 prefill 的 token embedding 之前完成：

1. `HmImageProcessor::LoadAndProcess()` 读取和预处理图像。
2. 设置 `vision_input_map_`。
3. 执行 `vision_module_->Run()` 和同步。
4. 从 vision 输出中取出 image embeddings。
5. 扩展或替换 prompt 中的 image token embedding。
6. 设置多模态 position ids 或其他模型特有输入。

建议打点路径：

```text
generate.vision
generate.vision.preprocess
generate.vision.inference
generate.vision.postprocess
generate.prefill.common_setup
generate.prefill.preprocess_chunk
generate.prefill.inference_chunk
generate.prefill.postprocess
```

---

## 6. ASR 适配

ASR 模型必须继承 `ASRModel`，上下文必须继承 `ASRContext`。这样可以复用音频处理、转写接口和模板方法打点。

### 6.1 头文件模板

```cpp
#pragma once

#include "core/asr_model.h"

namespace houmo {

class YourASRModel : public ASRModel {
 public:
  explicit YourASRModel(const ModelConfig& config);

  std::unique_ptr<Context> create_context(int n_ctx = 0) override;

  Token sot_token_id() const override;
  Token lang_token_id(const std::string& language) const override;
  Token transcribe_token_id() const override;
  Token notimestamps_token_id() const override;
  std::vector<Token> eos_token_ids() const override;
  bool supports_language_detection() const override;

 private:
  void load();
  void init_encode_inputs();
  void init_prefill_inputs();
  void init_decode_inputs();
};

class YourASRContext : public ASRContext {
 public:
  explicit YourASRContext(ASRModel* model, int n_ctx);

  std::vector<float16> Encode(const std::vector<float>& mel_features,
                              int n_mels,
                              int n_frames) override;
  Token DetectLanguage() override;
  std::vector<Token> BuildPrompt(Token language_token) override;
  void Transcribe(const std::string& audio_path,
                  const SamplingParams& params,
                  ASRTokenCallback callback) override;
  void set_language(const std::string& language) override;

 private:
  void encode_preprocess_impl(const std::vector<float>& mel,
                              int n_mels,
                              int n_frames) override;
  void encode_inference_impl() override;
  void encode_postprocess_impl() override;

  void detect_lang_preprocess_impl() override;
  void detect_lang_inference_impl() override;
  Token detect_lang_postprocess_impl() override;

  void prefill_preprocess_impl(const std::vector<Token>& tokens) override;
  void prefill_inference_impl() override;
  Token prefill_postprocess_impl() override;

  void decode_preprocess_impl(Token prev_token) override;
  void decode_inference_impl() override;
  Token decode_postprocess_impl() override;
};

}  // namespace houmo
```

### 6.2 ASRModel::load() 要求

ASR 模型通常需要加载三类 TCIM 模块：

1. encode module：处理 Mel 特征，路径可通过 `extra_params` 传入。
2. prefill module：decoder prefill。
3. decode module：decoder 自回归 decode。

同时需要：

- 设置 `n_mels_`、`n_frames_`、`num_heads_`、`cache_max_len_`、`num_decode_layers_`。
- 加载 tokenizer 和可选 embedding。
- 初始化 encoder、prefill、decode 输入 tensor map。
- 初始化 decoder cache。

### 6.3 Transcribe() 推荐结构

```cpp
void YourASRContext::Transcribe(const std::string& audio_path,
                                const SamplingParams& params,
                                ASRTokenCallback callback) {
  profiler_.reset();
  profiler_.set_root_stage("transcribe");
  profiler_.start("transcribe");

  set_language(params.language);

  std::vector<MelFeatures> features;
  float audio_duration = 0.0f;
  {
    auto t = profiler_.scope("transcribe.audio_load");
    AudioProcessor processor;
    features = processor.Process(audio_path);
    for (const auto& f : features) audio_duration += f.duration;
  }

  for (const auto& feature : features) {
    do_encode(/* mel float data */, feature.feature_dim, feature.num_frames);

    Token language_token = do_detect_language();
    auto prompt = BuildPrompt(language_token);

    Token token = do_prefill(prompt);
    profiler_.record_ttft();

    if (!callback(token)) break;

    while (true) {
      token = do_decode(token);
      profiler_.add_output_token();

      // 子类应检查 eos_token_ids()、max_tokens 和 callback 返回值。
      if (!callback(token)) break;
    }
  }

  profiler_.stop("transcribe");
  fill_perf_info(audio_duration);
}
```

注意：`do_encode()` 当前接收 `std::vector<float>`，如果 `AudioProcessor::Process()` 返回 FP16 MelFeatures，子类需要按模型实现决定是否保留 FP16、转换为 float，或在 `Transcribe()` 中直接调用自定义特征路径。

### 6.4 ASR 打点规范

ASR 子类不要手写重复计时逻辑，优先调用基类模板方法：

```text
do_encode()
  ├── transcribe.encode.preprocess
  ├── transcribe.encode.inference
  └── transcribe.encode.postprocess

do_detect_language()
  ├── transcribe.detect_lang.preprocess
  ├── transcribe.detect_lang.inference
  └── transcribe.detect_lang.postprocess

do_prefill()
  ├── transcribe.prefill.preprocess
  ├── transcribe.prefill.inference
  └── transcribe.prefill.postprocess

do_decode()
  ├── transcribe.decode.preprocess
  ├── transcribe.decode.inference
  └── transcribe.decode.postprocess
```

`fill_perf_info(audio_duration)` 会计算 `ASRPerfInfo`，包括 `overall_rtf`、`inference_rtf`、`decode_tps`、`overall_tps`。

### 6.5 AudioProcessor 使用

```cpp
AudioProcessorConfig audio_config;
audio_config.sample_rate = 16000;
audio_config.n_mels = 80;
audio_config.chunk_seconds = 30;
audio_config.encoder_window_seconds = 30;

auto processor = AudioProcessor(audio_config);
auto features = processor.Process(audio_path);
```

需要 128 mel 或特定 padding 方式时，调整 `n_mels` 和 `feature_mode`。

---

## 7. 模型注册

模型实现文件末尾使用 `REGISTER_MODEL` 注册。

### LLM/VLM 注册

```cpp
#include "core/model_factory.h"

REGISTER_MODEL(LLMModel, your_llm_key, ModelSeries::kYourLLM,
               [](const ModelConfig& c) {
                 return std::make_unique<YourLLMModel>(c);
               },
               "Your LLM model");
```

VLM 仍注册到 `ModelFactory<LLMModel>`，因为 `VLMModel` 继承 `LLMModel`。

### ASR 注册

```cpp
#include "core/model_factory.h"

REGISTER_MODEL(ASRModel, your_asr_key, ModelSeries::kYourASR,
               [](const ModelConfig& c) {
                 return std::make_unique<YourASRModel>(c);
               },
               "Your ASR model");
```

如果新增模型系列，需要同步更新：

1. `ModelSeries` 枚举。
2. `ModelSeriesToString()`。
3. `StringToModelSeries()`。
4. 对应模型实现文件中的 `REGISTER_MODEL`。
5. CMake 源文件列表和测试目标。

---

## 8. 测试规范

### 8.1 通用测试原则

- 测试文件放在 `tests/` 或模型目录约定位置。
- 缺少真实模型文件时使用 `GTEST_SKIP()`，不要让测试崩溃。
- 模型加载和推理路径使用 `ASSERT_NO_THROW` 包裹。
- 路径检查函数必须覆盖测试依赖的所有文件。
- `ModelConfig` 必须显式设置 `devices` 和 `lazy_mode`。

### 8.2 LLM/VLM 测试项

| 测试名 | 说明 |
|--------|------|
| `LoadModel` | 模型加载和基础属性 |
| `Tokenize` | tokenizer 编解码 |
| `CreateContext` | context 创建 |
| `PrefillAndDecode` | prefill + decode 一轮 |
| `Generate` | token callback 生成 |
| `ResetContext` | reset 后状态清理 |
| `ImageProcessor` | VLM 图像预处理 |
| `VisionEncoder` | VLM 视觉编码 |
| `PrefillWithImage` | VLM 带图 prefill |

### 8.3 ASR 测试项

| 测试名 | 说明 |
|--------|------|
| `LoadModel` | ASR 模型加载和参数检查 |
| `CreateContext` | ASRContext 创建和类型转换 |
| `AudioProcessor` | 音频加载、切分、Mel 特征 |
| `Encode` | Encoder 前向 |
| `BuildPrompt` | 语言 token 和 prompt 构造 |
| `PrefillAndDecode` | Decoder prefill + decode |
| `Transcribe` | 完整音频转写入口 |
| `PerfInfo` | `ASRPerfInfo` RTF/TPS 指标填充 |

ASR 测试应额外检查：

- `SamplingParams::language` 为 `auto` 和具体语言时的行为。
- `eos_token_ids()` 中任一 token 触发停止。
- 多 chunk 音频的 `n_chunks` 和 `audio_duration` 统计。
- `profiler().has_stage("transcribe.encode.inference")` 等关键阶段存在。

### 8.4 CMake 注册测试

```cmake
if(BUILD_TESTS)
  add_executable(your_model_test tests/your_model_test.cc)
  target_include_directories(your_model_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(your_model_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME YourModelTest COMMAND your_model_test)
endif()
```

---

## 9. 性能采集检查清单

### 生成类模型

- [ ] `generate` E2E 计时
- [ ] `set_input_tokens()` 设置输入 token 数
- [ ] prefill 后调用 `record_ttft()`
- [ ] 每次 decode 后调用 `add_output_token()`
- [ ] `generate.prefill` 和 `generate.decode` 阶段存在
- [ ] 结束时 `perf_stats_ = profiler_.to_perf_stats()`

### ASR 模型

- [ ] `set_root_stage("transcribe")`
- [ ] `transcribe.audio_load` 覆盖音频加载和特征提取
- [ ] 使用 `do_encode()` 而非直接调用 encode 钩子
- [ ] 使用 `do_detect_language()` 或明确处理不支持语言检测的模型
- [ ] 使用 `do_prefill()` 和 `do_decode()`
- [ ] prefill 后调用 `record_ttft()`
- [ ] decode token 后调用 `add_output_token()`
- [ ] 结束时调用 `fill_perf_info(audio_duration)`
- [ ] `ASRPerfInfo` 中 `overall_rtf`、`inference_rtf`、`decode_tps` 合理

---

## 10. 适配完成检查清单

### 代码

- [ ] 选择了正确的基类：LLM、VLM 或 ASR。
- [ ] 没有遮蔽基类已有成员。
- [ ] `load()` 使用 `config_.devices`，没有硬编码设备号。
- [ ] `weight_manager_` 使用 `*dev_manager_` 创建。
- [ ] 所有 TCIM 输入 tensor map 在加载后初始化。
- [ ] `ModelInfo` 或 ASR 参数字段填充完整。
- [ ] tokenizer、embedding 按模型需要加载。

### 注册和构建

- [ ] `ModelSeries` 和字符串转换已更新。
- [ ] `REGISTER_MODEL` 使用正确的基类工厂。
- [ ] CMake 源文件列表包含新模型实现。
- [ ] CMake 测试目标已添加。
- [ ] whole-archive 链接策略能保留静态注册对象。

### 推理

- [ ] Prefill padding 和 position 逻辑符合模型要求。
- [ ] Decode 正确维护 context length 和 cache。
- [ ] Stop token、max tokens、callback 中断都能停止。
- [ ] VLM 清理或保留图像状态的策略明确。
- [ ] ASR 多 chunk、语言设置、EOS 集合和音频时长统计正确。

### 测试

- [ ] 缺少模型文件时测试 skip。
- [ ] 单元测试覆盖 tokenizer/embedding/sampler/profiler 或模型对应模块。
- [ ] LLM/VLM 覆盖 load、context、prefill/decode、generate。
- [ ] ASR 覆盖 audio processor、encode、prompt、transcribe、perf info。
- [ ] `ctest --output-on-failure` 通过，或失败原因与缺少外部模型文件明确相关。
