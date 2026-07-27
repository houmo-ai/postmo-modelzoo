# 新模型适配指南

本文档基于当前 `0.1.0` 实现，说明如何在模型目录中复用 Houmo Inference Framework。基础库不包含具体模型，也不会自动加载模型文件。

## 1. 选择基类

| 类型 | Model 基类 | Context 基类 | 说明 |
|------|------------|--------------|------|
| 文本生成 | `houmo::LLMModel` | `houmo::Context` | 实现 prefill、decode、generate |
| 视觉语言 | `houmo::VLMModel` | `houmo::Context` | 额外实现 vision encode 和图像状态 |
| 语音识别 | `houmo::ASRModel` | `houmo::ASRContext` | 实现 Transcribe 和 ASR 模板 hook |

不要在子类中重复声明基类已有成员，例如 `config_`、`tokenizer_`、`embedding_`、TCIM modules、input maps、`context_length_`、`generated_ids_` 和 `profiler_`。

## 2. 配置约定

```cpp
houmo::ModelConfig config;
config.devices = {0};
config.batch_size = 1;
config.lazy_mode = false;
config.prefill_path = "path/to/prefill.hmm";
config.decode_path = "path/to/decode.hmm";
config.embedding_path = "path/to/embedding.bin";
config.tokenizer_path = "path/to/tokenizer";
config.vision_path = "path/to/vision.hmm";
config.extra_params["encode_path"] = "path/to/encoder.hmm";
```

`ModelConfig` 没有内置校验。模型构造或 `load()` 应检查必需路径、设备列表、batch 和扩展参数，并给出明确错误。

## 3. LLM 适配

### Model 骨架

```cpp
#include "core/llm_model.h"

class YourModel final : public houmo::LLMModel {
 public:
  explicit YourModel(const houmo::ModelConfig& config)
      : LLMModel(config) {
    load();
  }

  std::unique_ptr<houmo::Context> create_context(int n_ctx = 0) override;

 private:
  void load();
  void init_prefill_inputs();
  void init_decode_inputs();
};
```

### load() 职责

当前基类构造函数只保存配置，子类通常需要完成：

1. 根据 `config_.devices` 创建 `tcim::DevManager`。
2. 创建 `tcim::Module::WeightManager`。
3. 加载 `prefill_module_` 和 `decode_module_`。
4. 按模型要求建立或共享 KV cache。
5. 创建 `Embedding(config_.embedding_path, hidden_dim, prefill_length)`。
6. 创建 `HfTokenizer(config_.tokenizer_path)`。
7. 填充 `info_`，尤其是 `type`、`n_vocab`、`n_embd`、`n_ctx` 和 `prefill_length`。
8. 设置 `prefill_length_` 和 `attn_idx_start_`。
9. 初始化 `prefill_input_map_` 和 `decode_input_map_`。

`Embedding` 的 `hidden_dim` 必须大于 0。批量 lookup 还要求构造时提供足够的 `max_seq_len`。

### Context 骨架

```cpp
#include "core/context.h"

class YourContext final : public houmo::Context {
 public:
  YourContext(YourModel* model, int n_ctx)
      : Context(model, n_ctx), model_(model) {}

  houmo::Token prefill(const std::vector<houmo::Token>& tokens) override;
  houmo::Token decode(houmo::Token prev_token) override;
  void generate(const std::vector<houmo::Token>& prompt,
                const houmo::SamplingParams& params,
                std::function<bool(houmo::Token)> callback) override;
  void reset() override;

 private:
  YourModel* model_;
};
```

`Context` 持有非 owning model 指针，必须保证 model 生命周期覆盖 context。

### generate() 要求

基础类没有默认循环。建议至少保证：

```cpp
void YourContext::generate(
    const std::vector<houmo::Token>& prompt,
    const houmo::SamplingParams& params,
    std::function<bool(houmo::Token)> callback) {
  profiler_.reset();
  profiler_.set_root_stage("generate");
  profiler_.start("generate");
  profiler_.set_input_tokens(static_cast<int>(prompt.size()));
  set_sampler(params);

  houmo::Token token;
  {
    auto timer = profiler_.scope("generate.prefill");
    token = prefill(prompt);
  }
  profiler_.record_ttft();
  profiler_.add_output_token();

  bool keep_going = callback(token);
  while (keep_going && !should_stop(token, params)) {
    {
      auto timer = profiler_.scope("generate.decode");
      token = decode(token);
    }
    profiler_.add_output_token();
    keep_going = callback(token);
  }

  profiler_.stop("generate");
  perf_stats_ = profiler_.to_perf_stats();
}
```

`should_stop()` 是示意逻辑，需要由适配层实现。停止条件应覆盖 EOS、`stop_tokens`、`max_tokens`、context 上限和 callback 中断。

当前 `Sampler` 最终使用 `argmax`，不是随机采样。不要仅凭 `temperature` 或 `top_p` 假设输出具有随机性。

### reset() 要求

先调用 `Context::reset()`，再清理模型相关状态：

- KV cache
- position/cache index
- 本轮临时 tensor
- VLM 图像状态
- 需要时重置 profiler 和 sampler

## 4. VLM 适配

### Model 骨架

```cpp
#include "core/vlm_model.h"

class YourVLM final : public houmo::VLMModel {
 public:
  explicit YourVLM(const houmo::ModelConfig& config)
      : VLMModel(config) {
    load();
  }

  std::unique_ptr<houmo::Context> create_context(int n_ctx = 0) override;
  std::vector<float16> encode_image(const uint8_t* data,
                                    int width,
                                    int height,
                                    int channels) override;

 private:
  void load();
};
```

基类 `encode_image()` 和 `create_context()` 都只是占位实现，必须覆盖。

### 请求级图像状态

图片路径、图片 embedding 和是否启用视觉输入应放在 Context 中，而不是 Model 全局状态中。这样才能避免并发请求和多轮对话互相污染。

```cpp
class YourVLMContext final : public houmo::Context {
 public:
  void set_image(const std::string& image_path) override {
    image_path_ = image_path;
  }

 private:
  std::string image_path_;
  std::vector<float16> image_embeddings_;
};
```

### HmImageProcessor 注意事项

`HmImageProcessor` 当前位于全局命名空间：

```cpp
HmImageProcessor processor(448, 448, true);
ProcessedImage image = processor.LoadAndProcess(path);
std::vector<float16> input = processor.ToFP16Tensor(image);
```

`ToFP16Tensor()` 输出 `[3, 2, H, W]` 原始 0..255 数据。模型需要的 mean/std normalization、layout 转换和 patch merge 需由适配层完成。

加载失败不会抛异常，而会返回 fallback 图像。若模型不允许静默 fallback，应在调用前检查路径或在适配层增加错误策略。

## 5. ASR 适配

### Model 骨架

```cpp
#include "core/asr_model.h"

class YourASRModel final : public houmo::ASRModel {
 public:
  explicit YourASRModel(const houmo::ModelConfig& config)
      : ASRModel(config) {
    load();
  }

  std::unique_ptr<houmo::Context> create_context(int n_ctx = 0) override;
  houmo::Token sot_token_id() const override;
  houmo::Token lang_token_id(const std::string& language) const override;
  houmo::Token transcribe_token_id() const override;
  houmo::Token notimestamps_token_id() const override;
  std::vector<houmo::Token> eos_token_ids() const override;
  bool supports_language_detection() const override;

 private:
  void load();
};
```

在 `load()` 中设置 `n_mels_`、`n_frames_`、`num_heads_`、`cache_max_len_` 和 `num_decode_layers_`，并在模型子类中保存 encoder/decoder modules、Tokenizer、Embedding 和 tensor maps。`ASRModel` 不继承 `LLMModel`，这些资源不会由基类提供。

### Context 骨架

```cpp
class YourASRContext final : public houmo::ASRContext {
 public:
  YourASRContext(YourASRModel* model, int n_ctx)
      : ASRContext(model, n_ctx), model_(model) {}

  std::vector<float16> Encode(const std::vector<float>& mel,
                              int n_mels,
                              int n_frames) override;
  houmo::Token DetectLanguage() override;
  std::vector<houmo::Token> BuildPrompt(
      houmo::Token language_token) override;
  void Transcribe(const std::string& path,
                  const houmo::SamplingParams& params,
                  houmo::ASRTokenCallback callback) override;
  void set_language(const std::string& language) override;

 private:
  void encode_preprocess_impl(const std::vector<float>& mel,
                              int n_mels,
                              int n_frames) override;
  void encode_inference_impl() override;
  void encode_postprocess_impl() override;
  void prefill_preprocess_impl(
      const std::vector<houmo::Token>& tokens) override;
  void prefill_inference_impl() override;
  houmo::Token prefill_postprocess_impl() override;
  void decode_preprocess_impl(houmo::Token token) override;
  void decode_inference_impl() override;
  houmo::Token decode_postprocess_impl() override;

  YourASRModel* model_;
};
```

只有支持语言检测的模型需要覆盖三个 `detect_lang_*_impl()` hook。默认实现不执行推理并返回 `0`。

### Transcribe() 结构

模型应通过 `do_*()` 调用 hook，不要直接调用 `_impl()`，否则不会生成统一性能数据：

```cpp
void YourASRContext::Transcribe(
    const std::string& path,
    const houmo::SamplingParams& params,
    houmo::ASRTokenCallback callback) {
  profiler_.reset();
  profiler_.set_root_stage("transcribe");
  profiler_.start("transcribe");
  set_language(params.language);

  houmo::AudioProcessorConfig audio_config;
  audio_config.n_mels = model_->n_mels();
  houmo::AudioProcessor processor(audio_config);

  std::vector<houmo::MelFeatures> chunks;
  float duration = 0.0f;
  {
    auto timer = profiler_.scope("transcribe.audio_load");
    chunks = processor.Process(path);
    for (const auto& chunk : chunks) {
      duration += chunk.duration;
    }
  }

  for (const auto& chunk : chunks) {
    std::vector<float> mel(chunk.data.begin(), chunk.data.end());
    do_encode(mel, chunk.feature_dim, chunk.num_frames);

    houmo::Token language_token = supports_detection()
        ? do_detect_language()
        : model_->lang_token_id(language_);
    auto prompt = BuildPrompt(language_token);

    houmo::Token token = do_prefill(prompt);
    profiler_.record_ttft();
    profiler_.add_output_token();
    if (!callback(token)) break;

    while (!is_eos(token) && !reached_limit(params)) {
      token = do_decode(token);
      profiler_.add_output_token();
      if (!callback(token)) break;
    }
  }

  profiler_.stop("transcribe");
  fill_perf_info(duration);
}
```

`supports_detection()`、`is_eos()` 和 `reached_limit()` 是适配层示意函数，不属于基础库。

`AudioProcessor` 输出 FP16，而 ASR hook 接收 float，示例进行了显式转换。高性能实现可复用转换 buffer，避免每个 chunk 重复分配。

## 6. 模型注册

当前 `ModelSeries` 已定义 Qwen3 LLM、Qwen3.5 MLLM、Qwen3 VLM、Whisper、GLM-ASR 和 Qwen3-ASR。新增系列时需要更新枚举和两个字符串转换函数。

LLM/VLM 注册到 `ModelFactory<LLMModel>`：

```cpp
#include "core/model_factory.h"

REGISTER_MODEL(LLMModel, your_model, houmo::ModelSeries::kQwen3LLM,
               [](const houmo::ModelConfig& config) {
                 return std::make_unique<YourModel>(config);
               },
               "Your model");
```

ASR 注册到 `ModelFactory<ASRModel>`：

```cpp
REGISTER_MODEL(ASRModel, your_asr, houmo::ModelSeries::kWhisperASR,
               [](const houmo::ModelConfig& config) {
                 return std::make_unique<YourASRModel>(config);
               },
               "Your ASR model");
```

`REGISTER_MODEL` 宏定义在 `houmo` 命名空间内，但宏参数 `series` 直接进入表达式。根据调用位置，可使用 `houmo::ModelSeries::...` 或处于 `namespace houmo` 内时使用 `ModelSeries::...`。

静态注册对象必须被最终链接保留。如果模型实现在静态库中，最终应用需要采用 whole-archive 或等效链接策略。

## 7. CMake 集成

基础工程只构建 `houmo_infer` 共享库，没有模型源码自动发现机制。模型目录需要显式链接：

```cmake
target_link_libraries(your_model_target PRIVATE houmo_infer)
target_include_directories(your_model_target PRIVATE
  /path/to/houmo_engine/include
)
```

若直接向基础工程添加通用源码，需要更新 `CORE_SOURCES` 或 `MODULE_SOURCES`。具体模型源码通常应留在模型自身 target 中，避免基础库绑定具体模型。

## 8. 验证清单

### 加载

- [ ] 所有必需路径在加载前检查。
- [ ] `config_.devices` 被实际使用，没有硬编码设备 ID。
- [ ] `DevManager` 生命周期覆盖所有 TCIM modules。
- [ ] `WeightManager` 和共享 cache 的所有权清晰。
- [ ] `ModelInfo::type` 和其他字段完整初始化。
- [ ] Embedding hidden dimension 和 max sequence length 正确。

### 推理

- [ ] prefill padding、position 和 logits 索引与模型一致。
- [ ] decode 正确更新 position、context length 和 KV cache。
- [ ] EOS、`stop_tokens`、`max_tokens`、context 上限和 callback 均可终止循环。
- [ ] `generated_ids_` 包含 Sampler penalty 所需的正确历史。
- [ ] `reset()` 清理了基础状态和模型特有状态。
- [ ] VLM 图像状态属于 Context，且多轮保留策略明确。
- [ ] ASR 多 chunk 之间的 cache 重置/复用策略明确。

### 性能

- [ ] 根 stage 使用 `generate` 或 `transcribe`。
- [ ] prefill 后调用 `record_ttft()`。
- [ ] 每个实际输出 token 调用一次 `add_output_token()`。
- [ ] LLM/VLM 结束时更新 `perf_stats_`。
- [ ] ASR 结束时调用 `fill_perf_info(actual_audio_duration)`。
- [ ] 禁用 `HOUOMO_ENABLE_PROFILING` 后业务逻辑仍可运行。

### 错误处理

- [ ] `ModelFactory::Create()` 返回 `nullptr` 时调用方有处理。
- [ ] AudioProcessor 返回空结果时停止转写并报告错误。
- [ ] 图像加载 fallback 是否符合产品需求已经确认。
- [ ] Tokenizer 或 Embedding 未加载时不会调用对应 getter。
