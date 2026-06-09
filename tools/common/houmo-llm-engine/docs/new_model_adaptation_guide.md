# 新模型适配指南

本文档描述如何基于 Houmo Inference Framework 适配新模型（LLM / VLM）。

---

## 1. 文件结构

```
include/models/
├── your_model_model.h         # 头文件
src/models/
├── your_model_model.cc        # 实现 + 注册
tests/
├── your_model_test.cc         # GTest 测试
```

---

## 2. 继承规则

### 2.1 继承层级

```
LLMModel (基类)
  ├── YourModelLLMModel        # 纯文本 LLM
  └── VLMModel (VLM 基类)
        ├── YourModelVLMModel  # 视觉语言模型
        └── ...

Context (基类)
  ├── YourModelLLMContext      # LLM 上下文
  └── YourModelVLMContext      # VLM 上下文
```

### 2.2 选择继承基类

| 模型类型 | Model 基类 | Context 基类 | 说明 |
|----------|-----------|-------------|------|
| 纯文本 LLM | `LLMModel` | `Context` | 无视觉处理 |
| 视觉语言 VLM | `VLMModel` | `Context` | 继承 `LLMModel` + 视觉编码器 |

**规则**：
- VLM 模型**必须**继承 `VLMModel`，**禁止**直接继承 `LLMModel`
- Context 类**必须**通过 `LLMModel*` 指针访问模型（多态）
- **禁止**在 Context 中重新声明基类已有的成员变量或虚方法（如 `keep_history_`、`context_length_`）

### 2.3 禁止遮蔽基类成员

```cpp
// ❌ 错误：遮蔽基类 keep_history_
class YourContext : public Context {
  bool keep_history_ = true;                    // 遮蔽 Context::keep_history_
  void set_keep_history(bool keep) { ... }      // 遮蔽 Context::set_keep_history()
};

// ✅ 正确：直接使用继承的成员
class YourContext : public Context {
  // 不声明 keep_history_，使用 Context 的
};
```

---

## 3. Model 类实现规范

### 3.1 头文件模板

```cpp
#ifndef HOUMO_YOUR_MODEL_MODEL_H
#define HOUMO_YOUR_MODEL_MODEL_H

#include "core/context.h"
// LLM: #include "core/llm_model.h"
// VLM: #include "core/vlm_model.h"

namespace houmo {

class YourModelContext : public Context {
 public:
  explicit YourModelContext(LLMModel* model, int n_ctx);
  ~YourModelContext() override = default;

  Token prefill(const std::vector<Token>& tokens) override;
  Token decode(Token prev_token) override;
  void generate(const std::vector<Token>& prompt,
                const SamplingParams& params,
                std::function<bool(Token)> callback) override;
  void reset() override;

 private:
  // ========== Prefill 拆分方法 ==========
  void prefill_preprocess_chunk(int chunk, ...);
  void prefill_inference_chunk();
  Token prefill_postprocess(Sampler* sampler, int32_t seq_length);

  // ========== Decode 拆分方法 ==========
  void decode_preprocess(Token prev_token);
  void decode_inference();
  Token decode_postprocess(Sampler* sampler);

  // ========== 内部推理方法 ==========
  Token do_prefill_inference(const std::vector<Token>& tokens, Sampler* sampler);
  Token do_decode_inference(Token prev_token, Sampler* sampler);

  // ========== 模型特有成员 ==========
  // ...
};

// LLM 模型
class YourModelLLMModel : public LLMModel {
 public:
  explicit YourModelLLMModel(const ModelConfig& config);
  ~YourModelLLMModel() override = default;

  std::unique_ptr<Context> create_context(int n_ctx = 0) override;
  void ClearKVCache();

 private:
  void load();
  int n_blocks_ = 0;
  int batch_ = 0;
  int embedding_length_ = 0;
  int context_max_length_ = 0;
};

// VLM 模型 (如需要)
// class YourModelVLMModel : public VLMModel { ... };

}  // namespace houmo
#endif
```

### 3.2 Model::load() 规范

```cpp
void YourModelLLMModel::load() {
  // 步骤1 - 初始化设备管理器（必须使用 config_.devices，禁止硬编码设备号）
  dev_manager_ = std::make_unique<tcim::DevManager>(
      tcim::DevManager::Create(config_.devices));
  weight_manager_ = std::make_unique<tcim::Module::WeightManager>(
      tcim::Module::WeightManager::CreateWeightManager(*dev_manager_));

  // 步骤2 - 加载 prefill 模型
  // 步骤3 - 加载 decode 模型
  // 步骤4 - 共享 KV Cache
  // 步骤5 - 加载 Embedding
  // 步骤6 - 初始化输入 tensors
  // 步骤7 - 填充模型信息
  // 步骤8 - 加载 Tokenizer
  // 步骤9 - 初始化 KV Cache（调用 ClearKVCache()）
}
```

**规则**：
- `dev_manager_` 和 `weight_manager_` **必须**使用基类的 protected 成员（禁止在子类中重新声明）
- `CreateWeightManager` **必须**使用 `*dev_manager_`（禁止 `CreateWeightManager(0)` 硬编码设备号）

### 3.3 Model::ClearKVCache() 规范

```cpp
void YourModelLLMModel::ClearKVCache() {
  if (!prefill_module_ || !decode_module_) return;

  // 遍历需要清零的 cache 类型
  for (int idx = 0; idx < prefill_module_->GetInputNum(); idx++) {
    const auto input_name = prefill_module_->GetInputName(idx);
    // 根据模型 cache 类型过滤（conv_cache / recurrent_state / kcache / vcache）
    if (input_name.find("your_cache_pattern") == std::string::npos) continue;

    auto info = prefill_module_->GetInputInfo(input_name).AsContiguous();
    auto tensor = tcim::Tensor::CreateHostTensor(info);
    std::vector<uint8_t> zeros(tensor.MemSize(), 0);
    tensor.Buffer().CopyFromHost(zeros.data(), zeros.size());
    prefill_module_->SetInput(input_name, tensor);
  }
}
```

---

## 4. 推理 Pipeline 规范

### 4.1 Context 方法拆分规则

每个 Context 类**必须**将推理逻辑拆分为以下方法，便于性能采集和调试：

```
prefill(tokens)
  └─► do_prefill_inference(tokens, sampler)
        ├─► [VLM] run_vision()                    // 视觉处理
        ├─► [VLM] prefill_common_setup(tokens)    // 通用设置（一次性）
        ├─► [循环] prefill_preprocess_chunk()      // 分块预处理
        │         prefill_inference_chunk()        // 分块推理
        └─► prefill_postprocess(sampler, seq_len)  // 后处理 + 采样

decode(prev_token)
  └─► do_decode_inference(prev_token, sampler)
        ├─► decode_preprocess(prev_token)          // 预处理
        ├─► decode_inference()                     // 推理
        └─► decode_postprocess(sampler)            // 后处理 + 采样
```

### 4.2 Prefill 分块处理

```cpp
Token YourModelContext::do_prefill_inference(const std::vector<Token>& tokens,
                                             Sampler* sampler) {
  auto* model = static_cast<LLMModel*>(model_);
  const int32_t seq_length = static_cast<int32_t>(tokens.size());
  const int prefill_length = model->prefill_length();
  const int prefill_loop_chunk =
      (seq_length + prefill_length - 1) / prefill_length;

  for (int chunk = 0; chunk < prefill_loop_chunk; chunk++) {
    prefill_preprocess_chunk(chunk, tokens, seq_length, prefill_length);
    prefill_inference_chunk();
  }

  Token sampled_token = prefill_postprocess(sampler, seq_length);
  return sampled_token;
}
```

### 4.3 Prefill Padding 规范

最后一个 chunk 如果输入 tokens 不足 `prefill_length`，**必须**使用 `pad_token_id` 填充：

```cpp
Token pad_token_id = model->tokenizer()->pad_token_id();
if (input_ids.size() < static_cast<size_t>(prefill_length)) {
    input_ids.resize(prefill_length, pad_token_id);
}
```

### 4.4 Generate 流式生成规范

`generate()` **必须**包含完整的性能采集逻辑（见第 5 节）：

```cpp
void YourModelContext::generate(const std::vector<Token>& prompt,
                                const SamplingParams& params,
                                std::function<bool(Token)> callback) {
  profiler_.reset();
  auto& p = profiler_;

  p.start("generate");
  p.set_input_tokens(static_cast<int>(prompt.size()));

  set_sampler(params);

  // Prefill
  Token token;
  { auto t = p.scope("generate.prefill"); token = prefill(prompt); }

  p.record_ttft();

  if (token == model_->eos_token_id() || token == model_->bos_token_id()) {
    p.stop("generate"); perf_stats_ = p.to_perf_stats(); return;
  }
  if (!callback(token)) {
    p.stop("generate"); perf_stats_ = p.to_perf_stats(); return;
  }

  // Decode
  while (true) {
    if (context_length_ >= model_->max_ctx_available()) break;
    if (params.max_tokens > 0 &&
        generated_ids_.size() >= static_cast<size_t>(params.max_tokens)) break;

    { auto t = p.scope("generate.decode"); token = decode(token); }
    p.add_output_token();

    if (token == model_->eos_token_id() || token == model_->bos_token_id()) break;
    if (!callback(token)) break;
  }

  p.stop("generate");
  perf_stats_ = p.to_perf_stats();
}
```

### 4.5 Reset 规范

```cpp
void YourModelContext::reset() {
  Context::reset();  // 必须调用基类 reset
  auto* model = static_cast<YourModelLLMModel*>(model_);
  model->ClearKVCache();  // 清空 KV Cache
}
```

---

## 5. 性能采集规范

### 5.1 计时层级结构

所有 Context 类**必须**使用 `profiler_`（基类成员）进行性能采集。计时路径**必须**遵循以下层级结构：

```
generate                              // E2E 总时间
├── generate.vision                   // [VLM] 视觉处理
│   ├── generate.vision.preprocess    // [VLM] 视觉预处理
│   ├── generate.vision.inference     // [VLM] 视觉推理
│   └── generate.vision.postprocess   // [VLM] 视觉后处理
├── generate.prefill                  // Prefill 阶段
│   ├── generate.prefill.common_setup // [VLM] 通用设置（一次性）
│   ├── generate.prefill.preprocess_chunk  // 分块预处理
│   ├── generate.prefill.inference_chunk   // 分块推理
│   └── generate.prefill.postprocess       // 后处理 + 采样
└── generate.decode                   // Decode 阶段
    ├── generate.decode.preprocess    // 预处理
    ├── generate.decode.inference     // 推理
    └── generate.decode.postprocess   // 后处理 + 采样
```

### 5.2 计时方法

```cpp
// 方式1: ScopedTimer（推荐，自动 stop）
{
  auto t = profiler_.scope("generate.prefill.inference_chunk");
  prefill_inference_chunk();
}  // 析构时自动 stop

// 方式2: 手动 start/stop
profiler_.start("generate");
// ... 工作 ...
profiler_.stop("generate");
```

### 5.3 必须采集的指标

| 指标路径 | 说明 | 必须 |
|----------|------|------|
| `generate` | E2E 总时间 | ✅ |
| `generate.prefill` | Prefill 阶段总时间 | ✅ |
| `generate.decode` | 单次 Decode 时间（循环内） | ✅ |
| `generate.vision` | 视觉处理总时间 | VLM 必须 |
| `generate.prefill.preprocess_chunk` | 分块预处理 | ✅ |
| `generate.prefill.inference_chunk` | 分块推理 | ✅ |
| `generate.prefill.postprocess` | 后处理 + 采样 | ✅ |

### 5.4 Token 统计

```cpp
// generate() 开始时
p.set_input_tokens(static_cast<int>(prompt.size()));

// prefill 完成后
p.record_ttft();

// 每次 decode 后
p.add_output_token();

// generate() 结束时
p.stop("generate");
perf_stats_ = p.to_perf_stats();
```

### 5.5 采集结果导出

```cpp
// 用户通过 ctx->perf_stats() 获取
PerfStats stats = ctx->perf_stats();
// stats 包含: prefill_time_ms, decode_time_ms, ttft_ms, tps 等
```

---

## 6. VLM 特有规范

### 6.1 VLM Context 额外方法

```cpp
class YourModelVLMContext : public Context {
 public:
  // 图像接口
  void set_image(const std::string& image_path);
  void set_images(const std::vector<std::string>& image_paths);
  bool has_image() const;

 private:
  // Vision 处理
  void run_vision();
  void vision_preprocess(int image_idx);
  void vision_inference();
  void vision_postprocess(int image_idx);

  // Embedding 处理
  void scatter_image_embeds(...);
  std::vector<Token> expand_image_tokens(const std::vector<Token>& tokens);

  // M-RoPE
  std::pair<std::vector<std::vector<int32_t>>, int32_t> get_rope_index(...);

  // 图像相关成员
  std::vector<std::string> image_paths_;
  std::vector<float16> flat_image_embeds_;
  std::vector<ImageGridTHW> image_grid_thw_;
  bool use_vlm_ = false;
};
```

### 6.2 VLM Prefill 流程

```cpp
Token YourModelVLMContext::do_prefill_inference(const std::vector<Token>& tokens,
                                                 Sampler* sampler) {
  auto& p = profiler_;

  // 1. 视觉处理
  if (use_vlm_ && !image_paths_.empty()) {
    auto t = p.scope("generate.vision");
    run_vision();
  }

  // 2. 通用设置（一次性）
  auto [position_ids, seq_length, chunk_count] = [&]() {
    auto t = p.scope("generate.prefill.common_setup");
    return prefill_common_setup(tokens);
  }();

  // 3. 分块执行
  for (int chunk = 0; chunk < chunk_count; chunk++) {
    { auto t = p.scope("generate.prefill.preprocess_chunk"); ... }
    { auto t = p.scope("generate.prefill.inference_chunk"); ... }
  }

  // 4. 后处理
  Token sampled_token;
  { auto t = p.scope("generate.prefill.postprocess"); ... }

  return sampled_token;
}
```

### 6.3 VLM Model 额外接口

```cpp
class YourModelVLMModel : public VLMModel {
 public:
  // 视觉编码
  std::vector<float16> encode_image(const std::vector<float16>& pixel_values);

  // Vision 模块（使用基类的 vision_module_，不要重新声明）
  // vision_input_map_ 同上

  void ClearKVCache();

 private:
  void load();
  void init_vision_inputs();
};
```

---

## 7. 测试规范

### 7.1 测试文件模板

```cpp
#include <gtest/gtest.h>
#include <filesystem>

#include "models/your_model_model.h"
#include "modules/sampler.h"
#include "test_utils.h"

namespace houmo {
namespace fs = std::filesystem;

class YourModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string base_path = test::GetBasePath();
    prefill_path_ = base_path + "/models/.../prefill.hmm";
    decode_path_ = base_path + "/models/.../decode.hmm";
    embedding_path_ = base_path + "/models/.../quant_embedding.bin";
    tokenizer_path_ = base_path + "/models/tokenizers/.../tokenizer.json";
  }

  ModelConfig GetConfig() {
    ModelConfig config;
    config.prefill_path = prefill_path_;
    config.decode_path = decode_path_;
    config.embedding_path = embedding_path_;
    config.tokenizer_path = tokenizer_path_;
    config.devices = {0};
    config.lazy_mode = false;
    return config;
  }

  bool CheckModelFiles() {
    return fs::exists(prefill_path_) && fs::exists(decode_path_) &&
           fs::exists(embedding_path_) && fs::exists(tokenizer_path_);
  }

  std::string prefill_path_;
  std::string decode_path_;
  std::string embedding_path_;
  std::string tokenizer_path_;
};
```

### 7.2 必须包含的测试用例

| 测试名 | 说明 | 必须 |
|--------|------|------|
| `LoadModel` | 模型加载 + 基本属性检查 | ✅ |
| `Tokenize` | 编解码一致性 | ✅ |
| `CreateContext` | Context 创建 | ✅ |
| `PrefillAndDecode` | Prefill + Decode 一轮 | ✅ |
| `Generate` | 流式生成 | ✅ |
| `ResetContext` | Context 重置 | ✅ |
| `ImageProcessor` | 图像处理 | VLM |
| `VisionEncoder` | 视觉编码器 | VLM |
| `PrefillWithImage` | 带图像的 Prefill | VLM |

### 7.3 测试规则

1. **所有测试必须使用 `ASSERT_NO_THROW` 包裹测试体**（防止模型加载失败导致崩溃）
2. **`CheckModelFiles()` 必须检查所有 4 个路径**（prefill、decode、embedding、tokenizer）
3. **`ModelConfig` 必须设置 `devices` 和 `lazy_mode`**
4. **VLM 测试中 `dynamic_cast` 后必须 `ASSERT_NE(ptr, nullptr)`**

```cpp
TEST_F(YourModelTest, PrefillWithImage) {
  if (!CheckModelFiles()) GTEST_SKIP() << "Model files not found";

  auto config = GetConfig();
  ASSERT_NO_THROW({
    YourModelVLMModel model(config);
    auto ctx = model.create_context();

    auto* vlm_ctx = dynamic_cast<YourModelVLMContext*>(ctx.get());
    ASSERT_NE(vlm_ctx, nullptr) << "Failed to cast to VLMContext";
    vlm_ctx->set_image(test_image_path_);

    auto tokens = model.tokenize("描述这张图片", true, false);
    Token token = ctx->prefill(tokens);
    EXPECT_GE(token, 0);
  });
}
```

### 7.4 CMakeLists.txt 注册

```cmake
if(BUILD_TESTS)
  add_executable(your_model_test tests/your_model_test.cc)
  target_include_directories(your_model_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(your_model_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME YourModelTest COMMAND your_model_test)
endif()
```

---

## 8. 模型注册

### 8.1 静态注册

```cpp
// 在 .cc 文件末尾
#include "core/model_factory.h"

REGISTER_LLM_MODEL(your_model, ModelSeries::kYourModel,
                   [](const ModelConfig& c) {
                     return std::make_unique<YourModelLLMModel>(c);
                   },
                   "YourModel 模型");
```

### 8.2 枚举注册

在 `include/base/houmo.h` 的 `ModelSeries` 枚举中添加：

```cpp
enum class ModelSeries {
  // ... 现有
  kYourModel,  // 新增
};
```

---

## 9. 模型差异处理清单

适配新模型时，确认以下差异点：

| 项目 | 需确认 | 说明 |
|------|--------|------|
| Padding 位置 | 后置/前置 | 不足 prefill_length 时填充位置 |
| Pad Token | `pad_token_id` 值 | Qwen 系列等于 `bos_token_id` |
| position_ids | 是否需要 | Decode 阶段是否需要 position_ids |
| M-RoPE | 是否需要 | 多模态 3D 位置编码 |
| Deepstack | 是否需要 | 视觉特征注入方式 |
| Cache 类型 | kcache/vcache/conv_cache/recurrent_state | KV Cache 清零策略 |

### 常见输入名称

```
input_1 / inputs_embeds       - Embedding 输入
valid_length                   - 已处理序列长度
current_length                 - 当前处理的长度
position_ids                   - 位置编码
time_position / height_position / width_position  - M-RoPE 3D 位置
deepstack_image_embed_0/1/2   - Deepstack 视觉特征
linear_attn_mask              - 线性注意力掩码
model_layers_*_self_attn_kcache_input  - KV Cache (key)
model_layers_*_self_attn_vcache_input  - KV Cache (value)
```

---

## 10. 检查清单

### 代码
- [ ] 头文件创建 (`include/models/your_model_model.h`)
- [ ] 实现文件创建 (`src/models/your_model_model.cc`)
- [ ] Model 继承正确的基类（LLM → `LLMModel`，VLM → `VLMModel`）
- [ ] `load()` 使用 `config_.devices`，不硬编码设备号
- [ ] `weight_manager_` 使用基类成员，不重新声明
- [ ] Context 不遮蔽基类成员（`keep_history_`、`context_length_` 等）

### 推理 Pipeline
- [ ] `prefill` 拆分为 `preprocess_chunk` / `inference_chunk` / `postprocess`
- [ ] `decode` 拆分为 `preprocess` / `inference` / `postprocess`
- [ ] `generate` 包含完整的性能采集逻辑
- [ ] `reset()` 调用基类 `Context::reset()` + `ClearKVCache()`

### 性能采集
- [ ] `generate` E2E 计时
- [ ] `generate.prefill` Prefill 计时
- [ ] `generate.decode` Decode 计时（循环内）
- [ ] `generate.vision` Vision 计时（VLM）
- [ ] `set_input_tokens()` / `record_ttft()` / `add_output_token()` 调用正确
- [ ] `perf_stats_ = profiler_.to_perf_stats()` 在 generate 结束时赋值

### 测试
- [ ] GTest 测试创建 (`tests/your_model_test.cc`)
- [ ] CMakeLists.txt 更新
- [ ] `CheckModelFiles()` 检查 4 个路径
- [ ] 所有测试使用 `ASSERT_NO_THROW`
- [ ] `ModelConfig` 设置 `devices` 和 `lazy_mode`
- [ ] VLM 测试 `dynamic_cast` 后检查 `nullptr`
- [ ] 测试通过 (`ctest --output-on-failure`)
