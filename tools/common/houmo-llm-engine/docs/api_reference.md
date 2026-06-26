# Houmo Inference Framework - API 文档

> 版本: v0.2.0
> 日期: 2026-06-01

---

## 文件结构

```
include/
├── base/
│   └── houmo.h              # 基础类型定义 (Token, ModelType, Exception)
├── core/
│   ├── model_factory.h      # ModelFactory 工厂类
│   ├── llm_model.h          # LLMModel 基类
│   ├── vlm_model.h          # VLMModel 基类
│   └── context.h            # Context 基类
├── modules/
│   ├── tokenizer.h          # HfTokenizer
│   ├── embedding.h          # Embedding
│   ├── sampler.h            # Sampler
│   ├── streaming_decoder.h  # StreamingDecoder
│   └── image_processor.h    # ImageProcessor
└── models/
    ├── qwen3_llm_model.h    # Qwen3 LLM
    ├── qwen35_mllm_model.h  # Qwen3.5 MLLM
    └── qwen3_vlm_model.h    # Qwen3-VL
```

**常用引用：**

```cpp
// 使用工厂创建模型
#include "core/model_factory.h"

// 直接使用模型类
#include "core/llm_model.h"
#include "core/context.h"
#include "modules/sampler.h"
#include "modules/streaming_decoder.h"
```

---

## 快速开始

### 最小示例

```cpp
#include "core/model_factory.h"
#include "modules/streaming_decoder.h"
#include <iostream>

int main() {
    // 1. 配置模型
    houmo::ModelConfig config;
    config.devices = {0};
    config.prefill_path = "models/qwen3-4b/qwen3-4b_prefill.hmm";
    config.decode_path = "models/qwen3-4b/qwen3-4b_decode.hmm";
    config.embedding_path = "models/qwen3-4b/hmquant/quant_embedding.bin";
    config.tokenizer_path = "models/tokenizers/Qwen3-4B/tokenizer.json";

    // 2. 创建模型
    auto model = houmo::ModelFactory::Create("qwen3_llm", config);
    auto ctx = model->create_context();

    // 3. 流式生成
    auto tokens = model->tokenize("你好，请介绍一下自己。", false, false);
    houmo::SamplingParams params;
    params.max_tokens = 256;

    houmo::StreamingDecoder decoder(model->tokenizer());
    ctx->generate(tokens, params, [&decoder](houmo::Token token) {
        std::cout << decoder.decode(token) << std::flush;
        return true;
    });

    return 0;
}
```

---

## 核心类参考

### ModelFactory

模型工厂，支持静态注册和运行时创建模型。

```cpp
#include "core/model_factory.h"
```

#### ModelSeries 枚举

```cpp
enum class ModelSeries {
    kUnknown,      // 未知类型
    kQwen3LLM,     // Qwen3 纯文本 LLM
    kQwen35MLLM,   // Qwen3.5 多模态 MLLM
    kQwen3VLM,     // Qwen3-VL 视觉语言
};
```

#### 方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `Create(series, config)` | `std::unique_ptr<LLMModel>` | 根据系列创建模型 |
| `Create(name, config)` | `std::unique_ptr<LLMModel>` | 根据名称创建模型 |
| `ListRegisteredTypes()` | `std::vector<std::string>` | 列出已注册模型 |
| `GetRegisteredModels()` | `std::vector<RegistryEntry>` | 获取注册详情 |
| `IsRegistered(series)` | `bool` | 检查系列是否已注册 |

#### REGISTER_LLM_MODEL 宏

```cpp
REGISTER_LLM_MODEL(your_model_llm, ModelSeries::kYourModelLLM,
                   [](const ModelConfig& c) {
                     return std::make_unique<YourModelLLMModel>(c);
                   },
                   "YourModel 纯文本 LLM");
```

**重要**: 静态注册需要 `--whole-archive` 链接：

```cmake
target_link_libraries(sample_infer PRIVATE
  -Wl,--whole-archive houmo_infer -Wl,--no-whole-archive
)
```

---

### LLMModel

纯文本大语言模型类。

```cpp
#include "core/llm_model.h"
```

#### 构造函数

```cpp
explicit LLMModel(const ModelConfig& config);
```

#### 方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `type()` | `ModelType` | 返回 `ModelType::LLM` |
| `model_info()` | `ModelInfo` | 获取模型元数据信息 |
| `vocab_size()` | `int` | 词表大小 |
| `embedding_dim()` | `int` | Embedding 维度 |
| `max_ctx_available()` | `int` | 最大可用上下文长度 |
| `prefill_length()` | `int` | Prefill 序列长度 |
| `create_context(n_ctx)` | `std::unique_ptr<Context>` | 创建推理上下文 |
| `tokenize(text, add_bos, add_eos)` | `std::vector<Token>` | 文本转 Token |
| `token_to_str(token)` | `std::string` | Token 转字符串 |
| `tokenizer()` | `std::shared_ptr<HfTokenizer>` | 获取 Tokenizer |
| `eos_token_id()` | `Token` | EOS token ID |
| `bos_token_id()` | `Token` | BOS token ID |

---

### VLMModel

视觉语言模型类。

```cpp
#include "core/vlm_model.h"
```

#### 构造函数

```cpp
explicit VLMModel(const ModelConfig& config);
```

#### 方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `type()` | `ModelType` | 返回 `ModelType::VLM` |
| `model_info()` | `ModelInfo` | 获取模型元数据信息 |
| `vocab_size()` | `int` | 词表大小 |
| `embedding_dim()` | `int` | Embedding 维度 |
| `max_ctx_available()` | `int` | 最大可用上下文长度 |
| `create_context()` | `std::unique_ptr<Context>` | 创建推理上下文 |
| `tokenize(text, add_bos, add_eos)` | `std::vector<Token>` | 文本转 Token |
| `token_to_str(token)` | `std::string` | Token 转字符串 |
| `tokens_to_str(tokens)` | `std::string` | Token 序列转字符串 |

---

### Context

推理上下文，管理单次推理的状态 (KV Cache、输入缓存、性能统计)。

```cpp
#include "core/context.h"
```

#### 方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `prefill(tokens)` | `Token` | Prefill 阶段：处理 prompt tokens，返回第一个生成 token |
| `decode(prev_token)` | `Token` | Decode 阶段：生成下一个 token |
| `generate(prompt, params, callback)` | `void` | 流式生成 (回调模式) |
| `context_length()` | `int` | 当前上下文长度 |
| `reset()` | `void` | 重置状态 |
| `perf_stats()` | `PerfStats` | 获取性能统计 |
| `reset_perf_stats()` | `void` | 重置性能统计 |

#### 流式生成示例

```cpp
// Token 回调模式
houmo::SamplingParams params;
params.max_tokens = 100;

ctx->generate(tokens, params, [&](houmo::Token token) {
    std::cout << model.token_to_str(token);
    std::cout.flush();
    return true;  // 返回 false 可中断生成
});

// 使用 StreamingDecoder 处理 UTF-8 多字节字符
houmo::StreamingDecoder decoder(model.tokenizer());
ctx->generate(tokens, params, [&decoder](houmo::Token token) {
    std::string chunk = decoder.decode(token);
    if (!chunk.empty()) {
        std::cout << chunk << std::flush;
    }
    return true;
});
```

---

### Sampler

Token 采样器，实现多种采样策略。

```cpp
#include "modules/sampler.h"
```

#### 构造函数

```cpp
explicit Sampler(const SamplingParams& params);
```

#### 方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `sample(logits, size)` | `Token` | 从 logits 采样一个 token |
| `sample(logits, size, previous_tokens)` | `Token` | 带重复惩罚的采样 |
| `set_params(params)` | `void` | 更新采样参数 |
| `params()` | `const SamplingParams&` | 获取当前采样参数 |

#### 采样流程

```
logits → repetition_penalty → presence_penalty → top_k → top_p → min_p → temperature → argmax
```

---

### HfTokenizer

HuggingFace Tokenizer 包装类。

```cpp
#include "modules/tokenizer.h"
```

#### 构造函数

```cpp
explicit HfTokenizer(const std::string& tokenizer_json_path);
```

#### 方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `encode(text, add_bos, add_eos)` | `std::vector<Token>` | 编码文本为 Token ID |
| `decode(token)` | `std::string` | 解码单个 Token |
| `decode(tokens)` | `std::string` | 解码 Token 序列 |
| `bos_token_id()` | `Token` | 获取 BOS token ID |
| `eos_token_id()` | `Token` | 获取 EOS token ID |
| `pad_token_id()` | `Token` | 获取 PAD token ID |
| `vocab_size()` | `int` | 获取词表大小 |

---

### Embedding

Embedding 查找表，支持 Token ID 到 embedding 向量的转换。

```cpp
#include "modules/embedding.h"
```

#### 构造函数

```cpp
Embedding(const std::string& path, int hidden_dim, int max_seq_len);
```

#### 方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `lookup(token)` | `const float*` | 查找单个 token 的 embedding |
| `lookup(tokens, output)` | `void` | 批量查找 |
| `vocab_size()` | `int` | 词表大小 |
| `hidden_dim()` | `int` | Embedding 维度 |

#### 特性

- **Zero-copy 优化**: 单 token 查找直接返回指针，无拷贝
- **自动计算 vocab_size**: 根据文件大小和 hidden_dim 自动计算
- **Half 精度**: 支持 float16 格式的 embedding 权重

---

### StreamingDecoder

UTF-8 流式解码器，处理多字节字符的滑动窗口解码。

```cpp
#include "modules/streaming_decoder.h"
```

#### 方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `decode(token)` | `std::string` | 解码 token，返回完整字符（可能为空） |

---

## 数据结构

### ModelInfo

模型元数据结构体。

```cpp
struct ModelInfo {
    ModelType type;              // 模型类型 (LLM/VLM)
    std::string model_name;      // 模型名称
    int n_batch = 0;             // Batch 大小
    int n_vocab = 0;             // 词表大小
    int n_embd = 0;              // Embedding 维度
    int n_layer = 0;             // Transformer 层数
    int n_ctx = 0;               // 上下文长度
    int prefill_length = 0;      // Prefill 序列长度
    int kv_cache_layers = 0;     // KV Cache 层数
    int n_logits = 0;            // Logits 维度
};
```

#### 支持的模型

| 模型名称 | 类型 | 说明 |
|----------|------|------|
| `qwen3-4b` | LLM | Qwen3 纯文本模型 |
| `qwen3.5-0.8b` | LLM/VLM | Qwen3.5 多模态模型 |
| `qwen3.5-2b` | LLM/VLM | Qwen3.5 多模态模型 |
| `qwen3.5-4b` | LLM/VLM | Qwen3.5 多模态模型 |
| `qwen3.6-27b` | LLM/VLM | Qwen3.6 多模态模型 |
| `qwen2.5-7b` | LLM | Qwen2.5 纯文本模型 |
| `qwen3-vl-4b` | VLM | Qwen3-VL 视觉语言模型 |
| `qwen3-vl-8b` | VLM | Qwen3-VL 视觉语言模型 |
| `qwen2.5-vl-7b` | VLM | Qwen2.5-VL 视觉语言模型 |
| `deepseek-8b` | LLM | DeepSeek 模型 |
| `gemma4-26b-a4b` | VLM | Gemma4 视觉语言模型 |
| `copaw-flash-9b` | VLM | CoPaw-Flash 模型 |

---

### ModelConfig

模型配置结构体。

```cpp
struct ModelConfig {
    // 运行时参数
    std::vector<int> devices = {0};  // 设备 ID 列表
    int batch_size = 1;              // Batch 大小
    bool lazy_mode = false;          // 延迟加载模式

    // 模型路径
    std::string prefill_path;        // Prefill 模型路径 (.hmm)
    std::string decode_path;         // Decode 模型路径 (.hmm)
    std::string embedding_path;      // Embedding 权重路径 (.bin)
    std::string tokenizer_path;      // Tokenizer 词汇表路径 (.json)
    std::string vision_path;         // Vision 模型路径 (.hmm)，VLM 专用

    // 扩展参数
    std::map<std::string, std::string> extra_params;
};
```

---

### SamplingParams

采样参数结构体。

```cpp
struct SamplingParams {
    float temperature = 1.0f;        // 温度参数
    float top_p = 1.0f;              // Top-P 采样
    int top_k = 1;                   // Top-K 采样 (1 = greedy)
    float repetition_penalty = 1.0f; // 重复惩罚
    int penalty_last_n = 64;         // 惩罚窗口
    int max_tokens = 0;              // 最大生成 token 数 (0 = 无限制)
    std::vector<Token> stop_tokens;  // 停止 token
    float frequency_penalty = 0.0f;  // 频率惩罚
    float presence_penalty = 1.5f;   // 存在惩罚
    float min_p = 0.0f;              // Min-P 采样
    bool greedy = false;             // 贪心采样

    // Tokenize 选项
    bool add_bos = false;
    bool add_eos = false;
};
```

---

### 基础类型

```cpp
namespace houmo {

using Token = int32_t;

constexpr Token TokenNull = -1;   // 空 Token
constexpr Token TokenBos = -2;    // 序列开始 Token
constexpr Token TokenEos = -3;    // 序列结束 Token

enum class ModelType {
    LLM,    // 纯文本大语言模型
    VLM,    // 视觉语言模型
    ASR,    // 语音识别模型
    TTS,    // 语音合成模型
};

}
```

---

## 构建依赖

### 依赖项

- C++17 编译器
- CMake >= 3.16
- TCIM Runtime (tcim_lite)
- tokenizers_cpp (HuggingFace tokenizers C++)
- half.hpp (半精度浮点数)
- GTest (单元测试)

### 环境变量

```bash
export TCIM_RUNTIME_PATH=$DADAO_VENV/lib/python3.12/site-packages/tcim_lite
```

### 构建命令

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```
