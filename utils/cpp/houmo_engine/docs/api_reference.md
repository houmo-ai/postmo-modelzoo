# Houmo Inference Framework API Reference

> 源码版本：`0.1.0`
>
> 文档更新：2026-07-22

本文档只描述当前公共头文件中已经存在的接口。具体模型目录可能在这些基类之上提供更多 API。

## 基础类型

头文件：`include/base/houmo.h`

### Token、异常与版本

```cpp
using Token = int32_t;

constexpr Token TokenNull = -1;
constexpr Token TokenBos = -2;
constexpr Token TokenEos = -3;

class Exception : public std::runtime_error;

std::string version();
std::string build_info();
```

`version()` 当前返回 `0.1.0`。

### ModelType / ModelKind / CheckResult

```cpp
enum class ModelType { LLM, VLM, ASR, TTS };
enum class ModelKind { LLM, VLM, ASR, TTS };

struct CheckResult {
  bool valid = false;
  std::string error_message;
};
```

`CheckResult` 当前仅定义数据结构，基础库没有对应的公共 `Check()` 函数。

### ModelConfig

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

字段只负责传递配置，基类不会自动校验路径或加载文件。

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

模型子类负责填充 `ModelInfo`。`type` 没有默认初始化值，使用前必须赋值。

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

当前 `Sampler` 使用 `top_k`、`top_p`、`temperature`、`repetition_penalty` 和 `presence_penalty`。`max_tokens`、`stop_tokens`、`penalty_last_n`、`add_bos`、`add_eos`、`language` 由调用方或模型实现消费。`frequency_penalty`、`min_p` 和 `greedy` 当前未被 `Sampler` 使用。

### PerfStats

`PerfStats` 保存 prefill、decode、E2E、TTFT、TPOT、TPS、视觉编码耗时、token 数和预留内存指标。`PerfProfiler::to_perf_stats()` 负责填充其中可从 profiler 推导的字段。

## ModelFactory

头文件：`include/core/model_factory.h`

### ModelSeries

当前枚举值：

| 枚举 | 规范字符串 | 额外输入别名 |
|------|------------|--------------|
| `kUnknown` | `unknown` | - |
| `kQwen3LLM` | `qwen3_llm` | `qwen3` |
| `kQwen35MLLM` | `qwen35_mllm` | `qwen35` |
| `kQwen3VLM` | `qwen3_vlm` | - |
| `kWhisperASR` | `whisper_asr` | `whisper` |
| `kGlmAsr` | `glm_asr` | `glm-asr` |
| `kQwen3Asr` | `qwen3_asr` | `qwen3-asr` |

转换接口：

```cpp
std::string ModelSeriesToString(ModelSeries series);
ModelSeries StringToModelSeries(const std::string& str);
```

### 工厂接口

```cpp
template <typename ModelT>
class ModelFactory {
 public:
  using Creator =
      std::function<std::unique_ptr<ModelT>(const ModelConfig&)>;

  static void Register(const std::string& name,
                       ModelSeries series,
                       Creator creator,
                       const std::string& description = "");
  static std::unique_ptr<ModelT> Create(ModelSeries series,
                                        const ModelConfig& config);
  static std::unique_ptr<ModelT> Create(const std::string& name,
                                        const ModelConfig& config);
  static std::vector<std::string> ListRegisteredTypes();
  static std::vector<RegistryEntry> GetRegisteredModels();
  static bool IsRegistered(ModelSeries series);
  static bool IsRegistered(const std::string& name);
};
```

每个 `ModelT` 都有独立的注册表。`Create()` 找不到注册项时输出错误并返回 `nullptr`，不会抛异常。重复名称注册会覆盖旧条目。

注册宏：

```cpp
REGISTER_MODEL(LLMModel, your_model, ModelSeries::kQwen3LLM,
               [](const ModelConfig& config) {
                 return std::make_unique<YourModel>(config);
               },
               "Your model");
```

宏会使用标识符文本作为注册名，即上例注册名为 `your_model`。

## LLMModel

头文件：`include/core/llm_model.h`

`LLMModel` 构造函数只保存 `ModelConfig`，不加载任何资源。

| 接口 | 说明 |
|------|------|
| `type()` | 默认返回 `ModelType::LLM` |
| `tokenize(text, add_bos, add_eos)` | 调用已加载的 `HfTokenizer` |
| `token_to_str(token)` | 解码单 token |
| `tokens_to_str(tokens)` | 解码 token 序列 |
| `vocab_size()` | 返回 embedding 词表大小 |
| `embedding_dim()` | 返回 embedding hidden dimension |
| `max_ctx_available()` | 返回 `ModelInfo::n_ctx` |
| `model_info()` | 返回 `ModelInfo` 副本 |
| `create_context(n_ctx)` | 基类实现返回空指针，模型子类应覆盖 |
| `has_tokenizer()` | 判断 tokenizer 是否已加载 |
| `bos_token_id()` / `eos_token_id()` | 返回特殊 token ID |
| `tokenizer()` / `tokenizer_module()` | 返回共享 tokenizer |
| `prefill_module()` / `decode_module()` | 返回 TCIM module |
| `embedding()` | 返回共享 embedding |
| `prefill_input_map()` / `decode_input_map()` | 返回可修改 tensor map 引用 |
| `prefill_length()` / `attn_idx_start()` | 返回模型子类设置的参数 |

模型子类可直接访问 `config_`、`tokenizer_`、`embedding_`、TCIM modules、device/weight manager、`info_` 和输入 map。

## VLMModel

头文件：`include/core/vlm_model.h`

`VLMModel` 继承 `LLMModel`，并提供：

| 接口 | 说明 |
|------|------|
| `type()` | 返回 `ModelType::VLM` |
| `vision_module()` | 返回 vision TCIM module |
| `encode_image(data, width, height, channels)` | 基类占位接口，当前返回空结果 |
| `create_context(n_ctx)` | 基类占位接口，当前返回空指针 |
| `vision_input_map()` | 返回可修改 vision tensor map 引用 |

默认视觉参数为 image size `448`、patch size `16`、hidden size `0`，模型子类可覆盖这些受保护成员。

## Context

头文件：`include/core/context.h`

`Context` 是请求级状态基类。下列推理接口在基类中都是占位实现：

```cpp
virtual Token prefill(const std::vector<Token>& tokens);  // TokenNull
virtual Token decode(Token prev_token);                   // TokenNull
virtual void set_image(const std::string& image_path);    // no-op
virtual void generate(const std::vector<Token>& prompt,
                      const SamplingParams& params,
                      std::function<bool(Token)> callback);  // no-op
```

通用状态接口：

| 接口 | 当前行为 |
|------|----------|
| `set_keep_history()` / `keep_history()` | 设置/读取历史保留标志，默认 `true` |
| `context_length()` | 返回当前上下文长度 |
| `reset()` | 清零上下文长度并清空 `generated_ids_` |
| `set_sampler(params)` | 新建 `Sampler` |
| `sampler()` | 返回裸指针，可为空 |
| `perf_stats()` / `reset_perf_stats()` | 读取/重置汇总性能数据 |
| `profiler()` | 访问层级 profiler |

`reset()` 不会重置 sampler、profiler、模型 KV cache 或子类状态；子类需要按模型需求覆盖。

## ASRModel / ASRContext

头文件：`include/core/asr_model.h`

### ASRModel

`ASRModel` 与 `LLMModel` 是相互独立的基类。当前公共 getter 包括 `n_mels()`、`n_frames()`、`num_heads()`、`cache_max_len()` 和 `num_decode_layers()`。

子类必须实现：

- `create_context()`
- `sot_token_id()`
- `lang_token_id()`
- `transcribe_token_id()`
- `notimestamps_token_id()`
- `eos_token_ids()`
- `supports_language_detection()`

### ASRContext

`ASRContext` 继承 `Context`，但通过 `Context(nullptr, n_ctx)` 构造，因此 `Context::model_` 为空；ASR 模型通过 `asr_model()` 访问。

子类必须实现用户入口：

- `Encode(const std::vector<float>&, int, int)`
- `DetectLanguage()`
- `BuildPrompt(Token)`
- `Transcribe(path, params, callback)`
- `set_language()`

基类提供受保护模板方法 `do_encode()`、`do_detect_language()`、`do_prefill()`、`do_decode()` 和 `fill_perf_info()`。它们统一生成以下 profiler path：

```text
transcribe.encode.preprocess
transcribe.encode.inference
transcribe.encode.postprocess
transcribe.detect_lang.preprocess
transcribe.detect_lang.inference
transcribe.detect_lang.postprocess
transcribe.prefill.preprocess
transcribe.prefill.inference
transcribe.prefill.postprocess
transcribe.decode.preprocess
transcribe.decode.inference
transcribe.decode.postprocess
```

语言检测三个 hook 有默认 no-op 实现，其他 hook 为纯虚函数。

`fill_perf_info()` 使用 inference 子阶段计算 encode/prefill/decode 时间；`detect_lang_time` 查询父路径 `transcribe.detect_lang`；`n_chunks` 使用 `transcribe.encode.inference` 的调用次数。

## AudioProcessor

头文件：`include/modules/audio_processor.h`

### 配置

默认值：16 kHz、80 mel、30 秒 PCM chunk、30 秒 encoder window、FFT 400、hop 160、window 400、4 个特征线程、`kCenterPad`、0 到 8000 Hz。

### 接口

| 接口 | 当前行为 |
|------|----------|
| `LoadAudio(path)` | miniaudio 直接解码为配置采样率、单声道、float32 PCM；失败返回空 `AudioData` |
| `ChunkPCM(audio)` | 按 `chunk_seconds` 切分，不在此阶段补零 |
| `ExtractFeatures(audio)` | PCM 补零/截断到 encoder window，计算 log-Mel 并输出 FP16 |
| `Process(path)` | 加载、分块并逐块提取特征 |
| `feature_dim()` / `n_mels()` | 返回配置的 mel 数 |

`kWhisper` 模式使用 Whisper 风格 padding，并将输出帧数限制为 `encoder_window_seconds * sample_rate / hop_length`；`kCenterPad` 使用中心反射 padding。

## HmImageProcessor

头文件：`include/modules/image_processor.h`

注意：`ImageDims`、`ProcessedImage` 和 `HmImageProcessor` 当前定义在全局命名空间，而不是 `houmo`。

| 接口 | 当前行为 |
|------|----------|
| `LoadAndProcess(path)` | 强制解码 RGB；失败时返回目标尺寸、像素值 114 的 fallback 图像 |
| `LoadAndProcessBatch(paths)` | 顺序处理多张图像 |
| `ToFP16Tensor(image)` | HWC RGB uint8 转 `[C=3,T=2,H,W]`，两帧重复，保留 0..255 原始范围 |
| `GetTargetDims()` | 返回目标宽高和 3 通道 |

`use_v1=true` 时保持宽高比，图像写入左上角，剩余区域填充 114；`use_v1=false` 时直接 resize 到目标尺寸。

## Embedding

头文件：`include/modules/embedding.h`

```cpp
Embedding(const std::string& path,
          int hidden_dim = 0,
          int max_seq_len = 0);
```

`hidden_dim` 必须大于 0。词表大小根据文件字节数和 FP16 hidden dimension 自动计算。

| 接口 | 说明 |
|------|------|
| `token_embedding(token)` | 返回单 token embedding；越界返回 `nullptr` |
| `token_embedding(tokens)` | 写入内部复用 buffer 并返回指针；超过 `max_seq_len` 时抛异常 |
| `vocab_size()` / `hidden_dim()` | 返回权重信息 |

批量接口使用内部可变 buffer，同一个 `Embedding` 实例上的并发调用需要调用方同步。

## HfTokenizer

头文件：`include/modules/tokenizer.h`

| 接口 | 说明 |
|------|------|
| `encode(text, add_bos=true, add_eos=false, add_special_tokens=false)` | 编码文本，并可手动添加 BOS/EOS |
| `decode(token, skip_special_tokens=false)` | 解码单 token |
| `decode(tokens, skip_special_tokens=false)` | 解码 token 序列 |
| `bos_token_id()` / `eos_token_id()` / `pad_token_id()` | 特殊 token ID |
| `token_to_id(token)` | token 字符串转 ID |

若 tokenizer 未提供 BOS，构造函数尝试使用 `<|endoftext|>`；若未提供 PAD，则使用 BOS。

## Sampler

头文件：`include/modules/sampler.h`

当前处理流水线：

```text
logits
  -> repetition/presence penalties
  -> top-k logits mask
  -> temperature
  -> softmax
  -> top-p probability mask and renormalization
  -> argmax
```

`top_k == 1` 时走快速路径：应用 penalty 后直接 `argmax`。其他路径仍以 `argmax` 选择最终 token，因此当前实现是确定性的，不进行概率随机抽样。

## StreamingDecoder

头文件：`include/modules/streaming_decoder.h`

`StreamingDecoder` 缓存尚不能组成完整有效 UTF-8 字符的 token，并在后续 token 到达后统一解码。

| 接口 | 说明 |
|------|------|
| `decode(token)` | 返回本次可安全输出的字符串，可能为空 |
| `init(tokens)` | 手动 prefill/decode 模式下初始化 token 计数 |
| `reset()` | 清空生成和待解码状态 |
| `token_count()` | 返回已接收生成 token 数 |

## PerfProfiler

头文件：`include/modules/perf_profiler.h`

性能统计由拼写为 `HOUOMO_ENABLE_PROFILING` 的宏控制，默认值为 `1`。

主要接口：

- `start(path)` / `stop(path)` / `scope(path)`
- `get_time_ms()` / `get_count()` / `get_avg_time_ms()`
- `get_children()` / `has_stage()`
- `set_root_stage()` / `root_stage()`
- `set_input_tokens()` / `add_output_token()`
- `record_ttft()`
- `e2e_ms()` / `ttft_ms()` / `prefill_tps()` / `decode_tps()` / `overall_tps()`
- `avg_decode_latency_ms()`
- `print_summary(Tree|Table|Compact)`
- `to_perf_stats()` / `reset()`

禁用宏时上述接口保留，但全部退化为 no-op 或返回零值。

## 构建产物

`CMakeLists.txt` 只创建共享库 `houmo_infer`，并链接 `tcim_runtime_lite`、`tokenizer_lib`、`${CMAKE_DL_LIBS}`，Linux 额外链接 `pthread`。

安装规则包含：

- `houmo_infer`
- `tokenizer_lib`
- `convert_embed.py`

当前工程没有 `BUILD_TESTS`、GTest 或 CTest 配置。
