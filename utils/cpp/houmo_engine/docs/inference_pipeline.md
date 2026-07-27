# 推理 Pipeline

本文档区分基础库已经实现的流程与模型子类需要实现的流程。当前基础库不提供可直接运行的 LLM/VLM `generate()` 或 ASR `Transcribe()`；它提供状态容器、模块访问器、预处理和性能打点工具。

## 生成类模型

### 基础库边界

`Context` 的下列方法当前是占位实现：

```text
prefill(tokens)              -> TokenNull
decode(prev_token)           -> TokenNull
set_image(path)              -> no-op
generate(prompt, params, cb) -> no-op
```

因此完整生成循环必须由具体模型 `Context` 子类实现。基础类仅提供：

- `context_length_` 和 `generated_ids_`
- `keep_history_`
- `Sampler`
- `PerfStats` 和 `PerfProfiler`
- 指向 `LLMModel` 的非 owning 指针

### 推荐生成流程

下面是适配层可采用的流程，不是基础类中的现成实现：

```text
generate(prompt, params, callback)
    |
    +-> profiler.reset()
    +-> profiler.set_root_stage("generate")
    +-> profiler.start("generate")
    +-> profiler.set_input_tokens(prompt.size())
    +-> set_sampler(params)
    |
    +-> model-specific prefill(prompt)
    |     +-> embedding lookup
    |     +-> bind prefill tensors
    |     +-> TCIM Run + Sync
    |     +-> sample first token
    |
    +-> profiler.record_ttft()
    +-> profiler.add_output_token()
    +-> callback(first_token)
    |
    +-> loop model-specific decode(previous_token)
    |     +-> embedding lookup
    |     +-> update position/cache input
    |     +-> TCIM Run + Sync
    |     +-> sample next token
    |     +-> profiler.add_output_token()
    |     +-> callback(token)
    |
    +-> profiler.stop("generate")
    +-> perf_stats_ = profiler.to_perf_stats()
```

模型实现需要自行处理：

- `SamplingParams::max_tokens`
- `SamplingParams::stop_tokens`
- 模型 BOS/EOS token
- context 上限
- callback 返回 `false`
- `generated_ids_` 和 `context_length_` 更新
- KV cache 清理或复用

`Context::reset()` 只清零上下文长度并清空生成 token，不会重置 KV cache、sampler、profiler 或图像状态。

## LLM Model 数据流

`LLMModel` 提供资源容器，但不规定加载顺序。典型子类流程如下：

```text
ModelConfig
  -> DevManager(config.devices)
  -> WeightManager
  -> load prefill/decode TCIM modules
  -> load Embedding(path, hidden_dim, prefill_length)
  -> load HfTokenizer(tokenizer_path)
  -> fill ModelInfo
  -> initialize prefill_input_map / decode_input_map
```

请求阶段通常通过以下 getter 访问资源：

```text
LLMModel
  +-> embedding()
  +-> tokenizer()
  +-> prefill_module()
  +-> decode_module()
  +-> prefill_input_map()
  +-> decode_input_map()
```

输入 tensor 名称、shape、position ID、attention mask、KV cache 和 logits 位置均由具体模型决定。

## VLM 扩展

`VLMModel` 只增加 vision module 和 input map。`encode_image()` 在基类中不执行实际编码。

推荐的请求级流程：

```text
set_image(path)
  -> store image path in VLM Context

prefill(prompt)
  -> HmImageProcessor::LoadAndProcess(path)
  -> optional ToFP16Tensor(processed_image)
  -> bind vision_input_map
  -> vision_module Run + Sync
  -> extract image embeddings
  -> inject/replace prompt embeddings
  -> run normal language prefill
```

### 图像预处理现状

`HmImageProcessor` 当前有两种模式：

| 模式 | 行为 |
|------|------|
| `use_v1=true` | 保持宽高比，将缩放结果放在目标图像左上角，其余区域填充 114 |
| `use_v1=false` | 直接 resize 到目标宽高 |

图片始终转换为 RGB。加载失败时返回填充值为 114 的 fallback 图像。`ToFP16Tensor()` 生成 `[3, 2, H, W]`，两个 temporal frame 相同，像素值保持 0..255，不进行 mean/std normalization。

具体模型如果需要归一化、patch reshape 或不同 layout，应在适配层继续处理。

## ASR Pipeline

### 基础库边界

`ASRContext::Transcribe()` 是纯虚函数。基础库实现的是以下受保护模板方法：

```text
do_encode(mel)
  +-> transcribe.encode.preprocess  -> encode_preprocess_impl()
  +-> transcribe.encode.inference   -> encode_inference_impl()
  +-> transcribe.encode.postprocess -> encode_postprocess_impl()

do_detect_language()
  +-> transcribe.detect_lang.preprocess  -> detect_lang_preprocess_impl()
  +-> transcribe.detect_lang.inference   -> detect_lang_inference_impl()
  +-> transcribe.detect_lang.postprocess -> detect_lang_postprocess_impl()

do_prefill(tokens)
  +-> transcribe.prefill.preprocess  -> prefill_preprocess_impl()
  +-> transcribe.prefill.inference   -> prefill_inference_impl()
  +-> transcribe.prefill.postprocess -> prefill_postprocess_impl()

do_decode(token)
  +-> transcribe.decode.preprocess  -> decode_preprocess_impl()
  +-> transcribe.decode.inference   -> decode_inference_impl()
  +-> transcribe.decode.postprocess -> decode_postprocess_impl()
```

语言检测 hook 默认是 no-op 并返回 `0`，其他 hook 必须由 ASR 子类实现。

### 音频预处理

```text
AudioProcessor::Process(path)
  -> LoadAudio(path)
       -> miniaudio decode
       -> configured sample rate
       -> mono float32 PCM
  -> ChunkPCM(audio)
       -> split by chunk_seconds
       -> preserve actual chunk duration
       -> no zero padding here
  -> ExtractFeatures(chunk)
       -> pad/truncate PCM to encoder_window_seconds
       -> FFT + Mel filterbank + log compression
       -> normalize log-Mel range
       -> convert to FP16 [n_mels, num_frames]
```

`MelFeatures::duration` 保存补零前的实际音频时长。

`AudioFeatureMode::kCenterPad` 使用中心反射 padding。`kWhisper` 使用 Whisper 风格 padding，并限制最终帧数为 encoder window 对应的帧数。

### 推荐 Transcribe 流程

以下流程由具体 ASR Context 实现：

```text
Transcribe(path, params, callback)
  -> profiler.reset()
  -> profiler.set_root_stage("transcribe")
  -> profiler.start("transcribe")
  -> set_language(params.language)
  -> scope("transcribe.audio_load")
       -> AudioProcessor::Process(path)
  -> for each MelFeatures chunk
       -> convert FP16 features if model hook expects float
       -> do_encode(mel, feature_dim, num_frames)
       -> DetectLanguage() or do_detect_language()
       -> BuildPrompt(language_token)
       -> first_token = do_prefill(prompt)
       -> profiler.record_ttft()
       -> profiler.add_output_token()
       -> callback(first_token)
       -> decode loop with do_decode()
  -> profiler.stop("transcribe")
  -> fill_perf_info(total_actual_audio_duration)
```

公共 API 存在一个类型边界：`AudioProcessor` 输出 `std::vector<float16>`，而 `ASRContext::Encode()` 和 `do_encode()` 接收 `std::vector<float>`。适配层必须显式转换，或使用自己的特征输入路径。

### ASR 性能汇总

`fill_perf_info()` 当前按以下方式汇总：

| 字段 | 数据源 |
|------|--------|
| `audio_load_time` | `transcribe.audio_load` |
| `encode_time` | `transcribe.encode.inference` |
| `detect_lang_time` | `transcribe.detect_lang` |
| `prefill_time` | `transcribe.prefill.inference` |
| `decode_time` | `transcribe.decode.inference` |
| `total_time` | profiler E2E |
| `ttft_time` | `record_ttft()` 结果 |
| `output_tokens` | `add_output_token()` 次数 |
| `n_chunks` | encode inference 调用次数 |

`inference_rtf` 使用 `(encode + prefill + decode) / audio_duration`，不包含语言检测、预处理和后处理耗时。

## 采样流程

当前 `Sampler` 是确定性的：

```text
top_k == 1
  -> penalties
  -> argmax

top_k != 1
  -> penalties
  -> top-k mask
  -> temperature
  -> softmax
  -> top-p mask + renormalize
  -> argmax
```

因此 top-p 和 temperature 会改变最终最大概率 token 的比较空间，但不会按概率分布随机抽样。`frequency_penalty`、`min_p`、`greedy` 和 `penalty_last_n` 当前没有在 `Sampler` 内实现。

## 字符串流式输出

`Context` callback 输出 token。需要字符串流式输出时使用 `StreamingDecoder`：

```cpp
houmo::StreamingDecoder decoder(model.tokenizer());

context->generate(prompt, params, [&](houmo::Token token) {
  std::cout << decoder.decode(token);
  return true;
});
```

开始新的会话前调用 `reset()`。手动执行 prefill + decode 时，可先调用 `init(prompt_tokens)` 初始化 token 计数。

## 性能打点建议

基础库不强制 LLM/VLM 的 stage 名称，但 `PerfProfiler` 的吞吐和 `PerfStats` 转换按默认根阶段 `generate` 工作。建议使用：

```text
generate
generate.vision
generate.vision.preprocess
generate.vision.inference
generate.vision.postprocess
generate.prefill
generate.prefill.preprocess
generate.prefill.inference
generate.prefill.postprocess
generate.decode
generate.decode.preprocess
generate.decode.inference
generate.decode.postprocess
```

ASR 应使用基类模板方法已经固定的 `transcribe.*` 路径。
