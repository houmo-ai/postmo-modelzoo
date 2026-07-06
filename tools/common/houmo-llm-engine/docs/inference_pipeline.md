# 推理 Pipeline 文档

本文档描述 Houmo Inference Framework 中生成类模型（LLM/VLM）与语音识别模型（ASR）的通用推理流程。具体模型的输入张量名称、位置编码、cache 结构和预处理细节由各模型实现决定。

## 目录

- [生成类模型通用流程](#生成类模型通用流程)
- [LLM Prefill / Decode](#llm-prefill--decode)
- [VLM 扩展点](#vlm-扩展点)
- [ASR 转写流程](#asr-转写流程)
- [ASR 模板方法打点](#asr-模板方法打点)
- [性能指标](#性能指标)
- [文件位置](#文件位置)

---

## 生成类模型通用流程

生成类模型通过 `Context::generate(prompt, params, callback)` 提供 token 级流式输出。模型子类通常在 `generate()` 中组织 `prefill()` 和循环 `decode()`，并用 `PerfProfiler` 记录阶段耗时。

```text
generate(prompt, params, callback)
    │
    ├─► profiler_.reset()
    ├─► profiler_.start("generate")
    ├─► profiler_.set_input_tokens(prompt.size())
    ├─► set_sampler(params)
    │
    ├─► prefill(prompt)
    │       ├─► do_prefill_inference(prompt, sampler)
    │       └─► profiler_.record_ttft()
    │
    ├─► callback(first_token)
    │
    └─► while 未达到停止条件:
            ├─► decode(prev_token)
            │       └─► do_decode_inference(prev_token, sampler)
            ├─► profiler_.add_output_token()
            └─► callback(token)
```

停止条件通常包括：

- 生成 EOS/BOS 等模型定义的停止 token
- 达到 `SamplingParams::max_tokens`
- `Context::context_length_` 达到模型可用上下文上限
- callback 返回 `false`

---

## LLM Prefill / Decode

### Prefill

Prefill 负责处理 prompt tokens，完成 embedding lookup、prefill module 输入设置和首 token 采样。

```text
prefill(tokens)
    │
    ├─► set_sampler()
    ├─► generated_ids_.clear()
    │
    └─► do_prefill_inference(tokens, sampler)
            │
            ├─► 计算 seq_length
            ├─► 计算 prefill_loop_chunk
            │
            └─► for chunk in chunks:
                    │
                    ├─► prefill_preprocess_chunk()
                    │       ├─► 提取当前 chunk tokens
                    │       ├─► 不足 prefill_length 时 padding
                    │       ├─► embedding lookup
                    │       └─► 设置 prefill input tensors
                    │
                    ├─► prefill_inference_chunk()
                    │       └─► prefill_module()->Run() + Sync()
                    │
                    └─► prefill_postprocess()
                            ├─► 获取 logits
                            ├─► sampler->sample()
                            ├─► 更新 context_length_
                            └─► return sampled_token
```

### Decode

Decode 每次处理一个 token，复用模型维护的 KV cache 或其他增量状态。

```text
decode(prev_token)
    │
    └─► do_decode_inference(prev_token, sampler)
            │
            ├─► decode_preprocess(prev_token)
            │       ├─► embedding lookup
            │       ├─► 计算当前位置
            │       └─► 设置 decode input tensors
            │
            ├─► decode_inference()
            │       └─► decode_module()->Run() + Sync()
            │
            └─► decode_postprocess(sampler)
                    ├─► 获取 logits
                    ├─► sampler->sample()
                    ├─► context_length_++
                    └─► return sampled_token
```

---

## VLM 扩展点

`VLMModel` 继承 `LLMModel`，增加 `vision_module_`、`vision_input_map_` 和 `encode_image()` 接口。VLM 子类通常在 Prefill 前或 Prefill 通用设置阶段插入视觉处理。

```text
VLM prefill(tokens)
    │
    ├─► 可选：读取 image_paths_
    ├─► vision_preprocess()
    │       ├─► HmImageProcessor::LoadAndProcess()
    │       └─► 设置 vision input tensors
    │
    ├─► vision_inference()
    │       └─► vision_module()->Run() + Sync()
    │
    ├─► vision_postprocess()
    │       └─► 得到 image embeddings
    │
    └─► LLM prefill path
            ├─► token embedding lookup
            ├─► 注入 image embeddings
            ├─► 设置 position ids / multimodal metadata
            └─► prefill module inference
```

VLM 的通用约束：

- 视觉输入路径和图像状态应属于请求级 `Context`，不要放入全局状态。
- 视觉特征注入应在 prefill 阶段完成，decode 阶段只处理自回归 token。
- 多轮对话中需明确清理或保留图像状态，避免下一轮误复用。
- 视觉阶段建议使用 `generate.vision`、`generate.vision.preprocess`、`generate.vision.inference`、`generate.vision.postprocess` 等 profiler path。

---

## ASR 转写流程

ASR 使用 `ASRModel` / `ASRContext`。与生成类模型不同，ASR 的入口是 `ASRContext::Transcribe(audio_path, params, callback)`，内部先处理音频，再执行 encoder、语言检测、decoder prefill 和 decoder decode。

```text
Transcribe(audio_path, params, callback)
    │
    ├─► profiler_.reset()
    ├─► profiler_.set_root_stage("transcribe")
    ├─► profiler_.start("transcribe")
    │
    ├─► AudioProcessor::Process(audio_path)
    │       ├─► LoadAudio(path)
    │       │       ├─► 读取 wav/mp3/flac 等格式
    │       │       ├─► 重采样到 16kHz
    │       │       ├─► 转单声道
    │       │       └─► 归一化到 [-1, 1]
    │       │
    │       ├─► ChunkPCM(audio)
    │       │       ├─► 按 chunk_seconds 切分
    │       │       └─► 短 chunk 补零
    │       │
    │       └─► ExtractFeatures(chunk)
    │               ├─► STFT
    │               ├─► Mel Filter Bank
    │               ├─► Log compression
    │               └─► FP16 MelFeatures
    │
    ├─► for each feature chunk:
    │       │
    │       ├─► do_encode(mel, n_mels, n_frames)
    │       │       ├─► encode_preprocess_impl()
    │       │       ├─► encode_inference_impl()
    │       │       └─► encode_postprocess_impl()
    │       │
    │       ├─► language_token = do_detect_language()
    │       │       ├─► detect_lang_preprocess_impl()
    │       │       ├─► detect_lang_inference_impl()
    │       │       └─► detect_lang_postprocess_impl()
    │       │
    │       ├─► prompt = BuildPrompt(language_token)
    │       │
    │       ├─► token = do_prefill(prompt)
    │       │       ├─► prefill_preprocess_impl()
    │       │       ├─► prefill_inference_impl()
    │       │       └─► prefill_postprocess_impl()
    │       │
    │       ├─► profiler_.record_ttft()
    │       ├─► callback(token)
    │       │
    │       └─► while token not in eos_token_ids():
    │               ├─► token = do_decode(token)
    │               │       ├─► decode_preprocess_impl()
    │               │       ├─► decode_inference_impl()
    │               │       └─► decode_postprocess_impl()
    │               ├─► profiler_.add_output_token()
    │               └─► callback(token)
    │
    ├─► profiler_.stop("transcribe")
    └─► fill_perf_info(audio_duration)
```

ASR 子类负责决定：

- encode 模型路径如何从 `ModelConfig` 或 `extra_params` 获取
- encoder 输出如何保存并供 prefill/decode 使用
- 是否支持语言检测
- prompt token 的组成方式
- EOS token 集合
- 多 chunk 音频之间是否复用 decoder 状态

---

## ASR 模板方法打点

`ASRContext` 将 ASR 关键阶段封装为模板方法，子类只实现 `_impl` 钩子，基类负责统一 profiler path。

```text
ASRContext::do_encode()
    ├─► transcribe.encode.preprocess  -> encode_preprocess_impl()
    ├─► transcribe.encode.inference   -> encode_inference_impl()
    └─► transcribe.encode.postprocess -> encode_postprocess_impl()

ASRContext::do_detect_language()
    ├─► transcribe.detect_lang.preprocess  -> detect_lang_preprocess_impl()
    ├─► transcribe.detect_lang.inference   -> detect_lang_inference_impl()
    └─► transcribe.detect_lang.postprocess -> detect_lang_postprocess_impl()

ASRContext::do_prefill()
    ├─► transcribe.prefill.preprocess  -> prefill_preprocess_impl()
    ├─► transcribe.prefill.inference   -> prefill_inference_impl()
    └─► transcribe.prefill.postprocess -> prefill_postprocess_impl()

ASRContext::do_decode()
    ├─► transcribe.decode.preprocess  -> decode_preprocess_impl()
    ├─► transcribe.decode.inference   -> decode_inference_impl()
    └─► transcribe.decode.postprocess -> decode_postprocess_impl()
```

`fill_perf_info(audio_duration)` 从 profiler 中汇总 `ASRPerfInfo`：

- `audio_load_time`：音频加载和特征提取耗时
- `encode_time`：encoder 推理耗时
- `detect_lang_time`：语言检测耗时
- `prefill_time`：decoder prefill 推理耗时
- `decode_time`：decoder decode 推理耗时
- `total_time`：端到端耗时
- `ttft_time`：首 token 延迟
- `n_chunks`：encoder 推理次数
- `overall_rtf`：`total_time / audio_duration`
- `inference_rtf`：`(encode + prefill + decode) / audio_duration`
- `decode_tps` / `overall_tps`：token 吞吐

---

## 性能指标

### 生成类模型建议路径

| 阶段 | 说明 |
|------|------|
| `generate` | 端到端生成耗时 |
| `generate.vision` | 视觉处理耗时，VLM 使用 |
| `generate.prefill` | Prefill 总耗时 |
| `generate.prefill.preprocess_chunk` | 分块预处理 |
| `generate.prefill.inference_chunk` | 分块推理 |
| `generate.prefill.postprocess` | 后处理与采样 |
| `generate.decode` | Decode 总耗时 |
| `generate.decode.preprocess` | Decode 预处理 |
| `generate.decode.inference` | Decode 推理 |
| `generate.decode.postprocess` | Decode 后处理 |

### ASR 建议路径

| 阶段 | 说明 |
|------|------|
| `transcribe` | 端到端转写耗时 |
| `transcribe.audio_load` | 音频加载、切分、特征提取 |
| `transcribe.encode.preprocess` | Encoder 输入准备 |
| `transcribe.encode.inference` | Encoder 推理 |
| `transcribe.encode.postprocess` | Encoder 输出整理 |
| `transcribe.detect_lang.*` | 语言检测，模型支持时使用 |
| `transcribe.prefill.*` | Decoder prefill |
| `transcribe.decode.*` | Decoder 自回归 decode |

---

## 文件位置

| 组件 | 头文件 | 源文件 |
|------|--------|--------|
| 基础类型 | `include/base/houmo.h` | - |
| Context 基类 | `include/core/context.h` | `src/core/context.cc` |
| LLMModel 基类 | `include/core/llm_model.h` | `src/core/llm_model.cc` |
| VLMModel 基类 | `include/core/vlm_model.h` | `src/core/vlm_model.cc` |
| ASRModel / ASRContext | `include/core/asr_model.h` | `src/core/asr_model.cc` |
| ModelFactory | `include/core/model_factory.h` | `src/core/model_factory.cc` |
| AudioProcessor | `include/modules/audio_processor.h` | `src/modules/audio_processor.cc` |
| ImageProcessor | `include/modules/image_processor.h` | `src/modules/image_processor.cc` |
| Tokenizer | `include/modules/tokenizer.h` | `src/modules/tokenizer.cc` |
| Embedding | `include/modules/embedding.h` | `src/modules/embedding.cc` |
| Sampler | `include/modules/sampler.h` | `src/modules/sampler.cc` |
| StreamingDecoder | `include/modules/streaming_decoder.h` | `src/modules/streaming_decoder.cc` |
| PerfProfiler | `include/modules/perf_profiler.h` | `src/modules/perf_profiler.cc` |
