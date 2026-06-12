# 推理 Pipeline 文档

本文档描述四个模型类的推理执行流程。

## 目录

- [Qwen3Context (LLM)](#qwen3context-llm)
- [Qwen35MLLMContext (VLM)](#qwen35mllmcontext-vlm)
- [Qwen3VLMContext (VLM)](#qwen3vlmcontext-vlm)
- [WhisperContext (ASR)](#whispercontext-asr)
- [对比总结](#对比总结)

---

## Qwen3Context (LLM)

纯文本语言模型，无视觉处理。

### Prefill Pipeline

```
prefill(tokens)
    │
    ├─► set_sampler()          // 设置采样器
    ├─► generated_ids_.clear()
    │
    └─► do_prefill_inference(tokens, sampler)
            │
            ├─► 计算 seq_length, prefill_loop_chunk
            │
            └─► for chunk in [0, prefill_loop_chunk):
                    │
                    ├─► prefill_preprocess_chunk(chunk, tokens, ...)
                    │       │
                    │       ├─► 提取当前 chunk tokens
                    │       ├─► Padding 到 prefill_length
                    │       ├─► 获取 embedding
                    │       └─► 设置 prefill 输入 tensors
                    │
                    └─► prefill_inference_chunk()
                            │
                            └─► prefill_module->Run() + Sync()
                │
                └─► prefill_postprocess(sampler, seq_length)
                        │
                        ├─► 获取输出 logits
                        ├─► sampler->sample() 采样
                        ├─► context_length_ += seq_length
                        └─► return sampled_token
```

### Decode Pipeline

```
decode(prev_token)
    │
    └─► do_decode_inference(prev_token, sampler)
            │
            ├─► decode_preprocess(prev_token)
            │       │
            │       ├─► 获取 prev_token embedding
            │       └─► 设置 decode 输入 tensors
            │
            ├─► decode_inference()
            │       │
            │       └─► decode_module->Run() + Sync()
            │
            └─► decode_postprocess(sampler)
                    │
                    ├─► 获取输出 logits
                    ├─► sampler->sample() 采样
                    ├─► context_length_++
                    └─► return sampled_token
```

---

## Qwen35MLLMContext (VLM)

视觉语言模型，支持多图输入，使用 M-RoPE。

### Prefill Pipeline

```
prefill(tokens)
    │
    ├─► use_vlm_ = false
    ├─► set_sampler()
    ├─► generated_ids_.clear()
    │
    └─► do_prefill_inference(tokens, sampler)
            │
            ├─► padded_tokens = pad_visual_token(tokens)  // 扩展 image tokens
            │
            ├─► run_vision()                              // ★ 视觉处理
            │       │
            │       ├─► vision_preprocess(image_idx)
            │       │       ├─► 加载并预处理图像
            │       │       ├─► 计算 image_grid_thw
            │       │       └─► 设置 vision 输入
            │       │
            │       ├─► vision_inference()
            │       │       └─► vision_module->Run() + Sync()
            │       │
            │       └─► vision_postprocess(image_idx)
            │               └─► 拼接 flat_image_embeds_
            │
            ├─► prefill_common_setup(padded_tokens)      // ★ 通用设置 (一次)
            │       │
            │       ├─► image_embed_offset_ = 0
            │       ├─► use_vlm_ = !flat_image_embeds_.empty()
            │       ├─► get_rope_index() → position_ids_3d, rope_deltas_
            │       └─► 计算 seq_length, prefill_loop_chunk
            │
            └─► for chunk in [0, prefill_loop_chunk):
                    │
                    ├─► prefill_preprocess_chunk(chunk, padded_tokens, ...)
                    │       │
                    │       ├─► 提取当前 chunk tokens
                    │       ├─► Padding 到 prefill_length
                    │       ├─► 获取 embedding
                    │       ├─► scatter_image_embeds()  // 替换 image embedding
                    │       └─► 设置 prefill 输入 (含 3D position_ids)
                    │
                    └─► prefill_inference_chunk()
                            └─► prefill_module->Run() + Sync()
                │
                └─► prefill_postprocess(sampler, seq_length)
                        │
                        ├─► 获取输出 logits
                        ├─► sampler->sample() 采样
                        ├─► flat_image_embeds_.clear()
                        ├─► image_paths_.clear()        // 清空，防止下轮误执行
                        ├─► context_length_ += seq_length
                        └─► return sampled_token
```

### Decode Pipeline

```
decode(prev_token)
    │
    └─► do_decode_inference(prev_token, sampler)
            │
            ├─► decode_preprocess(prev_token)
            │       │
            │       ├─► 获取 prev_token embedding
            │       ├─► 计算 position (含 rope_deltas_ 调整)
            │       └─► 设置 decode 输入 (含 3D position_ids)
            │
            ├─► decode_inference()
            │       └─► decode_module->Run() + Sync()
            │
            └─► decode_postprocess(sampler)
                    │
                    ├─► 获取输出 logits
                    ├─► sampler->sample() 采样
                    ├─► context_length_++
                    └─► return sampled_token
```

---

## Qwen3VLMContext (VLM)

视觉语言模型，支持 Deepstack 架构，使用 M-RoPE。

### Prefill Pipeline

```
prefill(tokens)
    │
    ├─► use_vlm_ = !image_paths_.empty()
    ├─► set_sampler()
    ├─► generated_ids_.clear()
    │
    └─► do_prefill_inference(tokens, sampler)
            │
            ├─► run_vision()                              // ★ 视觉处理 (如有)
            │       │
            │       ├─► vision_preprocess(image_idx)
            │       │       ├─► 加载并预处理图像
            │       │       └─► 设置 vision 输入
            │       │
            │       ├─► vision_inference()
            │       │       └─► vision_module->Run() + Sync()
            │       │
            │       └─► vision_postprocess(image_idx)
            │               ├─► 拼接 flat_image_embeds_
            │               ├─► 拼接 deepstack_0/1/2_
            │               └─► 计算 image_grid_thw
            │
            ├─► prefill_common_setup(tokens)              // ★ 通用设置 (一次)
            │       │
            │       ├─► input_ids = expand_image_tokens(tokens)
            │       ├─► 计算 3D position_ids (第一轮) 或 1D (多轮)
            │       ├─► Padding 到 prefill_length
            │       ├─► 获取整个序列 embedding → chunk_embeds_
            │       ├─► scatter_image_embeds()            // 替换 image embedding
            │       ├─► 扩展 position_ids
            │       └─► 准备 deepstack (scatter 到整个序列)
            │
            └─► for chunk in [0, prefill_loop_chunk):
                    │
                    ├─► prefill_preprocess_chunk(chunk, seq_length, ...)
                    │       │
                    │       ├─► 从 chunk_embeds_ 提取当前 chunk
                    │       ├─► 从 position_ids_3d 提取当前 chunk
                    │       ├─► 从 deepstack_0/1/2_ 提取当前 chunk
                    │       └─► 设置 prefill 输入
                    │
                    └─► prefill_inference_chunk()
                            └─► prefill_module->Run() + Sync()
                │
                └─► prefill_postprocess(sampler, seq_length)
                        │
                        ├─► 获取输出 logits
                        ├─► sampler->sample() 采样
                        ├─► context_length_ += seq_length
                        ├─► past_seq_len_ = context_length_
                        ├─► image_paths_.clear()        // 清空，防止下轮误执行
                        └─► return sampled_token
```

### Decode Pipeline

```
decode(prev_token)
    │
    └─► do_decode_inference(prev_token, sampler)
            │
            ├─► decode_preprocess(prev_token)
            │       │
            │       ├─► 获取 prev_token embedding
            │       ├─► 计算 position (含 rope_deltas_ 调整)
            │       ├─► 准备 deepstack zeros
            │       └─► 设置 decode 输入
            │
            ├─► decode_inference()
            │       └─► decode_module->Run() + Sync()
            │
            └─► decode_postprocess(sampler)
                    │
                    ├─► 获取输出 logits
                    ├─► sampler->sample() 采样
                    ├─► context_length_++
                    ├─► past_seq_len_++
                    └─► return sampled_token
```

---

## WhisperContext (ASR)

语音识别模型，Encoder-Decoder 架构。使用 `ASRContext` 基类的模板方法模式实现自动性能打点。

### 完整转录 Pipeline（带性能打点）

```
Transcribe(audio_path, params, callback)
    │
    │  profiler_.reset() + set_root_stage("transcribe")
    │  p.start("transcribe")   ← E2E 计时开始
    │
    ├─► audio_processor_->Process()/LoadAudio()   // ★ 音频加载+特征提取
    │       scope("transcribe.audio_load")
    │
    ├─► for each chunk:
    │   │
    │   ├─► do_encode(mel, n_mels, n_frames)      // ★ Encoder 推理
    │   │       ├── scope("transcribe.encode.preprocess")
    │   │       ├── scope("transcribe.encode.inference")
    │   │       └── scope("transcribe.encode.postprocess")
    │   │
    │   ├─► do_detect_language()                   // ★ 语言检测（首次，仅 Whisper）
    │   │       ├── scope("transcribe.detect_lang.preprocess")
    │   │       ├── scope("transcribe.detect_lang.inference")
    │   │       └── scope("transcribe.detect_lang.postprocess")
    │   │
    │   ├─► BuildPrompt(lang_id) → [sot, lang, transcribe, notimestamps]
    │   │
    │   ├─► do_prefill(prompt)                     // ★ Prefill 推理
    │   │       ├── scope("transcribe.prefill.preprocess")
    │   │       ├── scope("transcribe.prefill.inference")
    │   │       └── scope("transcribe.prefill.postprocess")
    │   │       record_ttft()
    │   │
    │   └─► while not eos:
    │           ├─► do_decode(prev_token)           // ★ 自回归 Decode
    │           │       ├── scope("transcribe.decode.preprocess")
    │           │       ├── scope("transcribe.decode.inference")
    │           │       └── scope("transcribe.decode.postprocess")
    │           ├─► add_output_token()
    │           └─► callback(token)
    │
    ├─► p.stop("transcribe")    ← E2E 计时结束
    ├─► fill_perf_info(audio_duration)  → RTF/吞吐计算
    └─► profiler().print_summary()      → 树形性能报告
```

### 性能打点架构（模板方法模式）

打点代码全部在 `ASRContext` 基类中，子类（Whisper/Qwen3Asr/GlmAsr）只需实现 `_impl` 虚钩子，自动获得性能计时：

```
ASRContext::do_encode()           ← 基类模板方法（含打点 scope）
  ├── encode_preprocess_impl()     ← 子类虚钩子（具体逻辑）
  ├── encode_inference_impl()      ← 子类虚钩子
  └── encode_postprocess_impl()    ← 子类虚钩子

ASRContext::do_prefill()          ← 基类模板方法
  ├── prefill_preprocess_impl()
  ├── prefill_inference_impl()
  └── prefill_postprocess_impl()

ASRContext::do_decode()           ← 基类模板方法
  ├── decode_preprocess_impl()
  ├── decode_inference_impl()
  └── decode_postprocess_impl()

ASRContext::do_detect_language()  ← 基类模板方法（默认空实现）
ASRContext::fill_perf_info()      ← RTF/吞吐指标计算
```

### 性能指标

| 指标 | 说明 |
|------|------|
| `audio_load_time` | 音频加载+特征提取耗时 |
| `encode_time` | Encoder 推理总耗时 |
| `detect_lang_time` | 语言检测耗时（Whisper 独有） |
| `prefill_time` | Prefill 推理总耗时 |
| `decode_time` | Decode 推理总耗时 |
| `total_time` | 端到端总耗时 |
| `ttft_time` | Time-to-First-Token |
| `overall_rtf` | 整体实时率：total_time / audio_duration |
| `inference_rtf` | 纯推理实时率：(total_time - audio_load_time) / audio_duration |
| `decode_tps` | Decode 吞吐量（tokens/s） |
| `overall_tps` | 整体吞吐量（tokens/s） |

---

## 对比总结

### 函数调用对比

| 阶段 | Qwen3Context (LLM) | Qwen35MLLMContext (VLM) | Qwen3VLMContext (VLM) | WhisperContext (ASR) | Qwen3AsrContext (ASR) | GlmAsrContext (ASR) |
|------|-------------------|------------------------|----------------------|---------------------|----------------------|---------------------|
| **视觉/音频处理** | - | `run_vision()` | `run_vision()` | `do_encode()` | `do_encode()` | `do_encode()` |
| **语言检测** | - | - | - | `do_detect_language()` | - | - |
| **通用设置** | - | `prefill_common_setup()` | `prefill_common_setup()` | - | - | - |
| **分块预处理** | `prefill_preprocess_chunk()` | `prefill_preprocess_chunk()` | `prefill_preprocess_chunk()` | - | - | - |
| **分块推理** | `prefill_inference_chunk()` | `prefill_inference_chunk()` | `prefill_inference_chunk()` | - | - | - |
| **后处理** | `prefill_postprocess()` | `prefill_postprocess()` | `prefill_postprocess()` | - | - | - |

### 关键差异

| 特性 | Qwen3Context | Qwen35MLLMContext | Qwen3VLMContext | WhisperContext | Qwen3AsrContext | GlmAsrContext |
|------|-------------|-------------------|-----------------|----------------|-----------------|---------------|
| **模型架构** | Decoder-only | Decoder-only | Decoder-only | Encoder-Decoder | Encoder-Decoder | Encoder-Decoder |
| **视觉编码** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **音频编码** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **语言检测** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **M-RoPE** | ❌ | ✅ (3D position) | ✅ (3D position) | ❌ | ❌ | ❌ |
| **Deepstack** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **分块处理** | ✅ | ✅ | ✅ | ✅ (固定 30s) | ✅ (per-loop) | ✅ (PCM chunks) |
| **特殊 tokens** | BOS/EOS | BOS/EOS | BOS/EOS | SOT/Lang/Transcribe | audio_pad/audio_start | audio_token |
| **性能打点** | 手动 do_* | 手动 do_* | 手动 do_* | ASRContext 自动 | ASRContext 自动 | ASRContext 自动 |

### Pipeline 结构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Prefill Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│  LLM:  do_prefill_inference()                                   │
│          └─► [循环] prefill_preprocess → prefill_inference      │
│                                                                 │
│  VLM:  do_prefill_inference()                                   │
│          ├─► run_vision()                    ★ 视觉处理         │
│          ├─► prefill_common_setup()          ★ 通用设置 (一次)  │
│          └─► [循环] prefill_preprocess → prefill_inference      │
│                                                                 │
│  ASR:  Encode() + prefill()                    ★ 音频编码       │
│          ├─► Encode(mel_features)             ★ Encoder 前向    │
│          ├─► DetectLanguage()                 ★ 语言检测        │
│          └─► prefill(prompt_tokens)           ★ Decoder prefill │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Decode Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│  All:  do_decode_inference()                                    │
│          ├─► decode_preprocess()                                │
│          ├─► decode_inference()                                 │
│          └─► decode_postprocess()                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 文件位置

| 模型 | 头文件 | 源文件 |
|------|--------|--------|
| Qwen3Context | `include/models/qwen3_llm_model.h` | `src/models/qwen3_llm_model.cc` |
| Qwen35MLLMContext | `include/models/qwen35_mllm_model.h` | `src/models/qwen35_mllm_model.cc` |
| Qwen3VLMContext | `include/models/qwen3_vlm_model.h` | `src/models/qwen3_vlm_model.cc` |
| WhisperContext | `include/models/whisper_model.h` | `src/models/whisper_model.cc` |
| Qwen3AsrContext | `include/models/qwen3_asr_model.h` | `src/models/qwen3_asr_model.cc` |
| GlmAsrContext | `include/models/glm_asr_model.h` | `src/models/glm_asr_model.cc` |
| ASRContext (基类) | `include/core/asr_model.h` | `src/models/asr_model.cc` |

---

## ASR 模型继承关系

```
ASRContext (ASR 基类，继承 Context)
  ├── WhisperContext     (Whisper 实现)
  ├── Qwen3AsrContext     (Qwen3-ASR 实现)
  └── GlmAsrContext       (GLM-ASR 实现)

ASRModel (ASR 模型基类)
  ├── WhisperModel        (Whisper 实现)
  ├── Qwen3AsrModel       (Qwen3-ASR 实现)
  └── GlmAsrModel         (GLM-ASR 实现)
```

**设计原理：**
- ASRModel 独立于 LLMModel，通过 ModelFactory\<ASRModel\> 工厂创建
- ASRContext 继承 Context，提供模板方法模式的性能打点框架
- 子类只需实现 10-13 个 `_impl` 虚钩子，打点由基类自动完成
- 后续新增 ASR 模型只需实现钩子，零打点代码
