# 推理 Pipeline 文档

本文档描述三个模型类的推理执行流程。

## 目录

- [Qwen3Context (LLM)](#qwen3context-llm)
- [Qwen35MLLMContext (VLM)](#qwen35mllmcontext-vlm)
- [Qwen3VLMContext (VLM)](#qwen3vlmcontext-vlm)
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

## 对比总结

### 函数调用对比

| 阶段 | Qwen3Context (LLM) | Qwen35MLLMContext (VLM) | Qwen3VLMContext (VLM) |
|------|-------------------|------------------------|----------------------|
| **视觉处理** | - | `run_vision()` | `run_vision()` |
| **通用设置** | - | `prefill_common_setup()` | `prefill_common_setup()` |
| **分块预处理** | `prefill_preprocess_chunk()` | `prefill_preprocess_chunk()` | `prefill_preprocess_chunk()` |
| **分块推理** | `prefill_inference_chunk()` | `prefill_inference_chunk()` | `prefill_inference_chunk()` |
| **后处理** | `prefill_postprocess()` | `prefill_postprocess()` | `prefill_postprocess()` |

### 关键差异

| 特性 | Qwen3Context | Qwen35MLLMContext | Qwen3VLMContext |
|------|-------------|-------------------|-----------------|
| **视觉编码** | ❌ | ✅ | ✅ |
| **M-RoPE** | ❌ | ✅ (3D position) | ✅ (3D position) |
| **Deepstack** | ❌ | ❌ | ✅ |
| **Embedding 获取** | 每轮 chunk | 每轮 chunk | 一次性全部 |
| **Image Embedding 替换** | - | 每轮 chunk | 一次性全部 |
| **多轮对话 position** | 简单递增 | rope_deltas 调整 | rope_deltas 调整 |

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
