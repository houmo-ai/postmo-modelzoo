/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_llm_model.h
 * Description:
 *   Qwen3 LLM model implementation. Defines Qwen3Context for inference
 *   and Qwen3LLMModel for model loading.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOUMO_QWEN3_LLM_MODEL_H
#define HOUMO_QWEN3_LLM_MODEL_H

#include "core/context.h"
#include "core/llm_model.h"

namespace houmo {

/**
 * @brief Qwen3 inference context
 */
class Qwen3Context : public Context {
 public:
  explicit Qwen3Context(LLMModel* model, int n_ctx);
  ~Qwen3Context() override = default;

  // Override inference methods
  Token prefill(const std::vector<Token>& tokens) override;
  Token decode(Token prev_token) override;

  // Token callback generation
  void generate(const std::vector<Token>& prompt, const SamplingParams& params,
                std::function<bool(Token)> callback) override;

  // Reset context state (including KV Cache)
  void reset() override;

 protected:
  // Split sub-methods (for internal profiling)
  void prefill_preprocess_chunk(int chunk, const std::vector<Token>& tokens,
                                int32_t seq_length, int prefill_length);
  void prefill_inference_chunk();
  Token prefill_postprocess(Sampler* sampler, int32_t seq_length);

  void decode_preprocess(Token prev_token);
  void decode_inference();
  Token decode_postprocess(Sampler* sampler);

  // Internal interface (protected, for subclass and internal use)
  Token do_prefill_inference(const std::vector<Token>& tokens,
                             Sampler* sampler);
  Token do_decode_inference(Token prev_token, Sampler* sampler);
};

/**
 * @brief Qwen3 LLM model
 *
 * Inherits LLMModel, implements complete loading workflow.
 */
class Qwen3LLMModel : public LLMModel {
 public:
  explicit Qwen3LLMModel(const ModelConfig& config);
  ~Qwen3LLMModel() override = default;

  // Override create_context to create Qwen3Context
  std::unique_ptr<Context> create_context(int n_ctx = 0) override;

 private:
  void load();

  // Internal state
  int n_blocks_ = 0;
  int batch_ = 0;
  int embedding_length_ = 0;
  int context_max_length_ = 0;
};

}  // namespace houmo

#endif  // HOUMO_QWEN3_LLM_MODEL_H
