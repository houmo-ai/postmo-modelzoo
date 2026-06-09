/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: context.h
 * Description:
 *   Base class for inference context, managing per-request state including
 *   KV Cache, generation history, and performance profiling.
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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "base/houmo.h"
#include "modules/perf_profiler.h"
#include "modules/sampler.h"

namespace houmo {

// Forward declaration
class LLMModel;

/**
 * @brief Inference context base class
 *
 * Provides token-level streaming generation via callback mode.
 * For string-level streaming output, use StreamingDecoder helper.
 */
class Context {
 public:
  Context(LLMModel* model, int n_ctx) : model_(model), n_ctx_(n_ctx) {}
  virtual ~Context() = default;

  // ========== Inference interface ==========
  virtual Token prefill(const std::vector<Token>& tokens) { return TokenNull; }
  virtual Token decode(Token prev_token) { return TokenNull; }
  virtual void set_image(const std::string& image_path) {}

  // ========== Generation interface (Token callback mode) ==========
  /**
   * @brief Stream-generate tokens
   * @param prompt Input tokens
   * @param params Sampling parameters
   * @param callback Per-token callback; return true to continue, false to stop
   *
   * Usage:
   *   ctx->generate(tokens, params, [](Token token) {
   *       std::cout << model.token_to_str(token);
   *       return true;
   *   });
   *
   * For string-level streaming (handles UTF-8 multi-byte characters):
   *   StreamingDecoder decoder(model.tokenizer());
   *   ctx->generate(tokens, params, [&](Token token) {
   *       std::cout << decoder.decode(token);
   *       return true;
   *   });
   */
  virtual void generate(const std::vector<Token>& prompt,
                        const SamplingParams& params,
                        std::function<bool(Token)> callback) {}

  // ========== State management ==========
  virtual void set_keep_history(bool keep) { keep_history_ = keep; }
  virtual bool keep_history() const { return keep_history_; }
  virtual int context_length() const { return context_length_; }
  virtual void reset() {
    context_length_ = 0;
    generated_ids_.clear();
  }

  // ========== Sampler management ==========
  virtual void set_sampler(const SamplingParams& params) {
    sampler_ = std::make_unique<Sampler>(params);
  }
  Sampler* sampler() const { return sampler_.get(); }

  // ========== Performance statistics ==========
  virtual PerfStats perf_stats() const { return perf_stats_; }
  virtual void reset_perf_stats() { perf_stats_ = PerfStats{}; }

  // ========== Performance profiler ==========
  PerfProfiler& profiler() { return profiler_; }
  const PerfProfiler& profiler() const { return profiler_; }

 protected:
  LLMModel* model_;
  int n_ctx_;
  int context_length_ = 0;
  bool keep_history_ = true;
  PerfStats perf_stats_;
  PerfProfiler profiler_;
  std::vector<Token> generated_ids_;
  std::unique_ptr<Sampler> sampler_;
};

}  // namespace houmo
