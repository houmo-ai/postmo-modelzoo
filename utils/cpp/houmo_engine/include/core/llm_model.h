/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: llm_model.h
 * Description:
 *   LLM model base class. Provides minimal interface and data storage
 *   without imposing a loading strategy. Subclasses implement their own
 *   loading workflow.
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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/houmo.h"
#include "base/tcim_utils.h"
#include "modules/embedding.h"
#include "modules/perf_profiler.h"
#include "modules/tokenizer.h"

namespace houmo {

// Forward declaration
class Context;

/**
 * @brief LLM model base class
 *
 * Provides only the most basic public interface and data storage.
 * Does not impose a loading strategy.
 */
class LLMModel {
 public:
  /**
   * @brief Constructor - stores config only, no loading
   * @param config Model configuration
   */
  explicit LLMModel(const ModelConfig& config) : config_(config) {}

  virtual ~LLMModel() = default;

  // ========== Type info ==========

  virtual ModelType type() const { return ModelType::LLM; }

  // ========== Public methods (available after subclass loading) ==========

  virtual std::vector<Token> tokenize(const std::string& text,
                                      bool add_bos = false,
                                      bool add_eos = false);

  virtual std::string token_to_str(Token token);

  virtual std::string tokens_to_str(const std::vector<Token>& tokens);

  // ========== Basic properties ==========

  virtual int vocab_size() const;

  virtual int embedding_dim() const;

  virtual int max_ctx_available() const;

  virtual ModelInfo model_info() const;

  // ========== Context creation ==========

  virtual std::unique_ptr<Context> create_context(int n_ctx = 0);

  // ========== Tokenizer checks ==========

  bool has_tokenizer() const;

  Token bos_token_id() const;

  Token eos_token_id() const;

  /**
   * @brief Get shared Tokenizer pointer (for StreamingDecoder)
   * @return HfTokenizer pointer, or nullptr if not loaded
   */
  std::shared_ptr<HfTokenizer> tokenizer() const { return tokenizer_; }

  // ========== Internal interface (for Context use) ==========

  std::shared_ptr<tcim::Module> prefill_module() const {
    return prefill_module_;
  }
  std::shared_ptr<tcim::Module> decode_module() const { return decode_module_; }
  std::shared_ptr<HfTokenizer> tokenizer_module() const { return tokenizer_; }
  std::shared_ptr<Embedding> embedding() const { return embedding_; }

  std::map<std::string, tcim::Tensor>& prefill_input_map() const;
  std::map<std::string, tcim::Tensor>& decode_input_map() const;

  int prefill_length() const { return prefill_length_; }
  int attn_idx_start() const { return attn_idx_start_; }

 protected:
  // Protected members, directly accessible by subclasses
  ModelConfig config_;
  std::shared_ptr<HfTokenizer> tokenizer_;
  std::shared_ptr<Embedding> embedding_;
  std::shared_ptr<tcim::Module> prefill_module_;
  std::shared_ptr<tcim::Module> decode_module_;
  std::unique_ptr<tcim::DevManager> dev_manager_;
  std::unique_ptr<tcim::Module::WeightManager> weight_manager_;
  ModelInfo info_;

  // Input tensors
  std::map<std::string, tcim::Tensor> prefill_input_map_;
  std::map<std::string, tcim::Tensor> decode_input_map_;

  int prefill_length_ = 0;
  int attn_idx_start_ = 0;
};

}  // namespace houmo
