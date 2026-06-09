/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: llm_model.cc
 * Description:
 *   LLM model base class implementation
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

#include "core/llm_model.h"

#include <iostream>

#include "core/context.h"

namespace houmo {

// ============================================================================
// LLMModel - Base class implementation (provides default behavior, subclasses can override)
// ============================================================================

std::vector<Token> LLMModel::tokenize(const std::string& text, bool add_bos,
                                      bool add_eos) {
  if (tokenizer_) {
    return tokenizer_->encode(text, add_bos, add_eos);
  }
  std::cerr << "Warning: Tokenizer not loaded, returning empty tokens"
            << std::endl;
  return {};
}

std::string LLMModel::token_to_str(Token token) {
  if (tokenizer_) {
    return tokenizer_->decode(token);
  }
  std::cerr << "Warning: Tokenizer not loaded, returning empty string"
            << std::endl;
  return "";
}

std::string LLMModel::tokens_to_str(const std::vector<Token>& tokens) {
  if (tokenizer_) {
    return tokenizer_->decode(tokens);
  }
  std::cerr << "Warning: Tokenizer not loaded, returning empty string"
            << std::endl;
  return "";
}

int LLMModel::vocab_size() const { return info_.n_vocab; }

int LLMModel::embedding_dim() const { return info_.n_embd; }

int LLMModel::max_ctx_available() const { return info_.n_ctx; }

ModelInfo LLMModel::model_info() const { return info_; }

bool LLMModel::has_tokenizer() const { return tokenizer_ != nullptr; }

Token LLMModel::bos_token_id() const {
  if (tokenizer_) {
    return tokenizer_->bos_token_id();
  }
  return TokenNull;
}

Token LLMModel::eos_token_id() const {
  if (tokenizer_) {
    return tokenizer_->eos_token_id();
  }
  return TokenNull;
}

std::map<std::string, tcim::Tensor>& LLMModel::prefill_input_map() const {
  return const_cast<LLMModel*>(this)->prefill_input_map_;
}

std::map<std::string, tcim::Tensor>& LLMModel::decode_input_map() const {
  return const_cast<LLMModel*>(this)->decode_input_map_;
}

std::unique_ptr<Context> LLMModel::create_context(int n_ctx) {
  if (n_ctx <= 0) {
    n_ctx = info_.n_ctx;
  }
  // Create base Context; subclasses can override this method to create custom Context
  return std::make_unique<Context>(this, n_ctx);
}

}  // namespace houmo
