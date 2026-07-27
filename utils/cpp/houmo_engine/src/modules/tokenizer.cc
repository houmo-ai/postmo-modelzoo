/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: tokenizer.cc
 * Description:
 *   HuggingFace Tokenizer wrapper implementation.
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

#include "modules/tokenizer.h"

#include <fstream>
#include <iostream>

namespace houmo {

HfTokenizer::HfTokenizer(const std::string& tokenizer_path) {
  // Load tokenizer
  tokenizer_ = tokenizer::AutoTokenizer::from_pretrained(tokenizer_path);

  // Get special token IDs
  bos_token_id_ = tokenizer_->bos_token_id();
  eos_token_id_ = tokenizer_->eos_token_id();

  // PAD token ID equals BOS token ID (Qwen series)
  pad_token_id_ = tokenizer_->pad_token_id();

  if (bos_token_id_ < 0) {
    bos_token_id_ = tokenizer_->token_to_id("<|endoftext|>");
  }
  if (pad_token_id_ < 0) {
    pad_token_id_ = bos_token_id_;
  }
}

HfTokenizer::~HfTokenizer() { tokenizer_.reset(); }

std::vector<Token> HfTokenizer::encode(const std::string& text, bool add_bos,
                                       bool add_eos, bool add_special_tokens) {
  std::vector<Token> ids;

  // Encode text
  auto encoded = tokenizer_->encode(text, add_special_tokens);

  // Convert to std::vector<Token>
  ids.reserve(encoded.size() + (add_bos ? 1 : 0) + (add_eos ? 1 : 0));
  if (add_bos && bos_token_id_ >= 0) {
    ids.push_back(bos_token_id_);
  }

  for (const auto& id : encoded) {
    ids.push_back(static_cast<Token>(id));
  }

  if (add_eos && eos_token_id_ >= 0) {
    ids.push_back(eos_token_id_);
  }

  return ids;
}

std::string HfTokenizer::decode(Token token, bool skip_special_tokens) {
  std::vector<int32_t> ids = {static_cast<int32_t>(token)};
  return tokenizer_->decode(ids, skip_special_tokens);
}

std::string HfTokenizer::decode(const std::vector<Token>& tokens,
                                bool skip_special_tokens) {
  std::vector<int32_t> ids;
  ids.reserve(tokens.size());
  for (const auto& token : tokens) {
    ids.push_back(static_cast<int32_t>(token));
  }
  return tokenizer_->decode(ids, skip_special_tokens);
}

int HfTokenizer::token_to_id(const std::string& token) const {
  return tokenizer_->token_to_id(token);
}

}  // namespace houmo
