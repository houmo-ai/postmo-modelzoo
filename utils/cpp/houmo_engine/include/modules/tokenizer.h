/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: tokenizer.h
 * Description:
 *   HuggingFace Tokenizer wrapper. Wraps the tokenizer.cpp library to
 *   provide text-to-token and token-to-text conversion.
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

#include <memory>
#include <string>
#include <tokenizer.hpp>
#include <vector>

#include "base/houmo.h"

namespace houmo {

/**
 * @brief HuggingFace Tokenizer wrapper
 *
 * Wraps the tokenizers_cpp library to provide text-to-token and
 * token-to-text conversion.
 */
class HfTokenizer {
 public:
  /**
   * @brief Constructor
   * @param tokenizer_path Path to download raw hugging-face model path.
   */
  explicit HfTokenizer(const std::string& tokenizer_path);

  /**
   * @brief Destructor
   */
  ~HfTokenizer();

  // Non-copyable
  HfTokenizer(const HfTokenizer&) = delete;
  HfTokenizer& operator=(const HfTokenizer&) = delete;

  // Movable
  HfTokenizer(HfTokenizer&&) noexcept = default;
  HfTokenizer& operator=(HfTokenizer&&) noexcept = default;

  /**
   * @brief Encode text to token IDs
   * @param text Input text
   * @param add_bos Whether to prepend BOS token
   * @param add_eos Whether to append EOS token
   * @return List of token IDs
   */
  std::vector<Token> encode(const std::string& text, bool add_bos = true,
                            bool add_eos = false,
                            bool add_special_tokens = false);

  /**
   * @brief Decode a single token
   * @param token Token ID
   * @return Decoded string
   */
  std::string decode(Token token, bool skip_special_tokens = false);

  /**
   * @brief Decode a sequence of tokens
   * @param tokens List of token IDs
   * @return Decoded string
   */
  std::string decode(const std::vector<Token>& tokens,
                     bool skip_special_tokens = false);

  /**
   * @brief Get BOS token ID
   */
  Token bos_token_id() const { return bos_token_id_; }

  /**
   * @brief Get EOS token ID
   */
  Token eos_token_id() const { return eos_token_id_; }

  /**
   * @brief Get PAD token ID
   *
   * PAD token is used to pad input sequences to fixed length.
   * For Qwen series, pad_token_id equals bos_token_id.
   */
  Token pad_token_id() const { return pad_token_id_; }

  /**
   * @brief Get token ID for a token string
   * @param token Token string (e.g., "<|endoftext|>")
   * @return Token ID, or -1 if not found
   */
  int token_to_id(const std::string& token) const;

 private:
  std::shared_ptr<tokenizer::PreTrainedTokenizer> tokenizer_;
  Token bos_token_id_ = -1;
  Token eos_token_id_ = -1;
  Token pad_token_id_ = -1;
};

}  // namespace houmo
