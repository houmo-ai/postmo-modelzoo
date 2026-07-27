/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: embedding.h
 * Description:
 *   Embedding lookup table for token ID to embedding vector conversion.
 *   Loads binary embedding weights and provides fast lookup.
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

#include "base/houmo.h"

#include <memory>
#include <string>
#include <vector>

namespace houmo {

/**
 * @brief Embedding lookup table
 *
 * Loads token embeddings from binary file and provides fast lookup.
 */
class Embedding {
 public:
  /**
   * @brief Constructor
   * @param path Embedding file path
   * @param hidden_dim Hidden dimension size
   * @param max_seq_len Maximum sequence length
   */
  Embedding(const std::string& path, int hidden_dim = 0, int max_seq_len = 0);

  ~Embedding();

  // Non-copyable
  Embedding(const Embedding&) = delete;
  Embedding& operator=(const Embedding&) = delete;

  // Movable
  Embedding(Embedding&&) noexcept = default;
  Embedding& operator=(Embedding&&) noexcept = default;

  /**
   * @brief Get embedding for a single token
   * @param token Token ID
   * @return Pointer to embedding vector
   */
  const float16* token_embedding(Token token) const;

  /**
   * @brief Get embeddings for multiple tokens
   * @param tokens List of token IDs
   * @return Pointer to embedding matrix
   */
  const float16* token_embedding(const std::vector<Token>& tokens) const;

  /**
   * @brief Get vocabulary size
   */
  int vocab_size() const { return vocab_size_; }

  /**
   * @brief Get hidden dimension size
   */
  int hidden_dim() const { return hidden_dim_; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  int vocab_size_ = 0;
  int hidden_dim_ = 0;
};

}  // namespace houmo
