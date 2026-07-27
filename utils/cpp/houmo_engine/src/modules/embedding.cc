/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: embedding.cc
 * Description:
 *   Embedding lookup table implementation.
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

#include <cstring>
#include <fstream>
#include <iostream>

#include "modules/embedding.h"

namespace houmo {

/**
 * Reads embedding weight file
 * @param path              Path to the file
 * @param n_elems_align     Number of empty elements to append at the end after
 * reading the file, default 0
 * @return                  Returns unique_ptr on success, otherwise nullptr
 */
template <typename T>
std::unique_ptr<T[]> readEmbeddingWeight(const std::string& path,
                                         uint32_t& weight_n_bytes,
                                         size_t n_elems_align = 0) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("invalid embedding weight file!");
  }

  ifs.seekg(0, std::ios::end);
  const std::size_t n_bytes = ifs.tellg();
  ifs.seekg(0);
  weight_n_bytes = n_bytes;

  const std::size_t n_elem =
      (n_bytes + sizeof(T) - 1) / sizeof(T) + n_elems_align;
  auto ptr = std::make_unique<T[]>(n_elem);
  ifs.read(reinterpret_cast<char*>(ptr.get()), n_bytes);
  ifs.close();
  memset(reinterpret_cast<char*>(ptr.get()) + n_bytes, 0,
         n_elems_align * sizeof(T));
  return ptr;
}

// Internal implementation of Embedding
class Embedding::Impl {
 public:
  Impl(const std::string& path, int hidden_dim, int max_seq_len)
      : hidden_dim_(hidden_dim), max_seq_len_(max_seq_len) {
    if (hidden_dim <= 0) {
      throw Exception("Embedding hidden_dim must be positive");
    }

    // Read embedding weights (fp16 precision)
    uint32_t n_bytes = 0;
    size_t n_elems_align = max_seq_len > 0 ? max_seq_len * hidden_dim : 0;
    table_ = readEmbeddingWeight<float16>(path, n_bytes, n_elems_align);

    // Auto-calculate vocab_size
    vocab_size_ = static_cast<int>(n_bytes / (sizeof(float16) * hidden_dim));

    std::cout << "[INFO] Embedding loaded: vocab_size=" << vocab_size_
              << ", hidden_dim=" << hidden_dim_
              << ", max_seq_len=" << max_seq_len << "\n";

    if (vocab_size_ <= 0) {
      throw Exception("Invalid embedding file: vocab_size must be positive");
    }

    // Allocate batch embedding buffer (data type: float16)
    if (max_seq_len > 0) {
      batch_buffer_.resize(max_seq_len * hidden_dim);
    }
  }

  ~Impl() = default;

  const float16* token_embedding(Token token) const {
    if (token < 0 || token >= vocab_size_) {
      return nullptr;
    }
    return &table_[token * hidden_dim_];
  }

  const float16* token_embedding(const std::vector<Token>& tokens) const {
    if (tokens.size() > static_cast<size_t>(max_seq_len_)) {
      throw Exception(
          "Number of tokens exceeds max_seq_len of the embedding, please split "
          "to chunks of size <= max_seq_len");
    }
    std::fill(batch_buffer_.begin(), batch_buffer_.end(),
              static_cast<float16>(0.0f));
    for (size_t i = 0; i < tokens.size(); i++) {
      Token token = tokens[i];
      if (token >= 0 && token < vocab_size_) {
        const uint64_t src_offset = static_cast<uint64_t>(token) * hidden_dim_;
        const uint64_t dst_offset = static_cast<uint64_t>(i) * hidden_dim_;

        std::copy(table_.get() + src_offset,
                  table_.get() + src_offset + hidden_dim_,
                  batch_buffer_.data() + dst_offset);
      }
    }
    return batch_buffer_.data();
  }

  int vocab_size() const { return vocab_size_; }
  int hidden_dim() const { return hidden_dim_; }

 private:
  std::unique_ptr<float16[]> table_;  // float16 format weights
  mutable std::vector<float16>
      batch_buffer_;  // Batch embedding buffer (mutable for const function)
  uint32_t vocab_size_ = 0;
  uint32_t hidden_dim_ = 0;
  uint32_t max_seq_len_ = 0;
};

// Public interface implementation
Embedding::Embedding(const std::string& path, int hidden_dim, int max_seq_len)
    : impl_(std::make_unique<Impl>(path, hidden_dim, max_seq_len)) {
  vocab_size_ = impl_->vocab_size();
  hidden_dim_ = impl_->hidden_dim();
}

Embedding::~Embedding() = default;

const float16* Embedding::token_embedding(Token token) const {
  return impl_->token_embedding(token);
}

const float16* Embedding::token_embedding(
    const std::vector<Token>& tokens) const {
  return impl_->token_embedding(tokens);
}

}  // namespace houmo
