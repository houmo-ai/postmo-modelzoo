/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: HmEmbedding.cc
 * Description:
 *   HmEmbedding Implementation - Handles embedding operations for token IDs to
 * embedding vectors conversion.
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
#include "utils/HmEmbedding.h"

HmEmbedding::HmEmbedding(const std::string& embeddingWeightPath,
                         const int& embedding_len, const int& prefill_len)
    : prefill_length(prefill_len), embedding_length(embedding_len) {
  // Read embedding.bin, allocate extra space, return pointer to
  // corresponding address of embed_w when decoding
  try {
    uint32_t n_bytes = 0;
    embed_w = readEmbeddingWeight<float16>(
        embeddingWeightPath, n_bytes, prefill_length * embedding_length);
    vocab_size = n_bytes / (sizeof(float16) * embedding_length);
    std::cout << "[INFO] vocab_size: " << vocab_size
              << ", embedding_length: " << embedding_length
              << ", prefill_length: " << prefill_length << "\n";
    if (vocab_size <= 0 ||
        (vocab_size * embedding_length * sizeof(float16) > n_bytes)) {
      throw std::runtime_error(
          "Invalid embedding weight file: vocab size must be positive and file "
          "size mismatch.");
    }

    if (vocab_size * embedding_length * sizeof(float16) < n_bytes) {
      std::cout << "[Warning] embedding weight file size is not a multiple of "
                   "embedding vector size. Some data may be ignored.\n";
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("readEmbeddingWeight Error:" +
                             std::string(e.what()));
  }
  ptr = std::make_unique<float16[]>(prefill_length * embedding_length);
}

HmEmbedding::~HmEmbedding() {
  embed_w.reset();
  ptr.reset();
}

float16* HmEmbedding::EmbeddingTokens(const std::vector<int>& ids) {
  if (ids.empty()) {
    return nullptr;
  }

  uint64_t num_tokens = ids.size();

  if (num_tokens == 1) {
    int offset = ids[0] * embedding_length;
    return embed_w.get() + offset;
  }

  std::fill(ptr.get(), ptr.get() + prefill_length * embedding_length,
            float16(0));
  for (uint32_t index = 0; index < ids.size(); index++) {
    const int token_id = ids[index];
    const uint64_t src_offset =
        static_cast<uint64_t>(token_id) * embedding_length;
    const uint64_t dst_offset = static_cast<uint64_t>(index) * embedding_length;

    std::copy(embed_w.get() + src_offset,
              embed_w.get() + src_offset + embedding_length,
              ptr.get() + dst_offset);
  }

  return ptr.get();
}