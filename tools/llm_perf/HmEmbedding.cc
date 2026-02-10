/*
 * Copyright (c) 2025 HOUMO AI
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
#include "HmEmbedding.h"

HmEmbedding::HmEmbedding(const std::string& embeddingWeightPath,
                         const int& embedding_len, const int& prefill_len)
    : prefill_length(prefill_len), embedding_length(embedding_len) {
  // Read embedding.bin, allocate extra space, return pointer to
  // corresponding address of embed_w when decoding
  try {
    embed_w = readEmbeddingWeight<tensor_type>(
        embeddingWeightPath, prefill_length * embedding_length);
  } catch (const std::exception& e) {
    throw std::runtime_error("readEmbeddingWeight Error:" +
                             std::string(e.what()));
  }
  ptr = std::make_unique<tensor_type[]>(prefill_length * embedding_length);
}

HmEmbedding::~HmEmbedding() {
  embed_w.reset();
  ptr.reset();
}

tensor_type* HmEmbedding::EmbeddingTokens(const std::vector<int>& ids) {
  if (ids.empty()) {
    return nullptr;
  }

  uint64_t num_tokens = ids.size();

  if (num_tokens == 1) {
    int offset = ids[0] * embedding_length;
    return embed_w.get() + offset;
  }

  std::fill(ptr.get(), ptr.get() + prefill_length * embedding_length,
            tensor_type(0));
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