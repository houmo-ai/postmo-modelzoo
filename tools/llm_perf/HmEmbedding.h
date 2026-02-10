/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: HmEmbedding.h
 * Description:
 *   HmEmbedding Header File - Defines the HmEmbedding class for embedding
 * operations.
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
#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__

#include <algorithm>
#include <cctype>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "half.hpp"
#include "utils.h"

// Define tensor_type as half precision floating point
using tensor_type = half_float::half;

/**
 * Reads embedding weight file
 * @param path              Path to the file
 * @param n_elems_align     Number of empty elements to append at the end after
 * reading the file, default 0
 * @return                  Returns unique_ptr on success, otherwise nullptr
 */
template <typename T>
std::unique_ptr<T[]> readEmbeddingWeight(const std::string &path,
                                         size_t n_elems_align = 0) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("invalid embedding weight file!");
  }

  ifs.seekg(0, std::ios::end);
  const std::size_t n_bytes = ifs.tellg();
  ifs.seekg(0);

  const std::size_t n_elem =
      (n_bytes + sizeof(T) - 1) / sizeof(T) + n_elems_align;
  auto ptr = std::make_unique<T[]>(n_elem);
  ifs.read(reinterpret_cast<char *>(ptr.get()), n_bytes);
  ifs.close();
  memset(reinterpret_cast<char *>(ptr.get()) + n_bytes, 0,
         n_elems_align * sizeof(T));
  return ptr;
}

/**
 * HmEmbedding class - Handles conversion from token IDs to embedding vectors
 */
class HmEmbedding {
 public:
  /**
   * Constructor for HmEmbedding
   * @param embeddingWeightPath Path to the embedding weight file
   * @param embedding_len Length of the embedding vectors
   * @param prefill_len Length for prefill operations
   */
  HmEmbedding(const std::string &embeddingWeightPath, const int &embedding_len,
              const int &prefill_len);

  HmEmbedding(const HmEmbedding &it) = delete;
  HmEmbedding &operator=(const HmEmbedding &it) = delete;
  HmEmbedding(HmEmbedding &&it) noexcept = default;
  HmEmbedding &operator=(HmEmbedding &&it) noexcept = default;

  ~HmEmbedding();

  /**
   * Converts token IDs to embedding vectors
   * @param ids Vector of token IDs to be converted
   * @return Pointer to the resulting embedding vectors
   */
  tensor_type *EmbeddingTokens(const std::vector<int> &ids);

 private:
  std::unique_ptr<tensor_type[]> embed_w;
  std::unique_ptr<tensor_type[]> ptr;

  int prefill_length = 0;
  int embedding_length = 0;
};

#endif  // __EMBEDDING_H__