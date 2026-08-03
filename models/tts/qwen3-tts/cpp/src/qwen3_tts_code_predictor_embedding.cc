/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_code_predictor_embedding.cc
 * Description:
 *   Qwen3-TTS CodePredictor codebook embedding lookup implementation.
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

#include "qwen3_tts_code_predictor_embedding.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace houmo {

Qwen3TTSCodePredictorEmbedding::Qwen3TTSCodePredictorEmbedding(
    const std::string& path, size_t hidden_dim)
    : hidden_dim_(hidden_dim) {
  if (hidden_dim_ == 0) {
    throw Exception(
        "CodePredictor embedding hidden dimension must be positive");
  }
  const size_t expected_elements = kCodebookCount * kVocabSize * hidden_dim_;
  const size_t expected_bytes = expected_elements * sizeof(float16);
  if (!std::filesystem::exists(path) ||
      std::filesystem::file_size(path) != expected_bytes) {
    throw Exception("Unexpected CodePredictor embedding file size: " + path);
  }
  weights_.resize(expected_elements);
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(weights_.data()), expected_bytes);
  if (!stream) {
    throw Exception("Failed to read CodePredictor embedding: " + path);
  }
}

Qwen3TTSHiddenSequence Qwen3TTSCodePredictorEmbedding::Lookup(
    size_t codebook_index, Token token_id) const {
  if (codebook_index >= kCodebookCount || token_id < 0 ||
      static_cast<size_t>(token_id) >= kVocabSize) {
    throw std::out_of_range("CodePredictor embedding index is out of range");
  }
  const size_t offset =
      (codebook_index * kVocabSize + static_cast<size_t>(token_id)) *
      hidden_dim_;
  Qwen3TTSHiddenSequence output;
  output.sequence_length = 1;
  output.hidden_dim = hidden_dim_;
  output.data.assign(
      weights_.begin() + static_cast<std::ptrdiff_t>(offset),
      weights_.begin() + static_cast<std::ptrdiff_t>(offset + hidden_dim_));
  return output;
}

}  // namespace houmo
