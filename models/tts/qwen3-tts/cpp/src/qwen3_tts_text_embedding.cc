/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_text_embedding.cc
 * Description:
 *   Qwen3-TTS text embedding lookup implementation.
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

#include "qwen3_tts_text_embedding.h"

#include <filesystem>

namespace houmo {

Qwen3TTSTextEmbedding::Qwen3TTSTextEmbedding(
    const std::string& embedding_path, int max_sequence_length) {
  if (!std::filesystem::exists(embedding_path)) {
    throw Exception("Text embedding file not found: " + embedding_path);
  }
  embedding_ = std::make_unique<Embedding>(
      embedding_path, kHiddenDim, max_sequence_length);
}

Qwen3TTSHiddenSequence Qwen3TTSTextEmbedding::Lookup(
    const std::vector<Token>& token_ids) const {
  for (Token token : token_ids) {
    if (token < 0 || token >= embedding_->vocab_size()) {
      throw Exception("Text token ID is outside the embedding vocabulary: " +
                      std::to_string(token));
    }
  }

  Qwen3TTSHiddenSequence output;
  output.sequence_length = token_ids.size();
  output.hidden_dim = kHiddenDim;
  if (token_ids.empty()) return output;

  const float16* data = embedding_->token_embedding(token_ids);
  if (data == nullptr) {
    throw Exception("Failed to look up text embeddings");
  }
  output.data.assign(data, data + output.sequence_length * output.hidden_dim);
  return output;
}

}  // namespace houmo
