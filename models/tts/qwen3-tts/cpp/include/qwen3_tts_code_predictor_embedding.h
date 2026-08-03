/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_code_predictor_embedding.h
 * Description:
 *   Qwen3-TTS CodePredictor codebook embedding lookup interface.
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

#include <string>
#include <vector>

#include "qwen3_tts_types.h"

namespace houmo {

class Qwen3TTSCodePredictorEmbedding {
 public:
  static constexpr size_t kCodebookCount = 15;
  static constexpr size_t kVocabSize = 2048;

  explicit Qwen3TTSCodePredictorEmbedding(const std::string& path,
                                          size_t hidden_dim);

  Qwen3TTSHiddenSequence Lookup(size_t codebook_index, Token token_id) const;
  size_t hidden_dim() const { return hidden_dim_; }

 private:
  size_t hidden_dim_;
  std::vector<float16> weights_;
};

}  // namespace houmo
