/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_text_embedding.h
 * Description:
 *   Qwen3-TTS text embedding lookup interface.
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
#include <vector>

#include "modules/embedding.h"
#include "qwen3_tts_types.h"

namespace houmo {

class Qwen3TTSTextEmbedding {
 public:
  static constexpr int kHiddenDim = 2048;

  explicit Qwen3TTSTextEmbedding(const std::string& embedding_path,
                                 int max_sequence_length = 256);

  Qwen3TTSHiddenSequence Lookup(const std::vector<Token>& token_ids) const;
  int vocab_size() const { return embedding_->vocab_size(); }

 private:
  std::unique_ptr<Embedding> embedding_;
};

}  // namespace houmo
