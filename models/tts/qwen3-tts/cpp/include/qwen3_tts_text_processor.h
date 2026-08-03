/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_text_processor.h
 * Description:
 *   Qwen3-TTS text tokenization and prompt token splitting interface.
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

#include "base/houmo.h"
#include "modules/tokenizer.h"

namespace houmo {

struct Qwen3TTSTextFeatures {
  std::vector<Token> input_ids;
  std::vector<Token> role_ids;
  std::vector<Token> body_ids;
};

class Qwen3TTSTextProcessor {
 public:
  static constexpr int kRoleTokenCount = 3;
  static constexpr int kTemplateTailTokenCount = 5;
  explicit Qwen3TTSTextProcessor(const std::string& tokenizer_path);

  Qwen3TTSTextFeatures Process(const std::string& text) const;

  static std::string BuildAssistantText(const std::string& text);

 private:
  std::unique_ptr<HfTokenizer> tokenizer_;
};

}  // namespace houmo
