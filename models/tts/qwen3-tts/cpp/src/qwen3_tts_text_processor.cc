/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_text_processor.cc
 * Description:
 *   Qwen3-TTS text tokenization and prompt token splitting implementation.
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

#include "qwen3_tts_text_processor.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace houmo {

Qwen3TTSTextProcessor::Qwen3TTSTextProcessor(
    const std::string& tokenizer_path) {
  const std::filesystem::path tokenizer_json =
      std::filesystem::path(tokenizer_path) / "tokenizer.json";
  if (!std::filesystem::exists(tokenizer_json)) {
    throw Exception("tokenizer.json not found in tokenizer directory: " +
                    tokenizer_path);
  }
  tokenizer_ = std::make_unique<HfTokenizer>(tokenizer_path);
}

std::string Qwen3TTSTextProcessor::BuildAssistantText(
    const std::string& text) {
  return "<|im_start|>assistant\n" + text +
         "<|im_end|>\n<|im_start|>assistant\n";
}

Qwen3TTSTextFeatures Qwen3TTSTextProcessor::Process(
    const std::string& text) const {
  Qwen3TTSTextFeatures features;

  // The assistant template already contains every required special token.
  features.input_ids =
      tokenizer_->encode(BuildAssistantText(text), false, false, false);

  constexpr size_t kMinimumTokenCount =
      kRoleTokenCount + kTemplateTailTokenCount;
  if (features.input_ids.size() < kMinimumTokenCount) {
    throw Exception("Tokenized assistant text is shorter than its template");
  }

  const auto role_end = features.input_ids.begin() + kRoleTokenCount;
  const auto body_end =
      features.input_ids.end() - kTemplateTailTokenCount;
  features.role_ids.assign(features.input_ids.begin(), role_end);
  features.body_ids.assign(role_end, body_end);

  return features;
}

}  // namespace houmo
