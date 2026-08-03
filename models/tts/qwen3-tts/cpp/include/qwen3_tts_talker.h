/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_talker.h
 * Description:
 *   Qwen3-TTS Talker prefill and decode inference interface.
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

#include "qwen3_tts_prefill_decode_runtime.h"

namespace houmo {

struct Qwen3TTSTalkerOutput {
  std::vector<float16> logits;
  Qwen3TTSHiddenSequence past_hidden;
};

class Qwen3TTSTalker {
 public:
  Qwen3TTSTalker(const std::string& prefill_path,
                 const std::string& decode_path, int device_id = 0);

  Qwen3TTSTalkerOutput Prefill(
      const Qwen3TTSHiddenSequence& prompt,
      const Qwen3TTSHiddenSequence& padding_hidden,
      int32_t past_sequence_length = 0);

  Qwen3TTSTalkerOutput Decode(
      const Qwen3TTSHiddenSequence& input, int32_t past_sequence_length);

  size_t prefill_length() const { return runtime_->prefill_length(); }
  size_t hidden_dim() const { return runtime_->hidden_dim(); }
  size_t context_length() const { return runtime_->context_length(); }

 private:
  static Qwen3TTSTalkerOutput ParseOutputs(
      std::vector<std::vector<float16>> outputs, size_t hidden_dim);

  std::unique_ptr<Qwen3TTSPrefillDecodeRuntime> runtime_;
};

}  // namespace houmo
