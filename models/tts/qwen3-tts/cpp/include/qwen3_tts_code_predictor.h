/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_code_predictor.h
 * Description:
 *   Qwen3-TTS CodePredictor prefill and decode inference interface.
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

class Qwen3TTSCodePredictor {
 public:
  Qwen3TTSCodePredictor(const std::string& prefill_path,
                        const std::string& decode_path, int device_id = 0);

  void ResetCaches();
  std::vector<float16> Prefill(const Qwen3TTSHiddenSequence& input,
                               const Qwen3TTSHiddenSequence& padding_hidden);
  std::vector<float16> Decode(const Qwen3TTSHiddenSequence& input,
                              int32_t context_length,
                              int32_t generation_step);

  size_t prefill_length() const { return runtime_->prefill_length(); }
  size_t hidden_dim() const { return runtime_->hidden_dim(); }
  size_t context_length() const { return runtime_->context_length(); }

 private:
  std::unique_ptr<Qwen3TTSPrefillDecodeRuntime> runtime_;
};

}  // namespace houmo
