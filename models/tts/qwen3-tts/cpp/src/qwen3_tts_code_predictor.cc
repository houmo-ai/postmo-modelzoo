/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_code_predictor.cc
 * Description:
 *   Qwen3-TTS CodePredictor prefill and decode inference implementation.
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

#include "qwen3_tts_code_predictor.h"

#include <algorithm>
#include <stdexcept>

namespace houmo {

Qwen3TTSCodePredictor::Qwen3TTSCodePredictor(
    const std::string& prefill_path, const std::string& decode_path,
    int device_id)
    : runtime_(std::make_unique<Qwen3TTSPrefillDecodeRuntime>(
          prefill_path, decode_path, device_id)) {
  if (runtime_->prefill_output_count() != 1 ||
      runtime_->decode_output_count() != 1) {
    throw Exception("CodePredictor must produce one logits output");
  }
}

void Qwen3TTSCodePredictor::ResetCaches() { runtime_->ResetCaches(); }

std::vector<float16> Qwen3TTSCodePredictor::Prefill(
    const Qwen3TTSHiddenSequence& input,
    const Qwen3TTSHiddenSequence& padding_hidden) {
  input.Validate();
  padding_hidden.Validate();
  if (input.sequence_length == 0 || input.hidden_dim != hidden_dim() ||
      padding_hidden.sequence_length != 1 ||
      padding_hidden.hidden_dim != hidden_dim()) {
    throw std::invalid_argument("Invalid CodePredictor prefill input");
  }

  const size_t chunk_length = prefill_length();
  const size_t chunk_count =
      (input.sequence_length + chunk_length - 1) / chunk_length;
  std::vector<std::vector<float16>> final_outputs;
  for (size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
    const size_t start = chunk_index * chunk_length;
    const size_t valid_tokens =
        std::min(chunk_length, input.sequence_length - start);
    Qwen3TTSHiddenSequence chunk;
    chunk.sequence_length = chunk_length;
    chunk.hidden_dim = hidden_dim();
    chunk.data.reserve(chunk_length * hidden_dim());
    const auto begin = input.data.begin() +
                       static_cast<std::ptrdiff_t>(start * hidden_dim());
    const auto end = begin +
                     static_cast<std::ptrdiff_t>(valid_tokens * hidden_dim());
    chunk.data.insert(chunk.data.end(), begin, end);
    for (size_t index = valid_tokens; index < chunk_length; ++index) {
      chunk.data.insert(chunk.data.end(), padding_hidden.data.begin(),
                        padding_hidden.data.end());
    }
    const bool is_last = chunk_index + 1 == chunk_count;
    auto outputs = runtime_->RunPrefill(
        chunk, static_cast<int32_t>(start),
        static_cast<int32_t>(valid_tokens), 0, is_last);
    if (is_last) final_outputs = std::move(outputs);
  }
  if (final_outputs.size() != 1) {
    throw Exception("Unexpected CodePredictor prefill output");
  }
  return std::move(final_outputs[0]);
}

std::vector<float16> Qwen3TTSCodePredictor::Decode(
    const Qwen3TTSHiddenSequence& input, int32_t context_length,
    int32_t generation_step) {
  input.Validate();
  if (input.sequence_length != 1 || input.hidden_dim != hidden_dim()) {
    throw std::invalid_argument("Invalid CodePredictor decode input");
  }
  auto outputs = runtime_->RunDecode(input, context_length, generation_step);
  if (outputs.size() != 1) {
    throw Exception("Unexpected CodePredictor decode output");
  }
  return std::move(outputs[0]);
}

}  // namespace houmo
