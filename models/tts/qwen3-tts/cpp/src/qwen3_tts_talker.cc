/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_talker.cc
 * Description:
 *   Qwen3-TTS Talker prefill and decode inference implementation.
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

#include "qwen3_tts_talker.h"

#include <algorithm>
#include <stdexcept>

namespace houmo {

Qwen3TTSTalker::Qwen3TTSTalker(const std::string& prefill_path,
                               const std::string& decode_path, int device_id)
    : runtime_(std::make_unique<Qwen3TTSPrefillDecodeRuntime>(
          prefill_path, decode_path, device_id)) {
  if (runtime_->prefill_output_count() != 2 ||
      runtime_->decode_output_count() != 2) {
    throw Exception("Talker must produce logits and past_hidden");
  }
}

Qwen3TTSTalkerOutput Qwen3TTSTalker::ParseOutputs(
    std::vector<std::vector<float16>> outputs, size_t hidden_dim) {
  if (outputs.size() != 2 || outputs[1].size() != hidden_dim) {
    throw Exception("Unexpected Talker output shape");
  }
  Qwen3TTSTalkerOutput output;
  output.logits = std::move(outputs[0]);
  output.past_hidden.sequence_length = 1;
  output.past_hidden.hidden_dim = hidden_dim;
  output.past_hidden.data = std::move(outputs[1]);
  return output;
}

Qwen3TTSTalkerOutput Qwen3TTSTalker::Prefill(
    const Qwen3TTSHiddenSequence& prompt,
    const Qwen3TTSHiddenSequence& padding_hidden,
    int32_t past_sequence_length) {
  prompt.Validate();
  padding_hidden.Validate();
  if (prompt.sequence_length == 0 || prompt.hidden_dim != hidden_dim() ||
      padding_hidden.sequence_length != 1 ||
      padding_hidden.hidden_dim != hidden_dim()) {
    throw std::invalid_argument("Invalid Talker prefill input");
  }

  runtime_->ResetCaches();
  const size_t chunk_length = prefill_length();
  const size_t chunk_count =
      (prompt.sequence_length + chunk_length - 1) / chunk_length;
  std::vector<std::vector<float16>> final_outputs;
  for (size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
    const size_t start = chunk_index * chunk_length;
    const size_t valid_tokens =
        std::min(chunk_length, prompt.sequence_length - start);
    Qwen3TTSHiddenSequence chunk;
    chunk.sequence_length = chunk_length;
    chunk.hidden_dim = hidden_dim();
    chunk.data.reserve(chunk_length * hidden_dim());
    const auto begin = prompt.data.begin() +
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
        chunk, past_sequence_length + static_cast<int32_t>(start),
        static_cast<int32_t>(valid_tokens), 0, is_last);
    if (is_last) final_outputs = std::move(outputs);
  }
  return ParseOutputs(std::move(final_outputs), hidden_dim());
}

Qwen3TTSTalkerOutput Qwen3TTSTalker::Decode(
    const Qwen3TTSHiddenSequence& input, int32_t past_sequence_length) {
  input.Validate();
  if (input.sequence_length != 1 || input.hidden_dim != hidden_dim()) {
    throw std::invalid_argument("Invalid Talker decode input");
  }
  return ParseOutputs(runtime_->RunDecode(input, past_sequence_length),
                      hidden_dim());
}

}  // namespace houmo
