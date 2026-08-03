/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_streaming_prompt_builder.cc
 * Description:
 *   Qwen3-TTS streaming Talker prompt construction implementation.
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

#include "qwen3_tts_streaming_prompt_builder.h"

#include <stdexcept>

namespace houmo {
namespace {

void RequireSingleToken(const Qwen3TTSHiddenSequence& value,
                        const char* name) {
  value.Validate();
  if (value.sequence_length != 1) {
    throw std::invalid_argument(std::string(name) + " must contain one token");
  }
}

void AppendTokenSum(std::vector<float16>* output,
                    const Qwen3TTSHiddenSequence& left, size_t left_index,
                    const Qwen3TTSHiddenSequence& right, size_t right_index) {
  for (size_t dim = 0; dim < left.hidden_dim; ++dim) {
    output->push_back(left.data[left_index * left.hidden_dim + dim] +
                      right.data[right_index * right.hidden_dim + dim]);
  }
}

}  // namespace

Qwen3TTSStreamingPrompt Qwen3TTSStreamingPromptBuilder::Build(
    const Qwen3TTSHiddenSequence& role_hidden,
    const Qwen3TTSHiddenSequence& body_hidden,
    const Qwen3TTSHiddenSequence& tts_bos_hidden,
    const Qwen3TTSHiddenSequence& tts_eos_hidden,
    const Qwen3TTSHiddenSequence& tts_pad_hidden,
    const Qwen3TTSHiddenSequence& codec_prompt_hidden) const {
  role_hidden.Validate();
  body_hidden.Validate();
  codec_prompt_hidden.Validate();
  RequireSingleToken(tts_bos_hidden, "TTS BOS hidden");
  RequireSingleToken(tts_eos_hidden, "TTS EOS hidden");
  RequireSingleToken(tts_pad_hidden, "TTS PAD hidden");
  if (role_hidden.sequence_length != 3 || body_hidden.sequence_length == 0 ||
      codec_prompt_hidden.sequence_length < 2) {
    throw std::invalid_argument("Invalid streaming prompt sequence lengths");
  }

  const size_t hidden_dim = role_hidden.hidden_dim;
  if (hidden_dim == 0 || body_hidden.hidden_dim != hidden_dim ||
      tts_bos_hidden.hidden_dim != hidden_dim ||
      tts_eos_hidden.hidden_dim != hidden_dim ||
      tts_pad_hidden.hidden_dim != hidden_dim ||
      codec_prompt_hidden.hidden_dim != hidden_dim) {
    throw std::invalid_argument("Streaming prompt hidden dimensions do not match");
  }

  Qwen3TTSStreamingPrompt output;
  output.initial_prompt.sequence_length =
      role_hidden.sequence_length + codec_prompt_hidden.sequence_length;
  output.initial_prompt.hidden_dim = hidden_dim;
  output.initial_prompt.data.reserve(output.initial_prompt.sequence_length *
                                     hidden_dim);
  output.initial_prompt.data.insert(output.initial_prompt.data.end(),
                                    role_hidden.data.begin(),
                                    role_hidden.data.end());

  const size_t codec_length = codec_prompt_hidden.sequence_length;
  for (size_t index = 0; index + 2 < codec_length; ++index) {
    AppendTokenSum(&output.initial_prompt.data, tts_pad_hidden, 0,
                   codec_prompt_hidden, index);
  }
  AppendTokenSum(&output.initial_prompt.data, tts_bos_hidden, 0,
                 codec_prompt_hidden, codec_length - 2);
  AppendTokenSum(&output.initial_prompt.data, body_hidden, 0,
                 codec_prompt_hidden, codec_length - 1);

  output.trailing_text_hidden.sequence_length = body_hidden.sequence_length;
  output.trailing_text_hidden.hidden_dim = hidden_dim;
  output.trailing_text_hidden.data.reserve(body_hidden.sequence_length *
                                           hidden_dim);
  output.trailing_text_hidden.data.insert(
      output.trailing_text_hidden.data.end(),
      body_hidden.data.begin() + static_cast<std::ptrdiff_t>(hidden_dim),
      body_hidden.data.end());
  output.trailing_text_hidden.data.insert(
      output.trailing_text_hidden.data.end(), tts_eos_hidden.data.begin(),
      tts_eos_hidden.data.end());
  output.text_pad_hidden = tts_pad_hidden;

  output.initial_prompt.Validate();
  output.trailing_text_hidden.Validate();
  return output;
}

}  // namespace houmo
