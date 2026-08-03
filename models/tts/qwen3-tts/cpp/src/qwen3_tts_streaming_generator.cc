/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_streaming_generator.cc
 * Description:
 *   Qwen3-TTS streaming codec-frame generation implementation.
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

#include "qwen3_tts_streaming_generator.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace houmo {

Qwen3TTSStreamingGenerator::Qwen3TTSStreamingGenerator(
    Qwen3TTSTalker* talker, Qwen3TTSCodecEmbedding* talker_embedding,
    Qwen3TTSCodePredictor* code_predictor,
    Qwen3TTSCodePredictorEmbedding* code_predictor_embedding,
    Qwen3TTSSampler talker_sampler, Qwen3TTSSampler predictor_sampler,
    Token eos_token_id)
    : talker_(talker),
      talker_embedding_(talker_embedding),
      code_predictor_(code_predictor),
      code_predictor_embedding_(code_predictor_embedding),
      talker_sampler_(std::move(talker_sampler)),
      predictor_sampler_(std::move(predictor_sampler)),
      eos_token_id_(eos_token_id) {
  if (talker_ == nullptr || talker_embedding_ == nullptr ||
      code_predictor_ == nullptr || code_predictor_embedding_ == nullptr) {
    throw std::invalid_argument("Streaming generator dependencies are null");
  }
}

Qwen3TTSHiddenSequence Qwen3TTSStreamingGenerator::Concatenate(
    const Qwen3TTSHiddenSequence& left,
    const Qwen3TTSHiddenSequence& right) {
  left.Validate();
  right.Validate();
  if (left.hidden_dim != right.hidden_dim) {
    throw std::invalid_argument("Cannot concatenate different hidden dimensions");
  }
  Qwen3TTSHiddenSequence output;
  output.sequence_length = left.sequence_length + right.sequence_length;
  output.hidden_dim = left.hidden_dim;
  output.data = left.data;
  output.data.insert(output.data.end(), right.data.begin(), right.data.end());
  return output;
}

void Qwen3TTSStreamingGenerator::AddInPlace(
    Qwen3TTSHiddenSequence* target,
    const Qwen3TTSHiddenSequence& value) {
  target->Validate();
  value.Validate();
  if (target->sequence_length != value.sequence_length ||
      target->hidden_dim != value.hidden_dim) {
    throw std::invalid_argument("Cannot add hidden sequences with different shapes");
  }
  for (size_t index = 0; index < target->data.size(); ++index) {
    target->data[index] += value.data[index];
  }
}

Qwen3TTSHiddenSequence Qwen3TTSStreamingGenerator::TokenAt(
    const Qwen3TTSHiddenSequence& sequence, size_t index) {
  sequence.Validate();
  if (index >= sequence.sequence_length) {
    throw std::out_of_range("Hidden sequence token index is out of range");
  }
  Qwen3TTSHiddenSequence output;
  output.sequence_length = 1;
  output.hidden_dim = sequence.hidden_dim;
  const auto begin = sequence.data.begin() +
                     static_cast<std::ptrdiff_t>(index * sequence.hidden_dim);
  output.data.assign(begin,
                     begin + static_cast<std::ptrdiff_t>(sequence.hidden_dim));
  return output;
}

Qwen3TTSHiddenSequence Qwen3TTSStreamingGenerator::BuildNextTalkerInput(
    const Qwen3TTSHiddenSequence& group_zero_hidden,
    const std::vector<Qwen3TTSHiddenSequence>& predictor_hiddens,
    const Qwen3TTSStreamingPrompt& prompt, size_t frame_index) {
  Qwen3TTSHiddenSequence next_talker_input = group_zero_hidden;
  for (const auto& hidden : predictor_hiddens) {
    AddInPlace(&next_talker_input, hidden);
  }
  Qwen3TTSHiddenSequence text_hidden = prompt.text_pad_hidden;
  if (frame_index < prompt.trailing_text_hidden.sequence_length) {
    text_hidden = TokenAt(prompt.trailing_text_hidden, frame_index);
  }
  AddInPlace(&next_talker_input, text_hidden);
  return next_talker_input;
}

std::vector<Qwen3TTSCodecFrame> Qwen3TTSStreamingGenerator::Generate(
    const Qwen3TTSStreamingPrompt& prompt, size_t max_frames) {
  std::vector<Qwen3TTSCodecFrame> frames;
  frames.reserve(max_frames);
  Generate(prompt, max_frames, [&](const Qwen3TTSCodecFrame& frame) {
    frames.push_back(frame);
    return true;
  });
  return frames;
}

size_t Qwen3TTSStreamingGenerator::Generate(
    const Qwen3TTSStreamingPrompt& prompt, size_t max_frames,
    const std::function<bool(const Qwen3TTSCodecFrame&)>& on_frame) {
  prompt.initial_prompt.Validate();
  prompt.trailing_text_hidden.Validate();
  prompt.text_pad_hidden.Validate();
  perf_ = {};
  if (max_frames == 0) return 0;

  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  auto talker_output =
      talker_->Prefill(prompt.initial_prompt,
                       talker_embedding_->Lookup({0}));
  perf_.talker_prefill_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  perf_.talker_prefill_count = 1;
  std::vector<Token> talker_tokens;
  start = Clock::now();
  Token group_zero = talker_sampler_.Sample(talker_output.logits, talker_tokens);
  perf_.talker_sampling_seconds +=
      std::chrono::duration<double>(Clock::now() - start).count();
  ++perf_.talker_sampling_count;
  talker_tokens.push_back(group_zero);
  int32_t past_sequence_length =
      static_cast<int32_t>(prompt.initial_prompt.sequence_length);

  const auto predictor_padding = code_predictor_embedding_->Lookup(0, 0);
  size_t generated_frames = 0;
  for (size_t frame_index = 0; frame_index < max_frames; ++frame_index) {
    if (group_zero == eos_token_id_) {
      perf_.reached_eos = true;
      perf_.eos_step = generated_frames;
      break;
    }

    start = Clock::now();
    const auto group_zero_hidden = talker_embedding_->Lookup({group_zero});
    const auto predictor_input =
        Concatenate(talker_output.past_hidden, group_zero_hidden);
    perf_.frame_prepare_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();

    start = Clock::now();
    code_predictor_->ResetCaches();
    perf_.predictor_prepare_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();

    start = Clock::now();
    auto predictor_logits =
        code_predictor_->Prefill(predictor_input, predictor_padding);
    perf_.predictor_prefill_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();
    ++perf_.predictor_prefill_count;

    Qwen3TTSCodecFrame frame{};
    std::vector<Token> predictor_tokens;
    std::vector<Qwen3TTSHiddenSequence> predictor_hiddens;
    start = Clock::now();
    Token predictor_token =
        predictor_sampler_.Sample(predictor_logits, predictor_tokens);
    perf_.predictor_sampling_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();
    ++perf_.predictor_sampling_count;
    predictor_tokens.push_back(predictor_token);

    int32_t predictor_context_length =
        static_cast<int32_t>(predictor_input.sequence_length);
    for (size_t step = 0;
         step < Qwen3TTSCodePredictorEmbedding::kCodebookCount - 1; ++step) {
      const auto predictor_hidden =
          code_predictor_embedding_->Lookup(step, predictor_token);
      predictor_hiddens.push_back(predictor_hidden);
      start = Clock::now();
      predictor_logits = code_predictor_->Decode(
          predictor_hidden, predictor_context_length,
          static_cast<int32_t>(step + 1));
      perf_.predictor_decode_seconds +=
          std::chrono::duration<double>(Clock::now() - start).count();
      ++perf_.predictor_decode_count;
      start = Clock::now();
      predictor_token =
          predictor_sampler_.Sample(predictor_logits, predictor_tokens);
      perf_.predictor_sampling_seconds +=
          std::chrono::duration<double>(Clock::now() - start).count();
      ++perf_.predictor_sampling_count;
      predictor_tokens.push_back(predictor_token);
      ++predictor_context_length;
    }
    start = Clock::now();
    frame[0] = group_zero;
    std::copy(predictor_tokens.begin(), predictor_tokens.end(),
              frame.begin() + 1);
    perf_.frame_prepare_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();
    ++generated_frames;
    if (!on_frame(frame)) break;
    if (generated_frames == max_frames) {
      perf_.reached_max_frames = true;
      break;
    }

    start = Clock::now();
    const auto next_talker_input = BuildNextTalkerInput(
        group_zero_hidden, predictor_hiddens, prompt, frame_index);
    perf_.frame_prepare_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();
    start = Clock::now();
    talker_output = talker_->Decode(next_talker_input, past_sequence_length);
    perf_.talker_decode_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();
    ++perf_.talker_decode_count;
    start = Clock::now();
    group_zero = talker_sampler_.Sample(talker_output.logits, talker_tokens);
    perf_.talker_sampling_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();
    ++perf_.talker_sampling_count;
    talker_tokens.push_back(group_zero);
    ++past_sequence_length;
  }
  if (group_zero == eos_token_id_ && !perf_.reached_eos) {
    perf_.reached_eos = true;
    perf_.eos_step = generated_frames;
  }
  return generated_frames;
}

}  // namespace houmo
