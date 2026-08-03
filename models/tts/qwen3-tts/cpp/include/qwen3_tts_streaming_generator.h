/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_streaming_generator.h
 * Description:
 *   Qwen3-TTS streaming codec-frame generation interface.
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

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

#include "qwen3_tts_code_predictor.h"
#include "qwen3_tts_code_predictor_embedding.h"
#include "qwen3_tts_codec_embedding.h"
#include "qwen3_tts_sampler.h"
#include "qwen3_tts_streaming_prompt_builder.h"
#include "qwen3_tts_talker.h"

namespace houmo {

using Qwen3TTSCodecFrame = std::array<Token, 16>;

struct Qwen3TTSGenerationPerf {
  double talker_prefill_seconds = 0.0;
  double talker_decode_seconds = 0.0;
  double talker_sampling_seconds = 0.0;
  double frame_prepare_seconds = 0.0;
  double predictor_prepare_seconds = 0.0;
  double predictor_prefill_seconds = 0.0;
  double predictor_decode_seconds = 0.0;
  double predictor_sampling_seconds = 0.0;
  size_t talker_prefill_count = 0;
  size_t talker_decode_count = 0;
  size_t talker_sampling_count = 0;
  size_t predictor_prefill_count = 0;
  size_t predictor_decode_count = 0;
  size_t predictor_sampling_count = 0;
  bool reached_eos = false;
  size_t eos_step = 0;
  bool reached_max_frames = false;
};

class Qwen3TTSStreamingGenerator {
 public:
  Qwen3TTSStreamingGenerator(
      Qwen3TTSTalker* talker, Qwen3TTSCodecEmbedding* talker_embedding,
      Qwen3TTSCodePredictor* code_predictor,
      Qwen3TTSCodePredictorEmbedding* code_predictor_embedding,
      Qwen3TTSSampler talker_sampler, Qwen3TTSSampler predictor_sampler,
      Token eos_token_id = 2150);

  std::vector<Qwen3TTSCodecFrame> Generate(
      const Qwen3TTSStreamingPrompt& prompt, size_t max_frames);
  size_t Generate(const Qwen3TTSStreamingPrompt& prompt, size_t max_frames,
                  const std::function<bool(const Qwen3TTSCodecFrame&)>&
                      on_frame);
  const Qwen3TTSGenerationPerf& perf() const { return perf_; }

 private:
  static Qwen3TTSHiddenSequence Concatenate(
      const Qwen3TTSHiddenSequence& left,
      const Qwen3TTSHiddenSequence& right);
  static void AddInPlace(Qwen3TTSHiddenSequence* target,
                         const Qwen3TTSHiddenSequence& value);
  static Qwen3TTSHiddenSequence TokenAt(
      const Qwen3TTSHiddenSequence& sequence, size_t index);
  static Qwen3TTSHiddenSequence BuildNextTalkerInput(
      const Qwen3TTSHiddenSequence& group_zero_hidden,
      const std::vector<Qwen3TTSHiddenSequence>& predictor_hiddens,
      const Qwen3TTSStreamingPrompt& prompt, size_t frame_index);

  Qwen3TTSTalker* talker_;
  Qwen3TTSCodecEmbedding* talker_embedding_;
  Qwen3TTSCodePredictor* code_predictor_;
  Qwen3TTSCodePredictorEmbedding* code_predictor_embedding_;
  Qwen3TTSSampler talker_sampler_;
  Qwen3TTSSampler predictor_sampler_;
  Token eos_token_id_;
  Qwen3TTSGenerationPerf perf_;
};

}  // namespace houmo
