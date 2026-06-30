/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: asr_model.cc
 * Description:
 *   ASRModel base class implementation
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

#include "core/asr_model.h"

namespace houmo {

// ============================================================================
// ASRModel
// ============================================================================

ASRModel::ASRModel(const ModelConfig& config) : config_(config) {}

ASRModel::~ASRModel() = default;

// ============================================================================
// ASRContext
// ============================================================================

ASRContext::ASRContext(ASRModel* model, int n_ctx)
    : Context(nullptr, n_ctx), asr_model_(model) {}

const ASRPerfInfo& ASRContext::perf_info() const { return perf_info_; }

// ============================================================================
// Profiling template methods
// ============================================================================

void ASRContext::do_encode(const std::vector<float>& mel, int n_mels,
                           int n_frames) {
  auto& p = profiler_;
  {
    auto t = p.scope("transcribe.encode.preprocess");
    encode_preprocess_impl(mel, n_mels, n_frames);
  }
  {
    auto t = p.scope("transcribe.encode.inference");
    encode_inference_impl();
  }
  {
    auto t = p.scope("transcribe.encode.postprocess");
    encode_postprocess_impl();
  }
}

Token ASRContext::do_detect_language() {
  auto& p = profiler_;
  {
    auto t = p.scope("transcribe.detect_lang.preprocess");
    detect_lang_preprocess_impl();
  }
  {
    auto t = p.scope("transcribe.detect_lang.inference");
    detect_lang_inference_impl();
  }
  Token token;
  {
    auto t = p.scope("transcribe.detect_lang.postprocess");
    token = detect_lang_postprocess_impl();
  }
  return token;
}

Token ASRContext::do_prefill(const std::vector<Token>& tokens) {
  auto& p = profiler_;
  {
    auto t = p.scope("transcribe.prefill.preprocess");
    prefill_preprocess_impl(tokens);
  }
  {
    auto t = p.scope("transcribe.prefill.inference");
    prefill_inference_impl();
  }
  Token token;
  {
    auto t = p.scope("transcribe.prefill.postprocess");
    token = prefill_postprocess_impl();
  }
  return token;
}

Token ASRContext::do_decode(Token prev_token) {
  auto& p = profiler_;
  {
    auto t = p.scope("transcribe.decode.preprocess");
    decode_preprocess_impl(prev_token);
  }
  {
    auto t = p.scope("transcribe.decode.inference");
    decode_inference_impl();
  }
  Token token;
  {
    auto t = p.scope("transcribe.decode.postprocess");
    token = decode_postprocess_impl();
  }
  return token;
}

void ASRContext::fill_perf_info(float audio_duration) {
  perf_info_.audio_load_time =
      static_cast<float>(profiler_.get_time_ms("transcribe.audio_load"));
  perf_info_.encode_time =
      static_cast<float>(profiler_.get_time_ms("transcribe.encode.inference"));
  perf_info_.detect_lang_time =
      static_cast<float>(profiler_.get_time_ms("transcribe.detect_lang"));
  perf_info_.prefill_time =
      static_cast<float>(profiler_.get_time_ms("transcribe.prefill.inference"));
  perf_info_.decode_time =
      static_cast<float>(profiler_.get_time_ms("transcribe.decode.inference"));
  perf_info_.total_time = static_cast<float>(profiler_.e2e_ms());
  perf_info_.ttft_time = static_cast<float>(profiler_.ttft_ms());
  perf_info_.output_tokens = profiler_.output_tokens();
  perf_info_.n_chunks = profiler_.get_count("transcribe.encode.inference");
  perf_info_.audio_duration = audio_duration;

  if (audio_duration > 0.0f) {
    perf_info_.overall_rtf = (perf_info_.total_time / 1000.0f) / audio_duration;
    perf_info_.inference_rtf =
        ((perf_info_.encode_time + perf_info_.prefill_time +
          perf_info_.decode_time) /
         1000.0f) /
        audio_duration;
  }
  if (perf_info_.decode_time > 0.0f) {
    perf_info_.decode_tps = static_cast<float>(perf_info_.output_tokens) /
                            (perf_info_.decode_time / 1000.0f);
  }
  if (perf_info_.total_time > 0.0f) {
    perf_info_.overall_tps = static_cast<float>(perf_info_.output_tokens) /
                             (perf_info_.total_time / 1000.0f);
  }
}

}  // namespace houmo
