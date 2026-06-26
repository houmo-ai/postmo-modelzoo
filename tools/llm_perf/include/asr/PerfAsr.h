/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: PerfAsr.h
 * Description:
 *   ASR performance test using simulated data (random mel + forced decode rounds).
 *   Unified GLM-ASR and Qwen3-ASR perf flow with zero model_type branching.
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

#ifndef PERF_ASR_H
#define PERF_ASR_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/houmo.h"
#include "core/asr_model.h"
#include "tcim/tcim_runtime.h"

struct AsrTranscribeResult {
  double encode_time_ms = 0.0;
  double prefill_time_ms = 0.0;
  double decode_time_ms = 0.0;
  double total_time_ms = 0.0;
  double ttft_ms = 0.0;
  float audio_duration_s = 0.0f;
  float overall_rtf = 0.0f;
  float inference_rtf = 0.0f;
  float decode_tps = 0.0f;
  float overall_tps = 0.0f;
  int output_tokens = 0;
};

class PerfAsrModel : public houmo::ASRModel {
 public:
  explicit PerfAsrModel(const houmo::ModelConfig& config);
  ~PerfAsrModel() override;

  PerfAsrModel(const PerfAsrModel&) = delete;
  PerfAsrModel& operator=(const PerfAsrModel&) = delete;

  std::unique_ptr<houmo::Context> create_context(int n_ctx = 0) override;

  houmo::Token sot_token_id() const override { return 0; }
  houmo::Token lang_token_id(const std::string&) const override { return 0; }
  houmo::Token transcribe_token_id() const override { return 0; }
  houmo::Token notimestamps_token_id() const override { return 0; }
  std::vector<houmo::Token> eos_token_ids() const override { return {}; }
  bool supports_language_detection() const override { return false; }

  std::shared_ptr<tcim::Module> encoder_module() const {
    return encoder_module_;
  }
  std::shared_ptr<tcim::Module> prefill_module() const {
    return prefill_module_;
  }
  std::shared_ptr<tcim::Module> decode_module() const {
    return decode_module_;
  }
  int max_prefill() const { return max_prefill_; }
  int hidden_size() const { return hidden_size_; }
  int encoder_window() const { return encoder_window_; }
  int n_mels_val() const { return n_mels_; }
  bool encoder_input_is_f16() const { return encoder_input_is_f16_; }
  bool encoder_has_input_lengths() const { return encoder_has_input_lengths_; }

  std::unordered_map<std::string, tcim::Tensor>& encoder_input_map() {
    return encoder_input_map_;
  }
  std::unordered_map<std::string, tcim::Tensor>& prefill_input_map() {
    return prefill_input_map_;
  }
  std::unordered_map<std::string, tcim::Tensor>& decode_input_map() {
    return decode_input_map_;
  }

 protected:
  std::unique_ptr<tcim::DevManager> dev_manager_;
  std::unique_ptr<tcim::Module::WeightManager> weight_manager_;
  std::shared_ptr<tcim::Module> encoder_module_;
  std::shared_ptr<tcim::Module> prefill_module_;
  std::shared_ptr<tcim::Module> decode_module_;

  int max_prefill_ = 0;
  int hidden_size_ = 0;
  int encoder_window_ = 0;
  bool encoder_input_is_f16_ = false;
  bool encoder_has_input_lengths_ = false;

  std::unordered_map<std::string, tcim::Tensor> encoder_input_map_;
  std::unordered_map<std::string, tcim::Tensor> prefill_input_map_;
  std::unordered_map<std::string, tcim::Tensor> decode_input_map_;

 private:
  void load();
};

class PerfAsrContext : public houmo::ASRContext {
 public:
  explicit PerfAsrContext(houmo::ASRModel* model, int n_ctx);
  ~PerfAsrContext() override = default;

  PerfAsrContext(const PerfAsrContext&) = delete;
  PerfAsrContext& operator=(const PerfAsrContext&) = delete;

  std::vector<float16> Encode(const std::vector<float>&, int, int) override {
    return {};
  }
  houmo::Token DetectLanguage() override { return 0; }
  std::vector<houmo::Token> BuildPrompt(houmo::Token) override { return {}; }
  void Transcribe(const std::string&, const houmo::SamplingParams&,
                  houmo::ASRTokenCallback) override {}
  void set_language(const std::string&) override {}

  AsrTranscribeResult PerfRun(float audio_len_seconds,
                               int token_per_second,
                               int sample_rate = 16000);

 private:
  void encode_preprocess_impl(const std::vector<float>& mel, int n_mels,
                              int n_frames) override;
  void encode_inference_impl() override;
  void encode_postprocess_impl() override;

  void prefill_preprocess_impl(
      const std::vector<houmo::Token>& tokens) override;
  void prefill_inference_impl() override;
  houmo::Token prefill_postprocess_impl() override;

  void decode_preprocess_impl(houmo::Token prev_token) override;
  void decode_inference_impl() override;
  houmo::Token decode_postprocess_impl() override;

  std::vector<float16> audio_embeds_;
  int encode_n_frames_ = 0;
  int prefill_seq_len_ = 0;
  int decode_position_ = 0;
  int T_out_ = 0;
};

#endif  // PERF_ASR_H
