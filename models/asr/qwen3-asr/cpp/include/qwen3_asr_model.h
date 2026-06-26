/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_asr_model.h
 * Description:
 *   Qwen3-ASR model implementation.
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

#ifndef HOUMO_QWEN3_ASR_MODEL_H
#define HOUMO_QWEN3_ASR_MODEL_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/tcim_utils.h"
#include "core/asr_model.h"
#include "modules/audio_processor.h"
#include "modules/embedding.h"
#include "modules/tokenizer.h"

namespace houmo {

class Qwen3AsrContext : public ASRContext {
 public:
  explicit Qwen3AsrContext(ASRModel* model, int n_ctx);
  ~Qwen3AsrContext() override = default;

  Qwen3AsrContext(const Qwen3AsrContext&) = delete;
  Qwen3AsrContext& operator=(const Qwen3AsrContext&) = delete;

  void set_audio_processor(int sample_rate, int chunk_seconds,
                           int encoder_window_seconds);

  std::vector<float16> Encode(const std::vector<float>& mel_features,
                              int n_mels, int n_frames) override;
  Token DetectLanguage() override;
  std::vector<Token> BuildPrompt(Token language_token) override;
  Token prefill(const std::vector<Token>& tokens) override;
  Token decode(Token prev_token) override;
  void Transcribe(const std::string& audio_path, const SamplingParams& params,
                  ASRTokenCallback callback) override;
  void set_language(const std::string& language) override;
  Token get_asr_start_token() const { return asr_start; }

 private:
  // --- ASRContext profiling hooks ---
  void encode_preprocess_impl(const std::vector<float>& mel, int n_mels,
                              int n_frames) override;
  void encode_inference_impl() override;
  void encode_postprocess_impl() override;

  void prefill_preprocess_impl(const std::vector<Token>& tokens) override;
  void prefill_inference_impl() override;
  Token prefill_postprocess_impl() override;

  void decode_preprocess_impl(Token prev_token) override;
  void decode_inference_impl() override;
  Token decode_postprocess_impl() override;

  std::shared_ptr<AudioProcessor> audio_processor_;
  std::vector<float16> audio_embeds_;
  int decode_position_ = 0;
  int asr_start = 0;

  // Temp encoding/prefill params
  int encode_n_mels_ = 0;
  int encode_n_frames_ = 0;
  int prefill_seq_len_ = 0;
};

class Qwen3AsrModel : public ASRModel {
 public:
  explicit Qwen3AsrModel(const ModelConfig& config);
  ~Qwen3AsrModel();

  Qwen3AsrModel(const Qwen3AsrModel&) = delete;
  Qwen3AsrModel& operator=(const Qwen3AsrModel&) = delete;

  std::unique_ptr<Context> create_context(int n_ctx = 0) override;

  Token sot_token_id() const override { return audio_pad_id_; }
  Token lang_token_id(const std::string& language) const override {
    (void)language;
    return 0;
  }
  Token transcribe_token_id() const override { return 0; }
  Token notimestamps_token_id() const override { return 0; }
  std::vector<Token> eos_token_ids() const override { return {eos_token_id_}; }
  bool supports_language_detection() const override { return true; }
  Token audio_pad_id() const { return audio_pad_id_; }
  int max_feature_per_loop() const { return max_feature_one_loop_; }

  std::shared_ptr<tcim::Module> encoder_module() const {
    return encoder_module_;
  }
  std::shared_ptr<tcim::Module> prefill_module() const {
    return prefill_module_;
  }
  std::shared_ptr<tcim::Module> decode_module() const { return decode_module_; }
  std::shared_ptr<HfTokenizer> tokenizer() const { return tokenizer_; }
  int max_prefill() const { return max_prefill_; }
  int hidden_size() const { return hidden_size_; }
  int num_decode_layers() const { return num_decode_layers_; }
  int max_new_tokens() const { return max_new_tokens_; }

  const float16* get_embedding(Token token) const;
  const float16* get_embedding(const std::vector<Token>& tokens) const;

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
  std::shared_ptr<HfTokenizer> tokenizer_;
  std::unique_ptr<Embedding> embedding_;

  int max_prefill_ = 0;
  int hidden_size_ = 0;
  int num_decode_layers_ = 0;
  int max_new_tokens_ = 2048;
  int max_feature_one_loop_ = 0;

  Token audio_pad_id_ = 151676;
  Token eos_token_id_ = 0;

  std::unordered_map<std::string, tcim::Tensor> encoder_input_map_;
  std::unordered_map<std::string, tcim::Tensor> prefill_input_map_;
  std::unordered_map<std::string, tcim::Tensor> decode_input_map_;

 private:
  void load();
};

}  // namespace houmo

#endif  // HOUMO_QWEN3_ASR_MODEL_H
