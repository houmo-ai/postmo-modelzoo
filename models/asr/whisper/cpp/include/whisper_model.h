/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: whisper_model.h
 * Description:
 *   Whisper ASR model implementation.
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

#ifndef HOUMO_WHISPER_MODEL_H
#define HOUMO_WHISPER_MODEL_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/tcim_utils.h"
#include "core/asr_model.h"
#include "modules/audio_processor.h"
#include "modules/tokenizer.h"

namespace houmo {

/**
 * @brief Whisper inference context
 *
 * Whisper-specific implementation:
 * - DetectLanguage: Run decoder forward pass, extract language from logits
 * - BuildPrompt: [sot, lang, transcribe, notimestamps]
 * - Encode: Fixed 3000 frames input
 * - AudioProcessor: Built-in audio processing with model config
 */
class WhisperContext : public ASRContext {
 public:
  explicit WhisperContext(ASRModel* model, int n_ctx);
  ~WhisperContext() override = default;

  // Non-copyable
  WhisperContext(const WhisperContext&) = delete;
  WhisperContext& operator=(const WhisperContext&) = delete;

  // ========== Audio Processing ==========

  /**
   * @brief Get audio processor (configured with model's n_mels)
   */
  std::shared_ptr<AudioProcessor> audio_processor() { return audio_processor_; }

  /**
   * @brief Set audio processor parameters
   * @param sample_rate Target sample rate in Hz (default: 16000)
   * @param chunk_seconds Audio chunk size in seconds for segmentation
   * @param encoder_window_seconds Encoder input window size in seconds
   * (default: 30 for Whisper)
   */
  void set_audio_processor(int sample_rate, int chunk_seconds,
                           int encoder_window_seconds);

  /**
   * @brief Get current target sample rate
   */
  int sample_rate() const;

  /**
   * @brief Get current chunk seconds
   */
  int chunk_seconds() const;

  /**
   * @brief Get encode window seconds
   */
  int encoder_window_seconds() const;

  /**
   * @brief Load audio file and extract mel features
   * @param audio_path Path to audio file
   * @return Mel features ready for transcription
   */
  MelFeatures LoadAudio(const std::string& audio_path);

  // ========== Whisper-specific Implementation ==========

  /**
   * @brief Run encode to get encode outputs (KV cache for cross attention)
   * @param mel_features Mel spectrogram features [n_mels * n_frames]
   * @param n_mels Number of mel bins
   * @param n_frames Number of frames
   * @return Encoder outputs (key_state and value_state for each layer)
   */
  std::vector<tcim::Tensor> RunEncoder(const std::vector<float16>& mel_features,
                                       int n_mels, int n_frames);

  std::vector<float16> Encode(const std::vector<float>& mel_features,
                              int n_mels, int n_frames) override;

  Token DetectLanguage() override;

  std::vector<Token> BuildPrompt(Token language_token) override;

  Token prefill(const std::vector<Token>& tokens) override;
  Token decode(Token prev_token) override;

  void Transcribe(const std::string& audio_path, const SamplingParams& params,
                  ASRTokenCallback callback) override;

  void set_language(const std::string& language) override;

 private:
  // --- ASRContext profiling hooks ---
  void encode_preprocess_impl(const std::vector<float>& mel, int n_mels,
                              int n_frames) override;
  void encode_inference_impl() override;
  void encode_postprocess_impl() override;

  void detect_lang_preprocess_impl() override;
  void detect_lang_inference_impl() override;
  Token detect_lang_postprocess_impl() override;

  void prefill_preprocess_impl(const std::vector<Token>& tokens) override;
  void prefill_inference_impl() override;
  Token prefill_postprocess_impl() override;

  void decode_preprocess_impl(Token prev_token) override;
  void decode_inference_impl() override;
  Token decode_postprocess_impl() override;

  std::shared_ptr<AudioProcessor> audio_processor_;

  // Internal inference state
  std::vector<tcim::Tensor>
      encoder_outputs_;         // Cached encode outputs for cross-attn
  Token detected_lang_id_ = 0;  // Cached language detection result
  int decode_position_ = 0;     // Current decode position

  // Temp encoding params (set before do_encode)
  int encode_n_mels_ = 0;
  int encode_n_frames_ = 0;

  // Temp prefill tokens (set before do_prefill)
  std::vector<Token> prefill_tokens_;
  int prefill_prompt_len_ = 0;
};

/**
 * @brief Whisper ASR model
 *
 * Whisper-specific features:
 * - Language detection from encode logits
 * - Fixed prompt tokens: [sot, lang, transcribe, notimestamps]
 * - Encoder output as KV cache (not embeddings)
 * - Fixed 3000 frames (30 seconds)
 *
 * Model files:
 * - encode.hmm: Audio encode
 * - prefill.hmm: Prefill decoder
 * - decode.hmm: Decode decoder
 */
class WhisperModel : public ASRModel {
 public:
  explicit WhisperModel(const ModelConfig& config);
  ~WhisperModel();

  // Non-copyable
  WhisperModel(const WhisperModel&) = delete;
  WhisperModel& operator=(const WhisperModel&) = delete;

  // ========== Context Creation ==========

  std::unique_ptr<Context> create_context(int n_ctx = 0) override;

  // ========== Whisper-specific Token IDs ==========

  Token sot_token_id() const override { return sot_token_id_; }
  Token lang_token_id(const std::string& language) const override;
  Token transcribe_token_id() const override { return transcribe_token_id_; }
  Token notimestamps_token_id() const override {
    return notimestamps_token_id_;
  }
  std::vector<Token> eos_token_ids() const override { return {eos_token_id_}; }

  // ========== Whisper-specific Features ==========

  bool supports_language_detection() const override { return true; }

  // ========== Model Accessors (for WhisperContext) ==========

  std::shared_ptr<tcim::Module> encoder_module() const {
    return encoder_module_;
  }
  std::shared_ptr<tcim::Module> prefill_module() const {
    return prefill_module_;
  }
  std::shared_ptr<tcim::Module> decode_module() const { return decode_module_; }
  std::shared_ptr<HfTokenizer> tokenizer() const { return tokenizer_; }

  int base_idx() const { return base_idx_; }
  int encoder_seq_len() const { return encoder_seq_len_; }
  const std::vector<Token>& lang_to_id() const { return lang_to_id_; }

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
  // TCIM modules
  std::unique_ptr<tcim::DevManager> dev_manager_;
  std::unique_ptr<tcim::Module::WeightManager> weight_manager_;
  std::shared_ptr<tcim::Module> encoder_module_;
  std::shared_ptr<tcim::Module> prefill_module_;
  std::shared_ptr<tcim::Module> decode_module_;

  // Tokenizer
  std::shared_ptr<HfTokenizer> tokenizer_;

  // Whisper-specific model parameters
  int base_idx_ = 0;
  int encoder_seq_len_ = 0;

  // Token IDs
  Token sot_token_id_ = 0;
  Token transcribe_token_id_ = 0;
  Token notimestamps_token_id_ = 0;
  Token eos_token_id_ = 0;
  Token default_lang_token_id_ = 0;

  // Language token map: "zh" -> <|zh|> token ID
  std::unordered_map<std::string, Token> lang_token_map_;
  std::vector<Token> lang_to_id_;  // All language token IDs for detection

  // Input tensor maps
  std::unordered_map<std::string, tcim::Tensor> encoder_input_map_;
  std::unordered_map<std::string, tcim::Tensor> prefill_input_map_;
  std::unordered_map<std::string, tcim::Tensor> decode_input_map_;

 private:
  void load();
};

}  // namespace houmo

#endif  // HOUMO_WHISPER_MODEL_H
