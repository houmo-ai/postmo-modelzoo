/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: asr_model.h
 * Description:
 *   ASRModel base class for Automatic Speech Recognition models.
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

#ifndef HOUMO_ASR_MODEL_H
#define HOUMO_ASR_MODEL_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "base/houmo.h"
#include "core/context.h"
#include "modules/audio_processor.h"
#include "modules/sampler.h"

namespace houmo {

// ============================================================================
// Tensor Type Definition
// ============================================================================

/**
 * @brief Simple tensor wrapper for ASR model I/O
 *
 * Uses Eigen::Tensor internally for compatibility with existing code.
 */
template <typename Scalar, int NumIndices>
using Tensor = Eigen::Tensor<Scalar, NumIndices>;

/**
 * @brief Float16 tensor (commonly used in ASR models)
 */
using TensorF16 = Tensor<float16, 3>;

/**
 * @brief Float32 tensor
 */
using TensorF32 = Tensor<float, 3>;

// ============================================================================
// ASR-specific Types
// ============================================================================

/**
 * @brief ASR performance metrics
 */
struct ASRPerfInfo {
  float audio_load_time = 0.0f;  ///< Audio load + feature extraction time (ms)
  float encode_time = 0.0f;      ///< Encoder inference total time (ms)
  float detect_lang_time =
      0.0f;                     ///< Language detection time (ms, Whisper only)
  float prefill_time = 0.0f;    ///< Prefill phase total time (ms)
  float decode_time = 0.0f;     ///< Decode phase total time (ms)
  float total_time = 0.0f;      ///< End-to-end total time (ms)
  float ttft_time = 0.0f;       ///< Time to first token (ms)
  int output_tokens = 0;        ///< Total output tokens
  int n_chunks = 0;             ///< Number of audio chunks/loops
  float audio_duration = 0.0f;  ///< Total audio duration (s)
  float overall_rtf =
      0.0f;  ///< Overall RTF: total_time_ms/1000 / audio_duration
  float inference_rtf = 0.0f;  ///< Pure inference RTF: (total_time -
                               ///< audio_load_time)/1000 / audio_duration
  float decode_tps =
      0.0f;  ///< Decode throughput: output_tokens / (decode_time_ms/1000)
  float overall_tps =
      0.0f;  ///< Overall throughput: output_tokens / (total_time_ms/1000)
};

/**
 * @brief Token callback for streaming output
 * @param token Generated token ID
 * @return true to continue, false to stop
 */
using ASRTokenCallback = std::function<bool(Token)>;

// ============================================================================
// ASRModel Base Class
// ============================================================================

/**
 * @brief ASRModel base class for Automatic Speech Recognition
 *
 * Provides common ASR functionality:
 * - Audio preprocessing (Mel Spectrogram)
 * - Sampling management
 * - Model info getters
 *
 * Specific ASR models (Whisper, GLM-ASR, Qwen3-ASR) inherit from ASRModel
 * and implement model-specific logic.
 */
class ASRModel {
 public:
  explicit ASRModel(const ModelConfig& config);
  virtual ~ASRModel();

  // Non-copyable
  ASRModel(const ASRModel&) = delete;
  ASRModel& operator=(const ASRModel&) = delete;

  // Moveable
  ASRModel(ASRModel&&) noexcept = default;
  ASRModel& operator=(ASRModel&&) noexcept = default;

  // ========== Common Interface (Base class implementation) ==========

  /**
   * @brief Get number of mel bins
   */
  int n_mels() const { return n_mels_; }

  int n_frames() const { return n_frames_; }

  /**
   * @brief Get number of attention heads
   */
  int num_heads() const { return num_heads_; }

  /**
   * @brief Get KV cache max length
   */
  int cache_max_len() const { return cache_max_len_; }

  /**
   * @brief Get number of decoder layers
   */
  int num_decode_layers() const { return num_decode_layers_; }

  // ========== Pure Virtual Interface (Subclass implementation) ==========

  /**
   * @brief Create inference context
   * @param n_ctx Context length (0 = use default from model)
   * @return Unique pointer to Context
   */
  virtual std::unique_ptr<Context> create_context(int n_ctx = 0) = 0;

  /**
   * @brief Get start-of-transcript token ID
   */
  virtual Token sot_token_id() const = 0;

  /**
   * @brief Get language token ID
   * @param language Language code (e.g., "zh", "en", "yue")
   * @return Language token ID, or 0 if not supported
   */
  virtual Token lang_token_id(const std::string& language) const = 0;

  /**
   * @brief Get transcribe token ID
   */
  virtual Token transcribe_token_id() const = 0;

  /**
   * @brief Get no timestamps token ID
   */
  virtual Token notimestamps_token_id() const = 0;

  /**
   * @brief Get EOS token ID(s)
   * @return Vector of EOS token IDs
   */
  virtual std::vector<Token> eos_token_ids() const = 0;

  /**
   * @brief Check if language detection is supported
   * @return true if supported, false otherwise
   */
  virtual bool supports_language_detection() const = 0;

 protected:
  // Protected members, directly accessible by subclasses
  ModelConfig config_;

  // Model parameters (set by subclass during load)
  int n_mels_ = 80;
  int n_frames_ = 0;
  int num_heads_ = 0;
  int cache_max_len_ = 0;
  int num_decode_layers_ = 0;
};

// ============================================================================
// ASRContext Base Class
// ============================================================================

/**
 * @brief ASRContext base class for ASR inference
 *
 * Manages per-request state for ASR inference:
 * - Encoder outputs
 * - Decoder state (KV cache)
 * - Generated tokens
 * - Performance metrics
 *
 * Profiling is handled by template methods in this base class.
 * Subclasses only implement _impl hooks — profiling is automatic.
 */
class ASRContext : public Context {
 public:
  explicit ASRContext(ASRModel* model, int n_ctx);
  ~ASRContext() override = default;

  // Non-copyable
  ASRContext(const ASRContext&) = delete;
  ASRContext& operator=(const ASRContext&) = delete;

  // ========== Pure Virtual Interface (Subclass implementation) ==========

  /**
   * @brief Encode audio features (encode forward pass)
   * @param mel_features Mel spectrogram features [n_mels * n_frames]
   * @param n_mels Number of mel bins
   * @param n_frames Number of time frames
   * @return Encoder outputs (as float vector)
   */
  virtual std::vector<float16> Encode(const std::vector<float>& mel_features,
                                      int n_mels, int n_frames) = 0;

  /**
   * @brief Detect language from encode outputs
   * @return Detected language token ID
   *
   * Note: For models without language detection (e.g., GLM-ASR),
   *       this returns the configured language token.
   */
  virtual Token DetectLanguage() = 0;

  /**
   * @brief Build prompt tokens for transcription
   * @param language_token Language token ID (from DetectLanguage or config)
   * @return Prompt token sequence
   */
  virtual std::vector<Token> BuildPrompt(Token language_token) = 0;

  /**
   * @brief Full transcription from audio file
   * @param audio_path Path to the audio file
   * @param params Sampling parameters
   * @param callback Token callback for streaming
   * @return Transcription text
   */
  virtual void Transcribe(const std::string& audio_path,
                          const SamplingParams& params,
                          ASRTokenCallback callback) = 0;

  /**
   * @brief Set language for transcription
   * @param language Language code (e.g., "zh", "en") or "auto"
   */
  virtual void set_language(const std::string& language) = 0;

  // ========== Common Interface (Base class implementation) ==========

  /**
   * @brief Get performance info
   */
  const ASRPerfInfo& perf_info() const;

  /**
   * @brief Get ASR model pointer
   */
  ASRModel* asr_model() const { return asr_model_; }

 protected:
  // ════════════════════════════════════════════════════════════════
  // Profiling template methods (base class implementation)
  // Call these from subclasses' Transcribe(); profiling is automatic.
  // ════════════════════════════════════════════════════════════════

  void do_encode(const std::vector<float>& mel, int n_mels, int n_frames);

  Token do_detect_language();

  Token do_prefill(const std::vector<Token>& tokens);

  Token do_decode(Token prev_token);

  void fill_perf_info(float audio_duration);

 private:
  // ════════════════════════════════════════════════════════════════
  // Virtual hooks — subclasses implement these for actual inference
  // The base class wraps them with profiling scopes automatically.
  // ════════════════════════════════════════════════════════════════

  // --- Encoder ---
  virtual void encode_preprocess_impl(const std::vector<float>& mel, int n_mels,
                                      int n_frames) = 0;
  virtual void encode_inference_impl() = 0;
  virtual void encode_postprocess_impl() = 0;

  // --- Language detection (default no-ops for non-Whisper models) ---
  virtual void detect_lang_preprocess_impl() {}
  virtual void detect_lang_inference_impl() {}
  virtual Token detect_lang_postprocess_impl() { return 0; }

  // --- Prefill ---
  virtual void prefill_preprocess_impl(const std::vector<Token>& tokens) = 0;
  virtual void prefill_inference_impl() = 0;
  virtual Token prefill_postprocess_impl() = 0;

  // --- Decode ---
  virtual void decode_preprocess_impl(Token prev_token) = 0;
  virtual void decode_inference_impl() = 0;
  virtual Token decode_postprocess_impl() = 0;

 protected:
  ASRModel* asr_model_;
  ASRPerfInfo perf_info_;
  std::string language_ = "auto";
};

}  // namespace houmo

#endif  // HOUMO_ASR_MODEL_H
