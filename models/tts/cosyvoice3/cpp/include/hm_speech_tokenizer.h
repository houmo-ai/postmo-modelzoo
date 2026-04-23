/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_speech_tokenizer.h
 * Description:
 *   Speech tokenizer module for CosyVoice3 TTS.
 *   Extracts speech tokens from prompt audio using speech_tokenizer.hmm model.
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

#ifndef HM_SPEECH_TOKENIZER_H_
#define HM_SPEECH_TOKENIZER_H_

#include <memory>
#include <vector>

#include "common_types.h"
#include "hm_audio.h"
#include "tcim_runtime_utils.h"

namespace houmo {

/**
 * @brief Speech tokenizer for extracting speech tokens from audio.
 *
 * This class uses the speech_tokenizer.hmm model to extract discrete
 * speech tokens from prompt audio. The tokens are used for voice cloning
 * and speech generation in the LLM module.
 *
 * Data flow:
 *   1. Input: PCM audio (16kHz mono)
 *   2. Compute 128-bin mel spectrogram
 *   3. Pad to fixed shape [1, 128, 3000] (max 30s audio)
 *   4. Create attention masks (mask, mask1)
 *   5. Run speech_tokenizer.hmm inference
 *   6. Output: speech token sequence [n_frames/4]
 *
 * Reference: demo.py lines 1286-1330
 */
class HmSpeechTokenizer {
 public:
  // Constants from Python implementation
  static constexpr int kMaxMelFrames =
      3000;                           ///< Max mel frames (30s audio at 16kHz)
  static constexpr int kNMels = 128;  ///< Number of mel bins
  static constexpr int kTokenRatio =
      4;  ///< Token/mel frame ratio (n_tokens = n_frames/4)
  static constexpr int kMaxTokens = 750;   ///< Max tokens (3000/4)
  static constexpr int kMaskDim2 = 20;     ///< mask shape dim 2
  static constexpr int kMask1Dim2 = 1280;  ///< mask1 shape dim 2

  // Float16 minimum value for attention mask
  static constexpr float kMaskMinValue =
      -65504.0f;  ///< float16 min for invalid attention

  /**
   * @brief Constructor with TCIM module reference.
   * @param speech_tokenizer_model_path Path to the speech_tokenizer
   * HmTcimModule model.
   * @param audio Pointer to HmAudio instance for mel computation.
   */
  HmSpeechTokenizer(std::string speech_tokenizer_model_path,
                    std::shared_ptr<HmAudio> audio);

  /**
   * @brief Destructor.
   */
  ~HmSpeechTokenizer();

  // Non-copyable
  HmSpeechTokenizer(const HmSpeechTokenizer&) = delete;
  HmSpeechTokenizer& operator=(const HmSpeechTokenizer&) = delete;

  // Moveable
  HmSpeechTokenizer(HmSpeechTokenizer&& other) noexcept;
  HmSpeechTokenizer& operator=(HmSpeechTokenizer&& other) noexcept;

  /**
   * @brief Extract speech tokens from audio.
   * @param pcm_data_16k PCM audio data at 16kHz (mono).
   * @return Vector of speech token IDs.
   *
   * Process:
   * 1. Check audio duration (max 30s)
   * 2. Compute 128-bin mel spectrogram
   * 3. Pad mel features to [1, 128, 3000]
   * 4. Create attention masks
   * 5. Run speech_tokenizer.hmm
   * 6. Extract and return token sequence
   *
   * Reference: demo.py lines 1286-1330
   */
  std::vector<int> Extract(const std::vector<float>& pcm_data_16k,
                           CosyVoice3Perf* perf = nullptr);

  /**
   * @brief Extract speech tokens with length info.
   * @param pcm_data_16k PCM audio data at 16kHz (mono).
   * @return Pair of (token_ids, token_count).
   */
  std::pair<std::vector<int>, int> ExtractWithLength(
      const std::vector<float>& pcm_data_16k, CosyVoice3Perf* perf = nullptr);

  /**
   * @brief Get last inference time in milliseconds.
   * @return Inference time in ms.
   */
  float GetInferenceTimeMs() const { return inference_time_ms_; }

 private:
  /**
   * @brief Prepare input tensors for speech tokenizer.
   * @param mel_features Mel features from HmAudio::ComputeMelSpectrogram128.
   * @param feat_len Number of valid mel frames.
   *
   * Creates three input tensors:
   * - padded_mel: [1, 128, 3000] float16
   * - mask: [1, 20, 750, 750] float16 (attention mask)
   * - mask1: [1, 750, 1280] float16 (encoder mask)
   *
   * Reference: demo.py lines 1298-1308
   */
  void PrepareInput(const MelFeatures128& mel_features, int feat_len);

  /**
   * @brief Create attention mask tensor.
   * @param valid_len Number of valid token positions (feat_len/4).
   * @return Vector of float16 mask values.
   *
   * Shape: [1, 20, 750, 750]
   * Values: 0 for valid positions, -65504 for invalid
   */
  std::vector<TensorType> CreateAttentionMask(int valid_len);

  /**
   * @brief Create encoder mask tensor.
   * @param valid_len Number of valid token positions (feat_len/4).
   * @return Vector of float16 mask values.
   *
   * Shape: [1, 750, 1280]
   * Values: 1.0 for valid positions, 0.0 for invalid
   */
  std::vector<TensorType> CreateEncoderMask(int valid_len);

  /**
   * @brief Convert mel features to padded input tensor.
   * @param mel_features Mel features from ComputeMelSpectrogram128.
   * @param feat_len Number of valid frames.
   * @return Vector of float16 padded mel values [1, 128, 3000].
   */
  std::vector<TensorType> PreparePaddedMel(const MelFeatures128& mel_features,
                                           int feat_len);

  /**
   * @brief Extract tokens from model output.
   * @param valid_len Number of valid tokens.
   * @return Vector of token IDs.
   */
  std::vector<int> ExtractTokens(int valid_len);

 private:
  // Audio processor for mel computation
  std::shared_ptr<HmAudio> audio_;

  // Input tensor (cached)
  std::map<std::string, tcim::Tensor> input_maps_;

  // Output tensor names (cached)
  std::vector<std::string> output_names_;

  // Performance tracking
  float inference_time_ms_;

  // Cache for last extraction
  int last_feat_len_;
  int last_token_len_;

  tcim::Module::WeightManager weight_manager_;
  std::shared_ptr<tcim::Module> speech_module_;
};

}  // namespace houmo

#endif  // HM_SPEECH_TOKENIZER_H_