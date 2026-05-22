/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_speaker_embedding.h
 * Description:
 *   Speaker embedding extraction for CosyVoice3 TTS.
 *   Extracts speaker embedding from prompt audio using CampPlus model.
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

#ifndef HM_SPEAKER_EMBEDDING_H_
#define HM_SPEAKER_EMBEDDING_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common_types.h"
#include "hm_audio.h"
#include "tcim_runtime_utils.h"

namespace houmo {

/**
 * @brief Speaker embedding extractor using CampPlus model.
 *
 * Extracts a 192-dimensional speaker embedding vector from prompt audio.
 * The embedding captures speaker characteristics (voice, tone, style)
 * for voice cloning in CosyVoice3.
 *
 * Data flow (from Python demo.py lines 1332-1367):
 * ```
 * Input: PCM audio (16kHz)
 *   |
 *   v
 * ComputeFbank() -> [n_frames, 80]
 *   |
 *   v
 * Mean normalization (subtract mean across time)
 *   |
 *   v
 * Pad/trim to [1000, 80] (fixed T)
 *   |
 *   v
 * campplus.hmm inference (input: [1, 1000, 80])
 *   |
 * Output: speaker embedding [1, 192]
 * ```
 */
class HmSpeakerEmbedding {
 public:
  /**
   * @brief Constructor with model path.
   * @param speaker_embedding_model_path Path to the campplus model.
   */
  explicit HmSpeakerEmbedding(const std::string& speaker_embedding_model_path);

  /**
   * @brief Destructor.
   */
  ~HmSpeakerEmbedding();

  // Non-copyable
  HmSpeakerEmbedding(const HmSpeakerEmbedding&) = delete;
  HmSpeakerEmbedding& operator=(const HmSpeakerEmbedding&) = delete;

  // Moveable
  HmSpeakerEmbedding(HmSpeakerEmbedding&& other) noexcept;
  HmSpeakerEmbedding& operator=(HmSpeakerEmbedding&& other) noexcept;

  /**
   * @brief Extract speaker embedding from audio.
   *
   * @param pcm_data_16k PCM audio data at 16kHz sample rate.
   * @return Speaker embedding vector [192].
   *
   * Process:
   * 1. Compute 80-bin FBANK features
   * 2. Apply mean normalization (CMVN)
   * 3. Pad/trim to fixed length (1000 frames)
   * 4. Run campplus model inference
   * 5. Return 192-dim embedding
   */
  std::vector<TensorType> Extract(const std::vector<float>& pcm_data_16k,
                                  CosyVoice3Perf* perf = nullptr);

  /**
   * @brief Extract speaker embedding from audio file.
   *
   * @param audio_path Path to audio file (WAV/MP3/etc).
   * @return Speaker embedding vector [192].
   *
   * Loads audio, resamples to 16kHz, and extracts embedding.
   */
  std::vector<TensorType> ExtractFromFile(const std::string& audio_path);

  /**
   * @brief Get last inference time in milliseconds.
   * @return Inference time for last Extract() call.
   */
  float GetLastInferenceTimeMs() const { return last_inference_time_ms_; }

  /**
   * @brief Get embedding dimension.
   * @return Dimension of speaker embedding (192).
   */
  static constexpr int GetEmbeddingDim() { return kEmbeddingDim; }

  /**
   * @brief Get fixed time length.
   * @return Fixed time frames for CampPlus input (1000).
   */
  static constexpr int GetFixedT() { return kFixedT; }

 private:
  /**
   * @brief Prepare FBANK features for model input.
   *
   * @param fbank FBANK features from HmAudio::ComputeFbank().
   * @return Prepared features as float16 vector [1, 1000, 80].
   *
   * Steps:
   * 1. Transpose from [n_mels, n_frames] to [n_frames, n_mels]
   * 2. Apply mean normalization (CMVN)
   * 3. Pad/trim to kFixedT (1000) frames
   * 4. Convert to float16
   */
  std::vector<TensorType> PrepareFeatures(const FbankFeatures& fbank);

  /**
   * @brief Pad or trim features to fixed length.
   *
   * @param features Features in [n_frames, n_mels] layout.
   * @param n_frames Current number of frames.
   * @param n_mels Number of mel bins.
   * @return Features with exactly kFixedT frames.
   */
  std::vector<float> PadOrTrim(std::vector<float>& features, int n_frames,
                               int n_mels);

  // Audio processor for loading files
  HmAudio audio_processor_;

  std::map<std::string, tcim::Tensor> input_maps_;
  std::vector<std::string> output_names_;

  tcim::Module::WeightManager weight_manager_;
  std::shared_ptr<tcim::Module> speaker_module_;

  // Constants (matching Python demo.py)
  static constexpr int kFixedT = 1000;       ///< Fixed time length for CampPlus
  static constexpr int kEmbeddingDim = 192;  ///< Speaker embedding dimension
  static constexpr int kSampleRate = 16000;  ///< Required sample rate

  // Performance tracking
  float last_inference_time_ms_;
};

}  // namespace houmo

#endif  // HM_SPEAKER_EMBEDDING_H_