/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_audio.h
 * Description:
 *   Audio preprocessing for CosyVoice3 TTS - audio loading,
 *   resampling, mel spectrogram extraction, and FBANK computation.
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

#ifndef HM_AUDIO_H_
#define HM_AUDIO_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "audio/miniaudio.h"
#include "half/half.hpp"
#include "kaldi-native-fbank/csrc/online-feature.h"

namespace houmo {

// Tensor type alias for float16 (used by TCIM models)
using TensorType = half_float::half;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Mel spectrogram features for Flow/Vocoder (80-bin at 24kHz)
 */
struct MelFeatures80 {
  std::vector<TensorType> data;  ///< Mel features [n_mels * n_frames] float16
  int n_mels;                    ///< Number of mel bins (80)
  int n_frames;                  ///< Number of time frames
};

/**
 * @brief Mel spectrogram features for Speech Tokenizer (128-bin at 16kHz)
 */
struct MelFeatures128 {
  std::vector<TensorType> data;  ///< Mel features [n_mels * n_frames] float16
  int n_mels;                    ///< Number of mel bins (128)
  int n_frames;                  ///< Number of time frames
};

/**
 * @brief FBANK features for CampPlus (80-bin at 16kHz)
 */
struct FbankFeatures {
  std::vector<float> data;  ///< FBANK features [n_frames * n_mels] float32
  int n_mels;               ///< Number of mel bins (80)
  int n_frames;             ///< Number of time frames
};

/**
 * @brief Audio preprocessing class for CosyVoice3 TTS
 *
 * Handles:
 * - Audio file loading (any format via miniaudio)
 * - Automatic resampling to target sample rate (16kHz or 24kHz)
 * - Mono conversion
 * - Mel spectrogram computation (80-bin for Flow/Vocoder at 24kHz)
 * - Mel spectrogram computation (128-bin for Speech Tokenizer at 16kHz)
 * - FBANK feature computation (80-bin for CampPlus at 16kHz)
 */
class HmAudio {
 public:
  // CosyVoice3 audio processing constants

  // Sample rates
  static constexpr int kSampleRate16k =
      16000;  ///< Sample rate for tokenizer/campplus
  static constexpr int kSampleRate24k = 24000;  ///< Sample rate for vocoder

  // Mel params for 24kHz (Flow/Vocoder)
  static constexpr int kNFFT24k = 1920;  ///< FFT size for 24kHz mel
  static constexpr int kHopSize24k =
      480;  ///< Hop size for 24kHz mel (20ms stride)
  static constexpr int kWinSize24k = 1920;  ///< Window size for 24kHz mel
  static constexpr int kNMels80 = 80;       ///< Mel bins for Flow/Vocoder
  static constexpr int kNFFTBins24k = 961;  ///< FFT output bins (n_fft/2 + 1)

  // Mel params for 16kHz (Speech Tokenizer - Whisper style)
  static constexpr int kNFFT16k = 400;  ///< FFT size for 16kHz mel
  static constexpr int kHopSize16k =
      160;  ///< Hop size for 16kHz mel (10ms stride)
  static constexpr int kNMels128 = 128;     ///< Mel bins for Speech Tokenizer
  static constexpr int kNFFTBins16k = 201;  ///< FFT output bins (n_fft/2 + 1)

  // FBANK params for 16kHz (CampPlus)
  static constexpr int kFbankNMels = 80;     ///< FBANK mel bins
  static constexpr int kFbankFixedT = 1000;  ///< Fixed time frames for CampPlus

  /**
   * @brief Default constructor
   */
  HmAudio();

  // Non-copyable
  HmAudio(const HmAudio&) = delete;
  HmAudio& operator=(const HmAudio&) = delete;

  // Moveable
  HmAudio(HmAudio&&) noexcept = default;
  HmAudio& operator=(HmAudio&&) noexcept = default;

  ~HmAudio();

  /**
   * @brief Load audio file, convert to mono, resample to target sample rate
   * @param audio_path Path to audio file (WAV/MP3/FLAC etc.)
   * @param target_sr Target sample rate (16000 or 24000)
   * @return true if successful
   */
  bool LoadAudio(const std::string& audio_path, int target_sr);

  /**
   * @brief Get loaded PCM data
   * @return PCM samples as float32 vector
   */
  const std::vector<float>& GetPcmData() const { return pcm_data_; }

  /**
   * @brief Get sample rate of loaded audio
   */
  int GetSampleRate() const { return sample_rate_; }

  /**
   * @brief Get number of samples
   */
  int GetNumSamples() const { return total_samples_; }

  /**
   * @brief Get audio duration in seconds
   */
  float GetDuration() const;

  /**
   * @brief Compute 80-bin mel spectrogram for Flow/Vocoder (24kHz)
   * @param pcm_data Input PCM samples (24kHz mono)
   * @return MelFeatures80 with mel spectrogram [n_mels, n_frames]
   */
  MelFeatures80 ComputeMelSpectrogram80(
      const std::vector<float>& pcm_data) const;

  /**
   * @brief Compute 128-bin mel spectrogram for Speech Tokenizer (16kHz)
   *        Uses Whisper-style parameters
   * @param pcm_data Input PCM samples (16kHz mono)
   * @return MelFeatures128 with mel spectrogram [n_mels, n_frames]
   */
  MelFeatures128 ComputeMelSpectrogram128(
      const std::vector<float>& pcm_data) const;

  /**
   * @brief Compute FBANK features for CampPlus (16kHz)
   * @param pcm_data Input PCM samples (16kHz mono)
   * @return FbankFeatures with FBANK features [n_frames, n_mels]
   */
  FbankFeatures ComputeFbank(const std::vector<float>& pcm_data) const;

  /**
   * @brief Get mel features for loaded audio (80-bin, 24kHz)
   */
  MelFeatures80 GetMelFeatures80() const;

  /**
   * @brief Get mel features for loaded audio (128-bin, 16kHz)
   */
  MelFeatures128 GetMelFeatures128() const;

  /**
   * @brief Get FBANK features for loaded audio (16kHz)
   */
  FbankFeatures GetFbankFeatures() const;

 private:
  std::vector<float> pcm_data_;
  int sample_rate_;
  int total_samples_;
  bool audio_loaded_;

  // Helper functions
  std::vector<float> ComputeHannWindow(int win_size) const;
  std::vector<float> ApplyReflectionPadding(const std::vector<float>& audio,
                                            int pad_len) const;

  // Hz to mel conversion
  float HzToMel(float hz) const;
  float MelToHz(float mel) const;
};

}  // namespace houmo

#endif  // HM_AUDIO_H_