/*
 * Copyright (c) 2026 HOUMO AI
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * File: hm_whisper_audio.hpp
 * Description: Audio preprocessing for Whisper ASR - audio loading, resampling,
 *              and mel spectrogram extraction
 */

#ifndef HM_WHISPER_AUDIO_HPP_
#define HM_WHISPER_AUDIO_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "half.hpp"

namespace houmo {

// Tensor type alias for Whisper model inference
using TensorType = half_float::half;

/**
 * @brief Audio chunk data structure
 */
struct AudioChunk {
  std::vector<float> pcm_data;  ///< PCM samples (float32, 16kHz mono)
  float duration;               ///< Duration in seconds
  int chunk_index;              ///< Chunk index (0-based)
  bool is_last;                 ///< Is this the last chunk
};

/**
 * @brief Mel spectrogram features
 */
struct MelFeatures {
  std::vector<TensorType> data;  ///< Mel features [n_mels * n_frames]
  int n_mels;                    ///< Number of mel bins (80)
  int n_frames;                  ///< Number of time frames (3000 for 30s)
};

/**
 * @brief Audio preprocessing class for Whisper ASR
 *
 * Handles:
 * - Audio file loading (via miniaudio)
 * - Automatic resampling to 16kHz mono
 * - Audio chunking for long files
 * - Mel spectrogram computation (STFT + mel filter bank)
 */
class HmWhisperAudio {
 public:
  // Whisper audio processing constants
  static constexpr int kSampleRate = 16000;  ///< Target sample rate

  /**
   * @brief Default constructor (30 second chunks, 80 mel bins)
   */
  HmWhisperAudio();

  /**
   * @brief Constructor with custom chunk size and mel bins
   * @param chunk_size_seconds Chunk size in seconds
   * @param n_mels Number of mel bins (default: 80, whisper-large-v3-turbo uses 128)
   */
  explicit HmWhisperAudio(int chunk_size_seconds, int n_mels = 80);

  // Non-copyable
  HmWhisperAudio(const HmWhisperAudio&) = delete;
  HmWhisperAudio& operator=(const HmWhisperAudio&) = delete;

  // Moveable
  HmWhisperAudio(HmWhisperAudio&&) noexcept = default;
  HmWhisperAudio& operator=(HmWhisperAudio&&) noexcept = default;

  ~HmWhisperAudio();

  /**
   * @brief Load audio file and prepare for processing
   * @param audio_path Path to audio file (any format)
   * @return true if successful
   */
  bool LoadAudio(const std::string& audio_path);

  /**
   * @brief Get total number of audio chunks
   */
  int GetNumChunks() const;

  /**
   * @brief Get specific audio chunk
   * @param chunk_index Chunk index (0-based)
   * @return AudioChunk with PCM data
   */
  AudioChunk GetChunk(int chunk_index) const;

  /**
   * @brief Compute mel spectrogram for PCM data
   * @param pcm_data Input PCM samples
   * @return MelFeatures with mel spectrogram
   */
  MelFeatures ComputeMelSpectrogram(const std::vector<float>& pcm_data) const;

  /**
   * @brief Get mel features for specific chunk
   * @param chunk_index Chunk index
   * @return MelFeatures for that chunk
   */
  MelFeatures GetChunkMelFeatures(int chunk_index) const;

  /**
   * @brief Get total audio duration
   * @return Duration in seconds
   */
  float GetTotalDuration() const;

  /**
   * @brief Get chunk size setting
   * @return Chunk size in seconds
   */
  int GetChunkSize() const;

 private:
  void DownmixToMono(const std::vector<float>& interleaved_pcm, int channels,
                     std::vector<float>* mono_pcm) const;
  bool ResampleAudio(const std::vector<float>& input_pcm,
                     uint32_t input_sample_rate,
                     std::vector<float>* output_pcm) const;

  int chunk_size_seconds_;
  int n_mels_;
  int n_samples_per_chunk_;
  std::vector<float> pcm_data_;
  int total_samples_;
  bool audio_loaded_;
};

// Keep backward compatible alias
using tensor_type = TensorType;

}  // namespace houmo

#endif  // HM_WHISPER_AUDIO_HPP_