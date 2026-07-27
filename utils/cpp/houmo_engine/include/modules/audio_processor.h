/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: audio_processor.h
 * Description:
 *   Audio processor for ASR models - supports Whisper, GLM-ASR, Qwen3-ASR
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

#ifndef HOUMO_AUDIO_PROCESSOR_H
#define HOUMO_AUDIO_PROCESSOR_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "base/houmo.h"

namespace houmo {

/**
 * @brief Audio data
 *
 * Stores PCM audio data and related metadata.
 */
struct AudioData {
  std::vector<float> pcm;   ///< PCM data (float32, 16kHz, mono)
  int sample_rate = 16000;  ///< Sample rate (Hz)
  float duration = 0.0f;    ///< Duration (seconds)
};

/**
 * @brief Audio features
 *
 * Stores audio feature data and dimension information.
 */
struct MelFeatures {
  std::vector<float16> data;  ///< Feature data (FP16)
  int feature_dim = 0;  ///< Feature dimension (80 or 128 for Mel Spectrogram)
  int num_frames = 0;   ///< Number of frames
  float duration =
      0.0f;  ///< Actual audio duration (seconds), excluding padding
};

enum class AudioFeatureMode {
  kCenterPad,
  kWhisper,
};

/**
 * @brief Audio processor configuration
 *
 * Supports different ASR models through configuration:
 * - Whisper, GLM-ASR, Qwen3-ASR: Mel Spectrogram
 */
struct AudioProcessorConfig {
  // ========== Basic parameters ==========

  int sample_rate = 16000;  ///< Target sample rate (Hz), default 16kHz

  // ========== Mel Spectrogram parameters (Whisper/GLM-ASR/Qwen3-ASR)
  // ==========

  int n_mels =
      80;  ///< Number of Mel bins (80 or 128 for whisper-large-v3-turbo)
  int chunk_seconds =
      30;  ///< PCM chunk size (seconds), used for audio segmentation
  int encoder_window_seconds =
      30;  ///< Encoder input window size (seconds), Whisper fixed at 30 seconds

  // ========== Advanced parameters (optional) ==========

  int fft_size = 400;        ///< FFT size (25ms at 16kHz)
  int hop_length = 160;      ///< Hop length (10ms at 16kHz)
  int win_length = 400;      ///< Window length (25ms at 16kHz)
  int feature_threads = 4;   ///< Worker threads for Mel feature extraction
  AudioFeatureMode feature_mode = AudioFeatureMode::kCenterPad;
  float mel_fmin = 0.0f;     ///< Mel filter minimum frequency
  float mel_fmax = 8000.0f;  ///< Mel filter maximum frequency
};

/**
 * @brief Audio processor
 *
 * Supports audio preprocessing for ASR models:
 * - Whisper (Mel Spectrogram)
 * - GLM-ASR (Mel Spectrogram)
 * - Qwen3-ASR (Mel Spectrogram)
 *
 * Typical workflow:
 * 1. LoadAudio() - Load audio file
 * 2. ChunkPCM() - PCM level chunking (30 seconds)
 * 3. ExtractFeatures() - Extract features
 *
 * Or use Process() for one-stop processing.
 *
 * Reference:
 * - Whisper: /hmdd/imodelzoo/models/asr/whisper/demo.py
 * - GLM-ASR: /hmdd/imodelzoo/models/asr/glm-asr/demo.py
 * - Qwen3-ASR: /hmdd/imodelzoo/models/asr/qwen3-asr/demo_asr.py
 */
class AudioProcessor {
 public:
  /**
   * @brief Default constructor (Mel Spectrogram mode)
   */
  AudioProcessor();

  /**
   * @brief Configuration constructor
   * @param config Processor configuration
   */
  explicit AudioProcessor(const AudioProcessorConfig& config);

  // Non-copyable
  AudioProcessor(const AudioProcessor&) = delete;
  AudioProcessor& operator=(const AudioProcessor&) = delete;

  // Moveable
  AudioProcessor(AudioProcessor&&) noexcept = default;
  AudioProcessor& operator=(AudioProcessor&&) noexcept = default;

  ~AudioProcessor();

  // ========== Audio loading ==========

  /**
   * @brief Load audio file
   * @param path Audio file path (supports wav, mp3, flac, etc.)
   * @return AudioData PCM data
   *
   * Internal workflow:
   * - Use miniaudio to read audio file
   * - Automatic resampling to 16kHz
   * - Convert to mono
   * - Normalize to [-1, 1] range
   */
  AudioData LoadAudio(const std::string& path);

  // ========== Feature extraction ==========

  /**
   * @brief Extract audio features
   * @param audio Input audio
   * @return MelFeatures Audio features
   *
   * Mel Spectrogram mode (Whisper/GLM-ASR/Qwen3-ASR):
   * - STFT (Short-Time Fourier Transform)
   * - Mel Filter Bank
   * - Log compression
   * - Output dimension: [n_mels, n_frames]
   */
  MelFeatures ExtractFeatures(const AudioData& audio);

  // ========== Chunk processing ==========

  /**
   * @brief PCM level chunking
   * @param audio Input audio
   * @return List of audio chunks
   *
   * Split long audio into fixed-length chunks (default 30 seconds):
   * - Chunks shorter than 30 seconds are zero-padded
   * - Each returned chunk is an independent AudioData
   */
  std::vector<AudioData> ChunkPCM(const AudioData& audio);

  // ========== One-stop processing ==========

  /**
   * @brief Load audio and extract features (supports automatic chunking)
   * @param path Audio file path
   * @return List of feature chunks
   *
   * Complete workflow:
   * 1. LoadAudio(path)
   * 2. ChunkPCM(audio)
   * 3. ExtractFeatures(chunk) for each chunk
   */
  std::vector<MelFeatures> Process(const std::string& path);

  // ========== Information retrieval ==========

  /**
   * @brief Get feature dimension
   * @return Feature dimension (80 or 128 for Mel Spectrogram)
   */
  int feature_dim() const;

  /**
   * @brief Get sample rate
   */
  int sample_rate() const { return config_.sample_rate; }

  /**
   * @brief Get number of Mel bins
   */
  int n_mels() const { return config_.n_mels; }

  /**
   * @brief Get configuration
   */
  const AudioProcessorConfig& config() const { return config_; }

 private:
  AudioProcessorConfig config_;

  MelFeatures ComputeMelSpectrogram(const AudioData& audio);
};

}  // namespace houmo

#endif  // HOUMO_AUDIO_PROCESSOR_H
