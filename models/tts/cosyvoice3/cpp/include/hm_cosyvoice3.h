/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_cosyvoice3.h
 * Description:
 *   Main CosyVoice3 TTS Pipeline class.
 *   Integrates all modules for end-to-end text-to-speech synthesis.
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

#ifndef HM_COSYVOICE3_H_
#define HM_COSYVOICE3_H_

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_types.h"
#include "hm_audio.h"
#include "hm_native_flow.h"
#include "hm_native_llm.h"
#include "hm_native_vocoder.h"
#include "hm_perf_monitor.h"
#include "hm_speaker_embedding.h"
#include "hm_speech_tokenizer.h"
#include "hm_text_frontend.h"
namespace houmo {

// ============================================================================
// WAV File Header Structure
// ============================================================================

/**
 * @brief WAV file header for saving audio.
 *
 * Standard WAV format:
 * - RIFF header
 * - fmt chunk (format information)
 * - data chunk (audio samples)
 */
struct WavHeader {
  char riff[4];           ///< "RIFF"
  uint32_t file_size;     ///< File size - 8
  char wave[4];           ///< "WAVE"
  char fmt[4];            ///< "fmt "
  uint32_t fmt_size;      ///< Format chunk size (16 for PCM)
  uint16_t audio_format;  ///< Audio format (1 for PCM)
  uint16_t num_channels;  ///< Number of channels (1 for mono)
  uint32_t sample_rate;   ///< Sample rate (24000)
  uint32_t byte_rate;     ///< Bytes per second (sample_rate * num_channels *
                          ///< bits_per_sample/8)
  uint16_t block_align;  ///< Block alignment (num_channels * bits_per_sample/8)
  uint16_t bits_per_sample;  ///< Bits per sample (16 or 32)
  char data[4];              ///< "data"
  uint32_t data_size;        ///< Data chunk size (num_samples * block_align)
};

// ============================================================================
// CosyVoice3 - Main TTS Pipeline Class
// ============================================================================

/**
 * @brief Main CosyVoice3 TTS Pipeline class.
 *
 * Integrates all modules for end-to-end text-to-speech synthesis:
 * - TextFrontend: Text normalization and tokenization
 * - Audio: Audio loading and feature extraction
 * - SpeechTokenizer: Extract speech tokens from prompt audio
 * - SpeakerEmbedding: Extract speaker embedding from prompt audio
 * - LLM: Generate speech tokens from text
 * - FlowDecoder: Convert speech tokens to mel spectrogram
 * - HiftVocoder: Convert mel spectrogram to audio waveform
 *
 * Provides three inference modes:
 * - InferenceZeroShot: Zero-shot voice cloning with prompt text and audio
 * - InferenceCrossLingual: Cross-lingual generation (no prompt text)
 * - InferenceInstruct2: Instruction-following synthesis (e.g., dialect
 * conversion)
 *
 * Reference: demo.py lines 1858-2047
 */
class CosyVoice3 {
 public:
  // ========================================================================
  // Construction
  // ========================================================================

  /**
   * @brief Constructor with configuration.
   * @param config CosyVoice3Config with model paths and settings.
   *
   * Initializes all modules:
   * 1. Create output directory
   * 2. Initialize TextFrontend (tokenizer)
   * 3. Initialize Audio processor
   * 4. Initialize SpeechTokenizer
   *
   * Reference: demo.py lines 1859-1924
   */
  explicit CosyVoice3(const CosyVoice3Config& config);

  /**
   * @brief Destructor.
   */
  ~CosyVoice3();

  // Non-copyable
  CosyVoice3(const CosyVoice3&) = delete;
  CosyVoice3& operator=(const CosyVoice3&) = delete;

  // Moveable
  CosyVoice3(CosyVoice3&& other) noexcept;
  CosyVoice3& operator=(CosyVoice3&& other) noexcept;

  // ========================================================================
  // Inference Methods
  // ========================================================================

  /**
   * @brief Zero-shot voice cloning synthesis.
   *
   * Uses prompt text and prompt audio to clone the speaker's voice
   * and generate speech for the target text.
   *
   * @param tts_text Target text to synthesize.
   * @param prompt_text Prompt text (with <|endofprompt|> marker).
   * @param prompt_wav_path Path to prompt audio file.
   * @param speed Playback speed ratio (default 1.0).
   * @param perf Optional performance metrics pointer.
   *
   * @return Generated audio waveform.
   *
   * Reference: demo.py lines 1956-1999
   *
   * Flow:
   * 1. Normalize prompt_text (split=false)
   * 2. Normalize tts_text (split=true) -> list of utterances
   * 3. For each utterance:
   *    a. frontend_zero_shot() -> model_input
   *    b. LLM inference -> speech_tokens
   *    c. FlowDecoder inference -> mel_features
   *    d. HiftVocoder inference -> waveform
   * 4. Concatenate all utterance waveforms
   */
  std::vector<float> InferenceZeroShot(const std::string& tts_text,
                                       const std::string& prompt_text,
                                       const std::string& prompt_wav_path,
                                       float speed = 1.0f,
                                       CosyVoice3Perf* perf = nullptr);

  /**
   * @brief Cross-lingual synthesis.
   *
   * Uses prompt audio only for speaker/style cloning, but removes prompt text
   * and prompt speech-token conditioning from the LLM path.
   *
   * @param tts_text Target text to synthesize.
   * @param prompt_wav_path Path to prompt audio file.
   * @param speed Playback speed ratio (default 1.0).
   * @param perf Optional performance metrics pointer.
   *
   * @return Generated audio waveform.
   */
  std::vector<float> InferenceCrossLingual(const std::string& tts_text,
                                           const std::string& prompt_wav_path,
                                           float speed = 1.0f,
                                           CosyVoice3Perf* perf = nullptr);

  /**
   * @brief Instruction-following zero-shot synthesis.
   *
   * Uses instruction text as prompt text while removing prompt speech-token
   * conditioning from the LLM path.
   *
   * @param tts_text Target text to synthesize.
   * @param instruct_text Instruction text.
   * @param prompt_wav_path Path to prompt audio file.
   * @param speed Playback speed ratio (default 1.0).
   * @param perf Optional performance metrics pointer.
   *
   * @return Generated audio waveform.
   */
  std::vector<float> InferenceInstruct2(const std::string& tts_text,
                                        const std::string& instruct_text,
                                        const std::string& prompt_wav_path,
                                        float speed = 1.0f,
                                        CosyVoice3Perf* perf = nullptr);

  // ========================================================================
  // Audio Output
  // ========================================================================

  /**
   * @brief Save audio waveform to WAV file.
   *
   * @param pcm_data Audio waveform samples.
   * @param output_path Output file path.
   * @param bits_per_sample Bits per sample (16 or 32, default 16).
   *
   * @return True if successful.
   *
   * Creates standard WAV file with:
   * - RIFF header
   * - PCM format (16-bit or 32-bit)
   * - Mono channel
   * - Sample rate from config (24000)
   */
  bool SaveWav(const std::vector<float>& pcm_data,
               const std::string& output_path, int bits_per_sample = 16);

  /**
   * @brief Save audio waveform to WAV file (int16 format).
   *
   * @param pcm_data_int16 Audio samples as int16.
   * @param output_path Output file path.
   *
   * @return True if successful.
   */
  bool SaveWavInt16(const std::vector<int16_t>& pcm_data_int16,
                    const std::string& output_path);

  // ========================================================================
  // Performance Reporting
  // ========================================================================

  /**
   * @brief Print performance summary for all inference runs.
   *
   * Prints aggregated statistics:
   * - Total inference count
   * - Average RTF (Real-Time Factor)
   * - Average LLM time
   * - Average Flow time
   * - Average Vocoder time
   *
   * Reference: demo.py lines 1926-1939
   */
  void PrintPerfSummary();

  /**
   * @brief Get performance history.
   * @return Vector of performance metrics for each inference run.
   */
  const std::vector<CosyVoice3Perf>& GetPerfHistory() const {
    return perf_history_;
  }

  /**
   * @brief Clear performance history.
   */
  void ClearPerfHistory() { perf_history_.clear(); }

  // ========================================================================
  // Configuration Accessors
  // ========================================================================

  /**
   * @brief Get sample rate.
   * @return Output audio sample rate (24000).
   */
  int GetSampleRate() const { return sample_rate_; }

  /**
   * @brief Get output directory.
   * @return Output directory path.
   */
  const std::string& GetOutputDir() const { return output_dir_; }

  /**
   * @brief Check if all modules are loaded.
   * @return True if all modules initialized successfully.
   */
  bool IsReady() const;

  // ========================================================================
  // Speaker Cache (Optional Feature)
  // ========================================================================

  /**
   * @brief Add zero-shot speaker to cache.
   *
   * Pre-computes speaker features for reuse across multiple synthesis.
   *
   * @param prompt_text Prompt text for speaker.
   * @param prompt_wav_path Path to prompt audio.
   * @param spk_id Speaker ID for caching.
   *
   * @return True if successful.
   *
   * Reference: demo.py lines 1941-1949
   */
  bool AddZeroShotSpeaker(const std::string& prompt_text,
                          const std::string& prompt_wav_path,
                          const std::string& spk_id);

  /**
   * @brief Check if speaker is cached.
   * @param spk_id Speaker ID.
   * @return True if speaker features are cached.
   */
  bool HasSpeaker(const std::string& spk_id) const;

  /**
   * @brief Get cached speaker features.
   * @param spk_id Speaker ID.
   * @return Cached frontend input (nullptr if not found).
   */
  const CosyVoice3FrontendInput* GetCachedSpeaker(
      const std::string& spk_id) const;

 private:
  // ========================================================================
  // Module Components
  // ========================================================================

  std::shared_ptr<HmTextFrontend> text_frontend_;
  std::shared_ptr<HmAudio> audio_processor_;
  std::shared_ptr<HmSpeechTokenizer> speech_tokenizer_;
  std::shared_ptr<HmSpeakerEmbedding> speaker_embedding_;

  std::shared_ptr<HmNativeLLM> llm_;
  std::shared_ptr<HmNativeFlow> flow_;
  std::shared_ptr<HmNativeVocoder> vocoder_;

  // ========================================================================
  // Configuration
  // ========================================================================

  CosyVoice3Config config_;  ///< Configuration with model paths
  int sample_rate_;          ///< Output sample rate (24000)
  std::string output_dir_;   ///< Output directory for generated audio

  // ========================================================================
  // Performance Tracking
  // ========================================================================

  std::vector<CosyVoice3Perf> perf_history_;  ///< Performance metrics history
  HmPerfMonitor perf_monitor_;                ///< Performance monitoring class

  // ========================================================================
  // Speaker Cache
  // ========================================================================

  std::map<std::string, CosyVoice3FrontendInput>
      speaker_cache_;  ///< Cached speaker features

  // ========================================================================
  // Frontend Methods (Internal)
  // ========================================================================

  /**
   * @brief Build frontend inputs for zero-shot synthesis.
   *
   * Extracts all features from prompt audio:
   * - Prompt text tokens
   * - Prompt speech tokens (from audio)
   * - Prompt mel features (from audio)
   * - Speaker embedding (from audio)
   *
   * @param tts_text Target text.
   * @param prompt_text Prompt text (with <|endofprompt|>).
   * @param prompt_wav_path Path to prompt audio.
   * @param use_cached_spk_id Cached speaker ID (empty = extract fresh).
   *
   * @return Frontend input structure.
   *
   * Reference: demo.py lines 1444-1495
   */
  CosyVoice3FrontendInput FrontendZeroShot(
      const std::string& tts_text, const std::string& prompt_text,
      const std::string& prompt_wav_path,
      const std::string& use_cached_spk_id = "",
      CosyVoice3Perf* perf = nullptr);

  /**
   * @brief Build frontend inputs for cross-lingual synthesis.
   */
  CosyVoice3FrontendInput FrontendCrossLingual(
      const std::string& tts_text, const std::string& prompt_wav_path,
      const std::string& use_cached_spk_id = "",
      CosyVoice3Perf* perf = nullptr);

  /**
   * @brief Build frontend inputs for instruction-following zero-shot synthesis.
   */
  CosyVoice3FrontendInput FrontendInstruct2(
      const std::string& tts_text, const std::string& instruct_text,
      const std::string& prompt_wav_path,
      const std::string& use_cached_spk_id = "",
      CosyVoice3Perf* perf = nullptr);

  // ========================================================================
  // Feature Extraction Helpers
  // ========================================================================

  /**
   * @brief Load prompt audio and extract all features.
   *
   * @param prompt_wav_path Path to prompt audio file.
   * @param[out] pcm_16k PCM at 16kHz (for speech tokenizer and speaker
   * embedding).
   * @param[out] pcm_24k PCM at 24kHz (for mel features).
   * @param[out] prompt_mel Mel features at 24kHz.
   * @param[out] prompt_speech_tokens Speech tokens.
   * @param[out] speaker_emb Speaker embedding.
   *
   * @return True if successful.
   */
  bool ExtractPromptFeatures(const std::string& prompt_wav_path,
                             std::vector<float>& pcm_16k,
                             std::vector<float>& pcm_24k,
                             std::vector<TensorType>& speech_feat,
                             std::vector<int>& prompt_speech_tokens,
                             std::vector<TensorType>& speech_embedding,
                             CosyVoice3Perf* perf = nullptr);

  /**
   * @brief Run TTS synthesis for a single utterance.
   *
   * @param frontend_input Frontend input structure.
   * @param speed Playback speed ratio.
   * @param perf Performance metrics pointer.
   *
   * @return Generated audio waveform for this utterance.
   */
  std::vector<float> SynthesizeUtterance(
      const CosyVoice3FrontendInput& frontend_input, float speed,
      CosyVoice3Perf* perf);

  // ========================================================================
  // WAV Helper
  // ========================================================================

  /**
   * @brief Create WAV file header.
   *
   * @param num_samples Number of audio samples.
   * @param sample_rate Sample rate.
   * @param bits_per_sample Bits per sample.
   *
   * @return WAV header structure.
   */
  WavHeader CreateWavHeader(size_t num_samples, int sample_rate,
                            int bits_per_sample);

  // ========================================================================
  // Constants
  // ========================================================================

  static constexpr int kSampleRate16k =
      16000;  ///< Sample rate for tokenizer/campplus
  static constexpr int kSampleRate24k = 24000;  ///< Sample rate for vocoder
  static constexpr int kMelBins = 80;           ///< Mel frequency bins
  static constexpr int kTokenMelRatio = 2;      ///< Mel frames per speech token
};

// ============================================================================
// Command Line Argument Parsing
// ============================================================================

/**
 * @brief Parse command line arguments to CosyVoice3Config.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 *
 * @return Configuration structure.
 *
 * Reference: demo.py lines 134-266 (get_args function)
 *
 * Supported arguments:
 * --tokenizer_dir: Tokenizer directory path
 * --model_dir: Model directory path (for all .hmm files)
 * --embedding_dir: Embedding directory path (for all .hemm files)
 * --output_dir: Output directory for generated audio
 * --prompt_wav_path: Prompt audio file path
 * --ndevice: Number of devices (default 1)
 */
CosyVoice3Config ParseArgs(int argc, char* argv[]);

/**
 * @brief Print usage information.
 */
void PrintUsage(const char* program_name);

}  // namespace houmo

#endif  // HM_COSYVOICE3_H_