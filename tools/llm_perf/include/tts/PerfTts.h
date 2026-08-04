/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: PerfTts.h
 * Description:
 *   Qwen3-TTS fixed-frame performance settings and inference interface.
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

#ifndef PERF_TTS_H
#define PERF_TTS_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct TtsPerfSettings {
  std::string text_projection_path;
  std::string talker_prefill_path;
  std::string talker_decode_path;
  std::string code_predictor_prefill_path;
  std::string code_predictor_decode_path;
  std::string stateful_decoder_path;
  std::string embedding_path;
  std::string code_embedding_path;
  std::string text_embedding_path;

  double requested_audio_length_s = 0.0;
  int token_per_second = 3;
  size_t body_text_tokens = 0;
  size_t text_projection_tokens = 0;
  size_t target_codec_frames = 0;
  double nominal_audio_length_s = 0.0;
  size_t expected_audio_samples = 0;
  size_t decoder_chunks = 0;

  int device_id = 0;
  int loop = 1;
  bool warm_up = true;
  int interval_ms = 500;
  std::string model_name = "qwen3-tts";
  std::string output_wav;
  std::string dump_file;

  std::string language = "Chinese";
  std::string speaker = "vivian";
  uint32_t seed = 42;
};

struct TtsStagePerf {
  double text_embedding_ms = 0.0;
  size_t text_embedding_count = 0;
  double text_projection_ms = 0.0;
  size_t text_projection_count = 0;
  double prompt_prepare_ms = 0.0;
  size_t prompt_prepare_count = 0;

  double talker_prefill_ms = 0.0;
  size_t talker_prefill_count = 0;
  double talker_decode_ms = 0.0;
  size_t talker_decode_count = 0;
  double talker_sampling_ms = 0.0;
  size_t talker_sampling_count = 0;
  double codec_frame_prepare_ms = 0.0;

  double code_predictor_prepare_ms = 0.0;
  double code_predictor_prefill_ms = 0.0;
  size_t code_predictor_prefill_count = 0;
  double code_predictor_decode_ms = 0.0;
  size_t code_predictor_decode_count = 0;
  double code_predictor_sampling_ms = 0.0;
  size_t code_predictor_sampling_count = 0;

  double stateful_decoder_ms = 0.0;
  size_t stateful_decoder_count = 0;
  double other_ms = 0.0;
};

struct TtsPerfResult {
  double e2e_ms = 0.0;
  double ttfa_ms = 0.0;
  double rtf = 0.0;
  double codec_generation_ms = 0.0;
  double codec_frames_per_second = 0.0;
  size_t generated_frames = 0;
  size_t audio_samples = 0;
  double audio_duration_s = 0.0;
  size_t decoder_chunks = 0;
  TtsStagePerf stages;
  std::vector<float> waveform;
};

using TtsProgressCallback =
    std::function<void(size_t completed_frames, size_t total_frames)>;

class PerfTts {
 public:
  static constexpr uint32_t kSeed = 42;
  static constexpr int kSampleRate = 24000;
  static constexpr size_t kSamplesPerFrame = 1920;
  static constexpr double kSecondsPerFrame = 0.08;
  static constexpr size_t kDecoderChunkSize = 12;

  explicit PerfTts(const TtsPerfSettings& settings);
  ~PerfTts();

  PerfTts(const PerfTts&) = delete;
  PerfTts& operator=(const PerfTts&) = delete;
  PerfTts(PerfTts&&) noexcept;
  PerfTts& operator=(PerfTts&&) noexcept;

  void ValidateWorkload() const;
  TtsPerfResult Run(bool keep_waveform,
                    const TtsProgressCallback& progress_callback = {});
  const TtsPerfSettings& settings() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

#endif  // PERF_TTS_H
