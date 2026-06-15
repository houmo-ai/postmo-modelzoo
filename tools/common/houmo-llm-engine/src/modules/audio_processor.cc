/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: audio_processor.cc
 * Description:
 *   Audio processor implementation for ASR models
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

// miniaudio implementation entry point (must be in exactly one translation
// unit)
#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MINIAUDIO_IMPLEMENTATION
#include "modules/audio_processor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "miniaudio.h"
#include "samplerate.h"

namespace houmo {

// ============================================================================
// AudioProcessor
// ============================================================================

AudioProcessor::AudioProcessor() : config_(AudioProcessorConfig{}) {}

AudioProcessor::AudioProcessor(const AudioProcessorConfig& config)
    : config_(config) {}

AudioProcessor::~AudioProcessor() = default;

// ========== Audio loading ==========

AudioData AudioProcessor::LoadAudio(const std::string& path) {
  AudioData audio;

  // 1. Open audio file using miniaudio
  ma_decoder_config decoder_config =
      ma_decoder_config_init(ma_format_f32, 0, 0);
  ma_decoder decoder;
  ma_result result =
      ma_decoder_init_file(path.c_str(), &decoder_config, &decoder);
  if (result != MA_SUCCESS) {
    std::cerr << "Error: Failed to load audio: " << path
              << ", miniaudio error=" << result << "\n";
    return audio;
  }

  // 2. Get audio metadata
  ma_format format;
  ma_uint32 channels = 0;
  ma_uint32 sample_rate = 0;
  result = ma_decoder_get_data_format(&decoder, &format, &channels,
                                      &sample_rate, nullptr, 0);
  if (result != MA_SUCCESS || format != ma_format_f32 || channels == 0 ||
      sample_rate == 0) {
    ma_decoder_uninit(&decoder);
    std::cerr << "Error: Invalid audio metadata: " << path << "\n";
    return audio;
  }

  // 3. Read PCM data
  std::vector<float> interleaved_pcm;
  ma_uint64 total_frames = 0;
  result = ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames);

  if (result == MA_SUCCESS && total_frames > 0) {
    interleaved_pcm.resize(static_cast<size_t>(total_frames) * channels);
    ma_uint64 frames_read = 0;
    result = ma_decoder_read_pcm_frames(&decoder, interleaved_pcm.data(),
                                        total_frames, &frames_read);
    if (result != MA_SUCCESS && result != MA_AT_END) {
      ma_decoder_uninit(&decoder);
      std::cerr << "Error: Failed to read audio samples: " << path << "\n";
      return audio;
    }
    interleaved_pcm.resize(static_cast<size_t>(frames_read) * channels);
  } else {
    // Stream reading (unknown length)
    constexpr ma_uint64 kFramesPerRead = 4096;
    std::vector<float> chunk(static_cast<size_t>(kFramesPerRead) * channels);
    while (true) {
      ma_uint64 frames_read = 0;
      result = ma_decoder_read_pcm_frames(&decoder, chunk.data(),
                                          kFramesPerRead, &frames_read);
      if (result != MA_SUCCESS && result != MA_AT_END) {
        ma_decoder_uninit(&decoder);
        std::cerr << "Error: Failed to stream audio samples: " << path << "\n";
        return audio;
      }
      if (frames_read == 0) break;
      interleaved_pcm.insert(
          interleaved_pcm.end(), chunk.begin(),
          chunk.begin() + static_cast<std::ptrdiff_t>(frames_read * channels));
      if (result == MA_AT_END) break;
    }
  }

  ma_decoder_uninit(&decoder);

  if (interleaved_pcm.empty()) {
    std::cerr << "Error: Failed to read audio samples: " << path << "\n";
    return audio;
  }

  // 4. Convert to mono
  std::vector<float> mono_pcm;
  DownmixToMono(interleaved_pcm, static_cast<int>(channels), &mono_pcm);

  // 5. Resample to target sample rate
  if (sample_rate == static_cast<ma_uint32>(config_.sample_rate)) {
    audio.pcm = std::move(mono_pcm);
  } else {
    if (!ResampleAudio(mono_pcm, static_cast<uint32_t>(sample_rate),
                       &audio.pcm)) {
      std::cerr << "Error: Failed to resample audio from " << sample_rate
                << " Hz to " << config_.sample_rate << " Hz: " << path << "\n";
      return audio;
    }
  }

  audio.sample_rate = config_.sample_rate;
  audio.duration = static_cast<float>(audio.pcm.size()) / config_.sample_rate;

  std::cout << "Audio loaded: " << path << ", channels: " << channels
            << ", sample rate: " << sample_rate << "Hz"
            << ", duration: " << audio.duration << "s"
            << ", samples: " << audio.pcm.size() << "\n";

  return audio;
}

// ========== Feature extraction ==========

MelFeatures AudioProcessor::ExtractFeatures(const AudioData& audio) {
  return ComputeMelSpectrogram(audio);
}

// ========== Chunk processing ==========

std::vector<AudioData> AudioProcessor::ChunkPCM(const AudioData& audio) {
  std::vector<AudioData> chunks;

  if (audio.pcm.empty()) {
    return chunks;
  }

  int chunk_samples = config_.chunk_seconds * audio.sample_rate;
  int total_samples = static_cast<int>(audio.pcm.size());
  int num_chunks = (total_samples + chunk_samples - 1) / chunk_samples;

  for (int i = 0; i < num_chunks; ++i) {
    AudioData chunk;
    chunk.sample_rate = audio.sample_rate;

    int start = i * chunk_samples;
    int end = std::min(start + chunk_samples, total_samples);
    int chunk_len = end - start;

    chunk.pcm.assign(audio.pcm.begin() + start, audio.pcm.begin() + end);
    chunk.duration = static_cast<float>(chunk_len) / audio.sample_rate;

    // Note: No padding here - ComputeMelSpectrogram will pad to
    // encoder_window_seconds This allows chunk_seconds and
    // encoder_window_seconds to be different

    chunks.push_back(std::move(chunk));
  }

  return chunks;
}

// ========== One-stop processing ==========

std::vector<MelFeatures> AudioProcessor::Process(const std::string& path) {
  std::vector<MelFeatures> results;

  AudioData audio = LoadAudio(path);
  if (audio.pcm.empty()) {
    return results;
  }

  std::vector<AudioData> chunks = ChunkPCM(audio);
  results.reserve(chunks.size());

  for (const auto& chunk : chunks) {
    results.push_back(ExtractFeatures(chunk));
  }

  return results;
}

// ========== Information retrieval ==========

int AudioProcessor::feature_dim() const { return config_.n_mels; }

// ========== Helper methods ==========

void AudioProcessor::DownmixToMono(const std::vector<float>& interleaved,
                                   int channels, std::vector<float>* mono) {
  if (mono == nullptr || channels <= 0) return;

  if (channels == 1) {
    *mono = interleaved;
    return;
  }

  const size_t num_frames = interleaved.size() / channels;
  mono->assign(num_frames, 0.0f);

  const float scale = 1.0f / static_cast<float>(channels);
  for (size_t frame = 0; frame < num_frames; ++frame) {
    float mixed_sample = 0.0f;
    const size_t base_index = frame * channels;
    for (int channel = 0; channel < channels; ++channel) {
      mixed_sample += interleaved[base_index + channel];
    }
    (*mono)[frame] = mixed_sample * scale;
  }
}

bool AudioProcessor::ResampleAudio(const std::vector<float>& input,
                                   uint32_t input_sr,
                                   std::vector<float>* output) {
  if (output == nullptr || input.empty() || input_sr == 0) return false;

  if (input_sr == static_cast<uint32_t>(config_.sample_rate)) {
    *output = input;
    return true;
  }

  const double ratio = static_cast<double>(config_.sample_rate) / input_sr;
  const size_t output_capacity =
      static_cast<size_t>(std::ceil(input.size() * ratio)) + 1;

  output->assign(output_capacity, 0.0f);

  SRC_DATA src_data;
  std::memset(&src_data, 0, sizeof(src_data));
  src_data.data_in = input.data();
  src_data.input_frames = static_cast<long>(input.size());
  src_data.data_out = output->data();
  src_data.output_frames = static_cast<long>(output->size());
  src_data.src_ratio = ratio;
  src_data.end_of_input = 1;

  const int result = src_simple(&src_data, SRC_SINC_FASTEST, 1);
  if (result != 0) {
    std::cerr << "Error: libsamplerate resampling failed: "
              << src_strerror(result) << "\n";
    output->clear();
    return false;
  }

  output->resize(static_cast<size_t>(src_data.output_frames_gen));
  return true;
}

// ========== Mel Spectrogram (WhisperFeatureExtractor style) ==========
// Compatible with models using WhisperFeatureExtractor: Whisper, GLM-ASR, Qwen3-ASR

MelFeatures AudioProcessor::ComputeMelSpectrogram(const AudioData& audio) {
  MelFeatures features;
  features.feature_dim = config_.n_mels;
  features.duration = audio.duration;  // Save actual audio duration

  if (audio.pcm.empty()) {
    return features;
  }

  // 1. Configure window size based on encoder_window_seconds (padding/truncate)
  // This is the window size required by encode input, independent of chunk_seconds segmentation logic
  std::vector<float> processed_pcm = audio.pcm;
  size_t window_samples =
      static_cast<size_t>(config_.sample_rate) * config_.encoder_window_seconds;

  if (processed_pcm.size() < window_samples) {
    processed_pcm.resize(window_samples, 0.0f);  // Padding with zeros
  } else if (processed_pcm.size() > window_samples) {
    processed_pcm.resize(window_samples);  // Truncate
  }

  // 2. Configure feature extractor (WhisperFeatureExtractor style)
  knf::WhisperFeatureOptions whisper_opts;
  whisper_opts.dim = config_.n_mels;

  knf::OnlineWhisperFbank whisper_fbank(whisper_opts);
  whisper_fbank.AcceptWaveform(static_cast<float>(config_.sample_rate),
                               processed_pcm.data(), processed_pcm.size());
  whisper_fbank.InputFinished();

  // 3. Extract features
  features.num_frames = whisper_fbank.NumFramesReady();
  if (features.num_frames <= 0) {
    return features;
  }

  std::vector<float> tmp_data(features.feature_dim * features.num_frames, 0.0f);
  float max_log_spec = -1e20f;

  for (int t = 0; t < features.num_frames; ++t) {
    const float* frame = whisper_fbank.GetFrame(t);
    for (int m = 0; m < features.feature_dim; ++m) {
      float log_spec = std::log10(std::max(frame[m], 1e-10f));
      tmp_data[m * features.num_frames + t] = log_spec;
      if (log_spec > max_log_spec) {
        max_log_spec = log_spec;
      }
    }
  }

  // 4. Normalize and convert to FP16
  // Whisper normalization: clamp to [max_log_spec - 8, max_log_spec], then (val + 4)
  // / 4
  features.data.resize(tmp_data.size());
  for (size_t i = 0; i < tmp_data.size(); ++i) {
    float val = tmp_data[i];
    // Clamp to [max_log_spec - 8, max_log_spec]
    val = std::max(val, max_log_spec - 8.0f);
    // Normalize: (val + 4) / 4
    val = (val + 4.0f) / 4.0f;
    features.data[i] = static_cast<float16>(val);
  }

  return features;
}

}  // namespace houmo
