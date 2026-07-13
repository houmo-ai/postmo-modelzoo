/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: audio_processor.cc
 * Description:
 *   Audio processor implementation for ASR models
 *
 * Portions of the mel spectrogram implementation are adapted from llama.cpp:
 *   https://github.com/ggml-org/llama.cpp
 *
 * The adapted llama.cpp audio preprocessing code notes that some portions are
 * copied from whisper.cpp:
 *   https://github.com/ggml-org/whisper.cpp
 *
 * llama.cpp and whisper.cpp are licensed under the MIT License:
 *   Copyright (c) 2023-2026 The ggml authors
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
 * SPDX-License-Identifier: Apache-2.0 AND MIT
 */

// miniaudio implementation entry point (must be in exactly one translation
// unit)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MINIAUDIO_IMPLEMENTATION
#include "modules/audio_processor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "audio/miniaudio.h"

namespace houmo {

namespace {

struct LlamaMelFilters {
  int64_t n_mel = 0;
  int64_t n_fft = 0;
  std::vector<float> data;
};

struct LlamaMelCache {
  std::vector<float> sin_vals;
  std::vector<float> cos_vals;
  std::vector<float> hann_window;
  LlamaMelFilters filters;
};

struct LlamaMelResult {
  int64_t n_len = 0;
  int64_t n_len_org = 0;
  int64_t n_mel = 0;
  std::vector<float> data;
};

struct FilterParams {
  int64_t n_mel = 128;
  int64_t n_fft_bins = 201;
  int32_t hann_window_size = 400;
  int32_t hop_length = 160;
  int32_t sample_rate = 16000;
  bool center_padding = true;
  bool use_natural_log = false;
  float mel_floor = 5.960464477539063e-08f;
};

void fill_sin_cos_table(LlamaMelCache& cache, uint32_t n) {
  cache.sin_vals.resize(n);
  cache.cos_vals.resize(n);
  for (uint32_t i = 0; i < n; i++) {
    double theta = (2.0 * M_PI * i) / n;
    cache.sin_vals[i] = std::sin(theta);
    cache.cos_vals[i] = std::cos(theta);
  }
}

void fill_hann_window(LlamaMelCache& cache, uint32_t length) {
  cache.hann_window.resize(length);
  for (uint32_t i = 0; i < length; i++) {
    cache.hann_window[i] = 0.5f * (1.0f - std::cos((2.0 * M_PI * i) / length));
  }
}

void fill_mel_filterbank_matrix(LlamaMelCache& cache, int64_t n_mel,
                                int64_t n_fft, int sample_rate) {
  auto hz_to_mel = [](double f_hz) -> double {
    const double min_log_hz = 1000.0;
    const double lin_slope = 3.0 / 200.0;
    const double min_log_mel = min_log_hz * lin_slope;
    const double log_step = std::log(6.4) / 27.0;
    return f_hz < min_log_hz
               ? f_hz * lin_slope
               : min_log_mel + std::log(f_hz / min_log_hz) / log_step;
  };
  auto mel_to_hz = [](double m) -> double {
    const double min_log_hz = 1000.0;
    const double lin_slope = 3.0 / 200.0;
    const double min_log_mel = min_log_hz * lin_slope;
    const double log_step = std::log(6.4) / 27.0;
    return m < min_log_mel
               ? m / lin_slope
               : min_log_hz * std::exp((m - min_log_mel) * log_step);
  };

  const double fmax = 0.5 * sample_rate;
  const double m_lo = hz_to_mel(0.0);
  const double m_hi = hz_to_mel(fmax);

  std::vector<double> hz_pts(n_mel + 2);
  for (int i = 0; i < n_mel + 2; ++i) {
    double mel = m_lo + (m_hi - m_lo) * (double(i) / (n_mel + 1));
    hz_pts[i] = mel_to_hz(mel);
  }

  const int64_t n_fft_bins = n_fft / 2 + 1;
  std::vector<float> out((size_t)n_mel * (size_t)n_fft_bins, 0.0f);
  const double bin_hz_step = double(sample_rate) / double(n_fft);

  for (int64_t m = 0; m < n_mel; ++m) {
    const double f_left = hz_pts[m];
    const double f_center = hz_pts[m + 1];
    const double f_right = hz_pts[m + 2];
    const double denom_l = std::max(1e-30, f_center - f_left);
    const double denom_r = std::max(1e-30, f_right - f_center);
    const double enorm = 2.0 / std::max(1e-30, f_right - f_left);

    for (int k = 0; k < n_fft_bins; ++k) {
      const double f = k * bin_hz_step;
      double w = 0.0;
      if (f >= f_left && f <= f_center) {
        w = (f - f_left) / denom_l;
      } else if (f > f_center && f <= f_right) {
        w = (f_right - f) / denom_r;
      }
      out[(size_t)m * (size_t)n_fft_bins + (size_t)k] = float(w * enorm);
    }
  }

  cache.filters.n_mel = n_mel;
  cache.filters.n_fft = n_fft;
  cache.filters.data = std::move(out);
}

template <bool Inverse, bool RealInput>
void dft_impl(const LlamaMelCache& cache, const float* in, int n, float* out) {
  const int n_sin_cos_vals = cache.sin_vals.size();
  const int sin_cos_step = n_sin_cos_vals / n;
  constexpr float sign = Inverse ? 1.0f : -1.0f;
  const float scale = Inverse ? (1.0f / n) : 1.0f;

  for (int k = 0; k < n; k++) {
    float re = 0.0f;
    float im = 0.0f;
    for (int i = 0; i < n; i++) {
      int idx = (k * i * sin_cos_step) % n_sin_cos_vals;
      float cos_val = cache.cos_vals[idx];
      float sin_val = cache.sin_vals[idx];
      if constexpr (RealInput) {
        float in_re = in[i];
        re += in_re * cos_val;
        im += sign * in_re * sin_val;
      } else {
        float in_re = in[i * 2 + 0];
        float in_im = in[i * 2 + 1];
        re += in_re * cos_val - sign * in_im * sin_val;
        im += sign * in_re * sin_val + in_im * cos_val;
      }
    }
    out[k * 2 + 0] = re * scale;
    out[k * 2 + 1] = im * scale;
  }
}

template <bool Inverse, bool RealInput>
void fft_impl(const LlamaMelCache& cache, float* in, int n, float* out) {
  if (n == 1) {
    out[0] = in[0];
    out[1] = RealInput ? 0.0f : in[1];
    return;
  }

  const int half_n = n / 2;
  if (n - half_n * 2 == 1) {
    dft_impl<Inverse, RealInput>(cache, in, n, out);
    return;
  }

  if constexpr (RealInput) {
    float* even = in + n;
    for (int i = 0; i < half_n; ++i) even[i] = in[2 * i];
    float* even_fft = out + 2 * n;
    fft_impl<Inverse, true>(cache, even, half_n, even_fft);

    float* odd = even;
    for (int i = 0; i < half_n; ++i) odd[i] = in[2 * i + 1];
    float* odd_fft = even_fft + n;
    fft_impl<Inverse, true>(cache, odd, half_n, odd_fft);
  } else {
    float* even = in + n * 2;
    for (int i = 0; i < half_n; ++i) {
      even[i * 2 + 0] = in[2 * i * 2 + 0];
      even[i * 2 + 1] = in[2 * i * 2 + 1];
    }
    float* even_fft = out + 2 * n;
    fft_impl<Inverse, false>(cache, even, half_n, even_fft);

    float* odd = even;
    for (int i = 0; i < half_n; ++i) {
      odd[i * 2 + 0] = in[(2 * i + 1) * 2 + 0];
      odd[i * 2 + 1] = in[(2 * i + 1) * 2 + 1];
    }
    float* odd_fft = even_fft + n;
    fft_impl<Inverse, false>(cache, odd, half_n, odd_fft);
  }

  float* even_fft = out + 2 * n;
  float* odd_fft = even_fft + n;
  const int sin_cos_step = cache.sin_vals.size() / n;
  constexpr float sign = Inverse ? 1.0f : -1.0f;
  constexpr float scale = Inverse ? 0.5f : 1.0f;

  for (int k = 0; k < half_n; k++) {
    int idx = k * sin_cos_step;
    float re = cache.cos_vals[idx];
    float im = sign * cache.sin_vals[idx];
    float re_odd = odd_fft[2 * k + 0];
    float im_odd = odd_fft[2 * k + 1];

    out[2 * k + 0] = scale * (even_fft[2 * k + 0] + re * re_odd - im * im_odd);
    out[2 * k + 1] = scale * (even_fft[2 * k + 1] + re * im_odd + im * re_odd);
    out[2 * (k + half_n) + 0] =
        scale * (even_fft[2 * k + 0] - re * re_odd + im * im_odd);
    out[2 * (k + half_n) + 1] =
        scale * (even_fft[2 * k + 1] - re * im_odd - im * re_odd);
  }
}

void fft(const LlamaMelCache& cache, float* in, int n, float* out) {
  fft_impl<false, true>(cache, in, n, out);
}

void mel_worker(int ith, const float* hann, const std::vector<float>& samples,
                int n_samples, int frame_size, int frame_step, int n_threads,
                const FilterParams& params, const LlamaMelCache& cache,
                LlamaMelResult& out) {
  std::vector<float> fft_in(frame_size * 2, 0.0f);
  std::vector<float> fft_out(frame_size * 2 * 2 * 2, 0.0f);
  int64_t n_fft_bins = params.n_fft_bins;
  const auto& filters = cache.filters;

  for (int64_t i = ith;
       i < std::min((int64_t)(n_samples / frame_step + 1), out.n_len);
       i += n_threads) {
    const int64_t offset = i * frame_step;
    const int valid_len =
        std::min(frame_size, std::max(0, n_samples - (int)offset));
    for (int j = 0; j < valid_len; j++) {
      fft_in[j] = hann[j] * samples[offset + j];
    }
    if (valid_len < frame_size) {
      std::fill(fft_in.begin() + valid_len, fft_in.end(), 0.0f);
    }

    fft(cache, fft_in.data(), frame_size, fft_out.data());

    for (int j = 0; j < n_fft_bins; j++) {
      float re = fft_out[2 * j + 0];
      float im = fft_out[2 * j + 1];
      fft_out[j] = re * re + im * im;
    }

    for (int64_t j = 0; j < out.n_mel; j++) {
      double sum = 0.0;
      int k = 0;
      for (; k < n_fft_bins - 3; k += 4) {
        size_t idx = (size_t)j * (size_t)n_fft_bins + (size_t)k;
        sum += fft_out[k + 0] * filters.data[idx + 0] +
               fft_out[k + 1] * filters.data[idx + 1] +
               fft_out[k + 2] * filters.data[idx + 2] +
               fft_out[k + 3] * filters.data[idx + 3];
      }
      for (; k < n_fft_bins; k++) {
        sum += fft_out[k] *
               filters.data[(size_t)j * (size_t)n_fft_bins + (size_t)k];
      }
      sum = std::max(sum, (double)params.mel_floor);
      out.data[(size_t)j * out.n_len + (size_t)i] =
          params.use_natural_log ? std::log(sum) : std::log10(sum);
    }
  }

  double sum = params.use_natural_log ? std::log(1e-10) : std::log10(1e-10);
  for (int64_t i = ith; i < out.n_len; i += n_threads) {
    if (i < std::min((int64_t)(n_samples / frame_step + 1), out.n_len))
      continue;
    for (int64_t j = 0; j < out.n_mel; j++) {
      out.data[(size_t)j * out.n_len + (size_t)i] = sum;
    }
  }
}

bool log_mel_spectrogram(const float* samples, int n_samples_in, int n_threads,
                         const FilterParams& params, const LlamaMelCache& cache,
                         LlamaMelResult& out) {
  out.n_len_org = n_samples_in;
  const int frame_size = (params.n_fft_bins - 1) * 2;
  const int frame_step = params.hop_length;
  const int pad = frame_size / 2;

  std::vector<float> padded;
  if (params.center_padding) {
    padded.assign(n_samples_in + 2 * pad, 0.0f);
    for (int i = 0; i < pad; i++) {
      int src = pad - i;
      padded[i] = src < n_samples_in ? samples[src] : 0.0f;
    }
    std::copy(samples, samples + n_samples_in, padded.begin() + pad);
    for (int i = 0; i < pad; i++) {
      int src = n_samples_in - 2 - i;
      padded[n_samples_in + pad + i] = src >= 0 ? samples[src] : 0.0f;
    }
  } else {
    const int64_t stage_1_pad = params.sample_rate * 30;
    padded.assign(n_samples_in + stage_1_pad + pad * 2, 0.0f);
    std::copy(samples, samples + n_samples_in, padded.begin() + pad);
    if (n_samples_in < pad + 1) return false;
    std::reverse_copy(samples + 1, samples + 1 + pad, padded.begin());
  }

  out.n_mel = params.n_mel;
  out.n_len = ((int)padded.size() - frame_size) / frame_step + 1;
  if (out.n_mel <= 0 || out.n_len <= 0) return false;
  out.data.assign((size_t)out.n_mel * (size_t)out.n_len, 0.0f);

  std::vector<std::thread> workers(std::max(0, n_threads - 1));
  for (int i = 0; i < n_threads - 1; ++i) {
    workers[i] = std::thread(mel_worker, i + 1, cache.hann_window.data(),
                             std::cref(padded), (int)padded.size(), frame_size,
                             frame_step, n_threads, std::cref(params),
                             std::cref(cache), std::ref(out));
  }
  mel_worker(0, cache.hann_window.data(), padded, (int)padded.size(),
             frame_size, frame_step, n_threads, params, cache, out);
  for (auto& worker : workers) worker.join();

  double mmax = -1e20;
  for (float value : out.data) {
    if (value > mmax) mmax = value;
  }
  mmax -= 8.0;
  for (float& value : out.data) {
    value = (std::max((double)value, mmax) + 4.0) / 4.0;
  }
  return true;
}

LlamaMelCache make_cache(const AudioProcessorConfig& config) {
  LlamaMelCache cache;
  fill_sin_cos_table(cache, config.fft_size);
  fill_hann_window(cache, config.win_length);
  fill_mel_filterbank_matrix(cache, config.n_mels, config.fft_size,
                             config.sample_rate);
  return cache;
}

LlamaMelResult ComputeLlamaMelSpectrogram(const std::vector<float>& pcm,
                                          const AudioProcessorConfig& config) {
  LlamaMelCache cache = make_cache(config);
  FilterParams params;
  params.n_mel = config.n_mels;
  params.n_fft_bins = 1 + (config.fft_size / 2);
  params.hann_window_size = config.win_length;
  params.hop_length = config.hop_length;
  params.center_padding = config.feature_mode == AudioFeatureMode::kCenterPad;

  LlamaMelResult mel;
  if (!log_mel_spectrogram(pcm.data(), static_cast<int>(pcm.size()),
                           config.feature_threads, params, cache, mel)) {
    return {};
  }
  return mel;
}

}  // namespace

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

  // 1. Open audio file using miniaudio, decoding directly to target PCM layout
  ma_decoder_config decoder_config = ma_decoder_config_init(
      ma_format_f32, 1, static_cast<ma_uint32>(config_.sample_rate));
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

  // 4. Store decoded mono PCM at target sample rate
  audio.pcm = std::move(interleaved_pcm);

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

// ========== Mel Spectrogram (WhisperFeatureExtractor style) ==========
// Compatible with models using WhisperFeatureExtractor: Whisper, GLM-ASR,
// Qwen3-ASR

MelFeatures AudioProcessor::ComputeMelSpectrogram(const AudioData& audio) {
  MelFeatures features;
  features.feature_dim = config_.n_mels;
  features.duration = audio.duration;

  if (audio.pcm.empty()) {
    return features;
  }

  std::vector<float> processed_pcm = audio.pcm;
  size_t window_samples =
      static_cast<size_t>(config_.sample_rate) * config_.encoder_window_seconds;

  if (processed_pcm.size() < window_samples) {
    processed_pcm.resize(window_samples, 0.0f);
  } else if (processed_pcm.size() > window_samples) {
    processed_pcm.resize(window_samples);
  }

  LlamaMelResult mel = ComputeLlamaMelSpectrogram(processed_pcm, config_);
  if (mel.data.empty() || mel.n_mel <= 0 || mel.n_len <= 0) {
    return features;
  }

  features.feature_dim = static_cast<int>(mel.n_mel);
  features.num_frames = static_cast<int>(mel.n_len);
  if (config_.feature_mode == AudioFeatureMode::kWhisper) {
    features.num_frames = std::min(
        features.num_frames, config_.encoder_window_seconds *
                                 config_.sample_rate / config_.hop_length);
  }

  features.data.resize(static_cast<size_t>(features.feature_dim) *
                       static_cast<size_t>(features.num_frames));
  for (int m = 0; m < features.feature_dim; ++m) {
    const size_t src_offset =
        static_cast<size_t>(m) * static_cast<size_t>(mel.n_len);
    const size_t dst_offset =
        static_cast<size_t>(m) * static_cast<size_t>(features.num_frames);
    for (int t = 0; t < features.num_frames; ++t) {
      features.data[dst_offset + static_cast<size_t>(t)] =
          static_cast<float16>(mel.data[src_offset + static_cast<size_t>(t)]);
    }
  }

  return features;
}

}  // namespace houmo
