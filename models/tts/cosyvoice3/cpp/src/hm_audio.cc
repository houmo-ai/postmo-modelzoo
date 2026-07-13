/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_audio.cc
 * Description:
 *   Implementation of audio preprocessing class for CosyVoice3 TTS.
 *   Handles audio loading, resampling, and mel spectrogram extraction.
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

#include "hm_audio.h"

#include <samplerate.h>
#include <sndfile.h>

#include <cstring>

#include "audio/librosa.h"

namespace houmo {

HmAudio::HmAudio() : sample_rate_(0), total_samples_(0), audio_loaded_(false) {}

HmAudio::~HmAudio() {}

bool HmAudio::LoadAudio(const std::string& audio_path, int target_sr) {
  SF_INFO sfinfo;
  sfinfo.format = 0;

  SNDFILE* sndfile = sf_open(audio_path.c_str(), SFM_READ, &sfinfo);
  if (!sndfile) {
    std::cerr << "ERROR: Failed to open audio file: " << audio_path << "\n";
    std::cerr << "libsndfile error: " << sf_strerror(nullptr) << "\n";
    return false;
  }

  if (sfinfo.channels > 1) {
    sf_close(sndfile);
    throw std::runtime_error(
        "Only one channel mono audio is supported. Please convert to mono "
        "before loading.");
  }

  std::vector<float> raw_pcm_data(sfinfo.frames);
  sf_count_t frames_read =
      sf_readf_float(sndfile, raw_pcm_data.data(), sfinfo.frames);
  sf_close(sndfile);

  if (frames_read <= 0) {
    std::cerr << "ERROR: Failed to read any audio frames from: " << audio_path
              << "\n";
    return false;
  }
  raw_pcm_data.resize(frames_read);

  if (target_sr != sfinfo.samplerate) {
    std::cout << "Resampling from " << sfinfo.samplerate << "Hz to "
              << target_sr << "Hz ...\n";

    double src_ratio = static_cast<double>(target_sr) / sfinfo.samplerate;
    long expected_output_frames =
        static_cast<long>(frames_read * src_ratio) + 256;
    pcm_data_.resize(expected_output_frames);

    SRC_DATA src_data;
    src_data.data_in = raw_pcm_data.data();
    src_data.input_frames = static_cast<long>(frames_read);
    src_data.data_out = pcm_data_.data();
    src_data.output_frames = expected_output_frames;
    src_data.src_ratio = src_ratio;
    src_data.end_of_input = 1;

    int error = src_simple(&src_data, SRC_SINC_BEST_QUALITY, 1);
    if (error != 0) {
      std::cerr << "ERROR: libsamplerate failed: " << src_strerror(error)
                << "\n";
      return false;
    }

    long frames_to_output = src_data.output_frames_gen;
    pcm_data_.resize(frames_to_output);
    std::cout << "Resampled " << src_data.input_frames_used << " frames to "
              << frames_to_output << " frames at " << target_sr << "Hz\n";
    total_samples_ = static_cast<int>(frames_to_output);
  } else {
    std::cout << "Audio sample rate matches target: " << target_sr << "Hz\n";
    pcm_data_ = std::move(raw_pcm_data);
    total_samples_ = static_cast<int>(frames_read);
  }

  sample_rate_ = target_sr;
  audio_loaded_ = true;
  std::cout << "Audio loaded: " << audio_path << ", duration: " << GetDuration()
            << "s"
            << ", samples: " << total_samples_
            << ", sample_rate: " << sample_rate_ << "Hz\n";

  return true;
}

float HmAudio::GetDuration() const {
  if (sample_rate_ <= 0) return 0.0f;
  return static_cast<float>(total_samples_) / sample_rate_;
}

MelFeatures80 HmAudio::ComputeMelSpectrogram80(
    const std::vector<float>& pcm_data) const {
  MelFeatures80 mel;
  mel.n_mels = kNMels80;

  int pad_len = (kNFFT24k - kHopSize24k) / 2;
  std::vector<float> librosa_input =
      pcm_data.empty() ? std::vector<float>(kNFFT24k, 0.0f)
                       : ApplyReflectionPadding(pcm_data, pad_len);
  auto mel_frames = librosa::Feature::melspectrogram(
      librosa_input, kSampleRate24k, kNFFT24k, kHopSize24k, "hann", false,
      "reflect", 1.0f, kNMels80, 0, kSampleRate24k / 2);

  mel.n_frames = std::max(1, static_cast<int>(mel_frames.size()));
  mel.data.resize(mel.n_mels * mel.n_frames);

  if (mel_frames.empty()) {
    std::fill(mel.data.begin(), mel.data.end(),
              static_cast<TensorType>(std::log(1e-5f)));
    return mel;
  }

  for (int t = 0; t < mel.n_frames; ++t) {
    const auto& frame = mel_frames[t];
    int mel_bins = std::min(mel.n_mels, static_cast<int>(frame.size()));
    for (int m = 0; m < mel_bins; ++m) {
      float log_spec = std::log(std::max(frame[m], 1e-5f));
      mel.data[t * mel.n_mels + m] = static_cast<TensorType>(log_spec);
    }
    for (int m = mel_bins; m < mel.n_mels; ++m) {
      mel.data[t * mel.n_mels + m] = static_cast<TensorType>(std::log(1e-5f));
    }
  }

  return mel;
}

MelFeatures128 HmAudio::ComputeMelSpectrogram128(
    const std::vector<float>& pcm_data) const {
  MelFeatures128 mel;
  mel.n_mels = kNMels128;

  std::vector<float> librosa_input =
      pcm_data.empty() ? std::vector<float>(kNFFT16k, 0.0f)
                       : std::vector<float>(pcm_data.begin(), pcm_data.end());
  auto mel_frames = librosa::Feature::melspectrogram(
      librosa_input, kSampleRate16k, kNFFT16k, kHopSize16k, "hann", true,
      "reflect", 2.0f, kNMels128, 0, kSampleRate16k / 2);

  mel.n_frames = std::max(1, static_cast<int>(mel_frames.size()));
  std::vector<float> tmp_mel_data(mel.n_mels * mel.n_frames, 0.0f);
  float max_log_spec = -1e20f;

  for (int t = 0; t < mel.n_frames; ++t) {
    const auto& frame = mel_frames[t];
    int mel_bins = std::min(mel.n_mels, static_cast<int>(frame.size()));
    for (int m = 0; m < mel_bins; ++m) {
      float log_spec = std::log10(std::max(frame[m], 1e-10f));
      tmp_mel_data[m * mel.n_frames + t] = log_spec;
      if (log_spec > max_log_spec) {
        max_log_spec = log_spec;
      }
    }
    for (int m = mel_bins; m < mel.n_mels; ++m) {
      float log_spec = std::log10(1e-10f);
      tmp_mel_data[m * mel.n_frames + t] = log_spec;
      if (log_spec > max_log_spec) {
        max_log_spec = log_spec;
      }
    }
  }

  // Whisper-style normalization
  mel.data.resize(mel.n_mels * mel.n_frames);
  for (int i = 0; i < mel.n_mels * mel.n_frames; i++) {
    float val = tmp_mel_data[i];
    val = std::max(val, max_log_spec - 8.0f);
    mel.data[i] = static_cast<TensorType>((val + 4.0f) / 4.0f);
  }

  return mel;
}

FbankFeatures HmAudio::ComputeFbank(const std::vector<float>& pcm_data) const {
  knf::FbankOptions opts;
  opts.frame_opts.samp_freq = kSampleRate16k;  // sample_frequency=16000
  opts.frame_opts.dither = 0.0f;               // dither=0
  opts.mel_opts.num_bins = kFbankNMels;
  // torchaudio defaults snip_edges=True, kaldi-native-fbank also defaults to
  // True
  opts.frame_opts.snip_edges = true;

  // 2. Initialize Fbank calculator
  knf::OnlineFbank fbank(opts);

  fbank.AcceptWaveform(kSampleRate16k, pcm_data.data(), pcm_data.size());
  fbank.InputFinished();

  // 4. Get feature matrix
  int num_frames = fbank.NumFramesReady();
  int feat_dim = opts.mel_opts.num_bins;  // 80

  FbankFeatures features;
  features.n_frames = num_frames;
  features.n_mels = feat_dim;
  features.data.reserve(num_frames * feat_dim);

  // Put features of all frames continuously into vector
  for (int i = 0; i < num_frames; ++i) {
    const float* frame = fbank.GetFrame(
        i);  // Returns float pointer of single frame (size is n_mels)
    features.data.insert(features.data.end(), frame, frame + feat_dim);
  }
  return features;
}

MelFeatures80 HmAudio::GetMelFeatures80() const {
  if (!audio_loaded_ || sample_rate_ != kSampleRate24k) {
    std::cerr
        << "Warning: Audio not loaded at 24kHz, returning empty features\n";
    return MelFeatures80();
  }
  return ComputeMelSpectrogram80(pcm_data_);
}

MelFeatures128 HmAudio::GetMelFeatures128() const {
  if (!audio_loaded_ || sample_rate_ != kSampleRate16k) {
    std::cerr
        << "Warning: Audio not loaded at 16kHz, returning empty features\n";
    return MelFeatures128();
  }
  return ComputeMelSpectrogram128(pcm_data_);
}

FbankFeatures HmAudio::GetFbankFeatures() const {
  if (!audio_loaded_ || sample_rate_ != kSampleRate16k) {
    std::cerr
        << "Warning: Audio not loaded at 16kHz, returning empty features\n";
    return FbankFeatures();
  }
  return ComputeFbank(pcm_data_);
}

std::vector<float> HmAudio::ComputeHannWindow(int win_size) const {
  std::vector<float> window(win_size);
  for (int i = 0; i < win_size; i++) {
    window[i] =
        0.5f *
        (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (win_size - 1)));
  }
  return window;
}

std::vector<float> HmAudio::ApplyReflectionPadding(
    const std::vector<float>& audio, int pad_len) const {
  if (audio.empty()) return audio;

  std::vector<float> padded(audio.size() + 2 * pad_len);

  // Reflection padding for start
  for (int i = 0; i < pad_len; i++) {
    int src_idx = std::min(i + 1, static_cast<int>(audio.size()) - 1);
    padded[pad_len - 1 - i] = audio[src_idx];
  }

  // Original audio
  for (size_t i = 0; i < audio.size(); i++) {
    padded[pad_len + i] = audio[i];
  }

  // Reflection padding for end
  int audio_size = static_cast<int>(audio.size());
  for (int i = 0; i < pad_len; i++) {
    int src_idx = std::max(audio_size - 2 - i, 0);
    padded[pad_len + audio_size + i] = audio[src_idx];
  }

  return padded;
}

float HmAudio::HzToMel(float hz) const {
  // HTK-style mel scale: mel = 2595 * log10(1 + hz/700)
  return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float HmAudio::MelToHz(float mel) const {
  // HTK-style inverse: hz = 700 * (10^(mel/2595) - 1)
  return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

}  // namespace houmo