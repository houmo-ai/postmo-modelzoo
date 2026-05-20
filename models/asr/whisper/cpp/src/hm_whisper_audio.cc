/*
 * Copyright (c) 2026 HOUMO AI
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * File: hm_whisper_audio.cpp
 * Description: Implementation of audio preprocessing class
 */

#include "hm_whisper_audio.hpp"

#include <cstring>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "miniaudio.h"
#include "samplerate.h"

namespace houmo {

HmWhisperAudio::HmWhisperAudio()
    : chunk_size_seconds_(30),
      n_mels_(80),
      n_samples_per_chunk_(kSampleRate * 30),
      total_samples_(0),
      audio_loaded_(false) {}

HmWhisperAudio::HmWhisperAudio(int chunk_size_seconds, int n_mels)
    : chunk_size_seconds_(chunk_size_seconds),
      n_mels_(n_mels),
      n_samples_per_chunk_(kSampleRate * chunk_size_seconds),
      total_samples_(0),
      audio_loaded_(false) {}

HmWhisperAudio::~HmWhisperAudio() {}

bool HmWhisperAudio::LoadAudio(const std::string& audio_path) {
  ma_decoder_config decoder_config =
      ma_decoder_config_init(ma_format_f32, 0, 0);
  ma_decoder decoder;
  ma_result result =
      ma_decoder_init_file(audio_path.c_str(), &decoder_config, &decoder);
  if (result != MA_SUCCESS) {
    std::cerr << "Error: Failed to load audio: " << audio_path
              << ", miniaudio error=" << result << "\n";
    return false;
  }

  ma_format format = ma_format_unknown;
  ma_uint32 channels = 0;
  ma_uint32 sample_rate = 0;
  result = ma_decoder_get_data_format(&decoder, &format, &channels,
                                      &sample_rate, nullptr, 0);
  if (result != MA_SUCCESS || format != ma_format_f32 || channels == 0 ||
      sample_rate == 0) {
    ma_decoder_uninit(&decoder);
    std::cerr << "Error: Invalid audio metadata: " << audio_path << "\n";
    return false;
  }

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
      std::cerr << "Error: Failed to read audio samples: " << audio_path
                << ", miniaudio error=" << result << "\n";
      return false;
    }

    interleaved_pcm.resize(static_cast<size_t>(frames_read) * channels);
  } else {
    constexpr ma_uint64 kFramesPerRead = 4096;
    std::vector<float> chunk(static_cast<size_t>(kFramesPerRead) * channels);

    while (true) {
      ma_uint64 frames_read = 0;
      result = ma_decoder_read_pcm_frames(&decoder, chunk.data(),
                                          kFramesPerRead, &frames_read);
      if (result != MA_SUCCESS && result != MA_AT_END) {
        ma_decoder_uninit(&decoder);
        std::cerr << "Error: Failed to stream audio samples: " << audio_path
                  << ", miniaudio error=" << result << "\n";
        return false;
      }

      if (frames_read == 0) {
        break;
      }

      interleaved_pcm.insert(
          interleaved_pcm.end(), chunk.begin(),
          chunk.begin() + static_cast<std::ptrdiff_t>(frames_read * channels));

      if (result == MA_AT_END) {
        break;
      }
    }
  }

  ma_decoder_uninit(&decoder);

  if (interleaved_pcm.empty()) {
    std::cerr << "Error: Failed to read audio samples: " << audio_path << "\n";
    return false;
  }

  std::vector<float> mono_pcm;
  DownmixToMono(interleaved_pcm, static_cast<int>(channels), &mono_pcm);

  if (sample_rate == static_cast<ma_uint32>(kSampleRate)) {
    pcm_data_ = std::move(mono_pcm);
  } else {
    if (!ResampleAudio(mono_pcm, static_cast<uint32_t>(sample_rate),
                       &pcm_data_)) {
      std::cerr << "Error: Failed to resample audio from " << sample_rate
                << " Hz to " << kSampleRate << " Hz: " << audio_path << "\n";
      return false;
    }
  }

  total_samples_ = static_cast<int>(pcm_data_.size());
  audio_loaded_ = true;

  std::cout << "Audio loaded: " << audio_path << ", channels: " << channels
            << ", sample rate: " << sample_rate << "Hz"
            << ", duration: " << GetTotalDuration() << "s"
            << ", samples: " << total_samples_ << "\n";

  return true;
}

int HmWhisperAudio::GetNumChunks() const {
  if (!audio_loaded_) return 0;
  return (total_samples_ + n_samples_per_chunk_ - 1) / n_samples_per_chunk_;
}

AudioChunk HmWhisperAudio::GetChunk(int chunk_index) const {
  AudioChunk chunk;
  chunk.chunk_index = chunk_index;
  chunk.is_last = (chunk_index == GetNumChunks() - 1);

  int start_sample = chunk_index * n_samples_per_chunk_;
  int end_sample =
      std::min(start_sample + n_samples_per_chunk_, total_samples_);
  int chunk_samples = end_sample - start_sample;

  chunk.pcm_data.assign(pcm_data_.begin() + start_sample,
                        pcm_data_.begin() + end_sample);

  // Pad to exact chunk size if needed
  if (chunk.pcm_data.size() < static_cast<size_t>(n_samples_per_chunk_)) {
    chunk.pcm_data.resize(n_samples_per_chunk_, 0.0f);
  }

  chunk.duration = static_cast<float>(chunk_samples) / kSampleRate;
  return chunk;
}

MelFeatures HmWhisperAudio::ComputeMelSpectrogram(
    const std::vector<float>& pcm_data) const {
  MelFeatures mel;
  mel.n_mels = n_mels_;

  // Whisper encoder input is fixed to 30 seconds (n_mels x 3000). Keep the
  // original behavior by padding shorter chunks and truncating longer chunks
  // to the fixed 30-second window expected by the model.
  std::vector<float> whisper_pcm = pcm_data;
  constexpr size_t kWhisperWindowSamples =
      static_cast<size_t>(kSampleRate * 30);
  if (whisper_pcm.size() < kWhisperWindowSamples) {
    whisper_pcm.resize(kWhisperWindowSamples, 0.0f);
  } else if (whisper_pcm.size() > kWhisperWindowSamples) {
    whisper_pcm.resize(kWhisperWindowSamples);
  }

  knf::WhisperFeatureOptions whisper_feature_opts;
  whisper_feature_opts.dim = mel.n_mels;

  knf::OnlineWhisperFbank whisper_fbank(whisper_feature_opts);
  whisper_fbank.AcceptWaveform(static_cast<float>(kSampleRate),
                               whisper_pcm.data(), whisper_pcm.size());
  whisper_fbank.InputFinished();

  mel.n_frames = whisper_fbank.NumFramesReady();
  if (mel.n_frames <= 0) {
    return mel;
  }

  // Mel computation
  std::vector<float> tmp_mel_data(mel.n_mels * mel.n_frames, 0.0f);
  float max_log_spec = -1e20f;

  for (int t = 0; t < mel.n_frames; ++t) {
    const float* frame = whisper_fbank.GetFrame(t);
    for (int m = 0; m < mel.n_mels; ++m) {
      float log_spec = std::log10(std::max(frame[m], 1e-10f));
      tmp_mel_data[m * mel.n_frames + t] = log_spec;

      if (log_spec > max_log_spec) {
        max_log_spec = log_spec;
      }
    }
  }

  // Normalization
  mel.data.resize(mel.n_mels * mel.n_frames);
  for (int i = 0; i < mel.n_mels * mel.n_frames; i++) {
    float val = tmp_mel_data[i];
    val = std::max(val, max_log_spec - 8.0f);
    mel.data[i] = static_cast<TensorType>((val + 4.0f) / 4.0f);
  }

  return mel;
}

MelFeatures HmWhisperAudio::GetChunkMelFeatures(int chunk_index) const {
  AudioChunk chunk = GetChunk(chunk_index);
  return ComputeMelSpectrogram(chunk.pcm_data);
}

float HmWhisperAudio::GetTotalDuration() const {
  return static_cast<float>(total_samples_) / kSampleRate;
}

int HmWhisperAudio::GetChunkSize() const { return chunk_size_seconds_; }

void HmWhisperAudio::DownmixToMono(const std::vector<float>& interleaved_pcm,
                                   int channels,
                                   std::vector<float>* mono_pcm) const {
  if (mono_pcm == nullptr || channels <= 0) {
    return;
  }

  if (channels == 1) {
    *mono_pcm = interleaved_pcm;
    return;
  }

  const size_t num_frames = interleaved_pcm.size() / channels;
  mono_pcm->assign(num_frames, 0.0f);

  const float scale = 1.0f / static_cast<float>(channels);
  for (size_t frame = 0; frame < num_frames; ++frame) {
    float mixed_sample = 0.0f;
    const size_t base_index = frame * channels;
    for (int channel = 0; channel < channels; ++channel) {
      mixed_sample += interleaved_pcm[base_index + channel];
    }
    (*mono_pcm)[frame] = mixed_sample * scale;
  }
}

bool HmWhisperAudio::ResampleAudio(const std::vector<float>& input_pcm,
                                   uint32_t input_sample_rate,
                                   std::vector<float>* output_pcm) const {
  if (output_pcm == nullptr || input_pcm.empty() || input_sample_rate == 0) {
    return false;
  }

  if (input_sample_rate == static_cast<uint32_t>(kSampleRate)) {
    *output_pcm = input_pcm;
    return true;
  }

  const double src_ratio = static_cast<double>(kSampleRate) / input_sample_rate;
  const size_t output_capacity =
      static_cast<size_t>(std::ceil(input_pcm.size() * src_ratio)) + 1;

  output_pcm->assign(output_capacity, 0.0f);

  SRC_DATA src_data;
  std::memset(&src_data, 0, sizeof(src_data));
  src_data.data_in = input_pcm.data();
  src_data.input_frames = static_cast<long>(input_pcm.size());
  src_data.data_out = output_pcm->data();
  src_data.output_frames = static_cast<long>(output_pcm->size());
  src_data.src_ratio = src_ratio;
  src_data.end_of_input = 1;

  const int result = src_simple(&src_data, SRC_SINC_FASTEST, 1);
  if (result != 0) {
    std::cerr << "Error: libsamplerate resampling failed: "
              << src_strerror(result) << "\n";
    output_pcm->clear();
    return false;
  }

  output_pcm->resize(static_cast<size_t>(src_data.output_frames_gen));
  return true;
}

}  // namespace houmo