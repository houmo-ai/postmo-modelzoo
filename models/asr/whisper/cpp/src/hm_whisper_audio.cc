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

namespace houmo {

HmWhisperAudio::HmWhisperAudio()
    : chunk_size_seconds_(30),
      n_samples_per_chunk_(kSampleRate * 30),
      total_samples_(0),
      audio_loaded_(false) {}

HmWhisperAudio::HmWhisperAudio(int chunk_size_seconds)
    : chunk_size_seconds_(chunk_size_seconds),
      n_samples_per_chunk_(kSampleRate * chunk_size_seconds),
      total_samples_(0),
      audio_loaded_(false) {}

HmWhisperAudio::~HmWhisperAudio() {}

bool HmWhisperAudio::LoadAudio(const std::string& audio_path) {
    ma_decoder decoder;
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, kSampleRate);

    if (ma_decoder_init_file(audio_path.c_str(), &config, &decoder) != MA_SUCCESS) {
        std::cerr << "Error: Failed to load audio: " << audio_path << "\n";
        return false;
    }

    ma_uint64 frame_count;
    ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);

    pcm_data_.resize(frame_count);
    ma_decoder_read_pcm_frames(&decoder, pcm_data_.data(), frame_count, &frame_count);
    ma_decoder_uninit(&decoder);

    pcm_data_.resize(frame_count);
    total_samples_ = static_cast<int>(frame_count);
    audio_loaded_ = true;

    std::cout << "Audio loaded: " << audio_path
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
    int end_sample = std::min(start_sample + n_samples_per_chunk_, total_samples_);
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

MelFeatures HmWhisperAudio::ComputeMelSpectrogram(const std::vector<float>& pcm_data) const {
    MelFeatures mel;
    mel.n_mels = kNMels;

    // Calculate frames
    int n_frames = pcm_data.size() / kHopLength - kNFFT / kHopLength + 1;
    if (pcm_data.size() >= static_cast<size_t>(n_samples_per_chunk_)) {
        n_frames = 3000;  // Fixed for 30s chunks
    }
    mel.n_frames = n_frames;

    // Hann window - use cached version if size matches
    if (hann_window_cache_.empty() || hann_window_size_ != kNFFT) {
        hann_window_cache_ = ComputeHannWindow(kNFFT);
        hann_window_size_ = kNFFT;
    }
    const std::vector<float>& window = hann_window_cache_;

    // Reflection padding
    int pad_len = kNFFT / 2;
    std::vector<float> padded_audio = ApplyReflectionPadding(pcm_data, pad_len);

    // Mel computation
    std::vector<float> tmp_mel_data(mel.n_mels * mel.n_frames, 0.0f);
    float max_log_spec = -1e20f;

    for (int t = 0; t < n_frames; t++) {
        int offset = t * kHopLength;
        std::vector<float> magnitudes(kNFFTBins, 0.0f);

        // Naive DFT for power spectrum
        for (int k = 0; k < kNFFTBins; k++) {
            float sum_real = 0.0f;
            float sum_imag = 0.0f;
            float angle_step = -2.0f * static_cast<float>(M_PI) * k / kNFFT;

            for (int n = 0; n < kNFFT; n++) {
                float val = padded_audio[offset + n] * window[n];
                sum_real += val * std::cos(angle_step * n);
                sum_imag += val * std::sin(angle_step * n);
            }
            magnitudes[k] = (sum_real * sum_real + sum_imag * sum_imag);
        }

        // Apply mel filter bank
        for (int m = 0; m < mel.n_mels; m++) {
            float sum_mel = 0.0f;
            for (int k = 0; k < kNFFTBins; k++) {
                sum_mel += MEL_FILTERS[m][k] * magnitudes[k];
            }

            float log_spec = std::log10(std::max(sum_mel, 1e-10f));
            tmp_mel_data[m * n_frames + t] = log_spec;

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

int HmWhisperAudio::GetChunkSize() const {
    return chunk_size_seconds_;
}

std::vector<float> HmWhisperAudio::ComputeHannWindow(int n_fft) const {
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; i++) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / n_fft));
    }
    return window;
}

std::vector<float> HmWhisperAudio::ApplyReflectionPadding(
    const std::vector<float>& audio, int pad_len) const {
    std::vector<float> padded(audio.size() + 2 * pad_len);

    // Reflection padding for start
    for (int i = 0; i < pad_len; i++) {
        padded[pad_len - 1 - i] = audio[i + 1];
    }

    // Original audio
    for (size_t i = 0; i < audio.size(); i++) {
        padded[pad_len + i] = audio[i];
    }

    // Reflection padding for end
    int audio_size = static_cast<int>(audio.size());
    for (int i = 0; i < pad_len; i++) {
        padded[pad_len + audio_size + i] = audio[audio_size - 2 - i];
    }

    return padded;
}

}  // namespace houmo