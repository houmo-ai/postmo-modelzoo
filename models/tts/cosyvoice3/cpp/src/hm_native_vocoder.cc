/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_native_vocoder.cc
 * Description:
 *   HiFT vocoder implementation for CosyVoice3 TTS.
 *   Converts mel spectrogram to audio waveform using HiFT vocoder.
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

#include "hm_native_vocoder.h"

#include <algorithm>
#include <chrono>
#include <complex>
#include <iostream>

#include "audio/librosa.h"

namespace {

size_t ShapeElementCount(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  return count;
}

std::vector<houmo::TensorType> FloatToTensorVector(
    const std::vector<float>& values) {
  std::vector<houmo::TensorType> out(values.size());
  std::transform(values.begin(), values.end(), out.begin(), [](float value) {
    return static_cast<houmo::TensorType>(value);
  });
  return out;
}

std::vector<float> TensorToFloatVector(const void* data, size_t count) {
  const auto* typed = static_cast<const houmo::TensorType*>(data);
  std::vector<float> out(count);
  std::transform(
      typed, typed + count, out.begin(),
      [](houmo::TensorType value) { return static_cast<float>(value); });
  return out;
}

std::vector<float> ResizeMelLinear(const std::vector<float>& mel, int mel_bins,
                                   int src_frames, int dst_frames) {
  if (src_frames <= 0 || dst_frames <= 0) {
    return {};
  }
  if (src_frames == dst_frames) {
    return mel;
  }

  std::vector<float> resized(static_cast<size_t>(mel_bins * dst_frames), 0.0f);
  for (int mel_idx = 0; mel_idx < mel_bins; ++mel_idx) {
    const float* src = mel.data() + static_cast<size_t>(mel_idx * src_frames);
    float* dst = resized.data() + static_cast<size_t>(mel_idx * dst_frames);
    if (dst_frames == 1) {
      dst[0] = src[0];
      continue;
    }
    for (int frame_idx = 0; frame_idx < dst_frames; ++frame_idx) {
      const float src_pos = static_cast<float>(frame_idx) *
                            static_cast<float>(src_frames - 1) /
                            static_cast<float>(dst_frames - 1);
      const int left = static_cast<int>(src_pos);
      const int right = std::min(left + 1, src_frames - 1);
      const float alpha = src_pos - static_cast<float>(left);
      dst[frame_idx] = src[left] * (1.0f - alpha) + src[right] * alpha;
    }
  }
  return resized;
}

}  // namespace

namespace houmo {

HmNativeVocoder::HmNativeVocoder(const std::string& hift_part1_path,
                                 const std::string& hift_part2_path) {
  weight_manager_ = tcim::Module::WeightManager::CreateWeightManager(0);
  auto hift_part1_option = tcim::Module::Option(weight_manager_);
  auto hift_part2_option = tcim::Module::Option(weight_manager_);
  hift_part1_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_RET_STATUS(
      hift_part1_->LoadModel(hift_part1_path, hift_part1_option));
  std::string dummy_input_name = hift_part1_->GetInputName(0);
  std::vector<std::string> dummy_names = {dummy_input_name};
  hift_part2_option.SetDummyTensors(dummy_names);
  hift_part2_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_RET_STATUS(
      hift_part2_->LoadModel(hift_part2_path, hift_part2_option));
  auto dummy_input = hift_part1_->GetDevInput(dummy_input_name);
  CHECK_TCIM_RET_STATUS(
      hift_part2_->SetDevInput(dummy_input_name, dummy_input));

  for (int idx = 0; idx < hift_part1_->GetInputNum(); idx++) {
    auto name = hift_part1_->GetInputName(idx);
    auto info = hift_part1_->GetInputInfo(name).AsContiguous();
    hift_part1_input_maps_[name] = tcim::Tensor::CreateHostTensor(info);
  }

  for (int idx = 0; idx < hift_part2_->GetInputNum(); idx++) {
    auto name = hift_part2_->GetInputName(idx);
    auto info = hift_part2_->GetInputInfo(name).AsContiguous();
    hift_part2_input_maps_[name] = tcim::Tensor::CreateHostTensor(info);
  }
}

HmNativeVocoder::~HmNativeVocoder() {}

std::vector<float> HmNativeVocoder::Inference(
    const std::vector<float>& mel_spectrogram, float speed,
    CosyVoice3Perf* perf) {
  auto start_time = std::chrono::high_resolution_clock::now();
  if (mel_spectrogram.empty()) {
    return {};
  }

  const int mel_len2 = static_cast<int>(mel_spectrogram.size() / mel_bins_);
  std::vector<float> speech_feat = mel_spectrogram;
  int mel_frames_for_model = mel_len2;

  if (speed != 1.0f && mel_len2 > 0) {
    mel_frames_for_model = std::max(1, static_cast<int>(mel_len2 / speed));
    speech_feat =
        ResizeMelLinear(speech_feat, mel_bins_, mel_len2, mel_frames_for_model);
  }

  std::vector<float> padded_speech_feat(
      static_cast<size_t>(mel_bins_ * padded_mel_frames_), 0.0f);
  const int copied_frames = std::min(mel_frames_for_model, padded_mel_frames_);
  for (int mel_idx = 0; mel_idx < mel_bins_; ++mel_idx) {
    const size_t src_offset =
        static_cast<size_t>(mel_idx * mel_frames_for_model);
    const size_t dst_offset = static_cast<size_t>(mel_idx * padded_mel_frames_);
    std::copy(speech_feat.begin() + src_offset,
              speech_feat.begin() + src_offset + copied_frames,
              padded_speech_feat.begin() + dst_offset);
  }

  const auto hift_part1_input_name = hift_part1_->GetInputName(0);
  auto speech_feat_tensor = FloatToTensorVector(padded_speech_feat);
  CHECK_TCIM_RET_STATUS(
      hift_part1_input_maps_[hift_part1_input_name].Buffer().CopyFromHost(
          speech_feat_tensor.data(),
          hift_part1_input_maps_[hift_part1_input_name].MemSize()));
  CHECK_TCIM_RET_STATUS(hift_part1_->SetInput(
      hift_part1_input_name, hift_part1_input_maps_[hift_part1_input_name]));
  CHECK_TCIM_RET_STATUS(hift_part1_->Run());
  CHECK_TCIM_RET_STATUS(hift_part1_->Sync());

  auto hift_part1_output =
      hift_part1_->GetDevOutput(hift_part1_->GetOutputName(0)).ToHost(true);
  const auto hift_part1_output_shape =
      hift_part1_->GetOutputInfo(hift_part1_->GetOutputName(0)).Shape();
  std::vector<float> stft_input =
      TensorToFloatVector(hift_part1_output.Buffer().Data(),
                          ShapeElementCount(hift_part1_output_shape));

  auto stft_frames =
      librosa::Feature::stft(stft_input, 16, 4, "hann", false, "reflect");
  const auto hift_part2_input_name = hift_part2_->GetInputName(0);
  const auto hift_part2_input_shape =
      hift_part2_->GetInputInfo(hift_part2_input_name).Shape();
  const int target_frames = static_cast<int>(hift_part2_input_shape[1]);
  const int target_bins = static_cast<int>(hift_part2_input_shape[2]);
  std::vector<float> stft_flat(
      static_cast<size_t>(target_frames * target_bins * 2), 0.0f);
  const int copied_stft_frames =
      std::min(target_frames, static_cast<int>(stft_frames.size()));
  for (int frame_idx = 0; frame_idx < copied_stft_frames; ++frame_idx) {
    const int copied_bins =
        std::min(target_bins, static_cast<int>(stft_frames[frame_idx].size()));
    for (int bin_idx = 0; bin_idx < copied_bins; ++bin_idx) {
      const size_t offset =
          static_cast<size_t>((frame_idx * target_bins + bin_idx) * 2);
      stft_flat[offset] = stft_frames[frame_idx][bin_idx].real();
      stft_flat[offset + 1] = stft_frames[frame_idx][bin_idx].imag();
    }
  }

  auto stft_tensor = FloatToTensorVector(stft_flat);
  CHECK_TCIM_RET_STATUS(
      hift_part2_input_maps_[hift_part2_input_name].Buffer().CopyFromHost(
          stft_tensor.data(),
          hift_part2_input_maps_[hift_part2_input_name].MemSize()));
  CHECK_TCIM_RET_STATUS(hift_part2_->SetInput(
      hift_part2_input_name, hift_part2_input_maps_[hift_part2_input_name]));
  CHECK_TCIM_RET_STATUS(hift_part2_->Run());
  CHECK_TCIM_RET_STATUS(hift_part2_->Sync());

  auto hift_part2_output =
      hift_part2_->GetDevOutput(hift_part2_->GetOutputName(0)).ToHost(true);
  const auto hift_part2_output_shape =
      hift_part2_->GetOutputInfo(hift_part2_->GetOutputName(0)).Shape();
  std::vector<float> utterance_audio =
      TensorToFloatVector(hift_part2_output.Buffer().Data(),
                          ShapeElementCount(hift_part2_output_shape));

  const size_t expected_samples =
      static_cast<size_t>(std::max(mel_len2, 0) * waveform_hop_);
  if (utterance_audio.size() > expected_samples) {
    utterance_audio.resize(expected_samples);
  }

  if (perf) {
    auto end_time = std::chrono::high_resolution_clock::now();
    perf->vocoder_ms +=
        std::chrono::duration<float, std::milli>(end_time - start_time).count();
  }
  return utterance_audio;
}
}  // namespace houmo
