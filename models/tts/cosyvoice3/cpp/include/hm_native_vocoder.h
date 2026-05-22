/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_native_vocoder.h
 * Description:
 *   HiFT vocoder module for CosyVoice3 TTS.
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

#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common_types.h"
#include "tcim_runtime_utils.h"

namespace houmo {
class HmNativeVocoder {
 public:
  HmNativeVocoder(const std::string& hift_part1_path,
                  const std::string& hift_part2_path);
  ~HmNativeVocoder();

  std::vector<float> Inference(const std::vector<float>& mel_spectrogram,
                               float speed = 1.0f,
                               CosyVoice3Perf* perf = nullptr);

 private:
  tcim::Module::WeightManager weight_manager_;
  std::shared_ptr<tcim::Module> hift_part1_;
  std::shared_ptr<tcim::Module> hift_part2_;
  std::unordered_map<std::string, tcim::Tensor> hift_part1_input_maps_;
  std::unordered_map<std::string, tcim::Tensor> hift_part2_input_maps_;
  int mel_bins_ = 80;
  int padded_mel_frames_ = 1024;
  int waveform_hop_ = 480;
};
}  // namespace houmo
