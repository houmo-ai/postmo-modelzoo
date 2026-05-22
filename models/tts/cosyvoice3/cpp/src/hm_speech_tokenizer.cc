/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_speech_tokenizer.cc
 * Description:
 *   Implementation of speech tokenizer module for CosyVoice3 TTS.
 *   Extracts speech tokens from prompt audio using speech_tokenizer.hmm model.
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

#include "hm_speech_tokenizer.h"

#include <chrono>
#include <cstring>
#include <iostream>

namespace houmo {

// ============================================================================
// Constructor / Destructor
// ============================================================================

HmSpeechTokenizer::HmSpeechTokenizer(std::string speech_tokenizer_model_path,
                                     std::shared_ptr<HmAudio> audio)
    : audio_(audio),
      inference_time_ms_(0.0f),
      last_feat_len_(0),
      last_token_len_(0) {
  weight_manager_ = tcim::Module::WeightManager::CreateWeightManager(0);

  auto option = tcim::Module::Option(weight_manager_);

  speech_module_ = std::make_shared<tcim::Module>();
  speech_module_->LoadModel(speech_tokenizer_model_path, option);

  for (int idx = 0; idx < speech_module_->GetInputNum(); ++idx) {
    auto name = speech_module_->GetInputName(idx);
    auto info = speech_module_->GetInputInfo(name).AsContiguous();
    input_maps_[name] = tcim::Tensor::CreateHostTensor(info);
  }

  for (int idx = 0; idx < speech_module_->GetOutputNum(); ++idx) {
    output_names_.push_back(speech_module_->GetOutputName(idx));
  }
}

HmSpeechTokenizer::~HmSpeechTokenizer() = default;

HmSpeechTokenizer::HmSpeechTokenizer(HmSpeechTokenizer&& other) noexcept
    : speech_module_(std::move(other.speech_module_)),
      audio_(std::move(other.audio_)),
      input_maps_(std::move(other.input_maps_)),
      output_names_(std::move(other.output_names_)),
      inference_time_ms_(other.inference_time_ms_),
      last_feat_len_(other.last_feat_len_),
      last_token_len_(other.last_token_len_) {
  other.inference_time_ms_ = 0.0f;
  other.last_feat_len_ = 0;
  other.last_token_len_ = 0;
}

HmSpeechTokenizer& HmSpeechTokenizer::operator=(
    HmSpeechTokenizer&& other) noexcept {
  if (this != &other) {
    // speech_module_ is a reference and cannot be reassigned
    audio_ = std::move(other.audio_);
    input_maps_ = std::move(other.input_maps_);
    output_names_ = std::move(other.output_names_);
    inference_time_ms_ = other.inference_time_ms_;
    last_feat_len_ = other.last_feat_len_;
    last_token_len_ = other.last_token_len_;

    other.inference_time_ms_ = 0.0f;
    other.last_feat_len_ = 0;
    other.last_token_len_ = 0;
  }
  return *this;
}

// ============================================================================
// Public Methods
// ============================================================================

std::vector<int> HmSpeechTokenizer::Extract(
    const std::vector<float>& pcm_data_16k, CosyVoice3Perf* perf) {
  auto result = ExtractWithLength(pcm_data_16k, perf);
  return result.first;
}

std::pair<std::vector<int>, int> HmSpeechTokenizer::ExtractWithLength(
    const std::vector<float>& pcm_data_16k, CosyVoice3Perf* perf) {
  auto t_start = std::chrono::high_resolution_clock::now();

  // Check audio duration (max 30s)
  float duration =
      static_cast<float>(pcm_data_16k.size()) / HmAudio::kSampleRate16k;
  if (duration > 30.0f) {
    std::cerr << "Error: Audio too long for speech token extraction: "
              << duration << "s (max 30s)\n";
    return {{}, 0};
  }

  // Compute 128-bin mel spectrogram

  MelFeatures128 mel = audio_->ComputeMelSpectrogram128(pcm_data_16k);
  int feat_len = mel.n_frames;
  last_feat_len_ = feat_len;

  // Prepare input tensors
  PrepareInput(mel, feat_len);

  // Run inference
  CHECK_TCIM_RET_STATUS(speech_module_->Run());
  CHECK_TCIM_RET_STATUS(speech_module_->Sync());

  auto t_end = std::chrono::high_resolution_clock::now();
  inference_time_ms_ =
      std::chrono::duration<float, std::milli>(t_end - t_start).count();

  // Extract tokens
  int valid_token_len = feat_len / kTokenRatio;
  last_token_len_ = valid_token_len;
  std::vector<int> tokens = ExtractTokens(valid_token_len);

  if (perf) {
    perf->speech_tokenizer_ms += inference_time_ms_;
  }

  return {tokens, valid_token_len};
}

// ============================================================================
// Private Methods
// ============================================================================

void HmSpeechTokenizer::PrepareInput(const MelFeatures128& mel_features,
                                     int feat_len) {
  // Prepare padded mel input [1, 128, 3000]
  std::vector<TensorType> padded_mel = PreparePaddedMel(mel_features, feat_len);

  // Calculate valid token length
  int valid_len = feat_len / kTokenRatio;

  // Create attention masks
  std::vector<TensorType> mask = CreateAttentionMask(valid_len);
  std::vector<TensorType> mask1 = CreateEncoderMask(valid_len);
  // Set inputs to module
  // Input 0: padded_mel [1, 128, 3000]
  // Input 1: mask [1, 20, 750, 750]
  // Input 2: mask1 [1, 750, 1280]

  if (input_maps_.size() >= 3) {
    auto padded_mel_tensor = input_maps_[speech_module_->GetInputName(0)];
    size_t padded_mel_memSize = padded_mel_tensor.MemSize();
    padded_mel_tensor.Buffer().CopyFromHost(padded_mel.data(),
                                            padded_mel_memSize);
    CHECK_TCIM_RET_STATUS(speech_module_->SetInput(
        speech_module_->GetInputName(0), padded_mel_tensor));
    auto mask_tensor = input_maps_[speech_module_->GetInputName(1)];
    size_t mask_memSize = mask_tensor.MemSize();
    mask_tensor.Buffer().CopyFromHost(mask.data(), mask_memSize);
    CHECK_TCIM_RET_STATUS(
        speech_module_->SetInput(speech_module_->GetInputName(1), mask_tensor));
    auto mask1_tensor = input_maps_[speech_module_->GetInputName(2)];
    size_t mask1_memSize = mask1_tensor.MemSize();
    mask1_tensor.Buffer().CopyFromHost(mask1.data(), mask1_memSize);
    CHECK_TCIM_RET_STATUS(speech_module_->SetInput(
        speech_module_->GetInputName(2), mask1_tensor));
  } else {
    std::cerr << "Speech tokenizer: unexpected number of inputs: "
              << input_maps_.size() << " (expected 3)\n";
  }
}

std::vector<TensorType> HmSpeechTokenizer::PreparePaddedMel(
    const MelFeatures128& mel_features, int feat_len) {
  std::vector<TensorType> padded_mel(kNMels * kMaxMelFrames,
                                     static_cast<TensorType>(0.0f));

  const auto& mel_data = mel_features.data;

  for (int t = 0; t < feat_len; t++) {
    for (int m = 0; m < kNMels; m++) {
      // Source index in mel_features.data
      int src_idx = m * feat_len + t;
      // Target index in padded_mel
      int dst_idx = m * kMaxMelFrames + t;
      if (src_idx < static_cast<int>(mel_data.size())) {
        padded_mel[dst_idx] = mel_data[src_idx];
      }
    }
  }

  return padded_mel;
}

std::vector<TensorType> HmSpeechTokenizer::CreateAttentionMask(int valid_len) {
  // Create attention mask [1, 20, 750, 750]
  // Total elements: 1 * 20 * 750 * 750 = 11250000
  int total_elements = kMaskDim2 * kMaxTokens * kMaxTokens;
  std::vector<TensorType> mask(total_elements,
                               static_cast<TensorType>(kMaskMinValue));

  for (int d2 = 0; d2 < kMaskDim2; d2++) {
    for (int row = 0; row < kMaxTokens; row++) {
      for (int col = 0; col < valid_len; col++) {
        // Index in flattened array
        // Layout assumption: [d2, row, col] -> d2 * 750 * 750 + row * 750 + col
        int idx = d2 * kMaxTokens * kMaxTokens + row * kMaxTokens + col;
        mask[idx] = static_cast<TensorType>(0.0f);
      }
    }
  }

  return mask;
}

std::vector<TensorType> HmSpeechTokenizer::CreateEncoderMask(int valid_len) {
  // Create encoder mask [1, 750, 1280]
  // Total elements: 1 * 750 * 1280 = 960000

  int total_elements = kMaxTokens * kMask1Dim2;
  std::vector<TensorType> mask1(total_elements, static_cast<TensorType>(0.0f));

  // Set valid region to 1.0
  // Python: mask1[:, 0:feat_len//4, :] = 1.0
  // This means for rows < valid_len, all columns are 1.0

  for (int row = 0; row < valid_len; row++) {
    for (int col = 0; col < kMask1Dim2; col++) {
      // Index in flattened array
      // Layout: [row, col] -> row * 1280 + col
      int idx = row * kMask1Dim2 + col;
      mask1[idx] = static_cast<TensorType>(1.0f);
    }
  }

  return mask1;
}

std::vector<int> HmSpeechTokenizer::ExtractTokens(int valid_len) {
  // Extract tokens from output tensor
  if (output_names_.empty()) {
    std::cerr << "No output names for speech tokenizer\n";
    return {};
  }

  // Get output from device
  // Python uses get_dev_output().to_host().numpy()
  // In C++, we use GetOutput which handles the cast to host tensor

  // The output is int32 token IDs [1, seq_len]
  // We extract tokens[:valid_len]

  // Try to get int output
  std::vector<int> all_tokens;
  try {
    auto dev_output_tensor = speech_module_->GetDevOutput(output_names_[0]);
    auto host_output_tensor = dev_output_tensor.ToHost(true);

    int* outData = static_cast<int*>(host_output_tensor.Buffer().Data());
    size_t memSize = host_output_tensor.MemSize();
    all_tokens.assign(outData, outData + memSize / sizeof(int));
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Failed to get output tensor: ") +
                             e.what());
  }

  // Extract valid tokens
  // Python: speech_token = speech_token[:, :feat_len//4]
  // Output shape is [1, max_tokens], we need first valid_len tokens

  if (static_cast<int>(all_tokens.size()) < valid_len) {
    std::cerr << "Warning: Output has fewer tokens than expected: "
              << all_tokens.size() << " < " << valid_len << "\n";
    valid_len = static_cast<int>(all_tokens.size());
  }

  // Output tensor shape is [1, max_tokens], flattened as [max_tokens]
  // So we just take first valid_len elements
  std::vector<int> tokens(all_tokens.begin(), all_tokens.begin() + valid_len);

  return tokens;
}

}  // namespace houmo