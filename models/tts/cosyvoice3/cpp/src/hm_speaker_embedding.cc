/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_speaker_embedding.cc
 * Description:
 *   Implementation of speaker embedding extraction for CosyVoice3 TTS.
 *   Extracts speaker embedding from prompt audio using CampPlus model.
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

#include "hm_speaker_embedding.h"

#include <chrono>
#include <iostream>

namespace houmo {

HmSpeakerEmbedding::HmSpeakerEmbedding(
    const std::string& speaker_embedding_model_path)
    : last_inference_time_ms_(0.0f) {
  weight_manager_ = tcim::Module::WeightManager::CreateWeightManager(0);
  auto option = tcim::Module::Option(weight_manager_);

  speaker_module_ = std::make_shared<tcim::Module>();
  speaker_module_->LoadModel(speaker_embedding_model_path, option);

  for (int idx = 0; idx < speaker_module_->GetInputNum(); ++idx) {
    auto name = speaker_module_->GetInputName(idx);
    auto info = speaker_module_->GetInputInfo(name).AsContiguous();
    input_maps_[name] = tcim::Tensor::CreateHostTensor(info);
  }

  for (int idx = 0; idx < speaker_module_->GetOutputNum(); ++idx) {
    output_names_.push_back(speaker_module_->GetOutputName(idx));
  }
}

HmSpeakerEmbedding::~HmSpeakerEmbedding() = default;

HmSpeakerEmbedding::HmSpeakerEmbedding(HmSpeakerEmbedding&& other) noexcept
    : speaker_module_(std::move(other.speaker_module_)),
      audio_processor_(std::move(other.audio_processor_)),
      input_maps_(std::move(other.input_maps_)),
      output_names_(std::move(other.output_names_)),
      last_inference_time_ms_(other.last_inference_time_ms_) {
  other.last_inference_time_ms_ = 0.0f;
}

HmSpeakerEmbedding& HmSpeakerEmbedding::operator=(
    HmSpeakerEmbedding&& other) noexcept {
  if (this != &other) {
    speaker_module_ = std::move(other.speaker_module_);
    audio_processor_ = std::move(other.audio_processor_);
    input_maps_ = std::move(other.input_maps_);
    output_names_ = std::move(other.output_names_);
    last_inference_time_ms_ = other.last_inference_time_ms_;
    other.last_inference_time_ms_ = 0.0f;
  }
  return *this;
}

std::vector<TensorType> HmSpeakerEmbedding::Extract(
    const std::vector<float>& pcm_data_16k, CosyVoice3Perf* perf) {
  auto start_time = std::chrono::high_resolution_clock::now();
  // Step 1: Compute FBANK features
  FbankFeatures fbank = audio_processor_.ComputeFbank(pcm_data_16k);

  if (fbank.n_frames == 0 || fbank.data.empty()) {
    std::cerr << "Error: Empty FBANK features\n";
    return std::vector<TensorType>();
  }

  // Step 2: Prepare features for model input
  std::vector<TensorType> prepared_features = PrepareFeatures(fbank);
  // Step 3: Run campplus model inference
  if (speaker_module_->GetInputNum() == 0) {
    std::cerr << "Error: campplus model has no inputs\n";
    return std::vector<TensorType>();
  }

  auto input_name = speaker_module_->GetInputName(0);
  auto input_tensor = input_maps_[input_name];
  size_t input_memSize = input_tensor.MemSize();
  input_tensor.Buffer().CopyFromHost(prepared_features.data(), input_memSize);

  CHECK_TCIM_RET_STATUS(speaker_module_->SetInput(input_name, input_tensor));

  // Run inference
  CHECK_TCIM_RET_STATUS(speaker_module_->Run());
  CHECK_TCIM_RET_STATUS(speaker_module_->Sync());

  // Get output tensor
  // Output shape: [1, 192] float16
  if (output_names_.empty()) {
    std::cerr << "Error: campplus model has no outputs\n";
    return std::vector<TensorType>();
  }

  auto output_tensor = speaker_module_->GetOutput(output_names_[0]);
  size_t output_elements = output_tensor.MemSize() / sizeof(TensorType);
  std::vector<TensorType> embedding(output_elements);
  output_tensor.Buffer().CopyToHost(embedding.data(), output_tensor.MemSize());

  auto end_time = std::chrono::high_resolution_clock::now();
  last_inference_time_ms_ =
      std::chrono::duration<float, std::milli>(end_time - start_time).count();

  if (perf) {
    perf->speaker_emb_ms += last_inference_time_ms_;
  }

  return embedding;
}

std::vector<TensorType> HmSpeakerEmbedding::ExtractFromFile(
    const std::string& audio_path) {
  // Load audio at 16kHz
  if (!audio_processor_.LoadAudio(audio_path, kSampleRate)) {
    std::cerr << "Error: Failed to load audio: " << audio_path << "\n";
    return std::vector<TensorType>();
  }

  // Get PCM data
  const std::vector<float>& pcm_data = audio_processor_.GetPcmData();

  return Extract(pcm_data);
}

std::vector<TensorType> HmSpeakerEmbedding::PrepareFeatures(
    const FbankFeatures& fbank) {
  // feat = feat - feat.mean(dim=0, keepdim=True)
  int n_frames = fbank.n_frames;
  int n_mels = fbank.n_mels;

  std::vector<float> features_transposed(n_frames * n_mels);
  std::vector<float> features_mean(n_mels, 0.0f);

  for (int i = 0; i < n_frames; i++) {
    for (int j = 0; j < n_mels; j++) {
      features_mean[j] += fbank.data[i * n_mels + j];
    }
  }
  for (int j = 0; j < n_mels; j++) {
    features_mean[j] /= static_cast<float>(n_frames);
  }

  for (int i = 0; i < n_frames; i++) {
    for (int j = 0; j < n_mels; j++) {
      features_transposed[i * n_mels + j] =
          fbank.data[i * n_mels + j] - features_mean[j];
    }
  }
  // Step 3: Pad or trim to fixed length (kFixedT = 1000)
  std::vector<float> features_fixed =
      PadOrTrim(features_transposed, n_frames, n_mels);

  // Step 4: Convert to float16
  std::vector<TensorType> features_float16(kFixedT * n_mels);
  for (int t = 0; t < kFixedT; t++) {
    for (int m = 0; m < n_mels; m++) {
      features_float16[t * n_mels + m] =
          static_cast<TensorType>(features_fixed[t * n_mels + m]);
    }
  }

  return features_float16;
}

std::vector<float> HmSpeakerEmbedding::PadOrTrim(std::vector<float>& features,
                                                 int n_frames, int n_mels) {
  std::vector<float> result(kFixedT * n_mels,
                            0.0f);  // Initialize with zeros for padding

  if (n_frames < kFixedT) {
    // Copy all frames, rest will be zeros (padding)
    std::copy(features.begin(), features.end(), result.begin());
  } else {
    // Copy only first kFixedT frames (trim)
    std::copy(features.begin(), features.begin() + kFixedT * n_mels,
              result.begin());
  }

  return result;
}

}  // namespace houmo