/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_stateful_decoder.cc
 * Description:
 *   Qwen3-TTS stateful streaming audio decoder implementation.
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

#include "qwen3_tts_stateful_decoder.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include "base/tcim_utils.h"

namespace houmo {

namespace {

Qwen3TTSDecoderOutput DecodeEmptyChunk(Qwen3TTSDecoderState state,
                                       bool is_final) {
  if (!is_final) {
    throw std::invalid_argument("An empty decoder chunk must be final");
  }
  Qwen3TTSDecoderOutput output;
  output.audio = std::move(state.latent_audio);
  state.latent_audio.clear();
  output.state = std::move(state);
  return output;
}

void ValidateChunk(const std::vector<Qwen3TTSCodecFrame>& frames,
                   const Qwen3TTSDecoderState& state, bool is_final) {
  if (frames.size() > Qwen3TTSStatefulDecoder::kChunkSize ||
      (!is_final && frames.size() != Qwen3TTSStatefulDecoder::kChunkSize) ||
      state.kv_cache.size() != 16) {
    throw std::invalid_argument("Invalid stateful decoder chunk or state");
  }
}

}  // namespace

Qwen3TTSStatefulDecoder::Qwen3TTSStatefulDecoder(
    const std::string& model_path, int device_id)
    : weight_manager_(
          tcim::Module::WeightManager::CreateWeightManager(device_id)) {
  if (!std::filesystem::exists(model_path)) {
    throw Exception("Stateful decoder model not found: " + model_path);
  }
  auto option = tcim::Module::Option(weight_manager_);
  module_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_RET_STATUS(module_->LoadModel(model_path, option));
  for (int index = 0; index < module_->GetInputNum(); ++index) {
    const std::string name = module_->GetInputName(index);
    input_names_.push_back(name);
    if (index == 0 || (index >= 4 && index <= 6)) {
      host_inputs_.emplace(name, tcim::Tensor::CreateHostTensor(
                                     module_->GetInputInfo(name).AsContiguous()));
    }
  }
  for (int index = 0; index < module_->GetOutputNum(); ++index) {
    output_names_.push_back(module_->GetOutputName(index));
  }
  const auto shape = module_->GetInputInfo(input_names_[0]).Shape();
  if (shape.size() != 3 || shape[1] != kChunkSize ||
      shape[2] != kCodebookCount || input_names_.size() != 23 ||
      output_names_.size() != 21) {
    throw Exception("Unexpected stateful decoder interface");
  }
}

tcim::Tensor Qwen3TTSStatefulDecoder::CreateZeroDeviceInput(
    size_t input_index) {
  const std::string& name = input_names_.at(input_index);
  auto host = tcim::Tensor::CreateHostTensor(
      module_->GetInputInfo(name).AsContiguous());
  CHECK_TCIM_RET_STATUS(host.Buffer().MemSet(0, host.MemSize(), 0));
  CHECK_TCIM_RET_STATUS(module_->SetInput(name, host));
  return module_->GetDevInput(name);
}

Qwen3TTSDecoderState Qwen3TTSStatefulDecoder::CreateState() {
  Qwen3TTSDecoderState state;
  state.pre_conv_history = CreateZeroDeviceInput(1);
  state.latent_buffer = CreateZeroDeviceInput(2);
  state.conv_history = CreateZeroDeviceInput(3);
  for (size_t index = 7; index < input_names_.size(); ++index) {
    state.kv_cache.push_back(CreateZeroDeviceInput(index));
  }
  return state;
}

void Qwen3TTSStatefulDecoder::SetScalar(
    const std::string& name, const void* value, size_t size) {
  auto& tensor = host_inputs_.at(name);
  CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(value, size));
  CHECK_TCIM_RET_STATUS(module_->SetInput(name, tensor));
}

Qwen3TTSDecoderOutput Qwen3TTSStatefulDecoder::Decode(
    const std::vector<Qwen3TTSCodecFrame>& frames,
    Qwen3TTSDecoderState state, bool is_final) {
  if (frames.empty()) {
    return DecodeEmptyChunk(std::move(state), is_final);
  }
  ValidateChunk(frames, state, is_final);

  std::vector<int32_t> codes(kChunkSize * kCodebookCount, 0);
  for (size_t frame = 0; frame < frames.size(); ++frame) {
    std::copy(frames[frame].begin(), frames[frame].end(),
              codes.begin() + static_cast<std::ptrdiff_t>(
                                  frame * kCodebookCount));
  }
  auto& codes_tensor = host_inputs_.at(input_names_[0]);
  CHECK_TCIM_RET_STATUS(codes_tensor.Buffer().CopyFromHost(
      codes.data(), codes_tensor.MemSize()));
  CHECK_TCIM_RET_STATUS(module_->SetInput(input_names_[0], codes_tensor));
  CHECK_TCIM_RET_STATUS(module_->SetDevInput(input_names_[1],
                                             state.pre_conv_history));
  CHECK_TCIM_RET_STATUS(
      module_->SetDevInput(input_names_[2], state.latent_buffer));
  CHECK_TCIM_RET_STATUS(
      module_->SetDevInput(input_names_[3], state.conv_history));
  const float16 final_value(is_final ? 1.0f : 0.0f);
  const int32_t valid_frames = static_cast<int32_t>(frames.size());
  SetScalar(input_names_[4], &final_value, sizeof(final_value));
  SetScalar(input_names_[5], &state.kv_valid_length,
            sizeof(state.kv_valid_length));
  SetScalar(input_names_[6], &valid_frames, sizeof(valid_frames));
  for (size_t index = 0; index < state.kv_cache.size(); ++index) {
    CHECK_TCIM_RET_STATUS(
        module_->SetDevInput(input_names_[7 + index], state.kv_cache[index]));
  }

  CHECK_TCIM_RET_STATUS(module_->Run());
  CHECK_TCIM_RET_STATUS(module_->Sync());
  auto wav_host = module_->GetDevOutput(output_names_[0]).ToHost(true);
  auto valid_host = module_->GetDevOutput(output_names_[1]).ToHost(true);
  const int32_t valid_samples =
      *static_cast<const int32_t*>(valid_host.Buffer().Data());
  const auto* wav = static_cast<const float16*>(wav_host.Buffer().Data());
  const size_t wav_size = wav_host.MemSize() / sizeof(float16);

  Qwen3TTSDecoderOutput output;
  output.state.pre_conv_history = module_->GetDevOutput(output_names_[2]);
  output.state.latent_buffer = module_->GetDevOutput(output_names_[3]);
  output.state.conv_history = module_->GetDevOutput(output_names_[4]);
  for (size_t index = 5; index < output_names_.size(); ++index) {
    output.state.kv_cache.push_back(module_->GetDevOutput(output_names_[index]));
  }
  output.state.kv_valid_length = std::min<int32_t>(
      72, state.kv_valid_length + valid_frames);

  const size_t initial_skip =
      state.kv_valid_length == 0 ? 4 * kSamplesPerFrame : 0;
  const size_t start = std::min(initial_skip, wav_size);
  const size_t end = std::min(start + std::max(valid_samples, 0), wav_size);
  output.audio.reserve(end - start);
  for (size_t index = start; index < end; ++index) {
    output.audio.push_back(static_cast<float>(wav[index]));
  }
  if (!is_final) {
    output.state.latent_audio.reserve(wav_size - end);
    for (size_t index = end; index < wav_size; ++index) {
      output.state.latent_audio.push_back(static_cast<float>(wav[index]));
    }
  }
  if (state.skip_samples > 0) {
    const size_t skip = std::min(state.skip_samples, output.audio.size());
    output.audio.erase(output.audio.begin(),
                       output.audio.begin() + static_cast<std::ptrdiff_t>(skip));
    output.state.skip_samples = state.skip_samples - skip;
  }
  if (is_final) output.state.skip_samples = 4 * kSamplesPerFrame;
  return output;
}

}  // namespace houmo
