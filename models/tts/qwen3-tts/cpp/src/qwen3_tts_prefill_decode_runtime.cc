/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_prefill_decode_runtime.cc
 * Description:
 *   Shared Qwen3-TTS prefill/decode HMM runtime implementation.
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

#include "qwen3_tts_prefill_decode_runtime.h"

#include <filesystem>
#include <stdexcept>

#include "base/tcim_utils.h"

namespace houmo {

Qwen3TTSPrefillDecodeRuntime::Qwen3TTSPrefillDecodeRuntime(
    const std::string& prefill_path, const std::string& decode_path,
    int device_id)
    : weight_manager_(
          tcim::Module::WeightManager::CreateWeightManager(device_id)) {
  if (!std::filesystem::exists(prefill_path) ||
      !std::filesystem::exists(decode_path)) {
    throw Exception("Prefill or decode HMM file does not exist");
  }

  auto prefill_option = tcim::Module::Option(weight_manager_);
  prefill_module_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_RET_STATUS(prefill_module_->LoadModel(prefill_path, prefill_option));

  for (int index = 0; index < prefill_module_->GetInputNum(); ++index) {
    const std::string name = prefill_module_->GetInputName(index);
    if (IsCacheInput(name)) cache_names_.push_back(name);
  }

  auto decode_option = tcim::Module::Option(weight_manager_);
  decode_option.SetDummyTensors(cache_names_);
  decode_module_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_RET_STATUS(decode_module_->LoadModel(decode_path, decode_option));

  for (const auto& name : cache_names_) {
    CHECK_TCIM_RET_STATUS(
        decode_module_->SetDevInput(name, prefill_module_->GetDevInput(name)));
  }

  InitializeHostInputs(prefill_module_.get(), &prefill_inputs_);
  InitializeHostInputs(decode_module_.get(), &decode_inputs_);
  for (int index = 0; index < prefill_module_->GetOutputNum(); ++index) {
    prefill_output_names_.push_back(prefill_module_->GetOutputName(index));
  }
  for (int index = 0; index < decode_module_->GetOutputNum(); ++index) {
    decode_output_names_.push_back(decode_module_->GetOutputName(index));
  }

  const auto prefill_shape =
      prefill_module_->GetInputInfo(prefill_module_->GetInputName(0)).Shape();
  const auto decode_shape =
      decode_module_->GetInputInfo(decode_module_->GetInputName(0)).Shape();
  if (prefill_shape.size() != 3 || decode_shape.size() != 3 ||
      prefill_shape[0] != 1 || decode_shape[0] != 1 ||
      decode_shape[1] != 1 || prefill_shape[2] != decode_shape[2]) {
    throw Exception("Unexpected prefill/decode embedding input shape");
  }
  prefill_length_ = static_cast<size_t>(prefill_shape[1]);
  hidden_dim_ = static_cast<size_t>(prefill_shape[2]);
  if (!cache_names_.empty()) {
    const auto cache_shape =
        decode_module_->GetInputInfo(cache_names_.front()).Shape();
    if (cache_shape.size() >= 3) {
      context_length_ = static_cast<size_t>(cache_shape[2]);
    }
  }
}

bool Qwen3TTSPrefillDecodeRuntime::IsCacheInput(const std::string& name) {
  return name.find("model_layers_") == 0 &&
         (name.find("_kcache_input") != std::string::npos ||
          name.find("_vcache_input") != std::string::npos);
}

void Qwen3TTSPrefillDecodeRuntime::InitializeHostInputs(
    tcim::Module* module,
    std::unordered_map<std::string, tcim::Tensor>* inputs) {
  for (int index = 0; index < module->GetInputNum(); ++index) {
    const std::string name = module->GetInputName(index);
    if (!IsCacheInput(name)) {
      inputs->emplace(name, tcim::Tensor::CreateHostTensor(
                                module->GetInputInfo(name).AsContiguous()));
    }
  }
}

void Qwen3TTSPrefillDecodeRuntime::ResetCaches() {
  for (const auto& name : cache_names_) {
    auto cache = prefill_module_->GetDevInput(name);
    CHECK_TCIM_RET_STATUS(cache.Buffer().MemSet(0, cache.MemSize(), 0));
  }
}

void Qwen3TTSPrefillDecodeRuntime::CopyInput(
    const Qwen3TTSHiddenSequence& input, tcim::Tensor* tensor) {
  input.Validate();
  if (input.data.size() * sizeof(float16) != tensor->MemSize()) {
    throw std::invalid_argument("Model input data size mismatch");
  }
  CHECK_TCIM_RET_STATUS(tensor->Buffer().CopyFromHost(
      input.data.data(), tensor->MemSize()));
}

void Qwen3TTSPrefillDecodeRuntime::SetInt32(
    tcim::Module* module, const std::string& name, tcim::Tensor* tensor,
    int32_t value) {
  CHECK_TCIM_RET_STATUS(
      tensor->Buffer().CopyFromHost(&value, sizeof(value)));
  CHECK_TCIM_RET_STATUS(module->SetInput(name, *tensor));
}

std::vector<float16> Qwen3TTSPrefillDecodeRuntime::CopyFp16Output(
    tcim::Module* module, const std::string& name) {
  const auto info = module->GetOutputInfo(name);
  if (info.DataType() != tcim::DataType::FLOAT16) {
    throw Exception("Model output is not FP16: " + name);
  }
  auto output = module->GetDevOutput(name).ToHost(true);
  const size_t elements = output.MemSize() / sizeof(float16);
  const auto* data = static_cast<const float16*>(output.Buffer().Data());
  return std::vector<float16>(data, data + elements);
}

std::vector<std::vector<float16>>
Qwen3TTSPrefillDecodeRuntime::FetchOutputs(
    tcim::Module* module,
    const std::vector<std::string>& output_names) const {
  std::vector<std::vector<float16>> outputs;
  outputs.reserve(output_names.size());
  for (const auto& name : output_names) {
    outputs.push_back(CopyFp16Output(module, name));
  }
  return outputs;
}

std::vector<std::vector<float16>>
Qwen3TTSPrefillDecodeRuntime::RunPrefill(
    const Qwen3TTSHiddenSequence& input, int32_t valid_length,
    int32_t current_length, int32_t generation_steps, bool fetch_outputs) {
  const std::string input_name = prefill_module_->GetInputName(0);
  CopyInput(input, &prefill_inputs_.at(input_name));
  CHECK_TCIM_RET_STATUS(
      prefill_module_->SetInput(input_name, prefill_inputs_.at(input_name)));
  SetInt32(prefill_module_.get(), "valid_length",
           &prefill_inputs_.at("valid_length"), valid_length);
  SetInt32(prefill_module_.get(), "current_length",
           &prefill_inputs_.at("current_length"), current_length);
  if (prefill_inputs_.count("generate_steps") != 0) {
    SetInt32(prefill_module_.get(), "generate_steps",
             &prefill_inputs_.at("generate_steps"), generation_steps);
  }
  CHECK_TCIM_RET_STATUS(prefill_module_->Run());
  CHECK_TCIM_RET_STATUS(prefill_module_->Sync());
  return fetch_outputs
             ? FetchOutputs(prefill_module_.get(), prefill_output_names_)
             : std::vector<std::vector<float16>>{};
}

std::vector<std::vector<float16>>
Qwen3TTSPrefillDecodeRuntime::RunDecode(
    const Qwen3TTSHiddenSequence& input, int32_t valid_length,
    int32_t generation_steps) {
  const std::string input_name = decode_module_->GetInputName(0);
  CopyInput(input, &decode_inputs_.at(input_name));
  CHECK_TCIM_RET_STATUS(
      decode_module_->SetInput(input_name, decode_inputs_.at(input_name)));
  SetInt32(decode_module_.get(), "valid_length",
           &decode_inputs_.at("valid_length"), valid_length);
  SetInt32(decode_module_.get(), "current_length",
           &decode_inputs_.at("current_length"), 1);
  if (decode_inputs_.count("generate_steps") != 0) {
    SetInt32(decode_module_.get(), "generate_steps",
             &decode_inputs_.at("generate_steps"), generation_steps);
  }
  CHECK_TCIM_RET_STATUS(decode_module_->Run());
  CHECK_TCIM_RET_STATUS(decode_module_->Sync());
  return FetchOutputs(decode_module_.get(), decode_output_names_);
}

}  // namespace houmo
