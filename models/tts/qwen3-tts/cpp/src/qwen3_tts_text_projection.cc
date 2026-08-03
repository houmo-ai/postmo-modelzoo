/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_text_projection.cc
 * Description:
 *   Qwen3-TTS TextProjection HMM inference implementation.
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

#include "qwen3_tts_text_projection.h"

#include <filesystem>
#include <stdexcept>

#include "base/tcim_utils.h"

namespace houmo {

Qwen3TTSTextProjection::Qwen3TTSTextProjection(
    const std::string& model_path, int device_id)
    : weight_manager_(
          tcim::Module::WeightManager::CreateWeightManager(device_id)) {
  if (!std::filesystem::exists(model_path)) {
    throw Exception("TextProjection model not found: " + model_path);
  }

  module_ = std::make_shared<tcim::Module>();
  auto option = tcim::Module::Option(weight_manager_);
  CHECK_TCIM_RET_STATUS(module_->LoadModel(model_path, option));
  if (module_->GetInputNum() != 1 || module_->GetOutputNum() != 1) {
    throw Exception("TextProjection must have exactly one input and one output");
  }

  input_name_ = module_->GetInputName(0);
  output_name_ = module_->GetOutputName(0);
  const auto input_info = module_->GetInputInfo(input_name_).AsContiguous();
  const auto output_info = module_->GetOutputInfo(output_name_).AsContiguous();
  const auto input_shape = input_info.Shape();
  const auto output_shape = output_info.Shape();
  if (input_shape.size() != 3 || output_shape.size() != 3 ||
      input_shape[0] != 1 || output_shape[0] != 1 ||
      input_shape[1] != output_shape[1]) {
    throw Exception("Unexpected TextProjection input/output shape");
  }
  if (input_info.DataType() != tcim::DataType::FLOAT16 ||
      output_info.DataType() != tcim::DataType::FLOAT16) {
    throw Exception("TextProjection input/output must use FP16");
  }

  chunk_length_ = static_cast<size_t>(input_shape[1]);
  input_hidden_dim_ = static_cast<size_t>(input_shape[2]);
  output_hidden_dim_ = static_cast<size_t>(output_shape[2]);
  if (chunk_length_ == 0 || input_hidden_dim_ == 0 ||
      output_hidden_dim_ == 0) {
    throw Exception("TextProjection has an empty dimension");
  }
  input_tensor_ = tcim::Tensor::CreateHostTensor(input_info);
}

Qwen3TTSHiddenSequence Qwen3TTSTextProjection::Project(
    const Qwen3TTSHiddenSequence& input) const {
  input.Validate();
  if (input.hidden_dim != input_hidden_dim_) {
    throw std::invalid_argument("TextProjection input hidden dimension mismatch");
  }
  if (input.sequence_length % chunk_length_ != 0) {
    throw std::invalid_argument(
        "TextProjection sequence length is not divisible by model chunk length");
  }

  Qwen3TTSHiddenSequence output;
  output.sequence_length = input.sequence_length;
  output.hidden_dim = output_hidden_dim_;
  output.data.reserve(output.sequence_length * output.hidden_dim);

  const size_t input_chunk_elements = chunk_length_ * input_hidden_dim_;
  const size_t output_chunk_elements = chunk_length_ * output_hidden_dim_;
  for (size_t offset = 0; offset < input.data.size();
       offset += input_chunk_elements) {
    CHECK_TCIM_RET_STATUS(input_tensor_.Buffer().CopyFromHost(
        input.data.data() + offset, input_tensor_.MemSize()));
    CHECK_TCIM_RET_STATUS(module_->SetInput(input_name_, input_tensor_));
    CHECK_TCIM_RET_STATUS(module_->Run());
    CHECK_TCIM_RET_STATUS(module_->Sync());

    auto host_output = module_->GetDevOutput(output_name_).ToHost(true);
    if (host_output.MemSize() != output_chunk_elements * sizeof(float16)) {
      throw Exception("TextProjection output size mismatch");
    }
    const auto* output_data =
        static_cast<const float16*>(host_output.Buffer().Data());
    output.data.insert(output.data.end(), output_data,
                       output_data + output_chunk_elements);
  }
  output.Validate();
  return output;
}

}  // namespace houmo
