/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_text_projection.h
 * Description:
 *   Qwen3-TTS TextProjection HMM inference interface.
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

#include "qwen3_tts_types.h"
#include "tcim/tcim_runtime.h"

namespace houmo {

class Qwen3TTSTextProjection {
 public:
  explicit Qwen3TTSTextProjection(const std::string& model_path,
                                  int device_id = 0);

  Qwen3TTSHiddenSequence Project(
      const Qwen3TTSHiddenSequence& input) const;

  size_t input_hidden_dim() const { return input_hidden_dim_; }
  size_t output_hidden_dim() const { return output_hidden_dim_; }
  size_t chunk_length() const { return chunk_length_; }

 private:
  tcim::Module::WeightManager weight_manager_;
  std::shared_ptr<tcim::Module> module_;
  std::string input_name_;
  std::string output_name_;
  tcim::Tensor input_tensor_;
  size_t chunk_length_ = 0;
  size_t input_hidden_dim_ = 0;
  size_t output_hidden_dim_ = 0;
};

}  // namespace houmo
