/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_prefill_decode_runtime.h
 * Description:
 *   Shared Qwen3-TTS prefill/decode HMM runtime interface.
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

#include "qwen3_tts_types.h"
#include "tcim/tcim_runtime.h"

namespace houmo {

class Qwen3TTSPrefillDecodeRuntime {
 public:
  Qwen3TTSPrefillDecodeRuntime(const std::string& prefill_path,
                               const std::string& decode_path,
                               int device_id = 0);

  void ResetCaches();

  std::vector<std::vector<float16>> RunPrefill(
      const Qwen3TTSHiddenSequence& input, int32_t valid_length,
      int32_t current_length, int32_t generation_steps = 0,
      bool fetch_outputs = true);

  std::vector<std::vector<float16>> RunDecode(
      const Qwen3TTSHiddenSequence& input, int32_t valid_length,
      int32_t generation_steps = 0);

  size_t prefill_length() const { return prefill_length_; }
  size_t hidden_dim() const { return hidden_dim_; }
  size_t context_length() const { return context_length_; }
  size_t prefill_output_count() const { return prefill_output_names_.size(); }
  size_t decode_output_count() const { return decode_output_names_.size(); }
  size_t cache_count() const { return cache_names_.size(); }

 private:
  static bool IsCacheInput(const std::string& name);
  static std::vector<float16> CopyFp16Output(tcim::Module* module,
                                             const std::string& name);
  static void CopyInput(const Qwen3TTSHiddenSequence& input,
                        tcim::Tensor* tensor);
  static void SetInt32(tcim::Module* module, const std::string& name,
                       tcim::Tensor* tensor, int32_t value);

  void InitializeHostInputs(tcim::Module* module,
                            std::unordered_map<std::string, tcim::Tensor>* inputs);
  std::vector<std::vector<float16>> FetchOutputs(
      tcim::Module* module, const std::vector<std::string>& output_names) const;

  tcim::Module::WeightManager weight_manager_;
  std::shared_ptr<tcim::Module> prefill_module_;
  std::shared_ptr<tcim::Module> decode_module_;
  std::vector<std::string> cache_names_;
  std::vector<std::string> prefill_output_names_;
  std::vector<std::string> decode_output_names_;
  std::unordered_map<std::string, tcim::Tensor> prefill_inputs_;
  std::unordered_map<std::string, tcim::Tensor> decode_inputs_;
  size_t prefill_length_ = 0;
  size_t hidden_dim_ = 0;
  size_t context_length_ = 0;
};

}  // namespace houmo
