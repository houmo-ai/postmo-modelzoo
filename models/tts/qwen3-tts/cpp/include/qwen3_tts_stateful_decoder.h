/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_stateful_decoder.h
 * Description:
 *   Qwen3-TTS stateful streaming audio decoder interface.
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

#include "qwen3_tts_streaming_generator.h"
#include "tcim/tcim_runtime.h"

namespace houmo {

struct Qwen3TTSDecoderState {
  tcim::Tensor pre_conv_history;
  tcim::Tensor latent_buffer;
  tcim::Tensor conv_history;
  std::vector<tcim::Tensor> kv_cache;
  int32_t kv_valid_length = 0;
  size_t skip_samples = 0;
  std::vector<float> latent_audio;
};

struct Qwen3TTSDecoderOutput {
  std::vector<float> audio;
  Qwen3TTSDecoderState state;
};

class Qwen3TTSStatefulDecoder {
 public:
  static constexpr size_t kChunkSize = 12;
  static constexpr size_t kCodebookCount = 16;
  static constexpr size_t kSamplesPerFrame = 1920;

  explicit Qwen3TTSStatefulDecoder(const std::string& model_path,
                                   int device_id = 0);

  Qwen3TTSDecoderState CreateState();
  Qwen3TTSDecoderOutput Decode(const std::vector<Qwen3TTSCodecFrame>& frames,
                               Qwen3TTSDecoderState state, bool is_final);

 private:
  void SetScalar(const std::string& name, const void* value, size_t size);
  tcim::Tensor CreateZeroDeviceInput(size_t input_index);

  tcim::Module::WeightManager weight_manager_;
  std::shared_ptr<tcim::Module> module_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::unordered_map<std::string, tcim::Tensor> host_inputs_;
};

}  // namespace houmo
