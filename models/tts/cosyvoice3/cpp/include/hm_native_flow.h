/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_native_flow.h
 * Description:
 *   Flow decoder module for CosyVoice3 TTS.
 *   Converts speech tokens to mel spectrogram using flow matching.
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
#include "hm_embedding.h"
#include "tcim_runtime_utils.h"

namespace houmo {
class HmNativeFlow {
 public:
  HmNativeFlow(const std::string& input_embedding_path,
               const std::string& pre_lookahead_layer_path,
               const std::string& spk_embed_affine_layer_path,
               const std::string& flow_decoder_path);
  ~HmNativeFlow();

  std::vector<float> Inference(const CosyVoice3FrontendInput& frontend_input,
                               const std::vector<int>& tts_speech_tokens,
                               CosyVoice3Perf* perf = nullptr);

 private:
  tcim::Module::WeightManager weight_manager_;
  std::shared_ptr<HmEmbedding> input_embedding_;
  std::shared_ptr<tcim::Module> pre_lookahead_layer_;
  std::shared_ptr<tcim::Module> spk_embed_affine_layer_;
  std::shared_ptr<tcim::Module> flow_decoder_;
  std::unordered_map<std::string, tcim::Tensor> pre_lookahead_input_maps_;
  std::unordered_map<std::string, tcim::Tensor> spk_embed_affine_input_maps_;
  std::unordered_map<std::string, tcim::Tensor> flow_decoder_input_maps_;
  int pre_lookahead_seq_len_ = 0;
  int input_embedding_dim_ = 0;
  int flow_mel_bins_ = 80;
  int token_mel_ratio_ = 2;
  int flow_steps_ = 10;
  float inference_cfg_rate_ = 0.7f;
};
}  // namespace houmo
