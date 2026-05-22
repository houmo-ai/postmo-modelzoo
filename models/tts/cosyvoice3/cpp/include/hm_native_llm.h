/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_native_llm.h
 * Description:
 *   LLM module for CosyVoice3 TTS.
 *   Generates speech tokens from text using quantized LLM inference.
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
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "common_types.h"
#include "hm_embedding.h"
#include "hm_sampling.h"
#include "tcim_runtime_utils.h"

namespace houmo {
class HmNativeLLM {
 public:
  HmNativeLLM(const std::string& llm_decoder_path,
              const std::string& prefill_path, const std::string& decode_path,
              const std::string& embeddingWeightPath,
              const std::string& llm_speech_embedding_path,
              const std::string& sos_embedding_path,
              const std::string& task_id_embedding_path);
  ~HmNativeLLM();

  std::vector<int> Inference(const CosyVoice3FrontendInput& frontend_input,
                             CosyVoice3Perf* perf = nullptr);

 private:
  int get_nblocks();
  int get_attn_idx_start();

  tcim::Module::WeightManager llm_decoder_weight_manager_;
  std::shared_ptr<tcim::Module> llm_decoder_module_;
  std::unordered_map<std::string, tcim::Tensor> llm_decoder_input_maps_;

  tcim::Module::WeightManager prefill_weight_manager_;
  std::shared_ptr<tcim::Module> prefill_module_;
  std::unordered_map<std::string, tcim::Tensor> prefill_input_maps_;

  tcim::Module::WeightManager decode_weight_manager_;
  std::shared_ptr<tcim::Module> decode_module_;
  std::unordered_map<std::string, tcim::Tensor> decode_input_maps_;

  std::shared_ptr<HmEmbedding> quant_embedding;
  std::shared_ptr<HmEmbedding> llm_speech_embedding;
  std::shared_ptr<HmEmbedding> sos_embedding;
  std::shared_ptr<HmEmbedding> task_id_embedding;
  std::shared_ptr<HmSampling> sampling;
  std::vector<std::string> dummy_names;
  std::unique_ptr<TensorType[]> prefill_input_buffer;
  std::unique_ptr<TensorType[]> decode_input_buffer;
  int attn_idx_start_;
  int prefill_length;
  int embedding_length;
  int context_max_length;
  int batch;
  int argmax_dim_len;
  int llm_input_size;
  int context_length;

  int pad_token_id = 151645;
  std::vector<int> stop_token_ids;

  std::vector<int> silent_tokens = {1,   2,    28,   29,   55,  248,
                                    494, 2241, 2242, 2322, 2323};
};
}  // namespace houmo
