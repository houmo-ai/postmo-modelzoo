/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_native_llm.cc
 * Description:
 *   LLM module implementation for CosyVoice3 TTS.
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

#include "hm_native_llm.h"

#include <chrono>
#include <iostream>

namespace houmo {

HmNativeLLM::HmNativeLLM(const std::string& llm_decoder_path,
                         const std::string& prefill_path,
                         const std::string& decode_path,
                         const std::string& embeddingWeightPath,
                         const std::string& llm_speech_embedding_path,
                         const std::string& sos_embedding_path,
                         const std::string& task_id_embedding_path) {
  llm_decoder_weight_manager_ =
      tcim::Module::WeightManager::CreateWeightManager(0);
  prefill_weight_manager_ = tcim::Module::WeightManager::CreateWeightManager(0);
  decode_weight_manager_ = tcim::Module::WeightManager::CreateWeightManager(0);
  auto prefill_option = tcim::Module::Option(prefill_weight_manager_);
  auto decode_option = tcim::Module::Option(decode_weight_manager_);
  auto llm_decoder_option = tcim::Module::Option(llm_decoder_weight_manager_);
  llm_decoder_option.EnableHostLazyLoading(true);
  prefill_option.EnableHostLazyLoading(true);
  decode_option.EnableHostLazyLoading(true);
  llm_decoder_option.EnableIOLazyMode(true);
  prefill_option.EnableIOLazyMode(true);
  decode_option.EnableIOLazyMode(true);

  // llm_decoder
  llm_decoder_module_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_RET_STATUS(
      llm_decoder_module_->LoadModel(llm_decoder_path, llm_decoder_option));

  prefill_module_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_RET_STATUS(
      prefill_module_->LoadModel(prefill_path, prefill_option));
  // Get number of blocks in the model and create dummy names for cache inputs
  int n_blocks = get_nblocks();
  for (int i = 0; i < n_blocks; i++) {
    std::stringstream ss;
    ss << "model_layers_" << i << "_self_attn_kcache_input";
    dummy_names.emplace_back(ss.str());
  }

  for (int i = 0; i < n_blocks; i++) {
    std::stringstream ss;
    ss << "model_layers_" << i << "_self_attn_vcache_input";
    dummy_names.emplace_back(ss.str());
  }

  // Set dummy tensors for the decode module
  decode_option.SetDummyTensors(dummy_names);
  decode_module_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_RET_STATUS(decode_module_->LoadModel(decode_path, decode_option));

  attn_idx_start_ = get_attn_idx_start();
  this->prefill_length =
      prefill_module_->GetInputInfo(prefill_module_->GetInputName(0))
          .Shape()[1];
  this->embedding_length =
      prefill_module_->GetInputInfo(prefill_module_->GetInputName(0))
          .Shape()[2];
  this->context_max_length =
      prefill_module_
          ->GetInputInfo(prefill_module_->GetInputName(attn_idx_start_))
          .Shape()[2];
  this->batch =
      decode_module_->GetInputInfo(decode_module_->GetInputName(0)).Shape()[0];
  this->argmax_dim_len =
      llm_decoder_module_->GetOutputInfo(llm_decoder_module_->GetOutputName(0))
          .Shape()[1];
  this->llm_input_size = this->embedding_length;
  size_t total_mem_size = 0;
  for (int idx = attn_idx_start_; idx < 2 * n_blocks + attn_idx_start_; idx++) {
    auto prefill_name = prefill_module_->GetInputName(idx);
    auto decode_name = decode_module_->GetInputName(idx);
    auto cache = prefill_module_->GetDevInput(prefill_name);
    CHECK_TCIM_RET_STATUS(decode_module_->SetDevInput(decode_name, cache));
  }

  quant_embedding = std::make_shared<HmEmbedding>(
      embeddingWeightPath, this->embedding_length, this->prefill_length);
  int vocab_size = quant_embedding->get_vocab_size();
  llm_speech_embedding = std::make_shared<HmEmbedding>(
      llm_speech_embedding_path, this->embedding_length, 6761);
  sos_embedding = std::make_shared<HmEmbedding>(sos_embedding_path,
                                                this->embedding_length, 1);
  task_id_embedding = std::make_shared<HmEmbedding>(task_id_embedding_path,
                                                    this->embedding_length, 1);
  // init prefill input maps
  for (int idx = 0; idx < attn_idx_start_; idx++) {
    std::string input_name = prefill_module_->GetInputName(idx);
    auto info = prefill_module_->GetInputInfo(input_name).AsContiguous();
    prefill_input_maps_[input_name] = tcim::Tensor::CreateHostTensor(info);
  }
  prefill_input_buffer = std::make_unique<TensorType[]>(this->embedding_length *
                                                        this->prefill_length);
  // init decode input maps
  for (int idx = 0; idx < attn_idx_start_; idx++) {
    std::string input_name = decode_module_->GetInputName(idx);
    auto info = decode_module_->GetInputInfo(input_name).AsContiguous();
    decode_input_maps_[input_name] = tcim::Tensor::CreateHostTensor(info);
  }

  sampling = std::make_shared<HmSampling>();

  for (int tok_id = 0; tok_id < 200; tok_id++) {
    stop_token_ids.push_back(tok_id + 6561);
  }
}

int HmNativeLLM::get_nblocks() {
  int count = 0;
  static const std::regex pattern(
      R"(^model_layers_(\d+)_self_attn_kcache_input$)");
  int input_num = prefill_module_->GetInputNum();
  for (int idx = 0; idx < input_num; idx++) {
    std::string input_name = prefill_module_->GetInputName(idx);
    if (std::regex_match(input_name, pattern)) {
      ++count;
    }
  }
  return count;
}

int HmNativeLLM::get_attn_idx_start() {
  int start = 0;
  static const std::regex pattern(
      R"(^model_layers_(\d+)_self_attn_kcache_input$)");
  int input_num = prefill_module_->GetInputNum();
  for (int idx = 0; idx < input_num; idx++) {
    std::string input_name = prefill_module_->GetInputName(idx);
    if (std::regex_match(input_name, pattern)) {
      start = idx;
      break;
    }
  }
  return start;
}

HmNativeLLM::~HmNativeLLM() {}

std::vector<int> HmNativeLLM::Inference(
    const CosyVoice3FrontendInput& frontend_input, CosyVoice3Perf* perf) {
  auto start_time = std::chrono::high_resolution_clock::now();
  auto prefill_start_time = start_time;

  std::vector<int> tts_speech_tokens;
  tts_speech_tokens.clear();
  std::vector<int> text(
      frontend_input.prompt_text_len + frontend_input.text_len, 0);
  std::copy(frontend_input.prompt_text_tokens.begin(),
            frontend_input.prompt_text_tokens.begin() +
                frontend_input.prompt_text_len,
            text.begin());
  std::copy(frontend_input.text_tokens.begin(),
            frontend_input.text_tokens.begin() + frontend_input.text_len,
            text.begin() + frontend_input.prompt_text_len);

  TensorType* text_embedding = quant_embedding->EmbeddingTokens(text);
  TensorType* llm_prompt_speech_token_embedding;
  if (frontend_input.llm_prompt_speech_token_len != 0) {
    llm_prompt_speech_token_embedding = llm_speech_embedding->EmbeddingTokens(
        frontend_input.llm_prompt_speech_tokens);
  }
  int seq_length =
      1 + text.size() + 1 + frontend_input.llm_prompt_speech_token_len;
  TensorType* lm_input = new TensorType[this->embedding_length * seq_length];
  // [sos_emb, text_emb, task_id_emb, llm_prompt_speech_token_emb]
  std::copy(sos_embedding->get_embedding_ptr(),
            sos_embedding->get_embedding_ptr() + this->embedding_length,
            lm_input);
  std::copy(text_embedding,
            text_embedding + this->embedding_length * text.size(),
            lm_input + this->embedding_length);
  std::copy(task_id_embedding->get_embedding_ptr(),
            task_id_embedding->get_embedding_ptr() + this->embedding_length,
            lm_input + this->embedding_length * (1 + text.size()));
  std::copy(
      llm_prompt_speech_token_embedding,
      llm_prompt_speech_token_embedding +
          this->embedding_length * frontend_input.llm_prompt_speech_token_len,
      lm_input + this->embedding_length * (1 + text.size() + 1));

  // min_len max_len
  int min_len = frontend_input.text_len * 2;
  int max_len = frontend_input.text_len * 20;

  this->context_length = 0;

  if (seq_length > this->context_max_length) {
    throw std::runtime_error(
        "Input sequence length exceeds model's context window.");
  }

  int prefill_loop_round = std::ceil((float)seq_length / (float)prefill_length);
  int valid_length = 0;
  int current_length = 0;
  std::cout << prefill_loop_round
            << " rounds of prefill needed for total sequence length "
            << seq_length << "\n";
  for (int round_idx = 0; round_idx < prefill_loop_round; round_idx++) {
    valid_length = round_idx * prefill_length + this->context_length;
    std::fill(prefill_input_buffer.get(),
              prefill_input_buffer.get() +
                  this->embedding_length * this->prefill_length,
              0);
    if (round_idx == prefill_loop_round - 1) {
      current_length = seq_length - round_idx * prefill_length;
      std::copy(lm_input + round_idx * prefill_length * this->embedding_length,
                lm_input + seq_length * this->embedding_length,
                prefill_input_buffer.get());
    } else {
      current_length = this->prefill_length;
      std::copy(
          lm_input + round_idx * prefill_length * this->embedding_length,
          lm_input + (round_idx + 1) * prefill_length * this->embedding_length,
          prefill_input_buffer.get());
    }
    {
      CHECK_TCIM_RET_STATUS(
          prefill_input_maps_[prefill_module_->GetInputName(0)]
              .Buffer()
              .CopyFromHost(
                  prefill_input_buffer.get(),
                  prefill_input_maps_[prefill_module_->GetInputName(0)]
                      .MemSize()));
      CHECK_TCIM_RET_STATUS(
          prefill_input_maps_[prefill_module_->GetInputName(1)]
              .Buffer()
              .CopyFromHost(
                  &valid_length,
                  prefill_input_maps_[prefill_module_->GetInputName(1)]
                      .MemSize()));
      CHECK_TCIM_RET_STATUS(
          prefill_input_maps_[prefill_module_->GetInputName(2)]
              .Buffer()
              .CopyFromHost(
                  &current_length,
                  prefill_input_maps_[prefill_module_->GetInputName(2)]
                      .MemSize()));
      for (auto& it : prefill_input_maps_) {
        CHECK_TCIM_RET_STATUS(prefill_module_->SetInput(it.first, it.second));
      }

      // Run prefill module for the current chunk
      CHECK_TCIM_RET_STATUS(prefill_module_->Run());
      CHECK_TCIM_RET_STATUS(prefill_module_->Sync());
    }
  }

  // Record prefill time
  auto prefill_end_time = std::chrono::high_resolution_clock::now();
  float prefill_ms = std::chrono::duration<float, std::milli>(
                         prefill_end_time - prefill_start_time)
                         .count();
  int prefill_tokens = seq_length;

  auto output_name = prefill_module_->GetOutputName(0);
  auto dev_prefill_opt = prefill_module_->GetDevOutput(output_name);
  auto host_prefill_opt = dev_prefill_opt.ToHost(true);

  auto llm_decoder_input_name = llm_decoder_module_->GetInputName(0);
  auto llm_decoder_input_info =
      llm_decoder_module_->GetInputInfo(llm_decoder_input_name).AsContiguous();
  auto llm_decoder_input_tensor =
      tcim::Tensor::CreateHostTensor(llm_decoder_input_info);
  CHECK_TCIM_RET_STATUS(llm_decoder_input_tensor.Buffer().CopyFromHost(
      host_prefill_opt.Buffer().Data(), host_prefill_opt.MemSize()));
  // run llm decoder
  CHECK_TCIM_RET_STATUS(llm_decoder_module_->SetInput(
      llm_decoder_module_->GetInputName(0), llm_decoder_input_tensor));
  CHECK_TCIM_RET_STATUS(llm_decoder_module_->Run());
  CHECK_TCIM_RET_STATUS(llm_decoder_module_->Sync());
  auto llm_output_name = llm_decoder_module_->GetOutputName(0);
  auto llm_dev_output_opt = llm_decoder_module_->GetDevOutput(llm_output_name);
  auto llm_host_output_opt = llm_dev_output_opt.ToHost(true);

  constexpr int kSampling = 25;
  int top_ids = sampling->SamplingIds(
      llm_host_output_opt, tts_speech_tokens, this->argmax_dim_len, kSampling,
      static_cast<int>(tts_speech_tokens.size()) < min_len);

  if (std::find(stop_token_ids.begin(), stop_token_ids.end(), top_ids) !=
      stop_token_ids.end()) {
    delete[] lm_input;
    return tts_speech_tokens;
  }

  tts_speech_tokens.push_back(top_ids);
  this->context_length += seq_length;

  // Record TTFT (Time to First Token)
  auto ttft_time = std::chrono::high_resolution_clock::now();
  float ttft_ms =
      std::chrono::duration<float, std::milli>(ttft_time - start_time).count();

  // Start decode timing
  auto decode_start_time = std::chrono::high_resolution_clock::now();

  int decode_token_cnt = 0;
  TensorType* decode_input_buffer = nullptr;
  int decode_current_length = 1;
  std::vector<int> decode_ids;
  for (int i = 1; i < max_len; ++i) {
    decode_ids.clear();
    decode_ids.push_back(top_ids);
    decode_input_buffer = llm_speech_embedding->EmbeddingTokens(decode_ids);
    CHECK_TCIM_RET_STATUS(
        decode_input_maps_[decode_module_->GetInputName(0)]
            .Buffer()
            .CopyFromHost(
                decode_input_buffer,
                decode_input_maps_[decode_module_->GetInputName(0)].MemSize()));
    CHECK_TCIM_RET_STATUS(decode_module_->SetInput(
        decode_module_->GetInputName(0),
        decode_input_maps_[decode_module_->GetInputName(0)]));
    CHECK_TCIM_RET_STATUS(
        decode_input_maps_[decode_module_->GetInputName(1)]
            .Buffer()
            .CopyFromHost(
                &this->context_length,
                decode_input_maps_[decode_module_->GetInputName(1)].MemSize()));
    CHECK_TCIM_RET_STATUS(decode_module_->SetInput(
        decode_module_->GetInputName(1),
        decode_input_maps_[decode_module_->GetInputName(1)]));
    CHECK_TCIM_RET_STATUS(
        decode_input_maps_[decode_module_->GetInputName(2)]
            .Buffer()
            .CopyFromHost(
                &decode_current_length,
                decode_input_maps_[decode_module_->GetInputName(2)].MemSize()));
    CHECK_TCIM_RET_STATUS(decode_module_->SetInput(
        decode_module_->GetInputName(2),
        decode_input_maps_[decode_module_->GetInputName(2)]));

    CHECK_TCIM_RET_STATUS(decode_module_->Run());
    CHECK_TCIM_RET_STATUS(decode_module_->Sync());
    auto decode_output_name = decode_module_->GetOutputName(0);
    auto decode_dev_output_opt =
        decode_module_->GetDevOutput(decode_output_name);
    auto decode_host_output_opt = decode_dev_output_opt.ToHost(true);

    // run llm_decoder
    CHECK_TCIM_RET_STATUS(llm_decoder_input_tensor.Buffer().CopyFromHost(
        decode_host_output_opt.Buffer().Data(),
        decode_host_output_opt.MemSize()));

    CHECK_TCIM_RET_STATUS(llm_decoder_module_->SetInput(
        llm_decoder_module_->GetInputName(0), llm_decoder_input_tensor));
    CHECK_TCIM_RET_STATUS(llm_decoder_module_->Run());
    CHECK_TCIM_RET_STATUS(llm_decoder_module_->Sync());
    auto llm_decode_output_name = llm_decoder_module_->GetOutputName(0);
    auto llm_decode_dev_output_opt =
        llm_decoder_module_->GetDevOutput(llm_decode_output_name);
    auto llm_decode_host_output_opt = llm_decode_dev_output_opt.ToHost(true);
    top_ids =
        sampling->SamplingIds(llm_decode_host_output_opt, tts_speech_tokens,
                              this->argmax_dim_len, kSampling, i < min_len);
    this->context_length += 1;
    if (std::find(stop_token_ids.begin(), stop_token_ids.end(), top_ids) !=
        stop_token_ids.end()) {
      break;
    }
    tts_speech_tokens.push_back(top_ids);
    decode_token_cnt++;
  }

  int max_silent_token_num = 5;
  std::vector<int> result_tts_speech_tokens;
  int cur_silent_token_num = 0;
  for (int i = 0; i < tts_speech_tokens.size(); i++) {
    int cur_token_id = tts_speech_tokens[i];
    if (std::find(silent_tokens.begin(), silent_tokens.end(), cur_token_id) !=
        silent_tokens.end()) {
      cur_silent_token_num++;
      if (cur_silent_token_num > max_silent_token_num) {
        continue;
      }
    } else {
      cur_silent_token_num = 0;
    }
    result_tts_speech_tokens.emplace_back(cur_token_id);
  }
  if (perf) {
    auto end_time = std::chrono::high_resolution_clock::now();
    perf->llm_total_ms +=
        std::chrono::duration<float, std::milli>(end_time - start_time).count();
    perf->prefill_ms += prefill_ms;
    perf->prefill_tokens += prefill_tokens;
    perf->ttft_ms += ttft_ms;
    perf->decode_ms +=
        std::chrono::duration<float, std::milli>(end_time - decode_start_time)
            .count();
    perf->decode_tokens += result_tts_speech_tokens.size();
  }
  return result_tts_speech_tokens;
}
}  // namespace houmo
