/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_llm_model.cc
 * Description:
 *   Qwen3 LLM model implementation
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

#include "models/qwen3_llm_model.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <regex>
#include <sstream>

#include "base/tcim_utils.h"
#include "core/model_factory.h"

namespace fs = std::filesystem;

namespace houmo {

// ============================================================================
// Qwen3Context Implementation
// ============================================================================

Qwen3Context::Qwen3Context(LLMModel* model, int n_ctx) : Context(model, n_ctx) {
  std::cout << "Qwen3Context created with n_ctx=" << n_ctx << std::endl;
}

Token Qwen3Context::prefill(const std::vector<Token>& tokens) {
  // If sampler is not set, use default greedy
  if (!sampler_) {
    set_sampler(SamplingParams{});  // default top_k=1 (greedy)
  }
  generated_ids_.clear();
  Token token = do_prefill_inference(tokens, sampler_.get());
  generated_ids_.push_back(token);
  // context_length_ already updated in do_prefill_inference
  return token;
}

Token Qwen3Context::decode(Token prev_token) {
  // If sampler is not set, use default greedy
  if (!sampler_) {
    set_sampler(SamplingParams{});  // default top_k=1 (greedy)
  }
  Token token = do_decode_inference(prev_token, sampler_.get());
  generated_ids_.push_back(token);
  // context_length_ already updated in do_decode_inference
  return token;
}

void Qwen3Context::generate(const std::vector<Token>& prompt,
                            const SamplingParams& params,
                            std::function<bool(Token)> callback) {
  profiler_.reset();  // Auto-reset single-run statistics
  auto& p = profiler_;

  // Start E2E timing
  p.start("generate");
  p.set_input_tokens(static_cast<int>(prompt.size()));

  // Set sampler (created once)
  set_sampler(params);

  // Prefill stage
  Token token;
  {
    auto t = p.scope("generate.prefill");
    token = prefill(prompt);
  }

  // Record TTFT
  p.record_ttft();

  // Check if first token is EOS
  if (token == model_->eos_token_id() || token == model_->bos_token_id()) {
    p.stop("generate");
    perf_stats_ = p.to_perf_stats();
    return;
  }

  // Output first token
  if (!callback(token)) {
    p.stop("generate");
    perf_stats_ = p.to_perf_stats();
    return;
  }

  // Decode stage
  while (true) {
    if (context_length_ >= model_->max_ctx_available()) {
      std::cerr << "Reached maximum context length of the model." << std::endl;
      break;
    }

    if (params.max_tokens > 0 &&
        generated_ids_.size() >= static_cast<size_t>(params.max_tokens)) {
      break;
    }

    {
      auto t = p.scope("generate.decode");
      token = decode(token);
    }
    p.add_output_token();

    // Check EOS (after generation, don't output stop token)
    if (token == model_->eos_token_id() || token == model_->bos_token_id()) {
      break;
    }

    if (!callback(token)) break;
  }

  p.stop("generate");
  perf_stats_ = p.to_perf_stats();
}

void Qwen3Context::reset() {
  // Call base class reset to reset counters
  Context::reset();
}

void Qwen3Context::prefill_preprocess_chunk(int chunk,
                                            const std::vector<Token>& tokens,
                                            int32_t seq_length,
                                            int prefill_length) {
  auto* model = static_cast<LLMModel*>(model_);

  int32_t valid_length = chunk * prefill_length + context_length_;
  int start = chunk * prefill_length;
  int end =
      std::min((chunk + 1) * prefill_length, static_cast<int>(seq_length));
  int32_t current_length = end - start;

  std::vector<Token> input_ids(tokens.begin() + start, tokens.begin() + end);

  // If current chunk input is less than prefill_length, padding is needed
  if (input_ids.size() < static_cast<size_t>(prefill_length)) {
    input_ids.resize(prefill_length,
                     model_->tokenizer_module()->pad_token_id());
  }

  // Get embedding
  const float16* embed_data = model->embedding()->token_embedding(input_ids);

  // Set prefill inputs
  auto& input_map = model->prefill_input_map();
  auto prefill_module = model->prefill_module();
  const int attn_idx_start = model->attn_idx_start();

  for (int idx = 0; idx < attn_idx_start; idx++) {
    const std::string& input_name = prefill_module->GetInputName(idx);
    auto& tensor = input_map[input_name];
    size_t mem_size = tensor.MemSize();

    if (input_name.find("input_1") != std::string::npos) {
      tensor.Buffer().CopyFromHost(embed_data, mem_size);
    } else if (input_name.find("valid_length") != std::string::npos) {
      tensor.Buffer().CopyFromHost(&valid_length, mem_size);
    } else if (input_name.find("current_length") != std::string::npos) {
      tensor.Buffer().CopyFromHost(&current_length, mem_size);
    }
    prefill_module->SetInput(input_name, tensor);
  }
}

void Qwen3Context::prefill_inference_chunk() {
  auto* model = static_cast<LLMModel*>(model_);
  auto prefill_module = model->prefill_module();
  prefill_module->Run();
  prefill_module->Sync();
}

Token Qwen3Context::prefill_postprocess(Sampler* sampler, int32_t seq_length) {
  auto* model = static_cast<LLMModel*>(model_);
  auto prefill_module = model->prefill_module();

  // Get output
  const std::string& output_name = prefill_module->GetOutputName(0);
  auto dev_output = prefill_module->GetDevOutput(output_name);
  auto host_output = dev_output.ToHost(true);
  void* out_data = host_output.Buffer().Data();

  // Sample using Sampler
  const int vocab_size = model->vocab_size();
  float16* logits = static_cast<float16*>(out_data);
  Token sampled_token = sampler->sample(logits, vocab_size, generated_ids_);

  // Update context_length_
  context_length_ += seq_length;

  return sampled_token;
}

Token Qwen3Context::do_prefill_inference(const std::vector<Token>& tokens,
                                         Sampler* sampler) {
  auto& p = profiler_;
  auto* model = static_cast<LLMModel*>(model_);
  const int32_t seq_length = static_cast<int32_t>(tokens.size());
  const int prefill_length = model->prefill_length();
  const int prefill_loop_chunk =
      (seq_length + prefill_length - 1) / prefill_length;

  for (int chunk = 0; chunk < prefill_loop_chunk; chunk++) {
    {
      auto t = p.scope("generate.prefill.preprocess_chunk");
      prefill_preprocess_chunk(chunk, tokens, seq_length, prefill_length);
    }
    {
      auto t = p.scope("generate.prefill.inference_chunk");
      prefill_inference_chunk();
    }
  }

  Token sampled_token;
  {
    auto t = p.scope("generate.prefill.postprocess");
    sampled_token = prefill_postprocess(sampler, seq_length);
  }

  return sampled_token;
}

void Qwen3Context::decode_preprocess(Token prev_token) {
  auto* model = static_cast<LLMModel*>(model_);

  // Get embedding
  std::vector<Token> input_ids = {prev_token};
  const float16* embed_data = model->embedding()->token_embedding(input_ids);

  // Set decode inputs
  auto& input_map = model->decode_input_map();
  auto decode_module = model->decode_module();
  const int attn_idx_start = model->attn_idx_start();

  int32_t current_length = 1;

  for (int idx = 0; idx < attn_idx_start; idx++) {
    const std::string& input_name = decode_module->GetInputName(idx);
    auto& tensor = input_map[input_name];
    size_t mem_size = tensor.MemSize();

    if (input_name.find("input_1") != std::string::npos) {
      tensor.Buffer().CopyFromHost(embed_data, mem_size);
    } else if (input_name.find("valid_length") != std::string::npos) {
      tensor.Buffer().CopyFromHost(&context_length_, mem_size);
    } else if (input_name.find("current_length") != std::string::npos) {
      tensor.Buffer().CopyFromHost(&current_length, mem_size);
    }
    decode_module->SetInput(input_name, tensor);
  }
}

void Qwen3Context::decode_inference() {
  auto* model = static_cast<LLMModel*>(model_);
  auto decode_module = model->decode_module();
  decode_module->Run();
  decode_module->Sync();
}

Token Qwen3Context::decode_postprocess(Sampler* sampler) {
  auto* model = static_cast<LLMModel*>(model_);
  auto decode_module = model->decode_module();

  // Get output
  const std::string& output_name = decode_module->GetOutputName(0);
  auto dev_output = decode_module->GetDevOutput(output_name);
  auto host_output = dev_output.ToHost(true);
  void* out_data = host_output.Buffer().Data();

  // Sample using Sampler
  const int vocab_size = model->vocab_size();
  float16* logits = static_cast<float16*>(out_data);
  Token sampled_token = sampler->sample(logits, vocab_size, generated_ids_);

  // Update context_length_
  context_length_++;

  return sampled_token;
}

Token Qwen3Context::do_decode_inference(Token prev_token, Sampler* sampler) {
  auto& p = profiler_;

  {
    auto t = p.scope("generate.decode.preprocess");
    decode_preprocess(prev_token);
  }
  {
    auto t = p.scope("generate.decode.inference");
    decode_inference();
  }
  Token sampled_token;
  {
    auto t = p.scope("generate.decode.postprocess");
    sampled_token = decode_postprocess(sampler);
  }

  return sampled_token;
}

// ============================================================================
// Qwen3LLMModel Implementation
// ============================================================================

Qwen3LLMModel::Qwen3LLMModel(const ModelConfig& config) : LLMModel(config) {
  load();
}

std::unique_ptr<Context> Qwen3LLMModel::create_context(int n_ctx) {
  if (n_ctx <= 0) {
    n_ctx = info_.n_ctx;
  }
  return std::make_unique<Qwen3Context>(this, n_ctx);
}

void Qwen3LLMModel::load() {
  // Step 1 - Initialize device manager
  dev_manager_ = std::make_unique<tcim::DevManager>(
      tcim::DevManager::Create(config_.devices));
  weight_manager_ = std::make_unique<tcim::Module::WeightManager>(
      tcim::Module::WeightManager::CreateWeightManager(*dev_manager_));

  // Step 2 - Load prefill model
  {
    const std::string& prefill_path = config_.prefill_path;

    auto option_prefill = tcim::Module::Option(*weight_manager_);
    option_prefill.EnableIOLazyMode(true);
    option_prefill.EnableHostLazyLoading(config_.lazy_mode);

    prefill_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(
        prefill_module_->LoadModel(prefill_path, option_prefill));
    std::cout << "Prefill model loaded: " << prefill_path << std::endl;

    // Parse prefill info
    auto input0_shape =
        prefill_module_->GetInputInfo(prefill_module_->GetInputName(0)).Shape();

    if (input0_shape.size() >= 3) {
      batch_ = input0_shape[0];
      prefill_length_ = input0_shape[1];
      embedding_length_ = input0_shape[2];
    }

    // Get n_blocks
    std::regex pattern("model_layers_(\\d+)_self_attn_kcache_input");
    for (int i = 0; i < prefill_module_->GetInputNum(); i++) {
      std::string name = prefill_module_->GetInputName(i);
      std::smatch match;
      if (std::regex_search(name, match, pattern)) {
        int layer_idx = std::stoi(match[1].str());
        n_blocks_ = std::max(n_blocks_, layer_idx + 1);
      }
    }

    // Get attn_idx_start
    for (int i = 0; i < prefill_module_->GetInputNum(); i++) {
      std::string name = prefill_module_->GetInputName(i);
      if (name.find("kcache_input") != std::string::npos ||
          name.find("vcache_input") != std::string::npos) {
        attn_idx_start_ = i;
        break;
      }
    }

    // Get context_max_length
    if (attn_idx_start_ > 0 &&
        attn_idx_start_ < prefill_module_->GetInputNum()) {
      auto attn_shape =
          prefill_module_
              ->GetInputInfo(prefill_module_->GetInputName(attn_idx_start_))
              .Shape();
      if (attn_shape.size() >= 3) {
        context_max_length_ = attn_shape[2];
      }
    }

    std::cout << "  batch: " << batch_ << std::endl;
    std::cout << "  prefill_length: " << prefill_length_ << std::endl;
    std::cout << "  embedding_length: " << embedding_length_ << std::endl;
    std::cout << "  context_max_length: " << context_max_length_ << std::endl;
    std::cout << "  n_blocks: " << n_blocks_ << std::endl;
  }

  // Step 3 - Load decode model
  {
    const std::string& decode_path = config_.decode_path;

    // Generate dummy names for KV cache
    std::vector<std::string> dummy_names;
    for (int i = 0; i < n_blocks_; i++) {
      std::stringstream ss;
      ss << "model_layers_" << i << "_self_attn_kcache_input";
      dummy_names.emplace_back(ss.str());
    }
    for (int i = 0; i < n_blocks_; i++) {
      std::stringstream ss;
      ss << "model_layers_" << i << "_self_attn_vcache_input";
      dummy_names.emplace_back(ss.str());
    }

    auto option_decode = tcim::Module::Option(*weight_manager_);
    option_decode.SetDummyTensors(dummy_names);
    option_decode.EnableIOLazyMode(true);
    option_decode.EnableHostLazyLoading(config_.lazy_mode);

    decode_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(
        decode_module_->LoadModel(decode_path, option_decode));
    std::cout << "Decode model loaded: " << decode_path << std::endl;

    // Parse vocab_size
    auto output0_shape =
        decode_module_->GetOutputInfo(decode_module_->GetOutputName(0)).Shape();
    if (output0_shape.size() >= 3) {
      info_.n_vocab = output0_shape[2];
      info_.n_logits = output0_shape[2];
    }
    std::cout << "  vocab_size: " << info_.n_vocab << std::endl;
  }

  // Step 4 - Share KV Cache
  {
    if (!prefill_module_ || !decode_module_) return;

    for (int idx = 0; idx < prefill_module_->GetInputNum(); idx++) {
      const std::string layer_name = prefill_module_->GetInputName(idx);

      // Share model_layers / past_key_cache / past_value_cache
      if (layer_name.find("model_layers") != std::string::npos) {
        auto cache = prefill_module_->GetDevInput(layer_name);
        CHECK_TCIM_RET_STATUS(decode_module_->SetDevInput(layer_name, cache));
      }
    }
    std::cout << "KV Cache shared" << std::endl;
  }

  // Step 5 - Load Embedding
  {
    const std::string& embedding_path = config_.embedding_path;
    embedding_ = std::make_shared<Embedding>(embedding_path, embedding_length_,
                                             prefill_length_);
    std::cout << "Embedding loaded: vocab_size=" << embedding_->vocab_size()
              << std::endl;
  }

  // Step 6 - Initialize input tensors
  {
    prefill_input_map_.clear();
    for (int idx = 0; idx < attn_idx_start_; ++idx) {
      auto input_name = prefill_module_->GetInputName(idx);
      auto input_info =
          prefill_module_->GetInputInfo(input_name).AsContiguous();
      tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
      prefill_input_map_[input_name] = input_tensor;
    }

    decode_input_map_.clear();
    for (int idx = 0; idx < attn_idx_start_; ++idx) {
      auto input_name = decode_module_->GetInputName(idx);
      auto input_info = decode_module_->GetInputInfo(input_name).AsContiguous();
      tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
      decode_input_map_[input_name] = input_tensor;
    }
    std::cout << "Input tensors initialized" << std::endl;
  }

  // Step 7 - Fill model info
  {
    info_.type = ModelType::LLM;
    info_.n_batch = batch_;
    info_.n_embd = embedding_length_;
    info_.n_layer = n_blocks_;
    info_.n_ctx = context_max_length_;
    info_.prefill_length = prefill_length_;
    info_.kv_cache_layers = n_blocks_;

    // Extract model name from prefill path
    size_t last_slash = config_.prefill_path.find_last_of('/');
    if (last_slash != std::string::npos) {
      std::string dir_path = config_.prefill_path.substr(0, last_slash);
      size_t second_last_slash = dir_path.find_last_of('/');
      if (second_last_slash != std::string::npos) {
        info_.model_name = dir_path.substr(second_last_slash + 1);
      } else {
        info_.model_name = dir_path;
      }
    } else {
      info_.model_name = "unknown";
    }
    std::cout << "Model info filled: " << info_.model_name << std::endl;
  }

  // Step 8 - Load Tokenizer
  {
    if (fs::exists(config_.tokenizer_path)) {
      try {
        tokenizer_ = std::make_shared<HfTokenizer>(config_.tokenizer_path);
        std::cout << "Tokenizer loaded from: " << config_.tokenizer_path
                  << std::endl;
      } catch (const Exception& e) {
        std::cerr << "Warning: Failed to load tokenizer from "
                  << config_.tokenizer_path << ": " << e.what() << std::endl;
      }
    } else {
      std::cerr << "Warning: Tokenizer path does not exist: "
                << config_.tokenizer_path << std::endl;
    }
  }
}

// ============================================================================
// Model Registration
// ============================================================================

// Static registration for Qwen3 LLM model
REGISTER_LLM_MODEL(
    qwen3_llm, ModelSeries::kQwen3LLM,
    [](const ModelConfig& c) { return std::make_unique<Qwen3LLMModel>(c); },
    "Qwen3 text-only LLM");

}  // namespace houmo
