/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_vlm_model.cc
 * Description:
 *   Qwen3-VL vision-language model implementation
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

#include "models/qwen3_vlm_model.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

#include "base/tcim_utils.h"
#include "core/model_factory.h"

namespace fs = std::filesystem;

namespace houmo {

// ============================================================================
// Qwen3VLMContext Implementation
// ============================================================================

Qwen3VLMContext::Qwen3VLMContext(LLMModel* model, int n_ctx)
    : Context(model, n_ctx) {
  auto* qwen_model = static_cast<Qwen3VLMModel*>(model_);
  image_processor_ = std::make_shared<HmImageProcessor>(
      qwen_model->vision_image_size_w(), qwen_model->vision_image_size_h(),
      true);  // use_v1 = true
}

void Qwen3VLMContext::set_image(const std::string& image_path) {
  image_paths_.clear();
  image_paths_.push_back(image_path);
}

void Qwen3VLMContext::set_images(const std::vector<std::string>& image_paths) {
  image_paths_ = image_paths;
}

Token Qwen3VLMContext::prefill(const std::vector<Token>& tokens) {
  if (!sampler_) {
    set_sampler(SamplingParams{});
  }
  generated_ids_.clear();
  use_vlm_ = !image_paths_.empty();

  Token token = do_prefill_inference(tokens, sampler_.get());
  generated_ids_.push_back(token);
  return token;
}

Token Qwen3VLMContext::decode(Token prev_token) {
  if (!sampler_) {
    set_sampler(SamplingParams{});
  }
  Token token = do_decode_inference(prev_token, sampler_.get());
  generated_ids_.push_back(token);
  // context_length_ and past_seq_len_ are updated in do_decode_inference
  return token;
}

void Qwen3VLMContext::generate(const std::vector<Token>& prompt,
                               const SamplingParams& params,
                               std::function<bool(Token)> callback) {
  profiler_.reset();  // Auto-reset single-run statistics
  auto& p = profiler_;

  // Start E2E timing
  p.start("generate");
  p.set_input_tokens(static_cast<int>(prompt.size()));

  set_sampler(params);

  Token token;
  {
    auto t = p.scope("generate.prefill");
    token = prefill(prompt);
  }

  p.record_ttft();

  if (token == model_->eos_token_id() || token == model_->bos_token_id()) {
    p.stop("generate");
    perf_stats_ = p.to_perf_stats();
    return;
  }

  if (!callback(token)) {
    p.stop("generate");
    perf_stats_ = p.to_perf_stats();
    return;
  }

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

    if (token == model_->eos_token_id() || token == model_->bos_token_id()) {
      break;
    }

    if (!callback(token)) break;
  }

  p.stop("generate");
  perf_stats_ = p.to_perf_stats();
}

void Qwen3VLMContext::reset() {
  Context::reset();
  image_paths_.clear();
  flat_image_embeds_.clear();
  deepstack_0_.clear();
  deepstack_1_.clear();
  deepstack_2_.clear();
  image_grid_thw_.clear();
  rope_deltas_ = 0;
  past_seq_len_ = 0;
  use_vlm_ = false;

  // Clear KV Cache
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  model->ClearKVCache();
}

std::tuple<std::vector<float16>, std::vector<float16>, std::vector<float16>,
           std::vector<float16>>
Qwen3VLMContext::run_vision(const std::vector<float16>& pixel_values) {
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  return model->encode_image(pixel_values);
}

void Qwen3VLMContext::run_vision() {
  if (image_paths_.empty()) return;

  auto& p = profiler_;

  flat_image_embeds_.clear();
  deepstack_0_.clear();
  deepstack_1_.clear();
  deepstack_2_.clear();
  image_grid_thw_.clear();

  for (size_t i = 0; i < image_paths_.size(); i++) {
    {
      auto t = p.scope("generate.vision.preprocess");
      vision_preprocess(i);
    }
    {
      auto t = p.scope("generate.vision.inference");
      vision_inference();
    }
    {
      auto t = p.scope("generate.vision.postprocess");
      vision_postprocess(i);
    }
  }
}

std::tuple<std::vector<std::vector<int32_t>>, int32_t, int>
Qwen3VLMContext::prefill_common_setup(const std::vector<Token>& tokens) {
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  int prefill_length = model->prefill_length();
  int embedding_length = model->embedding_dim();

  // 1. Expand image tokens
  std::vector<Token> input_ids = expand_image_tokens(tokens);

  // 2. Compute position IDs
  int32_t initial_context_length = past_seq_len_;

  std::vector<int32_t> time_pos;
  std::vector<int32_t> height_pos;
  std::vector<int32_t> width_pos;

  if (initial_context_length > 0) {
    // Multi-turn dialogue: use simple 1D position IDs
    int32_t delta = initial_context_length + rope_deltas_;
    time_pos.resize(input_ids.size());
    height_pos.resize(input_ids.size());
    width_pos.resize(input_ids.size());
    for (size_t i = 0; i < input_ids.size(); i++) {
      time_pos[i] = delta + i;
      height_pos[i] = delta + i;
      width_pos[i] = delta + i;
    }
  } else {
    // First turn: use 3D position IDs
    auto [position_ids, rope_delta] =
        get_rope_index(input_ids, image_grid_thw_);
    rope_deltas_ = rope_delta;
    time_pos = position_ids[0];
    height_pos = position_ids[1];
    width_pos = position_ids[2];
  }

  // 3. Padding to prefill_length
  int32_t seq_length = input_ids.size();
  int32_t input_seq_length =
      ((seq_length + prefill_length - 1) / prefill_length) * prefill_length;

  if (input_seq_length > seq_length) {
    input_ids.resize(input_seq_length, model->tokenizer()->pad_token_id());
  }

  // 4. Get embedding and store to temporary state
  chunk_embeds_.resize(input_seq_length * embedding_length);
  for (int chunk_start = 0; chunk_start < input_seq_length;
       chunk_start += prefill_length) {
    int chunk_end = std::min(chunk_start + prefill_length, input_seq_length);
    int current_chunk_size = chunk_end - chunk_start;

    std::vector<Token> chunk_tokens(input_ids.begin() + chunk_start,
                                    input_ids.begin() + chunk_end);
    const float16* chunk_embeds =
        model->embedding()->token_embedding(chunk_tokens);
    std::memcpy(chunk_embeds_.data() + chunk_start * embedding_length,
                chunk_embeds,
                current_chunk_size * embedding_length * sizeof(float16));
  }

  // 5. Replace image embedding
  scatter_image_embeds(chunk_embeds_, input_ids, input_seq_length,
                       embedding_length);

  // 6. Extend position_ids to input_seq_length
  if (input_seq_length > seq_length) {
    time_pos.resize(input_seq_length);
    height_pos.resize(input_seq_length);
    width_pos.resize(input_seq_length);
    for (size_t i = seq_length; i < static_cast<size_t>(input_seq_length);
         i++) {
      time_pos[i] = time_pos[i - 1] + 1;
      height_pos[i] = height_pos[i - 1] + 1;
      width_pos[i] = width_pos[i - 1] + 1;
    }
  }

  // 7. Prepare deepstack (scatter to full sequence length, reuse
  // deepstack_0/1/2 as temporary storage) Save original deepstack data first
  auto orig_ds0 = deepstack_0_;
  auto orig_ds1 = deepstack_1_;
  auto orig_ds2 = deepstack_2_;

  deepstack_0_.assign(input_seq_length * embedding_length,
                      static_cast<float16>(0.0f));
  deepstack_1_.assign(input_seq_length * embedding_length,
                      static_cast<float16>(0.0f));
  deepstack_2_.assign(input_seq_length * embedding_length,
                      static_cast<float16>(0.0f));

  if (!orig_ds0.empty()) {
    size_t ds_feature_idx = 0;
    size_t img_idx = 0;

    for (size_t i = 0; i < input_ids.size() && ds_feature_idx < orig_ds0.size();
         i++) {
      if (input_ids[i] == QWEN3_VL_IMAGE_TOKEN_ID &&
          img_idx < image_grid_thw_.size()) {
        auto [grid_t, grid_h, grid_w] = image_grid_thw_[img_idx];
        size_t num_image_tokens = grid_t * grid_h * grid_w;

        for (size_t j = 0;
             j < num_image_tokens && ds_feature_idx < orig_ds0.size(); j++) {
          size_t embed_offset = (i + j) * embedding_length;
          if (ds_feature_idx + embedding_length <= orig_ds0.size()) {
            std::memcpy(deepstack_0_.data() + embed_offset,
                        orig_ds0.data() + ds_feature_idx,
                        embedding_length * sizeof(float16));
            std::memcpy(deepstack_1_.data() + embed_offset,
                        orig_ds1.data() + ds_feature_idx,
                        embedding_length * sizeof(float16));
            std::memcpy(deepstack_2_.data() + embed_offset,
                        orig_ds2.data() + ds_feature_idx,
                        embedding_length * sizeof(float16));
          }
          ds_feature_idx += embedding_length;
        }

        i += num_image_tokens - 1;
        img_idx++;
      }
    }
  }

  // 8. Return results
  std::vector<std::vector<int32_t>> position_ids_3d = {time_pos, height_pos,
                                                       width_pos};
  int prefill_loop_chunk =
      (input_seq_length + prefill_length - 1) / prefill_length;

  return {position_ids_3d, seq_length, prefill_loop_chunk};
}

void Qwen3VLMContext::vision_preprocess(int image_idx) {
  auto* model = static_cast<Qwen3VLMModel*>(model_);

  // 1. Load and preprocess image
  const auto& path = image_paths_[image_idx];
  current_processed_image_ = image_processor_->LoadAndProcess(path);

  // 2. Convert to float16 tensor
  current_vision_tensor_ =
      image_processor_->ToFP16Tensor(current_processed_image_);

  // 3. Set vision inputs
  auto vision_module = model->vision_module();
  auto& vision_input_map = model->vision_input_map();
  auto input_name = vision_module->GetInputName(0);
  auto& input_tensor = vision_input_map[input_name];
  input_tensor.Buffer().CopyFromHost(
      current_vision_tensor_.data(),
      current_vision_tensor_.size() * sizeof(float16));
  vision_module->SetInput(input_name, input_tensor);
}

void Qwen3VLMContext::vision_inference() {
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  auto vision_module = model->vision_module();
  // Only Run + Sync
  vision_module->Run();
  vision_module->Sync();
}

void Qwen3VLMContext::vision_postprocess(int image_idx) {
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  auto vision_module = model->vision_module();

  // Get output
  auto get_output_data =
      [vision_module](int output_idx) -> std::vector<float16> {
    auto output_name = vision_module->GetOutputName(output_idx);
    auto dev_output = vision_module->GetDevOutput(output_name);
    auto host_output = dev_output.ToHost(true);
    size_t num_elements = host_output.Buffer().Size() / sizeof(float16);
    std::vector<float16> data(num_elements);
    std::memcpy(data.data(), host_output.Buffer().Data(),
                host_output.Buffer().Size());
    return data;
  };

  // Save to current_vision_output_
  current_vision_output_ = {
      get_output_data(0),  // image_features
      get_output_data(1),  // deepstack_image_feature_0
      get_output_data(2),  // deepstack_image_feature_1
      get_output_data(3)   // deepstack_image_feature_2
  };

  // Concatenate results to global cache
  auto& [feat, ds0, ds1, ds2] = current_vision_output_;
  flat_image_embeds_.insert(flat_image_embeds_.end(), feat.begin(), feat.end());
  deepstack_0_.insert(deepstack_0_.end(), ds0.begin(), ds0.end());
  deepstack_1_.insert(deepstack_1_.end(), ds1.begin(), ds1.end());
  deepstack_2_.insert(deepstack_2_.end(), ds2.begin(), ds2.end());

  // Compute image grid (t, h, w)
  int grid_h = model->vision_image_size_h() /
               (model->patch_size() * model->spatial_merge_size());
  int grid_w = model->vision_image_size_w() /
               (model->patch_size() * model->spatial_merge_size());
  image_grid_thw_.push_back({1, grid_h, grid_w});
}

std::vector<Token> Qwen3VLMContext::expand_image_tokens(
    const std::vector<Token>& tokens) {
  if (image_grid_thw_.empty()) return tokens;

  std::vector<Token> expanded;
  size_t image_idx = 0;

  for (const auto& token : tokens) {
    if (token == QWEN3_VL_IMAGE_TOKEN_ID &&
        image_idx < image_grid_thw_.size()) {
      // Compute the number of tokens needed for this image
      auto [t, h, w] = image_grid_thw_[image_idx];
      int num_tokens = h * w;

      // Insert multiple IMAGE_TOKEN_IDs
      for (int i = 0; i < num_tokens; i++) {
        expanded.push_back(QWEN3_VL_IMAGE_TOKEN_ID);
      }
      image_idx++;
    } else {
      expanded.push_back(token);
    }
  }

  return expanded;
}

void Qwen3VLMContext::scatter_image_embeds(std::vector<float16>& inputs_embeds,
                                           const std::vector<Token>& input_ids,
                                           size_t seq_len, size_t embed_dim) {
  if (flat_image_embeds_.empty()) return;

  size_t image_feature_idx = 0;
  for (size_t i = 0;
       i < input_ids.size() && image_feature_idx < flat_image_embeds_.size();
       i++) {
    if (input_ids[i] == QWEN3_VL_IMAGE_TOKEN_ID) {
      // Count consecutive image tokens
      size_t num_image_tokens = 0;
      for (size_t j = i;
           j < input_ids.size() && input_ids[j] == QWEN3_VL_IMAGE_TOKEN_ID;
           j++) {
        num_image_tokens++;
      }

      // Copy vision features to these positions
      for (size_t j = 0; j < num_image_tokens &&
                         image_feature_idx < flat_image_embeds_.size();
           j++) {
        size_t embed_offset = (i + j) * embed_dim;
        if (image_feature_idx + embed_dim <= flat_image_embeds_.size()) {
          std::memcpy(inputs_embeds.data() + embed_offset,
                      flat_image_embeds_.data() + image_feature_idx,
                      embed_dim * sizeof(float16));
        }
        image_feature_idx += embed_dim;
      }
      i += num_image_tokens - 1;  // Skip processed tokens
    }
  }
}

std::pair<std::vector<std::vector<int32_t>>, int32_t>
Qwen3VLMContext::get_rope_index(
    const std::vector<Token>& input_ids,
    const std::vector<ImageGridTHW>& image_grid_thw) {
  size_t seq_len = input_ids.size();
  std::vector<int32_t> time_pos(seq_len, 0);
  std::vector<int32_t> height_pos(seq_len, 0);
  std::vector<int32_t> width_pos(seq_len, 0);

  int pos_idx = 0;
  size_t img_idx = 0;

  for (size_t i = 0; i < seq_len;) {
    // Check if current position is an image token and we have a grid for it
    if (input_ids[i] == QWEN3_VL_IMAGE_TOKEN_ID &&
        img_idx < image_grid_thw.size()) {
      // Get the grid dimensions for this image
      auto [grid_t, grid_h, grid_w] = image_grid_thw[img_idx];
      int num_image_tokens = grid_t * grid_h * grid_w;

      // Set 3D position IDs for all tokens of this image
      for (int t = 0; t < grid_t; t++) {
        for (int h = 0; h < grid_h; h++) {
          for (int w = 0; w < grid_w; w++) {
            size_t token_offset = i + (t * grid_h * grid_w) + (h * grid_w) + w;
            if (token_offset < seq_len) {
              time_pos[token_offset] = pos_idx + t;
              height_pos[token_offset] = pos_idx + h;
              width_pos[token_offset] = pos_idx + w;
            }
          }
        }
      }

      // Move position index forward by max(grid_t, grid_h, grid_w)
      pos_idx += std::max({grid_t, grid_h, grid_w});
      i += num_image_tokens;
      img_idx++;
    } else {
      // Text token: all 3 dimensions use same position
      time_pos[i] = pos_idx;
      height_pos[i] = pos_idx;
      width_pos[i] = pos_idx;
      pos_idx++;
      i++;
    }
  }

  // Calculate rope_deltas = max_position_id + 1 - seq_length
  int32_t max_pos_id = 0;
  for (size_t i = 0; i < seq_len; i++) {
    max_pos_id = std::max(max_pos_id, time_pos[i]);
    max_pos_id = std::max(max_pos_id, height_pos[i]);
    max_pos_id = std::max(max_pos_id, width_pos[i]);
  }
  int32_t rope_deltas = max_pos_id + 1 - static_cast<int32_t>(seq_len);

  std::vector<std::vector<int32_t>> position_ids = {time_pos, height_pos,
                                                    width_pos};
  return {position_ids, rope_deltas};
}

void Qwen3VLMContext::set_prefill_inputs(
    const std::vector<float16>& inputs_embeds,
    const std::vector<int32_t>& time_pos_ids,
    const std::vector<int32_t>& height_pos_ids,
    const std::vector<int32_t>& width_pos_ids, int32_t valid_length,
    int32_t current_length, const std::vector<float16>& ds_0,
    const std::vector<float16>& ds_1, const std::vector<float16>& ds_2) {
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  auto& input_map = model->prefill_input_map();
  auto prefill_module = model->prefill_module();
  int attn_idx_start = model->attn_idx_start();

  for (int idx = 0; idx < attn_idx_start; idx++) {
    const std::string& input_name = prefill_module->GetInputName(idx);
    auto& tensor = input_map[input_name];
    size_t mem_size = tensor.MemSize();

    if (input_name.find("input_1") != std::string::npos ||
        input_name.find("inputs_embeds") != std::string::npos) {
      tensor.Buffer().CopyFromHost(inputs_embeds.data(), mem_size);
    } else if (input_name.find("time_position") != std::string::npos) {
      tensor.Buffer().CopyFromHost(time_pos_ids.data(),
                                   time_pos_ids.size() * sizeof(int32_t));
    } else if (input_name.find("height_position") != std::string::npos) {
      tensor.Buffer().CopyFromHost(height_pos_ids.data(),
                                   height_pos_ids.size() * sizeof(int32_t));
    } else if (input_name.find("width_position") != std::string::npos) {
      tensor.Buffer().CopyFromHost(width_pos_ids.data(),
                                   width_pos_ids.size() * sizeof(int32_t));
    } else if (input_name.find("valid_length") != std::string::npos) {
      tensor.Buffer().CopyFromHost(&valid_length, sizeof(int32_t));
    } else if (input_name.find("current_length") != std::string::npos) {
      tensor.Buffer().CopyFromHost(&current_length, sizeof(int32_t));
    } else if (input_name.find("deepstack_image_embed_0") !=
               std::string::npos) {
      tensor.Buffer().CopyFromHost(ds_0.data(), mem_size);
    } else if (input_name.find("deepstack_image_embed_1") !=
               std::string::npos) {
      tensor.Buffer().CopyFromHost(ds_1.data(), mem_size);
    } else if (input_name.find("deepstack_image_embed_2") !=
               std::string::npos) {
      tensor.Buffer().CopyFromHost(ds_2.data(), mem_size);
    }

    prefill_module->SetInput(input_name, tensor);
  }
}

void Qwen3VLMContext::set_decode_inputs(
    const std::vector<float16>& inputs_embeds,
    const std::vector<int32_t>& time_pos_ids,
    const std::vector<int32_t>& height_pos_ids,
    const std::vector<int32_t>& width_pos_ids, int32_t valid_length,
    const std::vector<float16>& ds_0, const std::vector<float16>& ds_1,
    const std::vector<float16>& ds_2) {
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  auto& input_map = model->decode_input_map();
  auto decode_module = model->decode_module();
  int attn_idx_start = model->attn_idx_start();

  for (int idx = 0; idx < attn_idx_start; idx++) {
    const std::string& input_name = decode_module->GetInputName(idx);
    auto& tensor = input_map[input_name];

    if (input_name.find("input_1") != std::string::npos ||
        input_name.find("inputs_embeds") != std::string::npos) {
      tensor.Buffer().CopyFromHost(inputs_embeds.data(),
                                   inputs_embeds.size() * sizeof(float16));
    } else if (input_name.find("time_position") != std::string::npos) {
      tensor.Buffer().CopyFromHost(time_pos_ids.data(),
                                   time_pos_ids.size() * sizeof(int32_t));
    } else if (input_name.find("height_position") != std::string::npos) {
      tensor.Buffer().CopyFromHost(height_pos_ids.data(),
                                   height_pos_ids.size() * sizeof(int32_t));
    } else if (input_name.find("width_position") != std::string::npos) {
      tensor.Buffer().CopyFromHost(width_pos_ids.data(),
                                   width_pos_ids.size() * sizeof(int32_t));
    } else if (input_name.find("valid_length") != std::string::npos) {
      tensor.Buffer().CopyFromHost(&valid_length, sizeof(int32_t));
    } else if (input_name.find("current_length") != std::string::npos) {
      int32_t current_length = 1;
      tensor.Buffer().CopyFromHost(&current_length, sizeof(int32_t));
    } else if (input_name.find("deepstack_image_embed_0") !=
               std::string::npos) {
      tensor.Buffer().CopyFromHost(ds_0.data(), ds_0.size() * sizeof(float16));
    } else if (input_name.find("deepstack_image_embed_1") !=
               std::string::npos) {
      tensor.Buffer().CopyFromHost(ds_1.data(), ds_1.size() * sizeof(float16));
    } else if (input_name.find("deepstack_image_embed_2") !=
               std::string::npos) {
      tensor.Buffer().CopyFromHost(ds_2.data(), ds_2.size() * sizeof(float16));
    }

    decode_module->SetInput(input_name, tensor);
  }
}

Token Qwen3VLMContext::do_prefill_inference(const std::vector<Token>& tokens,
                                            Sampler* sampler) {
  auto& p = profiler_;

  // 1. Vision processing (if applicable)
  if (use_vlm_ && !image_paths_.empty()) {
    auto t = p.scope("generate.vision");
    run_vision();
  }

  // 2. Prefill common setup (execute once)
  auto [position_ids_3d, seq_length, prefill_loop_chunk] = [&]() {
    auto t = p.scope("generate.prefill.common_setup");
    return prefill_common_setup(tokens);
  }();

  // 3. Execute chunked prefill
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  for (int chunk = 0; chunk < prefill_loop_chunk; chunk++) {
    {
      auto t = p.scope("generate.prefill.preprocess_chunk");
      prefill_preprocess_chunk(chunk, seq_length, model->prefill_length(),
                               position_ids_3d);
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

void Qwen3VLMContext::prefill_preprocess_chunk(
    int chunk, int32_t seq_length, int prefill_length,
    const std::vector<std::vector<int32_t>>& position_ids_3d) {
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  int embedding_length = model->embedding_dim();

  // Compute chunk range (use chunk_embeds_ actual length since it was
  // prepared in prefill_common_setup)
  int input_seq_length = chunk_embeds_.size() / embedding_length;
  int chunk_start = chunk * prefill_length;
  int chunk_end = std::min(chunk_start + prefill_length, input_seq_length);
  int current_chunk_size = chunk_end - chunk_start;

  // current_length is based on seq_length, not padded length
  int start = chunk * prefill_length;
  int end = std::min((chunk + 1) * prefill_length, seq_length);
  int current_length = end - start;

  // Prepare chunk data
  size_t token_span = static_cast<size_t>(current_chunk_size);
  if (token_span < static_cast<size_t>(prefill_length))
    token_span = prefill_length;
  size_t embed_span = token_span * static_cast<size_t>(embedding_length);
  size_t embed_start = static_cast<size_t>(chunk_start) * embedding_length;

  // Prepare embedding (extract current chunk from chunk_embeds_)
  std::vector<float16> chunk_embeds(prefill_length * embedding_length,
                                    static_cast<float16>(0.0f));
  if (embed_span > 0) {
    std::memcpy(chunk_embeds.data(), chunk_embeds_.data() + embed_start,
                embed_span * sizeof(float16));
  }

  // Prepare position ids
  std::vector<int32_t> chunk_time(prefill_length, 0);
  std::vector<int32_t> chunk_height(prefill_length, 0);
  std::vector<int32_t> chunk_width(prefill_length, 0);

  const auto& time_pos = position_ids_3d[0];
  const auto& height_pos = position_ids_3d[1];
  const auto& width_pos = position_ids_3d[2];

  std::memcpy(chunk_time.data(), time_pos.data() + chunk_start,
              token_span * sizeof(int32_t));
  std::memcpy(chunk_height.data(), height_pos.data() + chunk_start,
              token_span * sizeof(int32_t));
  std::memcpy(chunk_width.data(), width_pos.data() + chunk_start,
              token_span * sizeof(int32_t));

  // Prepare deepstack (extract current chunk from temporary state)
  std::vector<float16> chunk_ds0(prefill_length * embedding_length,
                                 static_cast<float16>(0.0f));
  std::vector<float16> chunk_ds1(prefill_length * embedding_length,
                                 static_cast<float16>(0.0f));
  std::vector<float16> chunk_ds2(prefill_length * embedding_length,
                                 static_cast<float16>(0.0f));

  if (!deepstack_0_.empty()) {
    if (embed_span > 0) {
      std::memcpy(chunk_ds0.data(), deepstack_0_.data() + embed_start,
                  embed_span * sizeof(float16));
      std::memcpy(chunk_ds1.data(), deepstack_1_.data() + embed_start,
                  embed_span * sizeof(float16));
      std::memcpy(chunk_ds2.data(), deepstack_2_.data() + embed_start,
                  embed_span * sizeof(float16));
    }
  }

  int32_t valid_length = chunk * prefill_length + past_seq_len_;

  // Set inputs
  set_prefill_inputs(chunk_embeds, chunk_time, chunk_height, chunk_width,
                     valid_length, current_length, chunk_ds0, chunk_ds1,
                     chunk_ds2);
}

void Qwen3VLMContext::prefill_inference_chunk() {
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  auto prefill_module = model->prefill_module();
  // Only Run + Sync
  prefill_module->Run();
  prefill_module->Sync();
}

Token Qwen3VLMContext::prefill_postprocess(Sampler* sampler,
                                           int32_t seq_length) {
  auto* model = static_cast<Qwen3VLMModel*>(model_);

  // Update context
  context_length_ += seq_length;
  past_seq_len_ = context_length_;

  // Get output and sample
  auto output_name = model->prefill_module()->GetOutputName(0);
  auto dev_output = model->prefill_module()->GetDevOutput(output_name);
  auto host_output = dev_output.ToHost(true);
  const float16* logits =
      static_cast<const float16*>(host_output.Buffer().Data());

  // Clear image_paths_ to avoid misuse in next round
  image_paths_.clear();

  return sampler->sample(logits, model->vocab_size(), generated_ids_);
}

void Qwen3VLMContext::decode_preprocess(Token prev_token) {
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  int embedding_length = model->embedding_dim();

  // 1. Get embedding of previous token
  std::vector<Token> tokens = {prev_token};
  const float16* embed = model->embedding()->token_embedding(tokens);

  // 2. Compute position
  int32_t position = past_seq_len_ + rope_deltas_;

  // 3. Prepare position_ids (store to temporary state)
  decode_time_pos_ = {position};
  decode_height_pos_ = {position};
  decode_width_pos_ = {position};

  // 4. Prepare deepstack (decode stage uses zeros)
  std::vector<float16> ds_0(embedding_length, static_cast<float16>(0.0f));
  std::vector<float16> ds_1(embedding_length, static_cast<float16>(0.0f));
  std::vector<float16> ds_2(embedding_length, static_cast<float16>(0.0f));

  // 5. Set decode inputs
  std::vector<float16> inputs_embeds(embed, embed + embedding_length);
  set_decode_inputs(inputs_embeds, decode_time_pos_, decode_height_pos_,
                    decode_width_pos_, context_length_, ds_0, ds_1, ds_2);
}

void Qwen3VLMContext::decode_inference() {
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  auto decode_module = model->decode_module();
  // Only Run + Sync
  decode_module->Run();
  decode_module->Sync();
}

Token Qwen3VLMContext::decode_postprocess(Sampler* sampler) {
  auto* model = static_cast<Qwen3VLMModel*>(model_);
  auto decode_module = model->decode_module();

  // Get output
  auto output_name = decode_module->GetOutputName(0);
  auto dev_output = decode_module->GetDevOutput(output_name);
  auto host_output = dev_output.ToHost(true);
  const float16* logits =
      static_cast<const float16*>(host_output.Buffer().Data());

  // Sample
  Token token = sampler->sample(logits, model->vocab_size(), generated_ids_);

  // Update state
  context_length_++;
  past_seq_len_++;

  return token;
}

Token Qwen3VLMContext::do_decode_inference(Token prev_token, Sampler* sampler) {
  auto& p = profiler_;

  {
    auto t = p.scope("generate.decode.preprocess");
    decode_preprocess(prev_token);
  }
  {
    auto t = p.scope("generate.decode.inference");
    decode_inference();
  }
  Token token;
  {
    auto t = p.scope("generate.decode.postprocess");
    token = decode_postprocess(sampler);
  }

  return token;
}

// ============================================================================
// Qwen3VLMModel Implementation
// ============================================================================

Qwen3VLMModel::Qwen3VLMModel(const ModelConfig& config) : VLMModel(config) {
  load();
}

void Qwen3VLMModel::load() {
  // 1. Initialize device manager and WeightManager
  dev_manager_ = std::make_unique<tcim::DevManager>(
      tcim::DevManager::Create(config_.devices));
  weight_manager_ = std::make_unique<tcim::Module::WeightManager>(
      tcim::Module::WeightManager::CreateWeightManager(*dev_manager_));

  // 2. Load visual encoder
  std::cout << "Loading visual model from " << config_.vision_path << std::endl;
  vision_module_ = std::make_shared<tcim::Module>();
  auto option_visual = tcim::Module::Option(*weight_manager_);
  vision_module_->LoadModel(config_.vision_path, option_visual);
  std::cout << "Visual model loaded" << std::endl;

  // 3. Load prefill model
  std::cout << "Loading prefill model from " << config_.prefill_path
            << std::endl;
  prefill_module_ = std::make_shared<tcim::Module>();
  auto option_prefill = tcim::Module::Option(*weight_manager_);
  prefill_module_->LoadModel(config_.prefill_path, option_prefill);
  std::cout << "Prefill model loaded" << std::endl;

  // 4. Get n_blocks and create dummy names
  n_blocks_ = get_n_blocks();
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

  // 5. Load decode model
  std::cout << "Loading decode model from " << config_.decode_path << std::endl;
  auto option_decode = tcim::Module::Option(*weight_manager_);
  option_decode.SetDummyTensors(dummy_names);
  decode_module_ = std::make_shared<tcim::Module>();
  decode_module_->LoadModel(config_.decode_path, option_decode);
  std::cout << "Decode model loaded" << std::endl;

  // 6. Get attention index start
  attn_idx_start_ = get_attn_idx_start();

  // 7. Get model parameters
  prefill_length_ =
      prefill_module_->GetInputInfo(prefill_module_->GetInputName(0))
          .Shape()[1];
  embedding_length_ =
      prefill_module_->GetInputInfo(prefill_module_->GetInputName(0))
          .Shape()[2];
  context_max_length_ =
      prefill_module_
          ->GetInputInfo(prefill_module_->GetInputName(attn_idx_start_))
          .Shape()[2];
  batch_ =
      decode_module_->GetInputInfo(decode_module_->GetInputName(0)).Shape()[0];
  argmax_dim_len_ =
      decode_module_->GetOutputInfo(decode_module_->GetOutputName(0))
          .Shape()[2];
  vision_input_nums_ = vision_module_->GetInputNum();

  // 8. Get vision model input shape
  auto vit_input_shape =
      vision_module_->GetInputInfo(vision_module_->GetInputName(0)).Shape();
  std::cout << "VIT input shape: ";
  for (size_t i = 0; i < vit_input_shape.size(); i++) {
    std::cout << vit_input_shape[i] << " ";
  }
  std::cout << std::endl;

  if (vit_input_shape.size() >= 5) {
    vision_image_size_h_ = vit_input_shape[3];  // Height
    vision_image_size_w_ = vit_input_shape[4];  // Width
  } else if (vit_input_shape.size() >= 4) {
    vision_image_size_h_ = vit_input_shape[2];  // Height
    vision_image_size_w_ = vit_input_shape[3];  // Width
  }

  std::cout << "Image dimensions: " << vision_image_size_w_ << "x"
            << vision_image_size_h_ << std::endl;
  std::cout << "Context max length: " << context_max_length_ << std::endl;
  std::cout << "Embedding length: " << embedding_length_ << std::endl;
  std::cout << "Prefill length: " << prefill_length_ << std::endl;
  std::cout << "N blocks: " << n_blocks_ << std::endl;

  // 9. Load tokenizer
  std::cout << "Loading tokenizer from " << config_.tokenizer_path << std::endl;
  tokenizer_ = std::make_shared<HfTokenizer>(config_.tokenizer_path);

  // 10. Load embedding
  std::cout << "Loading embedding from " << config_.embedding_path << std::endl;
  embedding_ = std::make_shared<Embedding>(config_.embedding_path,
                                           embedding_length_, prefill_length_);

  // 11. Initialize input tensors
  init_vision_inputs();
  init_prefill_inputs();
  init_decode_inputs();

  // 12. Set up KV Cache sharing
  for (int idx = attn_idx_start_; idx < 2 * n_blocks_ + attn_idx_start_;
       idx++) {
    const std::string cache_name = prefill_module_->GetInputName(idx);
    auto cache = prefill_module_->GetDevInput(cache_name);
    decode_module_->SetDevInput(decode_module_->GetInputName(idx), cache);
  }

  std::cout << "Qwen3VLMModel loaded successfully" << std::endl;

  // 13. Set model info
  info_.n_vocab = tokenizer_->vocab_size();
  info_.n_embd = embedding_length_;
  info_.n_ctx = context_max_length_;
}

int Qwen3VLMModel::get_n_blocks() {
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

int Qwen3VLMModel::get_attn_idx_start() {
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

void Qwen3VLMModel::init_vision_inputs() {
  vision_input_map_.clear();
  for (int idx = 0; idx < vision_input_nums_; idx++) {
    auto input_name = vision_module_->GetInputName(idx);
    auto input_info = vision_module_->GetInputInfo(input_name).AsContiguous();
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    vision_input_map_[input_name] = input_tensor;
  }
}

void Qwen3VLMModel::init_prefill_inputs() {
  prefill_input_map_.clear();
  for (int idx = 0; idx < attn_idx_start_; idx++) {
    auto input_name = prefill_module_->GetInputName(idx);
    auto input_info = prefill_module_->GetInputInfo(input_name).AsContiguous();
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    prefill_input_map_[input_name] = input_tensor;
  }
}

void Qwen3VLMModel::init_decode_inputs() {
  decode_input_map_.clear();
  for (int idx = 0; idx < attn_idx_start_; idx++) {
    auto input_name = decode_module_->GetInputName(idx);
    auto input_info = decode_module_->GetInputInfo(input_name).AsContiguous();
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    decode_input_map_[input_name] = input_tensor;
  }
}

std::unique_ptr<Context> Qwen3VLMModel::create_context(int n_ctx) {
  int ctx_len = n_ctx > 0 ? n_ctx : context_max_length_;
  return std::make_unique<Qwen3VLMContext>(this, ctx_len);
}

std::tuple<std::vector<float16>, std::vector<float16>, std::vector<float16>,
           std::vector<float16>>
Qwen3VLMModel::encode_image(const std::vector<float16>& pixel_values) {
  // Set inputs
  auto input_name = vision_module_->GetInputName(0);
  auto input_tensor = vision_input_map_[input_name];
  input_tensor.Buffer().CopyFromHost(pixel_values.data(),
                                     pixel_values.size() * sizeof(float16));
  vision_module_->SetInput(input_name, input_tensor);

  // Run inference
  vision_module_->Run();
  vision_module_->Sync();

  // Get output
  auto get_output_data = [this](int output_idx) -> std::vector<float16> {
    auto output_name = vision_module_->GetOutputName(output_idx);
    auto dev_output = vision_module_->GetDevOutput(output_name);
    auto host_output = dev_output.ToHost(true);
    size_t num_elements = host_output.Buffer().Size() / sizeof(float16);
    std::vector<float16> data(num_elements);
    std::memcpy(data.data(), host_output.Buffer().Data(),
                host_output.Buffer().Size());
    return data;
  };

  return {
      get_output_data(0),  // image_features
      get_output_data(1),  // deepstack_image_feature_0
      get_output_data(2),  // deepstack_image_feature_1
      get_output_data(3)   // deepstack_image_feature_2
  };
}

void Qwen3VLMModel::ClearKVCache() {
  // KV Cache is automatically reset during prefill; interface reserved here
  // If explicit clearing is needed, implement it here
}

// ============================================================================
// Model Registration
// ============================================================================

// Static registration for Qwen3 VLM model
REGISTER_LLM_MODEL(
    qwen3_vlm, ModelSeries::kQwen3VLM,
    [](const ModelConfig& c) { return std::make_unique<Qwen3VLMModel>(c); },
    "Qwen3-VL vision-language model");

}  // namespace houmo
