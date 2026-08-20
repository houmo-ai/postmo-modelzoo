/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen35_mllm_model.cc
 * Description:
 *   Qwen3.5 LLM/VLM model implementation
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

#include "qwen35_mllm_model.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <unordered_map>

#include "base/tcim_utils.h"
#include "core/model_factory.h"

namespace fs = std::filesystem;

namespace houmo {
namespace {

bool IsActivationInput(const std::string& name);

template <typename T>
void SetHostInput(const std::shared_ptr<tcim::Module>& module,
                  const std::string& name, const std::vector<T>& data) {
  auto info = module->GetInputInfo(name).AsContiguous();
  auto tensor = tcim::Tensor::CreateHostTensor(info);
  if (tensor.MemSize() != data.size() * sizeof(T)) {
    throw Exception("dynamic vision input size mismatch for " + name);
  }
  tensor.Buffer().CopyFromHost(data.data(), tensor.MemSize());
  CHECK_TCIM_RET_STATUS(module->SetInput(name, tensor));
}

void SetPrefillInput(
    const std::string& name, tcim::Tensor& tensor, const std::vector<float16>& embeds,
    const std::vector<int32_t>& time_pos, const std::vector<int32_t>& height_pos,
  const std::vector<int32_t>& width_pos, const std::vector<float16>& attn_mask,
    int32_t valid_length, int32_t current_length) {
  const size_t mem_size = tensor.MemSize();
  if (IsActivationInput(name)) {
    tensor.Buffer().CopyFromHost(embeds.data(), mem_size);
  } else if (name.find("time_position_ids") != std::string::npos) {
    tensor.Buffer().CopyFromHost(time_pos.data(), time_pos.size() * sizeof(int32_t));
  } else if (name.find("hight_position_ids") != std::string::npos ||
             name.find("height_position_ids") != std::string::npos) {
    tensor.Buffer().CopyFromHost(height_pos.data(), height_pos.size() * sizeof(int32_t));
  } else if (name.find("width_position_ids") != std::string::npos) {
    tensor.Buffer().CopyFromHost(width_pos.data(), width_pos.size() * sizeof(int32_t));
  } else if (name.find("valid_length") != std::string::npos) {
    tensor.Buffer().CopyFromHost(&valid_length, mem_size);
  } else if (name.find("current_length") != std::string::npos) {
    tensor.Buffer().CopyFromHost(&current_length, mem_size);
  } else if (name.find("linear_attn_mask") != std::string::npos) {
    tensor.Buffer().CopyFromHost(attn_mask.data(), mem_size);
  }
}

std::string BareTensorName(const std::string& name) {
  constexpr const char* suffix = ".hmcc.format";
  if (name.size() >= std::char_traits<char>::length(suffix) &&
      name.compare(name.size() - std::char_traits<char>::length(suffix),
                   std::char_traits<char>::length(suffix), suffix) == 0) {
    return name.substr(0, name.size() - std::char_traits<char>::length(suffix));
  }
  return name;
}

bool IsActivationInput(const std::string& name) {
  const std::string bare = BareTensorName(name);
  return bare == "input_1" || bare.find("input_embedding") != std::string::npos ||
         bare.find("next_token_embedding") != std::string::npos ||
         bare.find("hidden_states") != std::string::npos ||
         bare.find("post_norm_hidden") != std::string::npos ||
         bare.find("pre_norm_hidden") != std::string::npos;
}

bool IsCacheInput(const std::string& name) {
  const std::string bare = BareTensorName(name);
  return bare.find("model_layers") != std::string::npos ||
         bare.find("kcache_input") != std::string::npos ||
         bare.find("vcache_input") != std::string::npos ||
         bare.find("past_key_cache") != std::string::npos ||
         bare.find("past_value_cache") != std::string::npos ||
         bare.find("past_conv_cache") != std::string::npos ||
         bare.find("past_recurrent_state") != std::string::npos;
}

std::string FindInputByBare(const std::shared_ptr<tcim::Module>& module,
                            const std::string& bare_name) {
  for (int i = 0; i < module->GetInputNum(); ++i) {
    const std::string actual = module->GetInputName(i);
    if (BareTensorName(actual) == bare_name) return actual;
  }
  return {};
}

std::string FindOutputByBare(const std::shared_ptr<tcim::Module>& module,
                             const std::string& bare_name) {
  for (int i = 0; i < module->GetOutputNum(); ++i) {
    const std::string actual = module->GetOutputName(i);
    if (BareTensorName(actual) == bare_name) return actual;
  }
  return {};
}

}  // namespace

// ============================================================================
// Qwen35MLLMContext Implementation
// ============================================================================

Qwen35MLLMContext::Qwen35MLLMContext(LLMModel* model, int n_ctx)
    : Context(model, n_ctx) {
  img_processor_ = std::make_shared<qwen35::DynamicImageProcessor>(
      PATCH_SIZE, TEMPORAL_PATCH_SIZE, SPATIAL_MERGE_SIZE, 65536,
      1536 * SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE * PATCH_SIZE * PATCH_SIZE,
      static_cast<Qwen35MLLMModel*>(model_)->preprocessor_config_path());
}

Token Qwen35MLLMContext::prefill(const std::vector<Token>& tokens) {
  use_vlm_ = false;
  if (!sampler_) {
    set_sampler(SamplingParams{});
  }
  generated_ids_.clear();
  Token token = do_prefill_inference(tokens, sampler_.get());
  generated_ids_.push_back(token);
  // context_length_ already updated in do_prefill_inference
  return token;
}

Token Qwen35MLLMContext::decode(Token prev_token) {
  if (!sampler_) {
    set_sampler(SamplingParams{});
  }
  Token token = do_decode_inference(prev_token, sampler_.get());
  generated_ids_.push_back(token);
  // context_length_ already updated in do_decode_inference
  return token;
}

void Qwen35MLLMContext::generate(const std::vector<Token>& prompt,
                                 const SamplingParams& params,
                                 std::function<bool(Token)> callback) {
  profiler_.reset();  // Auto-reset single-run statistics
  auto& p = profiler_;

  // Start E2E timing
  p.start("generate");
  // Note: input_tokens will be updated after prefill to include expanded image tokens
  int initial_context_length = context_length_;

  set_sampler(params);

  Token token;
  // Do not wrap prefill/decode with outer scopes: leaf stages already
  // roll up into generate.prefill / generate.decode. Nested start+stop on
  // the same parent path double-counts total_time in PerfProfiler.
  token = prefill(prompt);

  // Update input_tokens with actual processed tokens (including expanded image tokens)
  p.set_input_tokens(context_length_ - initial_context_length);

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

    token = decode(token);
    p.add_output_token();

    if (token == model_->eos_token_id() || token == model_->bos_token_id()) {
      break;
    }

    if (!callback(token)) break;
  }

  p.stop("generate");
  perf_stats_ = p.to_perf_stats();
}

void Qwen35MLLMContext::reset() {
  // Call base class reset to reset counters
  Context::reset();

  // Reset VLM-related state
  image_paths_.clear();
  image_embed_offset_ = 0;
  flat_image_embeds_.clear();
  image_grid_thw_.clear();
  rope_deltas_ = 0;
  use_vlm_ = false;

  // Clear KV Cache
  auto* model = static_cast<Qwen35MLLMModel*>(model_);
  model->ClearKVCache();
}

std::vector<Token> Qwen35MLLMContext::pad_visual_token(
    const std::vector<Token>& tokens) {
  std::vector<Token> results;
  size_t image_index = 0;
  for (size_t i = 0; i < tokens.size(); ++i) {
    results.push_back(tokens[i]);
    if (tokens[i] == IMAGE_TOKEN_ID) {
      if (image_index >= image_grid_thw_.size()) {
        throw Exception("image token count exceeds processed image count");
      }
      const auto [t, h, w] = image_grid_thw_[image_index++];
      if (t <= 0 || h <= 0 || w <= 0 || h % SPATIAL_MERGE_SIZE != 0 ||
          w % SPATIAL_MERGE_SIZE != 0) {
        throw Exception("invalid image_grid_thw for image token expansion");
      }
      const int grid_size = t * (h / SPATIAL_MERGE_SIZE) *
                            (w / SPATIAL_MERGE_SIZE);
      for (int j = 0; j < grid_size - 1; ++j) {
        results.push_back(IMAGE_TOKEN_ID);
      }
    }
  }
  if (image_index != image_grid_thw_.size()) {
    throw Exception("processed image count exceeds image token count");
  }
  return results;
}

void Qwen35MLLMContext::vision_preprocess(int image_idx, int gear) {
  auto* model = static_cast<Qwen35MLLMModel*>(model_);
  auto vision_module = model->vision_module(gear);
  if (!vision_module) {
    throw Exception("Vision module not loaded");
  }

  const auto& path = image_paths_[image_idx];
  (void)path;
  const auto& image = current_processed_image_;
  const int patch_capacity = model->vision_patch_capacity(gear);
  const int valid_patches = image.grid_t * image.grid_h * image.grid_w;
  if (valid_patches > patch_capacity) {
    throw Exception("image exceeds selected dynamic vision gear capacity");
  }
  current_vision_tensor_ = image.pixel_values;
  current_vision_tensor_.resize(static_cast<size_t>(patch_capacity) * image.patch_dim,
                                static_cast<float16>(0.0f));

  const int side = static_cast<int>(std::sqrt(NUM_POSITION_EMBEDDINGS));
  if (side * side != NUM_POSITION_EMBEDDINGS) {
    throw Exception("NUM_POSITION_EMBEDDINGS must be a square");
  }
  std::vector<int32_t> position_ids(4 * patch_capacity, 0);
  std::vector<float16> position_weights(4 * patch_capacity, static_cast<float16>(0.0f));
  std::vector<int32_t> rotary_position_ids(2 * patch_capacity, 0);
  std::vector<float16> attention_mask(patch_capacity, static_cast<float16>(0.0f));
  auto merge_index = [&](int row, int col) {
    const int block_h = row / SPATIAL_MERGE_SIZE;
    const int block_w = col / SPATIAL_MERGE_SIZE;
    const int local_h = row % SPATIAL_MERGE_SIZE;
    const int local_w = col % SPATIAL_MERGE_SIZE;
    return (block_h * (image.grid_w / SPATIAL_MERGE_SIZE) + block_w) *
               SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE +
           local_h * SPATIAL_MERGE_SIZE + local_w;
  };
  for (int row = 0; row < image.grid_h; ++row) {
    const double y = image.grid_h == 1 ? 0.0 :
        static_cast<double>(row) * (side - 1) / (image.grid_h - 1);
    const int y0 = static_cast<int>(std::floor(y));
    const int y1 = std::min(y0 + 1, side - 1);
    const double dy = y - y0;
    for (int col = 0; col < image.grid_w; ++col) {
      const double x = image.grid_w == 1 ? 0.0 :
          static_cast<double>(col) * (side - 1) / (image.grid_w - 1);
      const int x0 = static_cast<int>(std::floor(x));
      const int x1 = std::min(x0 + 1, side - 1);
      const double dx = x - x0;
      const int index = merge_index(row, col);
      const int ids[4] = {y0 * side + x0, y0 * side + x1,
                          y1 * side + x0, y1 * side + x1};
      const double weights[4] = {(1 - dy) * (1 - dx), (1 - dy) * dx,
                                 dy * (1 - dx), dy * dx};
      for (int i = 0; i < 4; ++i) {
        position_ids[i * patch_capacity + index] = ids[i];
        position_weights[i * patch_capacity + index] =
            static_cast<float16>(weights[i]);
      }
      rotary_position_ids[index] = row;
      rotary_position_ids[patch_capacity + index] = col;
    }
  }
  for (int i = valid_patches; i < patch_capacity; ++i) {
    attention_mask[i] = static_cast<float16>(-65504.0f);
  }
  SetHostInput(vision_module, "pixel_values", current_vision_tensor_);
  SetHostInput(vision_module, "position_ids", position_ids);
  SetHostInput(vision_module, "position_weights", position_weights);
  SetHostInput(vision_module, "rotary_position_ids", rotary_position_ids);
  SetHostInput(vision_module, "attention_mask", attention_mask);
}

void Qwen35MLLMContext::vision_inference(int gear) {
  auto* model = static_cast<Qwen35MLLMModel*>(model_);
  auto vision_module = model->vision_module(gear);
  CHECK_TCIM_RET_STATUS(vision_module->Run());
  CHECK_TCIM_RET_STATUS(vision_module->Sync());
}

void Qwen35MLLMContext::vision_postprocess(int image_idx) {
  auto* model = static_cast<Qwen35MLLMModel*>(model_);
  const auto [t, h, w] = image_grid_thw_[image_idx];
  const int valid_tokens = t * (h / SPATIAL_MERGE_SIZE) * (w / SPATIAL_MERGE_SIZE);
  int selected_gear = 0;
  for (int gear : model->vision_gears()) {
    if (valid_tokens <= gear) {
      selected_gear = gear;
      break;
    }
  }
  if (selected_gear == 0) {
    throw Exception("image exceeds loaded dynamic vision gears");
  }
  auto vision_module = model->vision_module(selected_gear);

  auto output_name = vision_module->GetOutputName(0);
  auto dev_output = vision_module->GetDevOutput(output_name);
  auto host_output = dev_output.ToHost(true);

  size_t output_size = host_output.Buffer().Size() / sizeof(float16);
  const float16* output_data =
      static_cast<const float16*>(host_output.Buffer().Data());
  const int embed_dim = model->embedding_dim();
  if (output_size < static_cast<size_t>(valid_tokens) * embed_dim) {
    throw Exception("dynamic vision output is shorter than valid image tokens");
  }
  flat_image_embeds_.insert(flat_image_embeds_.end(), output_data,
                            output_data + static_cast<size_t>(valid_tokens) * embed_dim);
}

void Qwen35MLLMContext::run_vision() {
  if (image_paths_.empty()) {
    return;
  }

  auto& p = profiler_;
  auto* model = static_cast<Qwen35MLLMModel*>(model_);

  // Clear previous results
  flat_image_embeds_.clear();
  image_grid_thw_.clear();

  auto process_image = [&](size_t i) {
    current_processed_image_ = img_processor_->LoadAndProcess(image_paths_[i]);
    image_grid_thw_.emplace_back(current_processed_image_.grid_t,
                                 current_processed_image_.grid_h,
                                 current_processed_image_.grid_w);
    const int valid_tokens = current_processed_image_.grid_t *
                             (current_processed_image_.grid_h / SPATIAL_MERGE_SIZE) *
                             (current_processed_image_.grid_w / SPATIAL_MERGE_SIZE);
    int gear = 0;
    for (int candidate : model->vision_gears()) {
      if (valid_tokens <= candidate) {
        gear = candidate;
        break;
      }
    }
    if (gear == 0) throw Exception("image exceeds loaded dynamic vision gears");
    {
      auto t = p.scope("generate.vision.preprocess");
      vision_preprocess(i, gear);
    }
    {
      auto t = p.scope("generate.vision.inference");
      vision_inference(gear);
    }
    {
      auto t = p.scope("generate.vision.postprocess");
      vision_postprocess(i);
    }
  };
  for (size_t i = 0; i < image_paths_.size(); ++i) process_image(i);
}

std::tuple<std::vector<std::vector<int32_t>>, int32_t, int>
Qwen35MLLMContext::prefill_common_setup(
    const std::vector<Token>& padded_tokens) {
  // Reset image embed offset
  image_embed_offset_ = 0;

  // Determine whether to use VLM mode
  use_vlm_ = !flat_image_embeds_.empty();

  // Compute M-RoPE position_ids
  auto [position_ids_3d, rope_deltas] =
      get_rope_index(padded_tokens, image_grid_thw_);
  rope_deltas_ = rope_deltas;

  // Compute chunking parameters
  auto* model = static_cast<LLMModel*>(model_);
  const int32_t seq_length = static_cast<int32_t>(padded_tokens.size());
  const int prefill_length = model->prefill_length();
  const int prefill_loop_chunk =
      (seq_length + prefill_length - 1) / prefill_length;

  return {position_ids_3d, seq_length, prefill_loop_chunk};
}

void Qwen35MLLMContext::scatter_image_embeds(
    const std::vector<Token>& input_ids, std::vector<float16>& token_embeds,
    size_t seq_len, size_t embed_dim, size_t chunk_start_in_full_seq) {
  // 1. Count IMAGE_TOKEN_ID occurrences in input_ids
  size_t n_image_tokens = 0;
  for (Token id : input_ids) {
    if (id == IMAGE_TOKEN_ID) {
      n_image_tokens++;
    }
  }

  if (n_image_tokens == 0) {
    return;  // No images, no replacement needed
  }

  // 2. Compute boundaries for each image
  // image_grid_thw_ contains (t, h, w) for each image
  // Each image's grid_size = t * (h/SPATIAL_MERGE_SIZE) * (w/SPATIAL_MERGE_SIZE)
  std::vector<size_t> image_boundaries;
  size_t current_boundary = 0;
  for (const auto& [t, h, w] : image_grid_thw_) {
    int llm_grid_h = h / SPATIAL_MERGE_SIZE;
    int llm_grid_w = w / SPATIAL_MERGE_SIZE;
    size_t grid_size = t * llm_grid_h * llm_grid_w;
    current_boundary += grid_size;
    image_boundaries.push_back(current_boundary);
  }

  // 3. Replace embedding
  // image_embed_offset_ tracks the number of image embeds already used
  size_t image_idx = image_embed_offset_;

  for (size_t i = 0; i < input_ids.size(); ++i) {
    if (input_ids[i] == IMAGE_TOKEN_ID) {
      // Check if image_idx is out of bounds
      if (image_idx >= flat_image_embeds_.size() / embed_dim) {
        std::cerr << "[ERROR] image_idx " << image_idx << " out of range!"
                  << std::endl;
        break;
      }

      std::copy(flat_image_embeds_.begin() + image_idx * embed_dim,
                flat_image_embeds_.begin() + (image_idx + 1) * embed_dim,
                token_embeds.begin() + i * embed_dim);

      image_idx++;
    }
  }

  // 4. Update offset
  image_embed_offset_ = image_idx;
}

std::pair<std::vector<std::vector<int32_t>>, int32_t>
Qwen35MLLMContext::get_rope_index(
    const std::vector<Token>& input_ids,
    const std::vector<ImageGridTHW>& image_grid_thw) {
  const size_t seq_len = input_ids.size();

  // Initialize 3D position_ids
  std::vector<int32_t> time_pos(seq_len, 0);
  std::vector<int32_t> height_pos(seq_len, 0);
  std::vector<int32_t> width_pos(seq_len, 0);

  if (image_grid_thw.empty()) {
    // Pure text: all dimensions use the same incremental position
    for (size_t i = 0; i < seq_len; ++i) {
      time_pos[i] = static_cast<int32_t>(i);
      height_pos[i] = static_cast<int32_t>(i);
      width_pos[i] = static_cast<int32_t>(i);
    }
    return {{time_pos, height_pos, width_pos}, 0};
  }

  // Count total IMAGE_TOKEN_ID occurrences
  size_t total_image_tokens = 0;
  for (Token id : input_ids) {
    if (id == IMAGE_TOKEN_ID) {
      total_image_tokens++;
    }
  }

  // Iterate over each image segment
  size_t image_index = 0;
  size_t start = 0;         // Start position of current processing
  int32_t start_index = 0;  // Accumulated position value

  while (image_index < image_grid_thw.size()) {
    // Find the position of the next IMAGE_TOKEN_ID
    size_t end = seq_len;  // Default to end
    for (size_t i = start; i < seq_len; ++i) {
      if (input_ids[i] == IMAGE_TOKEN_ID) {
        end = i;
        break;
      }
    }

    if (end >= seq_len) {
      break;  // No more images found
    }

    // Get current image's grid info
    auto [t, h, w] = image_grid_thw[image_index];
    int llm_grid_t = t;
    int llm_grid_h = h / SPATIAL_MERGE_SIZE;
    int llm_grid_w = w / SPATIAL_MERGE_SIZE;
    int grid_size = llm_grid_t * llm_grid_h * llm_grid_w;

    // Text portion (from start to end)
    size_t text_len = end - start;
    for (size_t i = 0; i < text_len; ++i) {
      time_pos[start + i] = start_index + static_cast<int32_t>(i);
      height_pos[start + i] = start_index + static_cast<int32_t>(i);
      width_pos[start + i] = start_index + static_cast<int32_t>(i);
    }
    start_index += static_cast<int32_t>(text_len);

    // Image portion (starting from end, grid_size tokens total)
    // Compute 3D position: t_index, h_index, w_index
    // Note: start_index already includes text_len, no need to add again
    for (int gt = 0; gt < llm_grid_t; ++gt) {
      for (int gh = 0; gh < llm_grid_h; ++gh) {
        for (int gw = 0; gw < llm_grid_w; ++gw) {
          size_t idx =
              end + gt * llm_grid_h * llm_grid_w + gh * llm_grid_w + gw;
          if (idx < seq_len) {
            // Python: t_index + start_index (start_index already includes
            // text_len)
            time_pos[idx] = start_index + gt;
            // Python: h_index + start_index
            height_pos[idx] = start_index + gh;
            // Python: w_index + start_index
            width_pos[idx] = start_index + gw;
          }
        }
      }
    }

    // Update start_index: max position across three dimensions + 1
    // Python: start_index = llm_pos_ids_list[-1].max() + 1
    // Image portion max position = max(t, h, w) + text_len + start_index
    // So start_index should increase max(llm_grid_t, llm_grid_h, llm_grid_w) - 1 + 1
    int max_dim = std::max({llm_grid_t, llm_grid_h, llm_grid_w});
    start_index += max_dim;
    start = end + grid_size;
    image_index++;
  }

  // Remaining text portion
  for (size_t i = start; i < seq_len; ++i) {
    time_pos[i] = start_index + static_cast<int32_t>(i - start);
    height_pos[i] = start_index + static_cast<int32_t>(i - start);
    width_pos[i] = start_index + static_cast<int32_t>(i - start);
  }

  // Compute rope_deltas = max_position + 1 - seq_len
  int32_t max_pos = 0;
  for (int32_t p : time_pos) {
    max_pos = std::max(max_pos, p);
  }
  int32_t rope_deltas = max_pos + 1 - static_cast<int32_t>(seq_len);

  return {{time_pos, height_pos, width_pos}, rope_deltas};
}

void Qwen35MLLMContext::prefill_preprocess_chunk(
    int chunk, const std::vector<Token>& padded_tokens, int32_t seq_length,
    int prefill_length,
    const std::vector<std::vector<int32_t>>& position_ids_3d) {
  auto* model = static_cast<LLMModel*>(model_);
  const int embed_dim = model->embedding_dim();

  int32_t valid_length = chunk * prefill_length + context_length_;
  int start = chunk * prefill_length;
  int end =
      std::min((chunk + 1) * prefill_length, static_cast<int>(seq_length));
  int32_t current_length = end - start;

  // Extract current chunk's tokens
  std::vector<Token> chunk_tokens(padded_tokens.begin() + start,
                                  padded_tokens.begin() + end);

  // Pad tokens to prefill_length
  Token pad_token_id = model->tokenizer()->pad_token_id();
  if (chunk_tokens.size() < static_cast<size_t>(prefill_length)) {
    chunk_tokens.resize(prefill_length, pad_token_id);
  }

  // Get current chunk's embedding
  const float16* embed_data = model->embedding()->token_embedding(chunk_tokens);
  chunk_embeds_.assign(embed_data, embed_data + prefill_length * embed_dim);

  // Replace current chunk's image embedding
  if (use_vlm_) {
    scatter_image_embeds(chunk_tokens, chunk_embeds_, prefill_length, embed_dim,
                         start);
  }

  // Extract current chunk's 3D position_ids
  chunk_time_pos_.assign(position_ids_3d[0].begin() + start,
                         position_ids_3d[0].begin() + end);
  chunk_height_pos_.assign(position_ids_3d[1].begin() + start,
                           position_ids_3d[1].begin() + end);
  chunk_width_pos_.assign(position_ids_3d[2].begin() + start,
                          position_ids_3d[2].begin() + end);

  // Pad position_ids
  if (chunk_time_pos_.size() < static_cast<size_t>(prefill_length)) {
    int32_t last_pos = chunk_time_pos_.empty() ? 0 : chunk_time_pos_.back();
    chunk_time_pos_.resize(prefill_length, last_pos);
    chunk_height_pos_.resize(prefill_length, last_pos);
    chunk_width_pos_.resize(prefill_length, last_pos);
  }

  // Create linear_attn_mask
  linear_attn_mask_.assign(prefill_length, static_cast<float16>(0.0f));
  for (int i = 0; i < current_length; i++) {
    linear_attn_mask_[i] = static_cast<float16>(1.0f);
  }

  // Set prefill inputs
  auto& input_map = model->prefill_input_map();
  auto prefill_module = model->prefill_module();
  const int attn_idx_start = model->attn_idx_start();

  for (int idx = 0; idx < prefill_module->GetInputNum(); idx++) {
    const std::string& input_name = prefill_module->GetInputName(idx);
    if (IsCacheInput(input_name)) continue;
    auto& tensor = input_map[input_name];
    SetPrefillInput(input_name, tensor, chunk_embeds_, chunk_time_pos_,
                    chunk_height_pos_, chunk_width_pos_, linear_attn_mask_,
                    valid_length, current_length);
    prefill_module->SetInput(input_name, tensor);
  }
}

void Qwen35MLLMContext::prefill_inference_chunk() {
  auto* model = static_cast<LLMModel*>(model_);
  auto prefill_module = model->prefill_module();
  prefill_module->Run();
  prefill_module->Sync();
}

Token Qwen35MLLMContext::prefill_postprocess(Sampler* sampler,
                                             int32_t seq_length) {
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

  // Prefill complete, clear image embeds and image_paths_
  flat_image_embeds_.clear();
  image_paths_.clear();

  // Update context_length_
  context_length_ += seq_length;

  return sampled_token;
}

Token Qwen35MLLMContext::do_prefill_inference(const std::vector<Token>& tokens,
                                              Sampler* sampler) {
  auto& p = profiler_;

  // 1. Vision processing must run before token expansion so image grids are known.
  run_vision();

  // 2. Expand image tokens using the actual per-image grids.
  std::vector<Token> padded_tokens = pad_visual_token(tokens);

  // 3. Prefill common setup (execute once)
  auto [position_ids_3d, seq_length, prefill_loop_chunk] = [&]() {
    auto t = p.scope("generate.prefill.common_setup");
    return prefill_common_setup(padded_tokens);
  }();

  // 4. Chunked prefill
  for (int chunk = 0; chunk < prefill_loop_chunk; chunk++) {
    {
      auto t = p.scope("generate.prefill.preprocess_chunk");
      prefill_preprocess_chunk(chunk, padded_tokens, seq_length,
                               model_->prefill_length(), position_ids_3d);
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

void Qwen35MLLMContext::decode_preprocess(Token prev_token) {
  auto* model = static_cast<LLMModel*>(model_);

  std::vector<Token> input_ids = {prev_token};
  const float16* embed_data = model->embedding()->token_embedding(input_ids);

  int32_t current_length = 1;
  decode_linear_attn_mask_.assign(1, static_cast<float16>(1.0f));

  // Compute position_ids
  decode_time_pos_.assign(1, context_length_);
  decode_height_pos_.assign(1, context_length_);
  decode_width_pos_.assign(1, context_length_);

  // VLM mode: use rope_deltas_ to adjust position
  if (use_vlm_) {
    int32_t decode_pos = context_length_ + rope_deltas_;
    decode_time_pos_[0] = decode_pos;
    decode_height_pos_[0] = decode_pos;
    decode_width_pos_[0] = decode_pos;
  }

  auto& input_map = model->decode_input_map();
  auto decode_module = model->decode_module();
  const int attn_idx_start = model->attn_idx_start();

  for (int idx = 0; idx < decode_module->GetInputNum(); idx++) {
    const std::string& input_name = decode_module->GetInputName(idx);
    if (IsCacheInput(input_name)) continue;
    auto& tensor = input_map[input_name];
    size_t mem_size = tensor.MemSize();

    if (IsActivationInput(input_name)) {
      tensor.Buffer().CopyFromHost(embed_data, mem_size);
    } else if (input_name.find("time_position_ids") != std::string::npos) {
      tensor.Buffer().CopyFromHost(decode_time_pos_.data(),
                                   decode_time_pos_.size() * sizeof(int32_t));
    } else if (input_name.find("hight_position_ids") != std::string::npos ||
               input_name.find("height_position_ids") != std::string::npos) {
      tensor.Buffer().CopyFromHost(decode_height_pos_.data(),
                                   decode_height_pos_.size() * sizeof(int32_t));
    } else if (input_name.find("width_position_ids") != std::string::npos) {
      tensor.Buffer().CopyFromHost(decode_width_pos_.data(),
                                   decode_width_pos_.size() * sizeof(int32_t));
    } else if (input_name.find("valid_length") != std::string::npos) {
      tensor.Buffer().CopyFromHost(&context_length_, mem_size);
    } else if (input_name.find("current_length") != std::string::npos) {
      tensor.Buffer().CopyFromHost(&current_length, mem_size);
    } else if (input_name.find("linear_attn_mask") != std::string::npos) {
      tensor.Buffer().CopyFromHost(decode_linear_attn_mask_.data(), mem_size);
    }
    decode_module->SetInput(input_name, tensor);
  }
}

void Qwen35MLLMContext::decode_inference() {
  auto* model = static_cast<LLMModel*>(model_);
  auto decode_module = model->decode_module();
  decode_module->Run();
  decode_module->Sync();
}

Token Qwen35MLLMContext::decode_postprocess(Sampler* sampler) {
  auto* model = static_cast<LLMModel*>(model_);
  auto decode_module = model->decode_module();

  const std::string& output_name = decode_module->GetOutputName(0);
  auto dev_output = decode_module->GetDevOutput(output_name);
  auto host_output = dev_output.ToHost(true);
  void* out_data = host_output.Buffer().Data();

  const int vocab_size = model->vocab_size();
  float16* logits = static_cast<float16*>(out_data);
  Token sampled_token = sampler->sample(logits, vocab_size, generated_ids_);

  context_length_++;

  return sampled_token;
}

Token Qwen35MLLMContext::do_decode_inference(Token prev_token,
                                             Sampler* sampler) {
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
// Qwen35MLLMModel Implementation
// ============================================================================

Qwen35MLLMModel::Qwen35MLLMModel(const ModelConfig& config) : VLMModel(config) {
  load();
}

int Qwen35MLLMModel::vision_patch_capacity(int gear) const {
  auto it = vision_patch_capacities_.find(gear);
  if (it == vision_patch_capacities_.end()) {
    throw Exception("dynamic vision gear is not loaded: m" + std::to_string(gear));
  }
  return it->second;
}

std::shared_ptr<tcim::Module> Qwen35MLLMModel::vision_module(int gear) const {
  auto it = vision_modules_.find(gear);
  return it == vision_modules_.end() ? nullptr : it->second;
}

std::string Qwen35MLLMModel::preprocessor_config_path() const {
  fs::path path(config_.tokenizer_path);
  if (fs::is_regular_file(path)) path = path.parent_path();
  return (path / "preprocessor_config.json").string();
}

std::unique_ptr<Context> Qwen35MLLMModel::create_context(int n_ctx) {
  if (n_ctx <= 0) {
    n_ctx = info_.n_ctx;
  }
  return std::make_unique<Qwen35MLLMContext>(this, n_ctx);
}

void Qwen35MLLMModel::ClearKVCache() {
  if (!prefill_module_ || !decode_module_) return;

  // Clear conv_cache and recurrent_state (if they exist)
    for (int idx = 0; idx < prefill_module_->GetInputNum(); idx++) {
      const auto input_name = prefill_module_->GetInputName(idx);
      const auto bare_name = BareTensorName(input_name);
      if (bare_name.find("conv_cache") == std::string::npos &&
          bare_name.find("recurrent_state") == std::string::npos) {
        continue;
      }

    // Clear prefill cache
    auto prefill_input_info =
        prefill_module_->GetInputInfo(input_name).AsContiguous();
    auto prefill_host_tensor =
        tcim::Tensor::CreateHostTensor(prefill_input_info);
    std::vector<uint8_t> zeros(prefill_host_tensor.MemSize(), 0);
    prefill_host_tensor.Buffer().CopyFromHost(zeros.data(), zeros.size());
    prefill_module_->SetInput(input_name, prefill_host_tensor);

    // Clear decode cache
    auto decode_input_info =
        decode_module_->GetInputInfo(input_name).AsContiguous();
    auto decode_host_tensor = tcim::Tensor::CreateHostTensor(decode_input_info);
    std::vector<uint8_t> decode_zeros(decode_host_tensor.MemSize(), 0);
    decode_host_tensor.Buffer().CopyFromHost(decode_zeros.data(),
                                             decode_zeros.size());
    decode_module_->SetInput(input_name, decode_host_tensor);
  }
}

void Qwen35MLLMModel::load() {
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

    auto input0_shape =
        prefill_module_->GetInputInfo(prefill_module_->GetInputName(0)).Shape();

    if (input0_shape.size() >= 3) {
      batch_ = input0_shape[0];
      prefill_length_ = input0_shape[1];
      embedding_length_ = input0_shape[2];
    }

    std::regex pattern("model_layers_(\\d+)_self_attn_kcache_input");
    for (int i = 0; i < prefill_module_->GetInputNum(); i++) {
      std::string name = BareTensorName(prefill_module_->GetInputName(i));
      std::smatch match;
      if (std::regex_search(name, match, pattern)) {
        int layer_idx = std::stoi(match[1].str());
        n_blocks_ = std::max(n_blocks_, layer_idx + 1);
      }
    }

    for (int i = 0; i < prefill_module_->GetInputNum(); i++) {
      std::string name = BareTensorName(prefill_module_->GetInputName(i));
      if (name.find("kcache_input") != std::string::npos ||
          name.find("vcache_input") != std::string::npos) {
        attn_idx_start_ = i;
        break;
      }
    }

    if (attn_idx_start_ > 0 && attn_idx_start_ < prefill_module_->GetInputNum()) {
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

    std::vector<std::string> dummy_names;
    for (int i = 0; i < prefill_module_->GetInputNum(); ++i) {
      const std::string name = prefill_module_->GetInputName(i);
      if (BareTensorName(name).find("model_layers") != std::string::npos) {
        dummy_names.push_back(name);
      }
    }

    auto option_decode = tcim::Module::Option(*weight_manager_);
    option_decode.SetDummyTensors(dummy_names);
    option_decode.EnableIOLazyMode(true);
    option_decode.EnableHostLazyLoading(config_.lazy_mode);

    decode_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(
        decode_module_->LoadModel(decode_path, option_decode));
    std::cout << "Decode model loaded: " << decode_path << std::endl;

    auto output0_shape =
        decode_module_->GetOutputInfo(decode_module_->GetOutputName(0)).Shape();
    if (output0_shape.size() >= 3) {
      info_.n_vocab = output0_shape[2];
      info_.n_logits = output0_shape[2];
    }
    std::cout << "  vocab_size: " << info_.n_vocab << std::endl;
  }

  // Step 4 - Load vision model
  const std::string& vision_path = config_.vision_path;
  if (!vision_path.empty()) {
    auto option_vision = tcim::Module::Option(*weight_manager_);
    option_vision.EnableIOLazyMode(true);
    option_vision.EnableHostLazyLoading(config_.lazy_mode);
    std::vector<std::pair<int, std::string>> vision_paths;
    for (int gear : VISION_GEARS) {
      const std::string suffix = "_visual_m" + std::to_string(gear) + ".hmm";
      const auto pos = vision_path.rfind("_visual.hmm");
      if (pos != std::string::npos) {
        vision_paths.emplace_back(gear, vision_path.substr(0, pos) + suffix);
      }
    }
    if (fs::is_regular_file(vision_path)) {
      int gear = 1536;
      std::regex gear_re(".*_m([0-9]+)\\.hmm$");
      std::smatch match;
      if (std::regex_match(vision_path, match, gear_re)) {
        gear = std::stoi(match[1].str());
        vision_paths.clear();
        vision_paths.emplace_back(gear, vision_path);
      } else {
        vision_paths.clear();
        vision_paths.emplace_back(gear, vision_path);
      }
    } else if (fs::is_directory(vision_path)) {
      vision_paths.clear();
      for (const auto& entry : fs::recursive_directory_iterator(vision_path)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".hmm") continue;
        std::regex gear_re(".*_m([0-9]+)\\.hmm$");
        std::smatch match;
        const std::string path = entry.path().string();
        if (std::regex_match(path, match, gear_re)) {
          vision_paths.emplace_back(std::stoi(match[1].str()), path);
        }
      }
    }
    std::sort(vision_paths.begin(), vision_paths.end());
    vision_paths.erase(std::unique(vision_paths.begin(), vision_paths.end()),
                       vision_paths.end());
    if (vision_paths.empty()) {
      throw Exception("no dynamic vision HMMs found from " + vision_path);
    }
    for (const auto& [gear, path] : vision_paths) {
      if (gear != 96 && gear != 196 && gear != 384 && gear != 704 && gear != 1536) {
        continue;
      }
      auto module = std::make_shared<tcim::Module>();
      CHECK_TCIM_RET_STATUS(module->LoadModel(path, option_vision));
      const auto names = module->GetInputNum();
      const std::array<std::string, 5> expected = {
          "pixel_values", "position_ids", "position_weights",
          "rotary_position_ids", "attention_mask"};
      if (names != static_cast<int>(expected.size())) {
        throw Exception("dynamic vision graph must expose five inputs: " + path);
      }
      for (int i = 0; i < names; ++i) {
        if (module->GetInputName(i) != expected[static_cast<size_t>(i)]) {
          throw Exception("unexpected dynamic vision input name in " + path);
        }
      }
      const auto shape = module->GetInputInfo("pixel_values").Shape();
      const int expected_capacity = gear * SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE;
      if (shape.size() != 3 || shape[0] != 1 || shape[1] != expected_capacity || shape[2] != 1536) {
        throw Exception("dynamic vision pixel_values shape mismatch: " + path);
      }
      const std::map<std::string, std::vector<int64_t>> expected_shapes = {
          {"position_ids", {4, expected_capacity}},
          {"position_weights", {4, expected_capacity}},
          {"rotary_position_ids", {2, expected_capacity}},
          {"attention_mask", {1, 1, 1, expected_capacity}},
      };
      for (const auto& [name, expected_shape] : expected_shapes) {
      const auto actual_shape = module->GetInputInfo(name).Shape();
        if (actual_shape.size() != expected_shape.size() ||
            !std::equal(actual_shape.begin(), actual_shape.end(), expected_shape.begin())) {
          throw Exception("dynamic vision input shape mismatch for " + name + ": " + path);
        }
      }
      vision_patch_capacities_[gear] = expected_capacity;
      vision_modules_[gear] = module;
      vision_gears_.push_back(gear);
      vision_patch_dim_ = shape[2];
      vision_hidden_size_ = module->GetOutputInfo(module->GetOutputName(0)).Shape().back();
      std::cout << "Vision gear m" << gear << " loaded: " << path << std::endl;
    }
    std::sort(vision_gears_.begin(), vision_gears_.end());
    if (vision_gears_.empty()) throw Exception("no supported dynamic vision gears loaded");
    vision_module_ = vision_modules_.at(vision_gears_.back());
    vision_input_name_ = "pixel_values";
    vision_output_name_ = vision_module_->GetOutputName(0);
  }

  // Step 5 - Share KV Cache
  {
    if (!prefill_module_ || !decode_module_) return;

    for (int idx = 0; idx < prefill_module_->GetInputNum(); idx++) {
      const std::string layer_name = prefill_module_->GetInputName(idx);
      const std::string bare_layer_name = BareTensorName(layer_name);

      // Share model_layers / past_key_cache / past_value_cache
      if (bare_layer_name.find("model_layers") != std::string::npos ||
          bare_layer_name.find("past_key_cache_") != std::string::npos ||
          bare_layer_name.find("past_value_cache_") != std::string::npos) {
        auto cache = prefill_module_->GetDevInput(layer_name);
        const std::string decode_input = FindInputByBare(decode_module_, bare_layer_name);
        if (!decode_input.empty()) {
          CHECK_TCIM_RET_STATUS(decode_module_->SetDevInput(decode_input, cache));
        }
      }

      // conv_cache: set prefill output, decode input, decode output
      if (bare_layer_name.find("conv_cache") != std::string::npos) {
        std::string output_name = bare_layer_name;
        const std::string prefix = "past_conv_cache_";
        if (output_name.rfind(prefix, 0) == 0) {
          output_name.replace(0, prefix.size(), "conv_cache_out_");
        }
        auto cache = prefill_module_->GetDevInput(layer_name);
        const std::string actual_prefill_output =
            FindOutputByBare(prefill_module_, output_name);
        const std::string decode_input =
            FindInputByBare(decode_module_, bare_layer_name);
        const std::string actual_decode_output =
            FindOutputByBare(decode_module_, output_name);
        if (actual_prefill_output.empty() || decode_input.empty() ||
            actual_decode_output.empty()) {
          throw Exception("conv cache graph contract mismatch for " + bare_layer_name);
        }
        CHECK_TCIM_RET_STATUS(
            prefill_module_->SetDevOutput(actual_prefill_output, cache));
        CHECK_TCIM_RET_STATUS(decode_module_->SetDevInput(decode_input, cache));
        CHECK_TCIM_RET_STATUS(
            decode_module_->SetDevOutput(actual_decode_output, cache));
      }

      // recurrent_state: set prefill output, decode input, decode output
      if (bare_layer_name.find("recurrent_state") != std::string::npos) {
        std::string output_name = bare_layer_name;
        const std::string prefix = "past_recurrent_state_";
        if (output_name.rfind(prefix, 0) == 0) {
          output_name.replace(0, prefix.size(), "recurrent_state_out_");
        }
        auto cache = prefill_module_->GetDevInput(layer_name);
        // Newer Qwen graphs (including the Qwen3.8 input contract) can keep
        // recurrent state only as an input tensor and omit the prefill output.
        const std::string actual_prefill_output =
            FindOutputByBare(prefill_module_, output_name);
        const std::string decode_input =
            FindInputByBare(decode_module_, bare_layer_name);
        const std::string actual_decode_output =
            FindOutputByBare(decode_module_, output_name);
        if (!actual_prefill_output.empty()) {
          CHECK_TCIM_RET_STATUS(
              prefill_module_->SetDevOutput(actual_prefill_output, cache));
        }
        if (decode_input.empty()) {
          throw Exception("decode recurrent input is missing for " + bare_layer_name);
        }
        CHECK_TCIM_RET_STATUS(decode_module_->SetDevInput(decode_input, cache));
        if (!actual_decode_output.empty()) {
          CHECK_TCIM_RET_STATUS(
              decode_module_->SetDevOutput(actual_decode_output, cache));
        }
      }
    }
    std::cout << "KV Cache shared" << std::endl;
  }

  // Step 6 - Load Embedding
  {
    const std::string& embedding_path = config_.embedding_path;
    embedding_ = std::make_shared<Embedding>(embedding_path, embedding_length_,
                                             prefill_length_);
    std::cout << "Embedding loaded: vocab_size=" << embedding_->vocab_size()
              << std::endl;
  }

  // Step 7 - Initialize input tensors
  {
    prefill_input_map_.clear();
    for (int idx = 0; idx < prefill_module_->GetInputNum(); ++idx) {
      auto input_name = prefill_module_->GetInputName(idx);
      if (IsCacheInput(input_name)) continue;
      auto input_info =
          prefill_module_->GetInputInfo(input_name).AsContiguous();
      tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
      prefill_input_map_[input_name] = input_tensor;
    }

    decode_input_map_.clear();
    for (int idx = 0; idx < decode_module_->GetInputNum(); ++idx) {
      auto input_name = decode_module_->GetInputName(idx);
      if (IsCacheInput(input_name)) continue;
      auto input_info = decode_module_->GetInputInfo(input_name).AsContiguous();
      tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
      decode_input_map_[input_name] = input_tensor;
    }
    std::cout << "Input tensors initialized" << std::endl;
  }

  // Step 8 - Fill model info
  {
    info_.type = ModelType::VLM;  // Qwen35MLLM is a VLM model
    info_.n_batch = batch_;
    info_.n_embd = embedding_length_;
    info_.n_layer = n_blocks_;
    info_.n_ctx = context_max_length_;
    info_.prefill_length = prefill_length_;
    info_.kv_cache_layers = n_blocks_;

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

  // Step 9 - Load Tokenizer
  {
    if (fs::exists(config_.tokenizer_path)) {
      try {
        fs::path tokenizer_path(config_.tokenizer_path);
        if (fs::is_regular_file(tokenizer_path)) {
          tokenizer_path = tokenizer_path.parent_path();
        }
        tokenizer_ = std::make_shared<HfTokenizer>(tokenizer_path.string());
        std::cout << "Tokenizer loaded from: " << tokenizer_path.string()
                  << std::endl;
      } catch (const Exception& e) {
        std::cerr << "Warning: Failed to load tokenizer from "
                  << config_.tokenizer_path << ": " << e.what() << std::endl;
      }
    }
  }

  // Step 10 - Initialize KV Cache (zero-fill)
  ClearKVCache();
  std::cout << "KV Cache initialized" << std::endl;
}

// ============================================================================
// Model Registration
// ============================================================================

// Static registration for Qwen35 MLLM model
REGISTER_MODEL(LLMModel, qwen35_mllm, ModelSeries::kQwen35MLLM,
    [](const ModelConfig& c) { return std::make_unique<Qwen35MLLMModel>(c); },
    "Qwen3.5 multimodal MLLM");

}  // namespace houmo
