/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen35_mllm_model.h
 * Description:
 *   Qwen3.5 multimodal LLM/VLM model implementation. Supports both
 *   text-only and vision-language inference with M-RoPE position encoding.
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

#ifndef HOUMO_QWEN35_MODEL_H
#define HOUMO_QWEN35_MODEL_H

#include "core/context.h"
#include "core/vlm_model.h"
#include "qwen35_dynamic_image_processor.h"

#include <array>
#include <map>

namespace houmo {

// VLM constants
constexpr int IMAGE_TOKEN_ID = 248056;
constexpr int VIDEO_TOKEN_ID = 248057;
constexpr int VISION_START_TOKEN_ID = 248053;
constexpr int VISION_END_TOKEN_ID = 248054;
constexpr int SPATIAL_MERGE_SIZE = 2;
constexpr int PATCH_SIZE = 16;
/** Number of frames grouped into one temporal patch. */
constexpr int TEMPORAL_PATCH_SIZE = 2;
/** Number of learned 2D position embeddings. */
constexpr int NUM_POSITION_EMBEDDINGS = 2304;
/** Maximum position covered by the visual rotary cache. */
constexpr int VISUAL_ROPE_CACHE_LENGTH = 3072;
/** Supported post-merge visual token capacities. */
constexpr int VISION_GEARS[] = {96, 196, 384, 704, 1536};

using ImageGridTHW = std::tuple<int, int, int>;

/**
 * @brief Qwen3.5 inference context
 */
class Qwen35MLLMContext : public Context {
 public:
  explicit Qwen35MLLMContext(LLMModel* model, int n_ctx);
  ~Qwen35MLLMContext() override = default;

  // Override inference methods
  Token prefill(const std::vector<Token>& tokens) override;
  Token decode(Token prev_token) override;

  // Token callback generation
  void generate(const std::vector<Token>& prompt, const SamplingParams& params,
                std::function<bool(Token)> callback) override;

  // Reset context state (including KV Cache)
  void reset() override;

  void set_image(const std::string& image_path) override {
    image_paths_.emplace_back(image_path);
  }

  void set_images(const std::vector<std::string>& image_paths) {
    image_paths_ = image_paths;
  }

 private:
  // ========== Vision split methods (for internal profiling) ==========
  void vision_preprocess(int image_idx, int gear);
  void vision_inference(int gear);
  void vision_postprocess(int image_idx);

  // ========== Prefill split methods (for internal profiling) ==========
  /**
   * @brief Prefill common setup (execute once)
   * @param padded_tokens Expanded token sequence
   * @return (position_ids_3d, seq_length, prefill_loop_chunk)
   */
  std::tuple<std::vector<std::vector<int32_t>>, int32_t, int>
  prefill_common_setup(const std::vector<Token>& padded_tokens);

  void prefill_preprocess_chunk(
      int chunk, const std::vector<Token>& padded_tokens, int32_t seq_length,
      int prefill_length,
      const std::vector<std::vector<int32_t>>& position_ids_3d);
  void prefill_inference_chunk();
  Token prefill_postprocess(Sampler* sampler, int32_t seq_length);

  // Decode split methods (for internal profiling)
  void decode_preprocess(Token prev_token);
  void decode_inference();
  Token decode_postprocess(Sampler* sampler);

  // Internal interface
  Token do_prefill_inference(const std::vector<Token>& tokens,
                             Sampler* sampler);
  Token do_decode_inference(Token prev_token, Sampler* sampler);

  std::vector<Token> pad_visual_token(const std::vector<Token>& tokens);

  // VLM helper functions
  void run_vision();
  void process_image_inputs();
  void scatter_image_embeds(const std::vector<Token>& input_ids,
                            std::vector<float16>& token_embeds, size_t seq_len,
                            size_t embed_dim, size_t chunk_start_in_full_seq);

  // Compute M-RoPE position IDs
  // Returns: position_ids[3][seq_len], rope_deltas
  std::pair<std::vector<std::vector<int32_t>>, int32_t> get_rope_index(
      const std::vector<Token>& input_ids,
      const std::vector<ImageGridTHW>& image_grid_thw);

  int32_t rope_deltas_ = 0;
  bool use_vlm_ = false;
  size_t image_embed_offset_ = 0;  // Current image embed offset

  std::vector<std::string> image_paths_;
  std::vector<float16>
      flat_image_embeds_;  // Concatenated image embeds [total_patches * embed_dim]
  std::vector<ImageGridTHW> image_grid_thw_;  // Per-image (t, h, w)

  // Vision processing temp state
  qwen35::DynamicImageResult current_processed_image_;
  std::vector<float16> current_vision_tensor_;

  // Prefill chunk temp state
  std::vector<float16> chunk_embeds_;
  std::vector<int32_t> chunk_time_pos_;
  std::vector<int32_t> chunk_height_pos_;
  std::vector<int32_t> chunk_width_pos_;
  std::vector<float16> linear_attn_mask_;

  // Decode temp state
  std::vector<int32_t> decode_time_pos_;
  std::vector<int32_t> decode_height_pos_;
  std::vector<int32_t> decode_width_pos_;
  std::vector<float16> decode_linear_attn_mask_;

  // Image processor
  std::shared_ptr<qwen35::DynamicImageProcessor> img_processor_;
};

/**
 * @brief Qwen3.5 multimodal LLM/VLM model
 *
 * Inherits VLMModel, implements complete loading workflow.
 */
class Qwen35MLLMModel : public VLMModel {
 public:
  explicit Qwen35MLLMModel(const ModelConfig& config);
  ~Qwen35MLLMModel() override = default;

  // Override create_context to create Qwen35MLLMContext
  std::unique_ptr<Context> create_context(int n_ctx = 0) override;

  // Clear KV Cache (for resetting dialogue state)
  void ClearKVCache();

  // ========== VLM methods ==========

  // Run vision model to get image embeddings
  std::vector<float> run_vision(
      const std::vector<std::vector<float>>& pixel_values_list);

  // Image embedding replacement
  std::vector<float> scatter_image_embeds(
      const std::vector<Token>& input_ids,
      const std::vector<float>& token_embeds,
      const std::vector<float>& image_embeds,
      Token image_token_id = IMAGE_TOKEN_ID);

  // Compute M-RoPE position IDs
  std::pair<std::vector<int32_t>, int32_t> get_rope_index(
      const std::vector<Token>& input_ids,
      const std::vector<ImageGridTHW>& image_grid_thw);

  // Build message containing images
  std::string build_messages(const std::string& prompt,
                             const std::vector<std::string>& image_paths,
                             int max_size_h, int max_size_w);

  int vision_patch_dim() const { return vision_patch_dim_; }
  int vision_patch_capacity(int gear) const;
  std::shared_ptr<tcim::Module> vision_module(int gear) const;
  const std::vector<int>& vision_gears() const { return vision_gears_; }
  std::string preprocessor_config_path() const;

 private:
  void load();

  // Internal state
  int n_blocks_ = 0;
  int batch_ = 0;
  int embedding_length_ = 0;
  int context_max_length_ = 0;

  // Vision configuration
  std::string vision_input_name_;
  std::string vision_output_name_;
  std::map<int, std::shared_ptr<tcim::Module>> vision_modules_;
  std::map<int, int> vision_patch_capacities_;
  std::vector<int> vision_gears_;
  int vision_patch_dim_ = 0;
  int vision_hidden_size_ = 0;

};

}  // namespace houmo

#endif  // HOUMO_QWEN35_LLM_MODEL_H
