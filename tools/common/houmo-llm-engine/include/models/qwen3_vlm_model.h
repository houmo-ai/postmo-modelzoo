/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_vlm_model.h
 * Description:
 *   Qwen3-VL (Vision Language Model) implementation. Inherits VLMModel
 *   and implements vision encode, M-RoPE, and Deepstack architecture.
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

#ifndef HOUMO_QWEN3_VLM_MODEL_H
#define HOUMO_QWEN3_VLM_MODEL_H

#include "core/context.h"
#include "core/vlm_model.h"
#include "modules/image_processor.h"

namespace houmo {

// ============================================================================
// Qwen3-VL constants
// ============================================================================

// Token IDs (Qwen3-VL specific)
constexpr int QWEN3_VL_IMAGE_TOKEN_ID = 151655;
constexpr int QWEN3_VL_VIDEO_TOKEN_ID = 151656;
constexpr int QWEN3_VL_VISION_START_TOKEN_ID = 151652;
constexpr int QWEN3_VL_VISION_END_TOKEN_ID = 151653;

// Vision configuration
constexpr int QWEN3_VL_PATCH_SIZE = 16;
constexpr int QWEN3_VL_SPATIAL_MERGE_SIZE = 2;

// Image grid type: (t, h, w)
using ImageGridTHW = std::tuple<int, int, int>;

// ============================================================================
// Qwen3VLMContext
// ============================================================================

/**
 * @brief Qwen3-VL inference context
 *
 * Implements Qwen3-VL inference logic including:
 * - Vision encode inference
 * - M-RoPE position IDs computation
 * - Image embedding replacement
 * - Prefill/Decode inference
 */
class Qwen3VLMContext : public Context {
 public:
  /**
   * @brief Constructor
   * @param model Qwen3-VLM model pointer
   * @param n_ctx Context length
   */
  explicit Qwen3VLMContext(LLMModel* model, int n_ctx);
  ~Qwen3VLMContext() override = default;

  // ========== Inference interface ==========

  /**
   * @brief Prefill (process prompt + image)
   * @param tokens Input token sequence
   * @return First generated token
   */
  Token prefill(const std::vector<Token>& tokens) override;

  /**
   * @brief Decode (autoregressive generation)
   * @param prev_token Previously generated token
   * @return Next generated token
   */
  Token decode(Token prev_token) override;

  /**
   * @brief Stream generation (Token callback mode)
   * @param prompt Input tokens
   * @param params Sampling parameters
   * @param callback Per-token callback function
   */
  void generate(const std::vector<Token>& prompt, const SamplingParams& params,
                std::function<bool(Token)> callback) override;

  /**
   * @brief Reset context state
   */
  void reset() override;

  // ========== Image interface ==========

  /**
   * @brief Set input image path
   * @param image_path Image file path
   */
  void set_image(const std::string& image_path) override;

  /**
   * @brief Set multiple input image paths
   * @param image_paths List of image file paths
   */
  void set_images(const std::vector<std::string>& image_paths);

  /**
   * @brief Check if image is set
   */
  bool has_image() const { return !image_paths_.empty(); }

 private:
  // ========== Vision split methods (for internal profiling) ==========
  void vision_preprocess(int image_idx);
  void vision_inference();
  void vision_postprocess(int image_idx);

  // ========== Prefill split methods (for internal profiling) ==========
  /**
   * @brief Prefill common setup (execute once)
   * @param tokens Input token sequence
   * @return (position_ids_3d, seq_length, prefill_loop_chunk)
   */
  std::tuple<std::vector<std::vector<int32_t>>, int32_t, int>
  prefill_common_setup(const std::vector<Token>& tokens);

  void prefill_preprocess_chunk(
      int chunk, int32_t seq_length, int prefill_length,
      const std::vector<std::vector<int32_t>>& position_ids_3d);
  void prefill_inference_chunk();
  Token prefill_postprocess(Sampler* sampler, int32_t seq_length);

  // ========== Decode split methods (for internal profiling) ==========
  void decode_preprocess(Token prev_token);
  void decode_inference();
  Token decode_postprocess(Sampler* sampler);

  // ========== Internal inference methods ==========

  /**
   * @brief Execute prefill inference
   */
  Token do_prefill_inference(const std::vector<Token>& tokens,
                             Sampler* sampler);

  /**
   * @brief Execute decode inference
   */
  Token do_decode_inference(Token prev_token, Sampler* sampler);

  // ========== VLM helper methods ==========

  /**
   * @brief Run vision encode
   * @param pixel_values Image pixel data (NCHW YUV format)
   * @return Vision features (image_features, deepstack_0, deepstack_1,
   * deepstack_2)
   */
  std::tuple<std::vector<float16>, std::vector<float16>, std::vector<float16>,
             std::vector<float16>>
  run_vision(const std::vector<float16>& pixel_values);

  /**
   * @brief Run vision encode for all images
   * Populates flat_image_embeds_ and image_grid_thw_
   */
  void run_vision();

  /**
   * @brief Expand image tokens
   * Replaces single <|image_pad|> token with multiple tokens
   */
  std::vector<Token> expand_image_tokens(const std::vector<Token>& tokens);

  /**
   * @brief Replace image embeddings
   * Scatters vision features into image token positions
   */
  void scatter_image_embeds(std::vector<float16>& inputs_embeds,
                            const std::vector<Token>& input_ids, size_t seq_len,
                            size_t embed_dim);

  /**
   * @brief Compute M-RoPE Position IDs
   * @param input_ids Input token sequence
   * @param image_grid_thw Per-image (t, h, w)
   * @return position_ids[3][seq_len], rope_deltas
   */
  std::pair<std::vector<std::vector<int32_t>>, int32_t> get_rope_index(
      const std::vector<Token>& input_ids,
      const std::vector<ImageGridTHW>& image_grid_thw);

  /**
   * @brief Set prefill inputs
   */
  void set_prefill_inputs(const std::vector<float16>& inputs_embeds,
                          const std::vector<int32_t>& time_pos_ids,
                          const std::vector<int32_t>& height_pos_ids,
                          const std::vector<int32_t>& width_pos_ids,
                          int32_t valid_length, int32_t current_length,
                          const std::vector<float16>& deepstack_0,
                          const std::vector<float16>& deepstack_1,
                          const std::vector<float16>& deepstack_2);

  /**
   * @brief Set decode inputs
   */
  void set_decode_inputs(const std::vector<float16>& inputs_embeds,
                         const std::vector<int32_t>& time_pos_ids,
                         const std::vector<int32_t>& height_pos_ids,
                         const std::vector<int32_t>& width_pos_ids,
                         int32_t valid_length,
                         const std::vector<float16>& deepstack_0,
                         const std::vector<float16>& deepstack_1,
                         const std::vector<float16>& deepstack_2);

  // ========== Member variables ==========

  // Image related
  std::vector<std::string> image_paths_;
  std::shared_ptr<HmImageProcessor> image_processor_;

  // Vision outputs
  std::vector<float16> flat_image_embeds_;    // Concatenated image embeds
  std::vector<float16> deepstack_0_;          // Deepstack layer 0
  std::vector<float16> deepstack_1_;          // Deepstack layer 1
  std::vector<float16> deepstack_2_;          // Deepstack layer 2
  std::vector<ImageGridTHW> image_grid_thw_;  // Per-image (t, h, w)

  // Vision processing temp state
  ProcessedImage current_processed_image_;
  std::vector<float16> current_vision_tensor_;
  std::tuple<std::vector<float16>, std::vector<float16>, std::vector<float16>,
             std::vector<float16>>
      current_vision_output_;

  // Prefill chunk temp state
  std::vector<float16> chunk_embeds_;
  std::vector<int32_t> chunk_time_pos_;
  std::vector<int32_t> chunk_height_pos_;
  std::vector<int32_t> chunk_width_pos_;

  // Decode temp state
  std::vector<int32_t> decode_time_pos_;
  std::vector<int32_t> decode_height_pos_;
  std::vector<int32_t> decode_width_pos_;

  // RoPE state
  int32_t rope_deltas_ = 0;
  bool use_vlm_ = false;

  // KV Cache state
  int32_t past_seq_len_ = 0;
};

// ============================================================================
// Qwen3VLMModel
// ============================================================================

/**
 * @brief Qwen3-VL model
 *
 * Inherits VLMModel, implements complete loading and inference workflow.
 *
 * Model files:
 *   - *_visual_*.hmm: Vision encode
 *   - *_prefill.hmm: Prefill module
 *   - *_decode.hmm: Decode module
 *   - quant_embedding.bin: Embedding table
 */
class Qwen3VLMModel : public VLMModel {
 public:
  /**
   * @brief Constructor
   * @param config Model configuration
   *
   * config must contain:
   *   - prefill_path: Prefill module path
   *   - decode_path: Decode module path
   *   - embedding_path: Embedding table path
   *   - vision_path: Vision encode path
   *   - tokenizer_path: Tokenizer path
   */
  explicit Qwen3VLMModel(const ModelConfig& config);
  ~Qwen3VLMModel() override = default;

  // ========== Context creation ==========

  /**
   * @brief Create Qwen3-VLM inference context
   * @param n_ctx Context length (default from model config)
   * @return Qwen3VLMContext smart pointer
   */
  std::unique_ptr<Context> create_context(int n_ctx = 0) override;

  // ========== Vision encoding ==========

  /**
   * @brief Encode image
   * @param pixel_values Image pixel data (NCHW YUV format)
   * @return Vision features (image_features, deepstack_0, deepstack_1,
   * deepstack_2)
   */
  std::tuple<std::vector<float16>, std::vector<float16>, std::vector<float16>,
             std::vector<float16>>
  encode_image(const std::vector<float16>& pixel_values);

  // ========== Model info ==========

  int vision_image_size_h() const { return vision_image_size_h_; }
  int vision_image_size_w() const { return vision_image_size_w_; }
  int patch_size() const { return patch_size_; }
  int spatial_merge_size() const { return spatial_merge_size_; }

  // ========== Vision module access (internal use) ==========

  std::shared_ptr<tcim::Module> vision_module() const { return vision_module_; }
  std::map<std::string, tcim::Tensor>& vision_input_map() {
    return vision_input_map_;
  }

  // ========== KV Cache management ==========

  /**
   * @brief Clear KV Cache
   */
  void ClearKVCache();

 private:
  /**
   * @brief Load model
   */
  void load();

  /**
   * @brief Initialize vision encode inputs
   */
  void init_vision_inputs();

  /**
   * @brief Initialize prefill inputs
   */
  void init_prefill_inputs();

  /**
   * @brief Initialize decode inputs
   */
  void init_decode_inputs();

  /**
   * @brief Get number of transformer layers
   */
  int get_n_blocks();

  /**
   * @brief Get attention index start position
   */
  int get_attn_idx_start();

  // ========== Model parameters ==========
  int n_blocks_ = 0;
  int batch_ = 0;
  int embedding_length_ = 0;
  int context_max_length_ = 0;
  int argmax_dim_len_ = 0;

  // Vision parameters
  int vision_image_size_h_ = 448;
  int vision_image_size_w_ = 448;
  int patch_size_ = QWEN3_VL_PATCH_SIZE;
  int spatial_merge_size_ = QWEN3_VL_SPATIAL_MERGE_SIZE;
  int vision_input_nums_ = 0;

  // Vision tensor
  std::map<std::string, tcim::Tensor> vision_input_map_;
};

}  // namespace houmo

#endif  // HOUMO_QWEN3_VLM_MODEL_H
