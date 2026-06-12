/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: vlm_model.h
 * Description:
 *   VLM (Vision-Language Model) base class. Inherits from LLMModel and
 *   adds vision encode support.
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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/tcim_utils.h"
#include "core/llm_model.h"

namespace houmo {

/**
 * @brief VLM model base class
 *
 * Inherits LLMModel, adds vision encode (Vision Encoder) support.
 */
class VLMModel : public LLMModel {
 public:
  /**
   * @brief Constructor
   * @param config Model configuration (must include vision_path)
   */
  explicit VLMModel(const ModelConfig& config);
  ~VLMModel() override = default;

  // ========== Type info ==========

  ModelType type() const override { return ModelType::VLM; }

  // ========== Vision interface ==========

  /**
   * @brief Get vision encode module
   */
  std::shared_ptr<tcim::Module> vision_module() const { return vision_module_; }

  /**
   * @brief Get image embedding after encoding
   * @param image_data Image data (RGB format)
   * @param width Image width
   * @param height Image height
   * @param channels Number of channels (usually 3)
   * @return Vision embedding vector
   */
  virtual std::vector<float16> encode_image(const uint8_t* image_data,
                                            int width, int height,
                                            int channels);

  // ========== Context creation ==========

  /**
   * @brief Create VLM inference context
   * @param n_ctx Context length (default from model config)
   * @return Context smart pointer
   */
  std::unique_ptr<Context> create_context(int n_ctx = 0) override;

  // ========== Internal interface (for Context use) ==========

  using LLMModel::attn_idx_start;
  using LLMModel::decode_input_map;
  using LLMModel::prefill_input_map;

  /**
   * @brief Get vision encode input map
   */
  std::map<std::string, tcim::Tensor>& vision_input_map() const {
    return const_cast<std::map<std::string, tcim::Tensor>&>(vision_input_map_);
  }

 protected:
  // Vision encode
  std::shared_ptr<tcim::Module> vision_module_;
  std::map<std::string, tcim::Tensor> vision_input_map_;

  // Vision encode parameters
  int vision_image_size_ = 448;  // Default image size
  int vision_patch_size_ = 16;   // Patch size
  int vision_hidden_size_ = 0;   // Vision encode hidden dimension
};

}  // namespace houmo
