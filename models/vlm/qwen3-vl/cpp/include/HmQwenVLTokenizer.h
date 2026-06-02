/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: HmQwenVLTokenizer.h
 * Description:
 *   Tokenizer interface for Qwen3-VL model with vision support.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __HMQWEN_VL_TOKENIZER_H__
#define __HMQWEN_VL_TOKENIZER_H__

#include "utils.h"

using half_float::half;
using tokenizers::Tokenizer;
using tensor_type = half;

/**
 * @brief HmQwenVLTokenizer Class - Handles tokenization for vision-language
 * model
 *
 * This class provides functionalities for:
 * 1. Applying chat templates with vision support
 * 2. Encoding text into token IDs
 * 3. Decoding token IDs back into text
 * 4. Generating embeddings from token IDs
 * 5. Handling special vision tokens
 */
class HmQwenVLTokenizer {
 public:
  /**
   * @brief Constructor - Initializes the tokenizer
   * @param tokenizerJsonPath Path to tokenizer.json
   * @param embeddingWeightPath Path to embedding weights
   * @param embedding_len Length of each embedding vector
   * @param prefill_len Maximum prefill length
   */
  HmQwenVLTokenizer(const std::string &tokenizerJsonPath,
                    const std::string &embeddingWeightPath,
                    const int embedding_len, const int prefill_len);

  HmQwenVLTokenizer(const HmQwenVLTokenizer &it) = delete;
  HmQwenVLTokenizer &operator=(const HmQwenVLTokenizer &it) = delete;
  HmQwenVLTokenizer(HmQwenVLTokenizer &&it) noexcept = default;
  HmQwenVLTokenizer &operator=(HmQwenVLTokenizer &&it) noexcept = default;
  ~HmQwenVLTokenizer();

  /**
   * @brief Apply chat template with vision support
   * @param text User text prompt
   * @param image_paths Vector of image paths (can be empty for text-only)
   * @param add_generation_prompt Whether to add generation prompt
   * @return Formatted text string ready for encoding
   */
  std::string ApplyChatTemplate(const std::string &text,
                                const std::vector<std::string> &image_paths,
                                bool add_generation_prompt = true);
  std::string ApplyChatTemplate(const std::string &role,
                                const std::string &role_text,
                                const std::vector<std::string> &image_paths,
                                const std::string system_prompt,
                                bool add_generation_prompt = true);
  /**
   * @brief Encode text to token IDs
   * @param text Text to encode
   * @return Vector of token IDs
   */
  std::vector<int> Encode(const std::string &text);

  /**
   * @brief Decode token IDs to text
   * @param ids Vector of token IDs
   * @return Decoded text
   */
  std::string Decode(const std::vector<int32_t> &ids);

  /**
   * @brief Generate embeddings from token IDs
   * @param ids Vector of token IDs
   * @return Pointer to embedding tensor
   */
  tensor_type *EmbeddingTokens(const std::vector<int> &ids);

  /**
   * @brief Count vision tokens in text
   * @param text Text to analyze
   * @return Number of vision tokens
   */
  int CountVisionTokens(const std::string &text);

  /**
   * @brief Get special token IDs
   */
  int GetImageTokenId() const { return config_.image_token_id; }
  int GetVideoTokenId() const { return config_.video_token_id; }
  int GetVisionStartTokenId() const { return config_.vision_start_token_id; }
  int GetVisionEndTokenId() const { return config_.vision_end_token_id; }
  int GetEosTokenId() const { return config_.eos_token_id; }
  int GetPadTokenId() const { return config_.pad_token_id; }

 private:
  std::unique_ptr<Tokenizer> tok_;
  std::unique_ptr<tensor_type[]> embed_w_;
  tensor_type *ptr_ = nullptr;
  size_t ptr_size_ = 0;
  int prefill_length_ = 0;
  int embedding_length_ = 0;
  ModelConfig config_;
};

#endif  // __HMQWEN_VL_TOKENIZER_H__
