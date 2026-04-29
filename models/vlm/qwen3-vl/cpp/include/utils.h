/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: utils.h
 * Description:
 *   Utility functions and structures for Qwen3-VL common operations.
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

#ifndef __QWEN3VL_UTILS_H__
#define __QWEN3VL_UTILS_H__

#include <tokenizers_cpp.h>

#include <cassert>
#include <chrono>
#include <codecvt>
#include <cstring>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <half.hpp>
#include <iostream>
#include <locale>
#include <string>
#include <vector>

using half_float::half;

/**
 * @brief Structure to represent a message with role and content
 */
struct Message {
  std::string role;     // Role of the message sender
  std::string content;  // Content of the message
};

/**
 * @brief Structure for model parameters
 */
struct ModelConfig {
  int image_token_id = 151655;
  int video_token_id = 151656;
  int vision_start_token_id = 151652;
  int vision_end_token_id = 151653;
  int vision_token_id = 151654;
  int pad_token_id = 0;
  int eos_token_id = 151645;
  int spatial_merge_size = 2;
  int patch_size = 16;
  int temporal_patch_size = 2;
  int image_size_w = 448;
  int image_size_h = 448;
};

/**
 * @brief Structure for image dimensions
 */
struct ImageDims {
  int width;
  int height;
  int channels;
};

/**
 * @brief Load binary data from a file
 */
static std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

/**
 * @brief Read embedding weights from a binary file
 */
template <typename T>
std::unique_ptr<T[]> readEmbeddingWeight(const std::string &path,
                                         size_t n_elems_align = 0) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  const std::size_t n_bytes = ifs.tellg();
  ifs.seekg(0);

  const std::size_t n_elem = n_bytes / sizeof(T) + n_elems_align;
  auto ptr = std::make_unique<T[]>(n_elem);
  ifs.read(reinterpret_cast<char *>(ptr.get()), n_bytes);
  ifs.close();
  memset(reinterpret_cast<char *>(ptr.get()) + n_bytes, 0,
         n_elems_align * sizeof(T));
  return ptr;
}

/**
 * @brief Calculate the length of a UTF-8 string in UTF-32 encoding
 */
static std::size_t utf8_len(std::string_view u8) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  return conv.from_bytes(u8.data(), u8.data() + u8.size()).size();
}

/**
 * @brief Convert a UTF-8 string to UTF-32 string
 */
static std::u32string utf8_to_u32(const std::string &u8) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  return conv.from_bytes(u8);
}

/**
 * @brief Convert a UTF-32 string to UTF-8 string
 */
static std::string u32_to_utf8(const std::u32string &u32) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  return conv.to_bytes(u32);
}

/**
 * @brief Check if a Unicode code point is a valid character (CJK or ASCII
 * letters)
 */
static bool is_valid_char(char32_t cp) noexcept {
  return (cp >= 0x4E00u && cp <= 0x9FFFu) || (cp >= 0x3400u && cp <= 0x4DBFu) ||
         (cp >= 0x20000u && cp <= 0x2A6DFu) ||
         (cp >= 0x2A700u && cp <= 0x2B73Fu) ||
         (cp >= 0x2B740u && cp <= 0x2B81Fu) ||
         (cp >= 0x2B820u && cp <= 0x2CEAFu) ||
         (cp >= 0xF900u && cp <= 0xFAFFu) ||
         (cp >= 0x2F800u && cp <= 0x2FA1Fu) ||
         (cp >= 0x0041u && cp <= 0x005Au) || (cp >= 0x0061u && cp <= 0x007Au);
}

/**
 * @brief Compute the index of the maximum value in an array using Eigen library
 */
template <typename T>
static int eigen_argmax(const T *ptr, std::size_t n) {
  using Eigen::Tensor;
  using Eigen::TensorMap;

  TensorMap<Tensor<const T, 1>> tm(static_cast<const T *>(ptr), n);
  Eigen::Tensor<Eigen::Index, 0> t = tm.argmax();
  Eigen::Index idx = t(0);

  return static_cast<int>(idx);
}

/**
 * @brief Structure for performance metrics
 */
struct PerfInfos {
  int batch_size = 1;
  int num_images = 0;
  int input_tokens = 0;
  int output_tokens = 0;
  int visual_tokens = 0;

  float prefill_model_load_time = 0.0f;
  float decode_model_load_time = 0.0f;
  float vision_model_load_time = 0.0f;

  float vision_time = 0.0f;
  float vision_preprocess_time = 0.0f;
  float vision_set_input_time = 0.0f;
  float vision_infer_time = 0.0f;
  float vision_get_output_time = 0.0f;

  float prefill_time = 0.0f;
  float prefill_tokenization_time = 0.0f;
  float prefill_embedding_time = 0.0f;
  float prefill_set_input_time = 0.0f;
  float prefill_infer_time = 0.0f;
  float prefill_get_output_time = 0.0f;

  float decode_time = 0.0f;
  float decode_tokenization_time = 0.0f;
  float decode_embedding_time = 0.0f;
  float decode_set_input_time = 0.0f;
  float decode_infer_time = 0.0f;
  float decode_get_output_time = 0.0f;

  float embedding_time = 0.0f;
  float ttft_time = 0.0f;
  float total_time = 0.0f;
};

#endif  // __QWEN3VL_UTILS_H__