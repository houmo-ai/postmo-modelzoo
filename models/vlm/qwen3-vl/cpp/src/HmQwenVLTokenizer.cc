/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: HmQwenVLTokenizer.cc
 * Description:
 *   Tokenizer implementation for Qwen3-VL with vision support.
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

#include "HmQwenVLTokenizer.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

HmQwenVLTokenizer::HmQwenVLTokenizer(const std::string &tokenizerJsonPath,
                                     const std::string &embeddingWeightPath,
                                     const int embedding_len,
                                     const int prefill_len)
    : prefill_length_(prefill_len), embedding_length_(embedding_len) {
  // Load tokenizer using factory function
  auto tokenizer_json = LoadBytesFromFile(tokenizerJsonPath);
  tok_ = tokenizers::Tokenizer::FromBlobJSON(tokenizer_json);

  // Load embedding weights
  embed_w_ = readEmbeddingWeight<tensor_type>(embeddingWeightPath);
  if (!embed_w_) {
    std::cerr << "Failed to load embedding weights from " << embeddingWeightPath
              << std::endl;
    exit(1);
  }
}

HmQwenVLTokenizer::~HmQwenVLTokenizer() {
  if (ptr_) {
    delete[] ptr_;
    ptr_ = nullptr;
  }
}

std::string HmQwenVLTokenizer::ApplyChatTemplate(
    const std::string &role, const std::string &role_text,
    const std::vector<std::string> &image_paths,
    const std::string system_prompt, bool add_generation_prompt) {
  std::stringstream ss;

  if (!system_prompt.empty()) {
    // Python format: <|im_start|>system\n{content}<|im_end|>\n
    // No extra newline before <|im_end|>
    ss << "<|im_start|>system\n" << system_prompt << "<|im_end|>\n";
  }
  // User message with vision
  ss << "<|im_start|>" << role << "\n";

  // Add vision tokens for each image using <|image_pad|> placeholder
  for (size_t i = 0; i < image_paths.size(); i++) {
    ss << "<|vision_start|><|image_pad|><|vision_end|>";
  }

  // Add user text
  ss << role_text;
  ss << "<|im_end|>\n";

  // Add assistant prompt
  if (add_generation_prompt) {
    ss << "<|im_start|>assistant\n";
  }

  return ss.str();
}

std::string HmQwenVLTokenizer::ApplyChatTemplate(
    const std::string &text, const std::vector<std::string> &image_paths,
    bool add_generation_prompt) {
  std::stringstream ss;

  // User message with vision
  ss << "<|im_start|>user\n";

  // Add vision tokens for each image using <|image_pad|> placeholder
  for (size_t i = 0; i < image_paths.size(); i++) {
    ss << "<|vision_start|><|image_pad|><|vision_end|>";
  }

  // Add user text
  ss << text;
  ss << "<|im_end|>\n";

  // Add assistant prompt
  if (add_generation_prompt) {
    ss << "<|im_start|>assistant\n";
  }

  return ss.str();
}

std::vector<int> HmQwenVLTokenizer::Encode(const std::string &text) {
  std::vector<int32_t> ids = tok_->Encode(text);
  return std::vector<int>(ids.begin(), ids.end());
}

std::string HmQwenVLTokenizer::Decode(const std::vector<int32_t> &ids) {
  return tok_->Decode(ids);
}

tensor_type *HmQwenVLTokenizer::EmbeddingTokens(const std::vector<int> &ids) {
  size_t num_tokens = ids.size();
  size_t required_size = num_tokens * embedding_length_ * sizeof(tensor_type);

  if (ptr_size_ < required_size) {
    if (ptr_) {
      delete[] ptr_;
    }
    ptr_ = new tensor_type[num_tokens * embedding_length_];
    ptr_size_ = required_size;
  }

  // Lookup embedding for each token
  for (size_t i = 0; i < num_tokens; i++) {
    int token_id = ids[i];
    std::memcpy(ptr_ + i * embedding_length_,
                embed_w_.get() + token_id * embedding_length_,
                embedding_length_ * sizeof(tensor_type));
  }

  return ptr_;
}

int HmQwenVLTokenizer::CountVisionTokens(const std::string &text) {
  int count = 0;
  std::string vision_token = "<|image_pad|>";
  size_t pos = 0;
  while ((pos = text.find(vision_token, pos)) != std::string::npos) {
    count++;
    pos += vision_token.length();
  }
  return count;
}