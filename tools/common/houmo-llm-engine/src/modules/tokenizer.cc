/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: tokenizer.cc
 * Description:
 *   HuggingFace Tokenizer wrapper implementation.
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

#include "modules/tokenizer.h"

#include <tokenizers_cpp.h>

#include <fstream>
#include <iostream>

namespace houmo {

// 辅助函数：加载文件内容
static std::string loadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    throw Exception("Cannot open tokenizer file: " + path);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

HfTokenizer::HfTokenizer(const std::string& tokenizer_json_path) {
  // 加载 tokenizer.json
  auto blob = loadBytesFromFile(tokenizer_json_path);
  tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);

  // 获取特殊 token ID
  // 注意：不同模型的特殊 token 可能不同
  // Qwen 系列: BOS 通常不需要，EOS 用于结束
  bos_token_id_ = tokenizer_->Encode("<|endoftext|>")[0];
  eos_token_id_ = tokenizer_->Encode("<|im_end|>")[0];

  // PAD token ID 等于 BOS token ID (Qwen 系列)
  pad_token_id_ = bos_token_id_;
  // 获取词表大小
  vocab_size_ = static_cast<int>(tokenizer_->GetVocabSize());
  std::cout << "Tokenizer loaded. Vocab size: " << vocab_size_ << std::endl;
}

HfTokenizer::~HfTokenizer() { tokenizer_.reset(); }

std::vector<Token> HfTokenizer::encode(const std::string& text, bool add_bos,
                                       bool add_eos) {
  std::vector<Token> ids;

  // 编码文本
  auto encoded = tokenizer_->Encode(text);

  // 转换为 std::vector<Token>
  ids.reserve(encoded.size() + (add_bos ? 1 : 0) + (add_eos ? 1 : 0));

  if (add_bos && bos_token_id_ >= 0) {
    ids.push_back(bos_token_id_);
  }

  for (const auto& id : encoded) {
    ids.push_back(static_cast<Token>(id));
  }

  if (add_eos && eos_token_id_ >= 0) {
    ids.push_back(eos_token_id_);
  }

  return ids;
}

std::string HfTokenizer::decode(Token token) {
  std::vector<int32_t> ids = {static_cast<int32_t>(token)};
  return tokenizer_->Decode(ids);
}

std::string HfTokenizer::decode(const std::vector<Token>& tokens) {
  std::vector<int32_t> ids;
  ids.reserve(tokens.size());
  for (const auto& token : tokens) {
    ids.push_back(static_cast<int32_t>(token));
  }
  return tokenizer_->Decode(ids);
}

void HfTokenizer::set_pad_token_id(std::string text) {
  pad_token_id_ = tokenizer_->Encode(text)[0];
}
void HfTokenizer::set_bos_token_id(std::string text) {
  bos_token_id_ = tokenizer_->Encode(text)[0];
}
void HfTokenizer::set_eos_token_id(std::string text) {
  eos_token_id_ = tokenizer_->Encode(text)[0];
}

int HfTokenizer::token_to_id(const std::string& token) const {
  return tokenizer_->TokenToId(token);
}

}  // namespace houmo
