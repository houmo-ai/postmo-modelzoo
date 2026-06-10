/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: tokenizer_test.cc
 * Description:
 *   Tokenizer module unit tests
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

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
namespace houmo {

class TokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tokenizer_path_ = "../tokenizers/qwen3-0.6b/tokenizer.json";
    if (!std::filesystem::exists(tokenizer_path_)) {
      GTEST_SKIP() << "Test tokenizer.json not found at: " << tokenizer_path_;
    }
  }

  std::string tokenizer_path_;
};

TEST_F(TokenizerTest, LoadTokenizer) {
  EXPECT_NO_THROW({ HfTokenizer tokenizer(tokenizer_path_); });
}

TEST_F(TokenizerTest, EncodeDecode) {
  HfTokenizer tokenizer(tokenizer_path_);

  // Test encoding
  std::string text = "Hello, world!";
  auto tokens = tokenizer.encode(text, false, false);
  EXPECT_GT(tokens.size(), 0) << "Should produce at least one token";

  // Test decoding
  std::string decoded = tokenizer.decode(tokens);
  EXPECT_EQ(decoded, text) << "Encode/Decode should be reversible";
}

TEST_F(TokenizerTest, EncodeChinese) {
  HfTokenizer tokenizer(tokenizer_path_);

  std::string text = "你好世界";
  auto tokens = tokenizer.encode(text, false, false);
  EXPECT_GT(tokens.size(), 0) << "Chinese text should be tokenized";

  std::string decoded = tokenizer.decode(tokens);
  EXPECT_EQ(decoded, text);
}

TEST_F(TokenizerTest, EncodeWithBosEos) {
  HfTokenizer tokenizer(tokenizer_path_);

  std::string text = "test";
  auto tokens_no_special = tokenizer.encode(text, false, false);
  auto tokens_with_bos = tokenizer.encode(text, true, false);
  auto tokens_with_eos = tokenizer.encode(text, false, true);
  auto tokens_with_both = tokenizer.encode(text, true, true);

  // With BOS should have one more token
  EXPECT_EQ(tokens_with_bos.size(), tokens_no_special.size() + 1);
  // With EOS should have one more token
  EXPECT_EQ(tokens_with_eos.size(), tokens_no_special.size() + 1);
  // With both should have two more tokens
  EXPECT_EQ(tokens_with_both.size(), tokens_no_special.size() + 2);
}

TEST_F(TokenizerTest, VocabSize) {
  HfTokenizer tokenizer(tokenizer_path_);
  int vocab_size = tokenizer.vocab_size();
  EXPECT_GT(vocab_size, 0) << "Vocab size should be positive";
  // Qwen series typically has 151936 or 152064 tokens
  EXPECT_GT(vocab_size, 100000) << "Vocab size should be large for LLM";
}

TEST_F(TokenizerTest, SpecialTokens) {
  HfTokenizer tokenizer(tokenizer_path_);

  Token bos = tokenizer.bos_token_id();
  Token eos = tokenizer.eos_token_id();

  // Special token IDs should be valid
  EXPECT_GE(bos, 0);
  EXPECT_GE(eos, 0);
  EXPECT_NE(bos, eos) << "BOS and EOS should be different";
}

TEST_F(TokenizerTest, DecodeSingleToken) {
  HfTokenizer tokenizer(tokenizer_path_);

  // Decode single token
  std::string decoded = tokenizer.decode(100);
  EXPECT_FALSE(decoded.empty())
      << "Single token decode should return something";
}

}  // namespace houmo
