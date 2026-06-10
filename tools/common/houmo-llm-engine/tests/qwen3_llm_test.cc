/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_llm_test.cc
 * Description:
 *   Qwen3 LLM model tests
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

#include <gtest/gtest.h>

#include <filesystem>

#include "models/qwen3_llm_model.h"
#include "modules/sampler.h"

namespace fs = std::filesystem;

namespace houmo {

class Qwen3LLMModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Use environment variable path
    prefill_path_ = "../models/qwen3-0.6b/qwen3-0.6b_prefill.hmm";
    decode_path_ = "../models/qwen3-0.6b/qwen3-0.6b_decode.hmm";
    embedding_path_ = "../models/qwen3-0.6b/hmquant/quant_embedding.bin";
    tokenizer_path_ = "../tokenizers/qwen3-0.6b/tokenizer.json";
  }

  bool CheckModelFiles() {
    return fs::exists(prefill_path_) && fs::exists(decode_path_) &&
           fs::exists(embedding_path_) && fs::exists(tokenizer_path_);
  }

  std::string prefill_path_;
  std::string decode_path_;
  std::string embedding_path_;
  std::string tokenizer_path_;
};

TEST_F(Qwen3LLMModelTest, LoadModel) {
  if (!CheckModelFiles()) {
    GTEST_SKIP() << "Model files not found, skipping test";
  }

  ModelConfig config;
  config.prefill_path = prefill_path_;
  config.decode_path = decode_path_;
  config.embedding_path = embedding_path_;
  config.tokenizer_path = tokenizer_path_;
  config.devices = {0};
  config.lazy_mode = false;

  EXPECT_NO_THROW({
    Qwen3LLMModel model(config);
    EXPECT_GT(model.vocab_size(), 0);
    EXPECT_GT(model.embedding_dim(), 0);
    EXPECT_GT(model.max_ctx_available(), 0);
  });
}

TEST_F(Qwen3LLMModelTest, Tokenize) {
  if (!CheckModelFiles()) {
    GTEST_SKIP() << "Model files not found, skipping test";
  }

  ModelConfig config;
  config.prefill_path = prefill_path_;
  config.decode_path = decode_path_;
  config.embedding_path = embedding_path_;
  config.tokenizer_path = tokenizer_path_;
  config.devices = {0};
  config.lazy_mode = false;

  ASSERT_NO_THROW({
    Qwen3LLMModel model(config);

    auto tokens = model.tokenize("你好", true, false);
    EXPECT_FALSE(tokens.empty());
    EXPECT_EQ(tokens[0], model.bos_token_id());

    std::string decoded = model.token_to_str(tokens[0]);
    EXPECT_FALSE(decoded.empty());
  });
}

TEST_F(Qwen3LLMModelTest, CreateContext) {
  if (!CheckModelFiles()) {
    GTEST_SKIP() << "Model files not found, skipping test";
  }

  ModelConfig config;
  config.prefill_path = prefill_path_;
  config.decode_path = decode_path_;
  config.embedding_path = embedding_path_;
  config.tokenizer_path = tokenizer_path_;
  config.devices = {0};
  config.lazy_mode = false;

  ASSERT_NO_THROW({
    Qwen3LLMModel model(config);

    auto ctx = model.create_context();
    EXPECT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->context_length(), 0);
  });
}

TEST_F(Qwen3LLMModelTest, PrefillAndDecode) {
  if (!CheckModelFiles()) {
    GTEST_SKIP() << "Model files not found, skipping test";
  }

  ModelConfig config;
  config.prefill_path = prefill_path_;
  config.decode_path = decode_path_;
  config.embedding_path = embedding_path_;
  config.tokenizer_path = tokenizer_path_;
  config.devices = {0};
  config.lazy_mode = false;

  ASSERT_NO_THROW({
    Qwen3LLMModel model(config);
    auto ctx = model.create_context();

    auto tokens = model.tokenize("你好", true, false);
    EXPECT_FALSE(tokens.empty());

    Token first_token = ctx->prefill(tokens);
    EXPECT_NE(first_token, TokenNull);
    EXPECT_GT(ctx->context_length(), 0);

    Token next_token = ctx->decode(first_token);
    EXPECT_NE(next_token, TokenNull);
  });
}

TEST_F(Qwen3LLMModelTest, Generate) {
  if (!CheckModelFiles()) {
    GTEST_SKIP() << "Model files not found, skipping test";
  }

  ModelConfig config;
  config.prefill_path = prefill_path_;
  config.decode_path = decode_path_;
  config.embedding_path = embedding_path_;
  config.tokenizer_path = tokenizer_path_;
  config.devices = {0};
  config.lazy_mode = false;

  ASSERT_NO_THROW({
    Qwen3LLMModel model(config);
    auto ctx = model.create_context();

    auto tokens = model.tokenize("你好", true, false);

    SamplingParams params;
    params.max_tokens = 10;

    int token_count = 0;
    ctx->generate(tokens, params, [&token_count](Token token) {
      token_count++;
      return true;
    });

    EXPECT_GT(token_count, 0);
  });
}

TEST_F(Qwen3LLMModelTest, ResetContext) {
  if (!CheckModelFiles()) {
    GTEST_SKIP() << "Model files not found, skipping test";
  }

  ModelConfig config;
  config.prefill_path = prefill_path_;
  config.decode_path = decode_path_;
  config.embedding_path = embedding_path_;
  config.tokenizer_path = tokenizer_path_;
  config.devices = {0};
  config.lazy_mode = false;

  ASSERT_NO_THROW({
    Qwen3LLMModel model(config);
    auto ctx = model.create_context();

    auto tokens = model.tokenize("你好", true, false);
    ctx->prefill(tokens);
    EXPECT_GT(ctx->context_length(), 0);

    ctx->reset();
    EXPECT_EQ(ctx->context_length(), 0);
  });
}

}  // namespace houmo
