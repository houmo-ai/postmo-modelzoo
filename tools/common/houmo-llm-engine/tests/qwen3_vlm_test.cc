/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_vlm_test.cc
 * Description:
 *   Qwen3-VLM model tests
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
#include <iostream>

#include "models/qwen3_vlm_model.h"

namespace houmo {

class Qwen3VLMModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Use environment variable path
    vision_path_ = "../models/qwen3-vl-4b/qwen3-vl-4b_visual_448x448x2.hmm";
    prefill_path_ = "../models/qwen3-vl-4b/qwen3-vl-4b_prefill.hmm";
    decode_path_ = "../models/qwen3-vl-4b/qwen3-vl-4b_decode.hmm";
    tokenizer_path_ = "../tokenizers/qwen3-vl-4b/tokenizer.json";
    embedding_path_ = "../models/qwen3-vl-4b/hmquant/quant_embedding.bin";
    test_image_path_ = "../tests/data/a.png";

    // Check if model files exist
    if (!std::filesystem::exists(vision_path_)) {
      GTEST_SKIP() << "Vision model not found: " << vision_path_;
    }
    if (!std::filesystem::exists(prefill_path_)) {
      GTEST_SKIP() << "Prefill model not found: " << prefill_path_;
    }
    if (!std::filesystem::exists(decode_path_)) {
      GTEST_SKIP() << "Decode model not found: " << decode_path_;
    }
    if (!std::filesystem::exists(tokenizer_path_)) {
      GTEST_SKIP() << "Tokenizer not found: " << tokenizer_path_;
    }
    if (!std::filesystem::exists(embedding_path_)) {
      GTEST_SKIP() << "Embedding not found: " << embedding_path_;
    }
  }

  ModelConfig GetConfig() {
    ModelConfig config;
    config.vision_path = vision_path_;
    config.prefill_path = prefill_path_;
    config.decode_path = decode_path_;
    config.tokenizer_path = tokenizer_path_;
    config.embedding_path = embedding_path_;
    config.devices = {0};
    config.lazy_mode = false;
    return config;
  }

  std::string vision_path_;
  std::string prefill_path_;
  std::string decode_path_;
  std::string tokenizer_path_;
  std::string embedding_path_;
  std::string test_image_path_;
};

// Test model loading
TEST_F(Qwen3VLMModelTest, LoadModel) {
  auto config = GetConfig();
  ASSERT_NO_THROW({
    Qwen3VLMModel model(config);
    EXPECT_TRUE(model.vision_module() != nullptr);
    EXPECT_GT(model.vocab_size(), 0);
    EXPECT_GT(model.embedding_dim(), 0);
    EXPECT_GT(model.max_ctx_available(), 0);
  });
}

// Test creating Context
TEST_F(Qwen3VLMModelTest, CreateContext) {
  auto config = GetConfig();
  Qwen3VLMModel model(config);

  ASSERT_NO_THROW({
    auto ctx = model.create_context();
    EXPECT_NE(ctx, nullptr);
  });
}

// Test Tokenize
TEST_F(Qwen3VLMModelTest, Tokenize) {
  auto config = GetConfig();
  Qwen3VLMModel model(config);

  ASSERT_NO_THROW({
    auto tokens = model.tokenize("你好", true, false);
    EXPECT_GT(tokens.size(), 0);
    std::cout << "Token count: " << tokens.size() << std::endl;
  });
}

// Test image processing
TEST_F(Qwen3VLMModelTest, ImageProcessor) {
  if (!std::filesystem::exists(test_image_path_)) {
    GTEST_SKIP() << "Test image not found: " << test_image_path_;
  }

  auto config = GetConfig();
  Qwen3VLMModel model(config);

  // Create image processor
  HmImageProcessor processor(model.vision_image_size_w(),
                             model.vision_image_size_h(), true);

  ASSERT_NO_THROW({
    auto img = processor.LoadAndProcess(test_image_path_);
    EXPECT_GT(img.width, 0);
    EXPECT_GT(img.height, 0);
    EXPECT_EQ(img.channels, 3);

    auto tensor = processor.ToFP16Tensor(img);
    EXPECT_GT(tensor.size(), 0);

    std::cout << "Image dimensions: " << img.width << "x" << img.height
              << std::endl;
    std::cout << "Tensor size: " << tensor.size() << std::endl;
  });
}

// Test vision encoder
TEST_F(Qwen3VLMModelTest, VisionEncoder) {
  if (!std::filesystem::exists(test_image_path_)) {
    GTEST_SKIP() << "Test image not found: " << test_image_path_;
  }

  auto config = GetConfig();
  Qwen3VLMModel model(config);

  // Create image processor
  HmImageProcessor processor(model.vision_image_size_w(),
                             model.vision_image_size_h(), true);
  auto img = processor.LoadAndProcess(test_image_path_);
  auto tensor = processor.ToFP16Tensor(img);

  ASSERT_NO_THROW({
    auto result = model.encode_image(tensor);

    EXPECT_GT(std::get<0>(result).size(), 0);
    EXPECT_GT(std::get<1>(result).size(), 0);
    EXPECT_GT(std::get<2>(result).size(), 0);
    EXPECT_GT(std::get<3>(result).size(), 0);

    std::cout << "Image features size: " << std::get<0>(result).size()
              << std::endl;
    std::cout << "Deepstack 0 size: " << std::get<1>(result).size()
              << std::endl;
    std::cout << "Deepstack 1 size: " << std::get<2>(result).size()
              << std::endl;
    std::cout << "Deepstack 2 size: " << std::get<3>(result).size()
              << std::endl;
  });
}

// Test text-only Prefill
TEST_F(Qwen3VLMModelTest, PrefillTextOnly) {
  auto config = GetConfig();
  Qwen3VLMModel model(config);
  auto ctx = model.create_context();

  ASSERT_NO_THROW({
    auto tokens = model.tokenize("你好", true, false);
    Token token = ctx->prefill(tokens);

    EXPECT_GE(token, 0);
    std::cout << "First token: " << token << std::endl;
  });
}

// Test Prefill with image
TEST_F(Qwen3VLMModelTest, PrefillWithImage) {
  if (!std::filesystem::exists(test_image_path_)) {
    GTEST_SKIP() << "Test image not found: " << test_image_path_;
  }

  auto config = GetConfig();
  Qwen3VLMModel model(config);
  auto ctx = model.create_context();

  // Set image
  auto* qwen_ctx = dynamic_cast<Qwen3VLMContext*>(ctx.get());
  ASSERT_NE(qwen_ctx, nullptr) << "Failed to cast to Qwen3VLMContext";
  qwen_ctx->set_image(test_image_path_);

  ASSERT_NO_THROW({
    auto tokens = model.tokenize("描述这张图片", true, false);
    Token token = ctx->prefill(tokens);

    EXPECT_GE(token, 0);
    std::cout << "First token with image: " << token << std::endl;
  });
}

// Test Decode
TEST_F(Qwen3VLMModelTest, Decode) {
  auto config = GetConfig();
  Qwen3VLMModel model(config);
  auto ctx = model.create_context();

  auto tokens = model.tokenize("你好", true, false);
  Token token = ctx->prefill(tokens);

  ASSERT_NO_THROW({
    for (int i = 0; i < 5; i++) {
      token = ctx->decode(token);
      EXPECT_GE(token, 0);
      if (token == model.eos_token_id()) break;
    }
  });
}

// Test streaming generation
TEST_F(Qwen3VLMModelTest, Generate) {
  auto config = GetConfig();
  Qwen3VLMModel model(config);
  auto ctx = model.create_context();

  auto tokens = model.tokenize("你好", true, false);

  std::string response;
  int token_count = 0;

  ASSERT_NO_THROW({
    ctx->generate(tokens, {.max_tokens = 20}, [&](Token t) {
      response += model.token_to_str(t);
      token_count++;
      return token_count < 20;
    });
  });

  std::cout << "Response: " << response << std::endl;
  EXPECT_GT(response.size(), 0);
}

}  // namespace houmo
