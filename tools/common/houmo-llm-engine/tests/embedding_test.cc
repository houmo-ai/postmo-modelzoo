/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: embedding_test.cc
 * Description:
 *   Embedding module unit tests
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

#include "modules/embedding.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <half.hpp>

namespace houmo {

// Test constants
constexpr int kTestHiddenDim = 1024;
constexpr int kTestMaxSeqLen = 256;
using float16 = half_float::half;

// Generate a small embedding file for testing
void generateTestEmbedding(const std::string& path, int vocab_size,
                           int hidden_dim) {
  std::ofstream ofs(path, std::ios::binary);
  ASSERT_TRUE(ofs) << "Cannot create test embedding file";

  // Write simple linear data: each element = token_id * 0.001 + offset * 0.0001
  for (int token_id = 0; token_id < vocab_size; token_id++) {
    for (int offset = 0; offset < hidden_dim; offset++) {
      float16 value(static_cast<float>(token_id * 0.001 + offset * 0.0001));
      ofs.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
  ofs.close();
}

class EmbeddingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Use environment variable path
    std::string default_path =
        "../models/qwen3-0.6b/hmquant/quant_embedding.bin";

    // Create a small test file if none exists
    if (!std::filesystem::exists(default_path)) {
      // Create small test embedding (100 vocab, 1024 dim)
      test_vocab_size_ = 100;
      test_hidden_dim_ = 1024;
      test_embedding_path_ = "test_embedding.bin";
      generateTestEmbedding(test_embedding_path_, test_vocab_size_,
                            test_hidden_dim_);
    } else {
      test_embedding_path_ = default_path;
      test_vocab_size_ = 151936;  // Qwen default vocab_size
      test_hidden_dim_ = kTestHiddenDim;
    }
  }

  void TearDown() override {
    // Clean up test files
    if (test_embedding_path_ == "test_embedding.bin") {
      std::remove("test_embedding.bin");
    }
  }

  std::string test_embedding_path_;
  int test_vocab_size_ = 0;
  int test_hidden_dim_ = 0;
};

TEST_F(EmbeddingTest, LoadEmbedding) {
  EXPECT_NO_THROW({
    Embedding emb(test_embedding_path_, test_hidden_dim_, kTestMaxSeqLen);
  });
}

TEST_F(EmbeddingTest, VocabSizeAndHiddenDim) {
  Embedding emb(test_embedding_path_, test_hidden_dim_, kTestMaxSeqLen);

  EXPECT_EQ(emb.hidden_dim(), test_hidden_dim_);
  EXPECT_GT(emb.vocab_size(), 0);
}

TEST_F(EmbeddingTest, SingleLookup) {
  Embedding emb(test_embedding_path_, test_hidden_dim_, kTestMaxSeqLen);

  // Lookup single token
  const float16* vec = emb.token_embedding(0);
  ASSERT_NE(vec, nullptr) << "token_embedding should return valid pointer";

  // Verify vector is not all zeros
  bool has_nonzero = false;
  for (int i = 0; i < test_hidden_dim_; i++) {
    if (static_cast<float>(vec[i]) != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Embedding vector should not be all zeros";
}

TEST_F(EmbeddingTest, LookupMultipleTokens) {
  Embedding emb(test_embedding_path_, test_hidden_dim_, kTestMaxSeqLen);

  // Batch lookup
  std::vector<Token> tokens = {0, 1, 2, 3, 4};
  const float16* batch_emb = emb.token_embedding(tokens);
  ASSERT_NE(batch_emb, nullptr);

  // Verify each token's vector
  for (size_t i = 0; i < tokens.size(); i++) {
    const float16* single = emb.token_embedding(tokens[i]);
    ASSERT_NE(single, nullptr);

    // Batch results should match single lookups
    for (int j = 0; j < test_hidden_dim_; j++) {
      EXPECT_FLOAT_EQ(static_cast<float>(single[j]),
                      static_cast<float>(batch_emb[i * test_hidden_dim_ + j]))
          << "Mismatch at token " << tokens[i] << ", dim " << j;
    }
  }
}

TEST_F(EmbeddingTest, LookupInvalidToken) {
  Embedding emb(test_embedding_path_, test_hidden_dim_, kTestMaxSeqLen);

  // Lookup invalid token
  const float16* vec = emb.token_embedding(-1);
  EXPECT_EQ(vec, nullptr) << "Invalid token should return nullptr";

  vec = emb.token_embedding(emb.vocab_size() + 1000);
  EXPECT_EQ(vec, nullptr) << "Out of range token should return nullptr";
}

TEST_F(EmbeddingTest, ConsistencyCheck) {
  Embedding emb(test_embedding_path_, test_hidden_dim_, kTestMaxSeqLen);

  // Looking up the same token multiple times should return the same result
  const float16* vec1 = emb.token_embedding(10);
  const float16* vec2 = emb.token_embedding(10);

  ASSERT_NE(vec1, nullptr);
  ASSERT_NE(vec2, nullptr);

  for (int i = 0; i < test_hidden_dim_; i++) {
    EXPECT_FLOAT_EQ(static_cast<float>(vec1[i]), static_cast<float>(vec2[i]));
  }
}

}  // namespace houmo
