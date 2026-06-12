/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: sampler_test.cc
 * Description:
 *   Sampler module unit tests
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

#include "modules/sampler.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace houmo {

struct LogitsFixture {
  int chunk = 0;
  Token expected_token = 0;
  std::string path;
};

std::vector<float16> loadLogitsFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.is_open()) << "Cannot open logits file: " << path;

  const auto file_size = file.tellg();
  EXPECT_GE(file_size, 0) << "Invalid logits file size: " << path;
  EXPECT_EQ(static_cast<std::streamoff>(file_size) %
                static_cast<std::streamoff>(sizeof(float16)),
            0)
      << "Logits file size must be a multiple of sizeof(float16): " << path;

  file.seekg(0, std::ios::beg);
  std::vector<float16> logits(static_cast<size_t>(file_size) / sizeof(float16));
  if (!logits.empty()) {
    file.read(reinterpret_cast<char*>(logits.data()),
              static_cast<std::streamsize>(file_size));
    EXPECT_TRUE(file.good() || file.eof())
        << "Failed to read logits file: " << path;
  }

  return logits;
}

std::vector<LogitsFixture> discoverLogitsFixtures() {
  std::vector<LogitsFixture> fixtures;
  static const std::regex pattern(R"(logits_(\d+)_(\d+)\.bin)");

  std::string data_dir = "../tests/data";
  if (!std::filesystem::exists(data_dir) ||
      !std::filesystem::is_directory(data_dir)) {
    return fixtures;
  }

  for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    const std::string filename = entry.path().filename().string();
    std::smatch match;
    if (!std::regex_match(filename, match, pattern)) {
      continue;
    }

    fixtures.push_back(LogitsFixture{
        std::stoi(match[1].str()),
        static_cast<Token>(std::stoi(match[2].str())),
        entry.path().string(),
    });
  }

  std::sort(fixtures.begin(), fixtures.end(),
            [](const LogitsFixture& lhs, const LogitsFixture& rhs) {
              return lhs.chunk < rhs.chunk;
            });

  return fixtures;
}

class SamplerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Default parameters aligned with tests/python/SamplingManager.py
    // command-line defaults top_k=1 means greedy sampling
    default_params_.temperature = 1.0f;
    default_params_.top_k = 1;
    default_params_.top_p = 1.0f;
    default_params_.min_p = 0.0f;
    default_params_.repetition_penalty = 1.0f;
    default_params_.presence_penalty = 1.5f;
  }

  SamplingParams default_params_;
};

TEST_F(SamplerTest, LoadchunkLogitsFromDataFiles) {
  const auto fixtures = discoverLogitsFixtures();
  if (fixtures.empty()) {
    GTEST_SKIP() << "No sampler logits fixtures found";
  }

  std::cout << "[SamplerTest] Found " << fixtures.size() << " logits fixtures"
            << std::endl;

  Sampler sampler(default_params_);

  std::vector<Token> previous_tokens;
  previous_tokens.reserve(fixtures.size());

  for (size_t i = 0; i < fixtures.size(); ++i) {
    const auto& fixture = fixtures[i];
    EXPECT_EQ(fixture.chunk, static_cast<int>(i))
        << "Unexpected chunk order in logits fixtures";
    ASSERT_TRUE(std::filesystem::exists(fixture.path))
        << "Fixture file does not exist: " << fixture.path;

    std::cout << "[SamplerTest] chunk " << fixture.chunk
              << ", reading file: " << fixture.path << std::endl;

    const auto logits = loadLogitsFile(fixture.path);
    ASSERT_FALSE(logits.empty()) << "Logits fixture is empty: " << fixture.path;

    const Token actual =
        sampler.sample(logits.data(), logits.size(), previous_tokens);
    std::cout << "[SamplerTest] chunk " << fixture.chunk
              << ", expected token: " << fixture.expected_token
              << ", actual token: " << actual << std::endl;
    EXPECT_EQ(actual, fixture.expected_token)
        << "Sampling mismatch for chunk " << fixture.chunk
        << ", fixture: " << fixture.path;

    previous_tokens.push_back(actual);
  }
}

}  // namespace houmo
