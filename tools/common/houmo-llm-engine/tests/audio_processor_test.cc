/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: audio_processor_test.cc
 * Description:
 *   Unit tests for AudioProcessor
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

#include "modules/audio_processor.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

namespace houmo {

class AudioProcessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    audio_path_ = "../tests/data/audio.mp3";  // 相对于 build 目录
    processor_ = std::make_unique<AudioProcessor>();
  }

  std::string audio_path_;
  std::unique_ptr<AudioProcessor> processor_;
};

TEST_F(AudioProcessorTest, DefaultConstruction) {
  EXPECT_NO_THROW(AudioProcessor processor);
  EXPECT_EQ(processor_->sample_rate(), 16000);
  EXPECT_EQ(processor_->n_mels(), 80);
  EXPECT_EQ(processor_->feature_dim(), 80);
}

TEST_F(AudioProcessorTest, CustomConstruction) {
  AudioProcessorConfig config;
  config.n_mels = 128;
  config.chunk_seconds = 10;

  AudioProcessor processor(config);
  EXPECT_EQ(processor.n_mels(), 128);
  EXPECT_EQ(processor.feature_dim(), 128);
}

TEST_F(AudioProcessorTest, LoadAudio) {
  if (!fs::exists(audio_path_)) {
    GTEST_SKIP() << "Audio file not found: " << audio_path_;
  }

  AudioData audio = processor_->LoadAudio(audio_path_);

  EXPECT_FALSE(audio.pcm.empty());
  EXPECT_EQ(audio.sample_rate, 16000);
  EXPECT_GT(audio.duration, 0.0f);
  std::cout << "Audio duration: " << audio.duration
            << "s, samples: " << audio.pcm.size() << "\n";
}

TEST_F(AudioProcessorTest, LoadNonExistentFile) {
  AudioData audio = processor_->LoadAudio("non_existent_file.wav");
  EXPECT_TRUE(audio.pcm.empty());
  EXPECT_EQ(audio.duration, 0.0f);
}

TEST_F(AudioProcessorTest, ExtractFeatures) {
  if (!fs::exists(audio_path_)) {
    GTEST_SKIP() << "Audio file not found: " << audio_path_;
  }

  AudioData audio = processor_->LoadAudio(audio_path_);
  ASSERT_FALSE(audio.pcm.empty());

  MelFeatures features = processor_->ExtractFeatures(audio);

  EXPECT_EQ(features.feature_dim, 80);
  EXPECT_GT(features.num_frames, 0);
  EXPECT_FALSE(features.data.empty());
  EXPECT_EQ(features.data.size(),
            static_cast<size_t>(features.feature_dim * features.num_frames));

  std::cout << "Features: dim=" << features.feature_dim
            << ", frames=" << features.num_frames
            << ", size=" << features.data.size() << "\n";
}

TEST_F(AudioProcessorTest, ChunkPCM) {
  // 创建 60 秒的虚拟音频数据
  AudioData audio;
  audio.pcm.resize(16000 * 60, 0.5f);
  audio.sample_rate = 16000;
  audio.duration = 60.0f;

  std::vector<AudioData> chunks = processor_->ChunkPCM(audio);

  // 默认 30 秒一块，60 秒应该分成 2 块
  EXPECT_EQ(chunks.size(), 2);
  EXPECT_EQ(chunks[0].pcm.size(), static_cast<size_t>(16000 * 30));
  EXPECT_EQ(chunks[1].pcm.size(), static_cast<size_t>(16000 * 30));
}

TEST_F(AudioProcessorTest, ChunkPCMWithPadding) {
  // 创建 45 秒的虚拟音频数据
  AudioData audio;
  audio.pcm.resize(16000 * 45, 0.5f);
  audio.sample_rate = 16000;
  audio.duration = 45.0f;

  std::vector<AudioData> chunks = processor_->ChunkPCM(audio);

  // 45 秒应该分成 2 块：30秒 + 15秒（最后一块不填充，保持实际长度）
  EXPECT_EQ(chunks.size(), 2);
  EXPECT_EQ(chunks[0].pcm.size(), static_cast<size_t>(16000 * 30));
  EXPECT_EQ(chunks[1].pcm.size(), static_cast<size_t>(16000 * 15));
}

TEST_F(AudioProcessorTest, ProcessEndToEnd) {
  if (!fs::exists(audio_path_)) {
    GTEST_SKIP() << "Audio file not found: " << audio_path_;
  }

  std::vector<MelFeatures> features_list = processor_->Process(audio_path_);

  EXPECT_FALSE(features_list.empty());
  for (const auto& features : features_list) {
    EXPECT_EQ(features.feature_dim, 80);
    EXPECT_FALSE(features.data.empty());
  }
}

}  // namespace houmo
