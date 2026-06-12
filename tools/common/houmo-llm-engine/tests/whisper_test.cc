/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: whisper_test.cc
 * Description:
 *   Unit tests for Whisper ASR model
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

#include <memory>
#include <vector>

#include "core/asr_model.h"
#include "core/model_factory.h"
#include "models/whisper_model.h"
#include "modules/audio_processor.h"

namespace houmo {
namespace {

// ============================================================================
// AudioProcessor Tests
// ============================================================================

class AudioProcessorTest : public ::testing::Test {
 protected:
  void SetUp() override { audio_processor_ = std::make_unique<AudioProcessor>(); }

  std::unique_ptr<AudioProcessor> audio_processor_;
};

TEST_F(AudioProcessorTest, DefaultConstruction) {
  auto cfg = audio_processor_->config();
  EXPECT_EQ(cfg.sample_rate, 16000);
  EXPECT_EQ(cfg.n_mels, 80);
}

TEST_F(AudioProcessorTest, CustomConstruction) {
  AudioProcessorConfig cfg;
  cfg.sample_rate = 16000;
  cfg.n_mels = 128;
  AudioProcessor proc(cfg);
  EXPECT_EQ(proc.config().n_mels, 128);
}

TEST_F(AudioProcessorTest, LoadNonExistentFile) {
  auto audio = audio_processor_->LoadAudio("non_existent_file.wav");
  EXPECT_TRUE(audio.pcm.empty());
}

// ============================================================================
// ASRModel Inheritance Tests
// ============================================================================

TEST(ASRInheritanceTest, WhisperModelInheritsASRModel) {
  EXPECT_TRUE((std::is_base_of<ASRModel, WhisperModel>::value));
}

TEST(ASRInheritanceTest, WhisperContextInheritsASRContext) {
  EXPECT_TRUE((std::is_base_of<ASRContext, WhisperContext>::value));
  EXPECT_TRUE((std::is_base_of<Context, WhisperContext>::value));
}

// ============================================================================
// SamplingParams Tests
// ============================================================================

TEST(SamplingParamsTest, DefaultParams) {
  SamplingParams params;
  EXPECT_FLOAT_EQ(params.temperature, 1.0f);
  EXPECT_FLOAT_EQ(params.top_p, 1.0f);
  EXPECT_EQ(params.top_k, 1);
  EXPECT_EQ(params.language, "auto");
}

TEST(SamplingParamsTest, CustomParams) {
  SamplingParams params;
  params.temperature = 0.8f;
  params.max_tokens = 100;
  params.language = "zh";
  EXPECT_FLOAT_EQ(params.temperature, 0.8f);
  EXPECT_EQ(params.max_tokens, 100);
  EXPECT_EQ(params.language, "zh");
}

// ============================================================================
// ASRPerfInfo Tests
// ============================================================================

TEST(PerfInfoTest, ASRPerfInfoDefaults) {
  ASRPerfInfo perf;
  EXPECT_FLOAT_EQ(perf.encode_time, 0.0f);
  EXPECT_FLOAT_EQ(perf.prefill_time, 0.0f);
  EXPECT_FLOAT_EQ(perf.decode_time, 0.0f);
  EXPECT_EQ(perf.output_tokens, 0);
}

// ============================================================================
// ModelSeries Tests
// ============================================================================

TEST(ModelSeriesTest, WhisperASR) {
  EXPECT_NE(static_cast<int>(ModelSeries::kWhisperASR),
            static_cast<int>(ModelSeries::kQwen3LLM));
}

}  // namespace
}  // namespace houmo
