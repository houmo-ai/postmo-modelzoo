/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: perf_profiler_test.cc
 * Description:
 *   PerfProfiler module unit tests
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

#include "modules/perf_profiler.h"

#include <gtest/gtest.h>

#include <thread>

namespace houmo {
namespace {

TEST(PerfProfilerTest, BasicTiming) {
  PerfProfiler profiler;

  profiler.start("test");
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  profiler.stop("test");

  double time_ms = profiler.get_time_ms("test");
  EXPECT_GE(time_ms, 10.0);
  EXPECT_LT(time_ms, 100.0);  // Should not exceed 100ms
  EXPECT_EQ(profiler.get_count("test"), 1);
}

TEST(PerfProfilerTest, ScopedTimer) {
  PerfProfiler profiler;

  {
    auto timer = profiler.scope("scoped_test");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  double time_ms = profiler.get_time_ms("scoped_test");
  EXPECT_GE(time_ms, 5.0);
  EXPECT_EQ(profiler.get_count("scoped_test"), 1);
}

TEST(PerfProfilerTest, HierarchicalPath) {
  PerfProfiler profiler;

  profiler.start("prefill.preprocess");
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  profiler.stop("prefill.preprocess");

  profiler.start("prefill.inference");
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  profiler.stop("prefill.inference");

  // Check sub-stage timing
  EXPECT_GE(profiler.get_time_ms("prefill.preprocess"), 5.0);
  EXPECT_GE(profiler.get_time_ms("prefill.inference"), 10.0);

  // Check parent stage timing (should include sub-stages)
  double prefill_time = profiler.get_time_ms("prefill");
  EXPECT_GE(prefill_time, 15.0);

  // Check sub-stage list
  auto children = profiler.get_children("prefill");
  EXPECT_EQ(children.size(), 2);
}

TEST(PerfProfilerTest, MultipleCalls) {
  PerfProfiler profiler;

  for (int i = 0; i < 3; i++) {
    profiler.start("repeated");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    profiler.stop("repeated");
  }

  EXPECT_EQ(profiler.get_count("repeated"), 3);
  double total = profiler.get_time_ms("repeated");
  double avg = profiler.get_avg_time_ms("repeated");
  EXPECT_GE(total, 6.0);
  EXPECT_NEAR(avg, total / 3, 1.0);
}

TEST(PerfProfilerTest, TokenStats) {
  PerfProfiler profiler;

  profiler.set_input_tokens(100);
  for (int i = 0; i < 50; i++) {
    profiler.add_output_token();
  }

  EXPECT_EQ(profiler.input_tokens(), 100);
  EXPECT_EQ(profiler.output_tokens(), 50);
}

TEST(PerfProfilerTest, TTFT) {
  PerfProfiler profiler;

  profiler.start("generate");
  profiler.start("prefill");
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  profiler.stop("prefill");
  profiler.record_ttft();
  profiler.stop("generate");

  double ttft = profiler.ttft_ms();
  EXPECT_GE(ttft, 10.0);
}

TEST(PerfProfilerTest, TPS) {
  PerfProfiler profiler;

  profiler.set_input_tokens(10);
  for (int i = 0; i < 100; i++) {
    profiler.add_output_token();
  }

  // Simulate 1 second of inference time
  profiler.start("generate");
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  profiler.stop("generate");

  double tps = profiler.overall_tps();
  // Due to short time duration, TPS should be very large
  EXPECT_GT(tps, 0);
}

TEST(PerfProfilerTest, Reset) {
  PerfProfiler profiler;

  profiler.start("test");
  profiler.stop("test");
  profiler.set_input_tokens(100);

  profiler.reset();

  EXPECT_EQ(profiler.get_count("test"), 0);
  EXPECT_EQ(profiler.input_tokens(), 0);
}

TEST(PerfProfilerTest, HasStage) {
  PerfProfiler profiler;

  EXPECT_FALSE(profiler.has_stage("test"));

  profiler.start("test");
  profiler.stop("test");

  EXPECT_TRUE(profiler.has_stage("test"));
}

TEST(PerfProfilerTest, ToPerfStats) {
  PerfProfiler profiler;

  profiler.set_input_tokens(10);
  profiler.add_output_token();
  profiler.add_output_token();

  profiler.start("generate");
  profiler.start("generate.prefill");
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  profiler.stop("generate.prefill");
  profiler.start("generate.decode");
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  profiler.stop("generate.decode");
  profiler.stop("generate");

  PerfStats stats = profiler.to_perf_stats();
  EXPECT_EQ(stats.n_input_tokens, 10);
  EXPECT_EQ(stats.n_output_tokens, 2);
  EXPECT_GE(stats.prefill_time_ms, 5.0);
  EXPECT_GE(stats.decode_time_ms, 5.0);
}
}  // namespace
}  // namespace houmo
