/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: inference_perf_tracker.h
 * Description:
 *   inference_perf_tracker Header File - Defines the perf tracker class.
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

#ifndef INFERENCE_PERF_TRACKER_H
#define INFERENCE_PERF_TRACKER_H

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using DurationMs = std::chrono::duration<double, std::milli>;
using DurationS = std::chrono::duration<double>;

// Performance type enum (corresponds to Python's PERFTYPE)
enum class PerfType {
  PREFILL_TOTAL_TIME = 0,
  PREFILL_TOKEN_TIME = 1,
  PREFILL_EMBED_TIME = 2,
  PREFILL_INPUT_TIME = 3,
  PREFILL_INFER_TIME = 4,
  PREFILL_OUTPUT_TIME = 5,

  DECODE_TOTAL_TIME = 6,
  DECODE_TOKEN_TIME = 7,
  DECODE_EMBED_TIME = 8,
  DECODE_INPUT_TIME = 9,
  DECODE_INFER_TIME = 10,
  DECODE_OUTPUT_TIME = 11,

  VISION_TOTAL_TIME = 12,
  VISION_PREPROCESS_TIME = 13,
  VISION_INPUT_TIME = 14,
  VISION_INFER_TIME = 15,
  VISION_OUTPUT_TIME = 16
};

// Performance information structure (corresponds to Python's PerfInformations)
struct PerfInformations {
  // Time statistics (ms)
  double total_time = 0.0;
  double tokenizer_time = 0.0;
  double embedding_time = 0.0;
  double setinput_time = 0.0;
  double infer_time = 0.0;
  double getoutput_time = 0.0;

  double setinput_time_per_token = 0.0;
  double infer_time_per_token = 0.0;
  double getoutput_time_per_token = 0.0;

  // Speed (tokens/s)
  double total_speed = 0.0;
  double tokenizer_speed = 0.0;
  double embedding_speed = 0.0;
  double setinput_speed = 0.0;
  double infer_speed = 0.0;
  double getoutput_speed = 0.0;

  // Timing start points (s, storing timestamps)
  double total_time_start = 0.0;
  double tokenizer_time_start = 0.0;
  double embedding_time_start = 0.0;
  double setinput_time_start = 0.0;
  double infer_time_start = 0.0;
  double getoutput_time_start = 0.0;

  // Vision processing time (ms)
  double vision_total_time = 0.0;
  double vision_preprocess_time = 0.0;
  double vision_setinput_time = 0.0;
  double vision_infer_time = 0.0;
  double vision_getoutput_time = 0.0;

  // Vision processing speed (images/s)
  double vision_total_speed = 0.0;
  double vision_preprocess_speed = 0.0;
  double vision_infer_speed = 0.0;

  // Vision preprocessing timing start point (complementing missing field in
  // Python)
  double vision_preprocess_time_start = 0.0;

  PerfInformations& operator+=(const PerfInformations& other) {
    total_time += other.total_time;
    tokenizer_time += other.tokenizer_time;
    embedding_time += other.embedding_time;
    setinput_time += other.setinput_time;
    infer_time += other.infer_time;
    getoutput_time += other.getoutput_time;

    vision_total_time += other.vision_total_time;
    vision_preprocess_time += other.vision_preprocess_time;
    vision_setinput_time += other.vision_setinput_time;
    vision_infer_time += other.vision_infer_time;
    vision_getoutput_time += other.vision_getoutput_time;

    return *this;
  }
};

// Inference metrics structure (corresponds to Python's InferenceMetrics)
struct InferenceMetrics {
  // Basic configuration
  int batch_size = 0;
  int input_seq_length = 0;
  int output_seq_length = 0;
  int num_images = 0;  // Number of images processed in vision stage

  // Performance information
  PerfInformations prefill_perf_infos;
  PerfInformations decode_perf_infos;
  PerfInformations vision_perf_infos;  // Vision performance information

  // Summary metrics
  double ttft = 0.0;      // Time to first token (ms)
  double tpot = 0.0;      // Time per output token (ms/token)
  double e2e_time = 0.0;  // End-to-end time (s)
  double e2e_tps = 0.0;   // End-to-end token throughput (tokens/s)

  InferenceMetrics& operator+=(const InferenceMetrics& other) {
    prefill_perf_infos += other.prefill_perf_infos;
    decode_perf_infos += other.decode_perf_infos;
    vision_perf_infos += other.vision_perf_infos;

    ttft += other.ttft;
    tpot += other.tpot;
    e2e_time += other.e2e_time;
    e2e_tps += other.e2e_tps;

    return *this;
  }
};

// Inference performance tracker class
class InferencePerformanceTracker {
 public:
  InferencePerformanceTracker();
  ~InferencePerformanceTracker() = default;
  InferencePerformanceTracker(const InferencePerformanceTracker&) = delete;
  InferencePerformanceTracker& operator=(const InferencePerformanceTracker&) =
      delete;

  // Allow move constructor and assignment
  InferencePerformanceTracker(InferencePerformanceTracker&&) = default;
  InferencePerformanceTracker& operator=(InferencePerformanceTracker&&) =
      default;

  // Start timing
  void perfStart(PerfType perf_type);

  // End timing and accumulate elapsed time
  void perfEnd(PerfType perf_type);

  // Calculate derived metrics (speed, TTFT, TPOT, TPS, etc.)
  void calculateMetrics(InferenceMetrics& metrics);

  // Set basic inference configuration
  void setBasicInfo(int batch_size, int input_seq_length, int output_seq_length,
                    int num_images = 0);

  // Print formatted performance summary report
  void showSummary(bool average = false);  // show average summary

  // Get current metrics (read-only)
  const InferenceMetrics& getCurrentMetrics() const;

  // Reset tracker (start timing again)
  void reset();
  void pref_delete_warmup();

 private:
  InferenceMetrics current_metrics;
  InferenceMetrics total_metrics;
  InferenceMetrics average_metrics;
  size_t num_collected_runs = 0;

  // Helper function: Get current timestamp (seconds)
  double getCurrentTime() const;

  // Helper function: Set timing start based on PerfType
  void setStartTime(PerfType perf_type);

  // Helper function: Get timing start based on PerfType
  double getStartTime(PerfType perf_type) const;

  // Helper function: Accumulate elapsed time (ms) based on PerfType
  void accumulateTime(PerfType perf_type, double time_diff_ms);
};

#endif  // INFERENCE_PERF_TRACKER_H