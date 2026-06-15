/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: inference_perf_tracker.cc
 * Description:
 *   inference_perf_tracker Implementation - Perf tracker rules of llm models.
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
#include "utils/perf_tracker/inference_perf_tracker.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>

// Helper function: Get current timestamp (seconds)
double InferencePerformanceTracker::getCurrentTime() const {
  return DurationS(Clock::now().time_since_epoch()).count();
}

// Helper function: Set timing start based on PerfType
void InferencePerformanceTracker::setStartTime(PerfType perf_type) {
  double now = getCurrentTime();
  switch (perf_type) {
    case PerfType::PREFILL_TOTAL_TIME:
      current_metrics.prefill_perf_infos.total_time_start = now;
      break;
    case PerfType::PREFILL_TOKEN_TIME:
      current_metrics.prefill_perf_infos.tokenizer_time_start = now;
      break;
    case PerfType::PREFILL_EMBED_TIME:
      current_metrics.prefill_perf_infos.embedding_time_start = now;
      break;
    case PerfType::PREFILL_INPUT_TIME:
      current_metrics.prefill_perf_infos.setinput_time_start = now;
      break;
    case PerfType::PREFILL_INFER_TIME:
      current_metrics.prefill_perf_infos.infer_time_start = now;
      break;
    case PerfType::PREFILL_OUTPUT_TIME:
      current_metrics.prefill_perf_infos.getoutput_time_start = now;
      break;

    case PerfType::DECODE_TOTAL_TIME:
      current_metrics.decode_perf_infos.total_time_start = now;
      break;
    case PerfType::DECODE_TOKEN_TIME:
      current_metrics.decode_perf_infos.tokenizer_time_start = now;
      break;
    case PerfType::DECODE_EMBED_TIME:
      current_metrics.decode_perf_infos.embedding_time_start = now;
      break;
    case PerfType::DECODE_INPUT_TIME:
      current_metrics.decode_perf_infos.setinput_time_start = now;
      break;
    case PerfType::DECODE_INFER_TIME:
      current_metrics.decode_perf_infos.infer_time_start = now;
      break;
    case PerfType::DECODE_OUTPUT_TIME:
      current_metrics.decode_perf_infos.getoutput_time_start = now;
      break;

    case PerfType::VISION_TOTAL_TIME:
      current_metrics.vision_perf_infos.total_time_start = now;
      break;
    case PerfType::VISION_PREPROCESS_TIME:
      current_metrics.vision_perf_infos.vision_preprocess_time_start = now;
      break;
    case PerfType::VISION_INPUT_TIME:
      current_metrics.vision_perf_infos.setinput_time_start = now;
      break;
    case PerfType::VISION_INFER_TIME:
      current_metrics.vision_perf_infos.infer_time_start = now;
      break;
    case PerfType::VISION_OUTPUT_TIME:
      current_metrics.vision_perf_infos.getoutput_time_start = now;
      break;
    case PerfType::PREFILL_LOAD_TIME:
      prefill_load_time_start = now;
      break;
    case PerfType::DECODE_LOAD_TIME:
      decode_load_time_start = now;
      break;
    case PerfType::VISION_LOAD_TIME:
      vision_load_time_start = now;
      break;

    default:
      throw std::invalid_argument("Invalid PerfType for setStartTime");
  }
}

// Helper function: Get timing start based on PerfType
double InferencePerformanceTracker::getStartTime(PerfType perf_type) const {
  switch (perf_type) {
    case PerfType::PREFILL_TOTAL_TIME:
      return current_metrics.prefill_perf_infos.total_time_start;
    case PerfType::PREFILL_TOKEN_TIME:
      return current_metrics.prefill_perf_infos.tokenizer_time_start;
    case PerfType::PREFILL_EMBED_TIME:
      return current_metrics.prefill_perf_infos.embedding_time_start;
    case PerfType::PREFILL_INPUT_TIME:
      return current_metrics.prefill_perf_infos.setinput_time_start;
    case PerfType::PREFILL_INFER_TIME:
      return current_metrics.prefill_perf_infos.infer_time_start;
    case PerfType::PREFILL_OUTPUT_TIME:
      return current_metrics.prefill_perf_infos.getoutput_time_start;

    case PerfType::DECODE_TOTAL_TIME:
      return current_metrics.decode_perf_infos.total_time_start;
    case PerfType::DECODE_TOKEN_TIME:
      return current_metrics.decode_perf_infos.tokenizer_time_start;
    case PerfType::DECODE_EMBED_TIME:
      return current_metrics.decode_perf_infos.embedding_time_start;
    case PerfType::DECODE_INPUT_TIME:
      return current_metrics.decode_perf_infos.setinput_time_start;
    case PerfType::DECODE_INFER_TIME:
      return current_metrics.decode_perf_infos.infer_time_start;
    case PerfType::DECODE_OUTPUT_TIME:
      return current_metrics.decode_perf_infos.getoutput_time_start;

    case PerfType::VISION_TOTAL_TIME:
      return current_metrics.vision_perf_infos.total_time_start;
    case PerfType::VISION_PREPROCESS_TIME:
      return current_metrics.vision_perf_infos.vision_preprocess_time_start;
    case PerfType::VISION_INPUT_TIME:
      return current_metrics.vision_perf_infos.setinput_time_start;
    case PerfType::VISION_INFER_TIME:
      return current_metrics.vision_perf_infos.infer_time_start;
    case PerfType::VISION_OUTPUT_TIME:
      return current_metrics.vision_perf_infos.getoutput_time_start;
    case PerfType::PREFILL_LOAD_TIME:
      return prefill_load_time_start;
    case PerfType::DECODE_LOAD_TIME:
      return decode_load_time_start;
    case PerfType::VISION_LOAD_TIME:
      return vision_load_time_start;

    default:
      throw std::invalid_argument("Invalid PerfType for getStartTime");
  }
}

// Helper function: Accumulate elapsed time (ms) based on PerfType
void InferencePerformanceTracker::accumulateTime(PerfType perf_type,
                                                 double time_diff_ms) {
  switch (perf_type) {
    case PerfType::PREFILL_TOTAL_TIME:
      current_metrics.prefill_perf_infos.total_time += time_diff_ms;
      break;
    case PerfType::PREFILL_TOKEN_TIME:
      current_metrics.prefill_perf_infos.tokenizer_time += time_diff_ms;
      break;
    case PerfType::PREFILL_EMBED_TIME:
      current_metrics.prefill_perf_infos.embedding_time += time_diff_ms;
      break;
    case PerfType::PREFILL_INPUT_TIME:
      current_metrics.prefill_perf_infos.setinput_time += time_diff_ms;
      break;
    case PerfType::PREFILL_INFER_TIME:
      current_metrics.prefill_perf_infos.infer_time += time_diff_ms;
      break;
    case PerfType::PREFILL_OUTPUT_TIME:
      current_metrics.prefill_perf_infos.getoutput_time += time_diff_ms;
      break;

    case PerfType::DECODE_TOTAL_TIME:
      current_metrics.decode_perf_infos.total_time += time_diff_ms;
      break;
    case PerfType::DECODE_TOKEN_TIME:
      current_metrics.decode_perf_infos.tokenizer_time += time_diff_ms;
      break;
    case PerfType::DECODE_EMBED_TIME:
      current_metrics.decode_perf_infos.embedding_time += time_diff_ms;
      break;
    case PerfType::DECODE_INPUT_TIME:
      current_metrics.decode_perf_infos.setinput_time += time_diff_ms;
      break;
    case PerfType::DECODE_INFER_TIME:
      current_metrics.decode_perf_infos.infer_time += time_diff_ms;
      break;
    case PerfType::DECODE_OUTPUT_TIME:
      current_metrics.decode_perf_infos.getoutput_time += time_diff_ms;
      break;

    case PerfType::VISION_TOTAL_TIME:
      current_metrics.vision_perf_infos.vision_total_time += time_diff_ms;
      break;
    case PerfType::VISION_PREPROCESS_TIME:
      current_metrics.vision_perf_infos.vision_preprocess_time += time_diff_ms;
      break;
    case PerfType::VISION_INPUT_TIME:
      current_metrics.vision_perf_infos.setinput_time += time_diff_ms;
      break;
    case PerfType::VISION_INFER_TIME:
      current_metrics.vision_perf_infos.infer_time += time_diff_ms;
      break;
    case PerfType::VISION_OUTPUT_TIME:
      current_metrics.vision_perf_infos.getoutput_time += time_diff_ms;
      break;
    case PerfType::PREFILL_LOAD_TIME:
      prefill_load_time += time_diff_ms;
      break;
    case PerfType::DECODE_LOAD_TIME:
      decode_load_time += time_diff_ms;
      break;
    case PerfType::VISION_LOAD_TIME:
      vision_load_time += time_diff_ms;
      break;

    default:
      throw std::invalid_argument("Invalid PerfType for accumulateTime");
  }
}

// Constructor
InferencePerformanceTracker::InferencePerformanceTracker() {
  current_metrics.e2e_time = getCurrentTime();
}

// Start timing
void InferencePerformanceTracker::perfStart(PerfType perf_type) {
  try {
    setStartTime(perf_type);
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string("perfStart failed: ") + e.what());
  }
}

// End timing and accumulate elapsed time
void InferencePerformanceTracker::perfEnd(PerfType perf_type) {
  try {
    double start_time = getStartTime(perf_type);
    double current_time = getCurrentTime();
    double time_diff_ms = (current_time - start_time) * 1000.0;
    accumulateTime(perf_type, time_diff_ms);
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string("perfEnd failed: ") + e.what());
  }
}

// Calculate derived metrics
void InferencePerformanceTracker::calculateMetrics(InferenceMetrics& metrics) {
  // 1. Calculate Prefill stage speed (tokens/s)
  if (metrics.prefill_perf_infos.tokenizer_time > 0) {
    metrics.prefill_perf_infos.tokenizer_speed =
        metrics.input_seq_length /
        (metrics.prefill_perf_infos.tokenizer_time / 1000.0);
  }
  if (metrics.prefill_perf_infos.embedding_time > 0) {
    metrics.prefill_perf_infos.embedding_speed =
        metrics.input_seq_length /
        (metrics.prefill_perf_infos.embedding_time / 1000.0);
  }
  if (metrics.prefill_perf_infos.setinput_time > 0) {
    metrics.prefill_perf_infos.setinput_speed =
        metrics.input_seq_length /
        (metrics.prefill_perf_infos.setinput_time / 1000.0);
  }
  if (metrics.prefill_perf_infos.infer_time > 0) {
    int padded_length = metrics.input_seq_length;
    metrics.prefill_perf_infos.infer_speed =
        (padded_length * metrics.batch_size) /
        (metrics.prefill_perf_infos.infer_time / 1000.0);
  }
  if (metrics.prefill_perf_infos.getoutput_time > 0) {
    metrics.prefill_perf_infos.getoutput_speed =
        metrics.input_seq_length /
        (metrics.prefill_perf_infos.getoutput_time / 1000.0);
  }
  if (metrics.prefill_perf_infos.total_time > 0) {
    int padded_length = metrics.input_seq_length;
    metrics.prefill_perf_infos.total_speed =
        (padded_length * metrics.batch_size) /
        (metrics.prefill_perf_infos.total_time / 1000.0);
  }

  // 2. Calculate Decode stage speed (tokens/s)
  if (metrics.decode_perf_infos.tokenizer_time > 0) {
    metrics.decode_perf_infos.tokenizer_speed =
        (metrics.batch_size * metrics.output_seq_length) /
        (metrics.decode_perf_infos.tokenizer_time / 1000.0);
  }
  if (metrics.decode_perf_infos.embedding_time > 0) {
    metrics.decode_perf_infos.embedding_speed =
        (metrics.batch_size * metrics.output_seq_length) /
        (metrics.decode_perf_infos.embedding_time / 1000.0);
  }
  if (metrics.decode_perf_infos.setinput_time > 0) {
    metrics.decode_perf_infos.setinput_speed =
        (metrics.batch_size * metrics.output_seq_length) /
        (metrics.decode_perf_infos.setinput_time / 1000.0);
    metrics.decode_perf_infos.setinput_time_per_token =
        metrics.decode_perf_infos.setinput_time /
        (metrics.batch_size * metrics.output_seq_length);
  }
  if (metrics.decode_perf_infos.infer_time > 0) {
    metrics.decode_perf_infos.infer_speed =
        (metrics.batch_size * metrics.output_seq_length) /
        (metrics.decode_perf_infos.infer_time / 1000.0);
    metrics.decode_perf_infos.infer_time_per_token =
        metrics.decode_perf_infos.infer_time /
        (metrics.batch_size * metrics.output_seq_length);
  }
  if (metrics.decode_perf_infos.getoutput_time > 0) {
    metrics.decode_perf_infos.getoutput_speed =
        (metrics.batch_size * metrics.output_seq_length) /
        (metrics.decode_perf_infos.getoutput_time / 1000.0);
    metrics.decode_perf_infos.getoutput_time_per_token =
        metrics.decode_perf_infos.getoutput_time /
        (metrics.batch_size * metrics.output_seq_length);
  }
  if (metrics.decode_perf_infos.total_time > 0) {
    metrics.decode_perf_infos.total_speed =
        (metrics.batch_size * metrics.output_seq_length) /
        (metrics.decode_perf_infos.total_time / 1000.0);
  }

  // 3. Calculate Vision stage speed (images/s)
  if (metrics.vision_perf_infos.vision_preprocess_time > 0 &&
      metrics.num_images > 0) {
    metrics.vision_perf_infos.vision_preprocess_speed =
        metrics.num_images /
        (metrics.vision_perf_infos.vision_preprocess_time / 1000.0);
  }
  if (metrics.vision_perf_infos.infer_time > 0 && metrics.num_images > 0) {
    metrics.vision_perf_infos.vision_infer_speed =
        metrics.num_images / (metrics.vision_perf_infos.infer_time / 1000.0);
  }
  if (metrics.vision_perf_infos.vision_total_time > 0 &&
      metrics.num_images > 0) {
    metrics.vision_perf_infos.vision_total_speed =
        metrics.num_images /
        (metrics.vision_perf_infos.vision_total_time / 1000.0);
  }

  // 4. Calculate summary metrics
  metrics.ttft = metrics.prefill_perf_infos.total_time;
  if (metrics.output_seq_length > 0) {
    metrics.tpot = metrics.decode_perf_infos.total_time /
                   (metrics.batch_size * metrics.output_seq_length);
  }

  // End-to-end TPS = Total processed tokens / Total time (tokens/s)
  int total_tokens = (metrics.output_seq_length);
  if (metrics.e2e_time > 0) {
    metrics.e2e_tps =
        (metrics.batch_size * metrics.output_seq_length) / metrics.e2e_time;
  }
}

// Set basic inference configuration
void InferencePerformanceTracker::setBasicInfo(int batch_size,
                                               int input_seq_length,
                                               int output_seq_length,
                                               int num_images) {
  current_metrics.batch_size = batch_size;
  current_metrics.input_seq_length = input_seq_length;
  current_metrics.output_seq_length = output_seq_length;
  current_metrics.num_images = num_images;
  total_metrics.batch_size = batch_size;
  total_metrics.input_seq_length = input_seq_length;
  total_metrics.output_seq_length = output_seq_length;
  total_metrics.num_images = num_images;
  // Calculate end-to-end time (seconds)
  current_metrics.e2e_time = getCurrentTime() - current_metrics.e2e_time;
}

// Print formatted performance summary report
void InferencePerformanceTracker::showSummary(bool average) {
  InferenceMetrics metrics;
  if (average) {
    metrics = total_metrics;
    metrics.e2e_time = metrics.e2e_time / num_collected_runs;
    metrics.e2e_tps = metrics.e2e_tps / num_collected_runs;
    metrics.ttft = metrics.ttft / num_collected_runs;
    metrics.tpot = metrics.tpot / num_collected_runs;
    metrics.prefill_perf_infos.total_time /= num_collected_runs;
    metrics.prefill_perf_infos.tokenizer_time /= num_collected_runs;
    metrics.prefill_perf_infos.embedding_time /= num_collected_runs;
    metrics.prefill_perf_infos.setinput_time /= num_collected_runs;
    metrics.prefill_perf_infos.infer_time /= num_collected_runs;
    metrics.prefill_perf_infos.getoutput_time /= num_collected_runs;
    metrics.prefill_perf_infos.vision_total_time /= num_collected_runs;
    metrics.prefill_perf_infos.vision_preprocess_time /= num_collected_runs;
    metrics.prefill_perf_infos.vision_setinput_time /= num_collected_runs;
    metrics.prefill_perf_infos.vision_infer_time /= num_collected_runs;
    metrics.prefill_perf_infos.vision_getoutput_time /= num_collected_runs;

    metrics.decode_perf_infos.total_time /= num_collected_runs;
    metrics.decode_perf_infos.tokenizer_time /= num_collected_runs;
    metrics.decode_perf_infos.embedding_time /= num_collected_runs;
    metrics.decode_perf_infos.setinput_time /= num_collected_runs;
    metrics.decode_perf_infos.infer_time /= num_collected_runs;
    metrics.decode_perf_infos.getoutput_time /= num_collected_runs;
    metrics.decode_perf_infos.vision_total_time /= num_collected_runs;
    metrics.decode_perf_infos.vision_preprocess_time /= num_collected_runs;
    metrics.decode_perf_infos.vision_setinput_time /= num_collected_runs;
    metrics.decode_perf_infos.vision_infer_time /= num_collected_runs;
    metrics.decode_perf_infos.vision_getoutput_time /= num_collected_runs;

    metrics.vision_perf_infos.total_time /= num_collected_runs;
    metrics.vision_perf_infos.tokenizer_time /= num_collected_runs;
    metrics.vision_perf_infos.embedding_time /= num_collected_runs;
    metrics.vision_perf_infos.setinput_time /= num_collected_runs;
    metrics.vision_perf_infos.infer_time /= num_collected_runs;
    metrics.vision_perf_infos.getoutput_time /= num_collected_runs;
    metrics.vision_perf_infos.vision_total_time /= num_collected_runs;
    metrics.vision_perf_infos.vision_preprocess_time /= num_collected_runs;
    metrics.vision_perf_infos.vision_setinput_time /= num_collected_runs;
    metrics.vision_perf_infos.vision_infer_time /= num_collected_runs;
    metrics.vision_perf_infos.vision_getoutput_time /= num_collected_runs;
    calculateMetrics(metrics);
    average_metrics = metrics;
  } else {
    calculateMetrics(current_metrics);
    total_metrics += current_metrics;
    num_collected_runs++;
    metrics = current_metrics;
  }
}

// Get current metrics (read-only)
const InferenceMetrics& InferencePerformanceTracker::getCurrentMetrics() const {
  return current_metrics;
}

void InferencePerformanceTracker::set_kvcache_mem(double mem_size) {
  kvcache_mem = mem_size;
}

void InferencePerformanceTracker::reset() {
  memset(&current_metrics, 0, sizeof(current_metrics));
  current_metrics.e2e_time = getCurrentTime();
}

void InferencePerformanceTracker::pref_delete_warmup() {
  memset(&total_metrics, 0, sizeof(total_metrics));
  num_collected_runs = 0;
  // total_metrics.e2e_time = getCurrentTime();
}
