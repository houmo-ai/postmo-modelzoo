/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_perf_monitor.h
 * Description:
 *   Performance monitoring class for tracking TTS inference metrics.
 *   Provides thread-safe recording, aggregation, and formatted reporting.
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

#ifndef HM_PERF_MONITOR_H_
#define HM_PERF_MONITOR_H_

#include <mutex>
#include <string>
#include <vector>

#include "common_types.h"

namespace houmo {

// ============================================================================
// Helper Functions for Performance Formatting
// ============================================================================

/**
 * @brief Convert seconds to milliseconds.
 * @param seconds Time in seconds.
 * @return Time in milliseconds.
 *
 * Reference: demo.py line 73-74 (_ms function)
 */
inline float ToMilliseconds(float seconds) { return seconds * 1000.0f; }

/**
 * @brief Format milliseconds string.
 * @param seconds Time in seconds.
 * @param digits Number of decimal digits (default 3).
 * @return Formatted string like "123.456 ms".
 *
 * Reference: demo.py lines 77-78 (_fmt_ms function)
 */
std::string FormatMs(float seconds, int digits = 3);

/**
 * @brief Format seconds string.
 * @param seconds Time in seconds.
 * @param digits Number of decimal digits (default 3).
 * @return Formatted string like "1.234 s".
 *
 * Reference: demo.py lines 81-82 (_fmt_s function)
 */
std::string FormatSeconds(float seconds, int digits = 3);

/**
 * @brief Format tokens per second string.
 * @param tokens Number of tokens.
 * @param seconds Time in seconds.
 * @param digits Number of decimal digits (default 2).
 * @return Formatted string like "25.34 tokens/s" or "inf tokens/s" if seconds
 * <= 0.
 *
 * Reference: demo.py lines 85-88 (_fmt_toks_per_s function)
 */
std::string FormatTokensPerSec(float tokens, float seconds, int digits = 2);

/**
 * @brief Generate full performance report string.
 * @param perf Performance metrics structure.
 * @return Multi-line formatted performance report.
 *
 * Reference: demo.py lines 91-131 (_format_perf_report function)
 *
 * Output format:
 * LLM Total Cost: XXX ms
 * LLM Prefill Speed: XXX tokens/s
 * TTFT (Time to First Token): XXX ms
 * TPOT (Time Per Output Token): XXX tokens/s
 * TTS Total Cost: XXX ms
 * TTS Real-Time Factor(RTF): X.XXXXXX
 * TTS Generate Speed: X.XX x real-time
 * E2E Latency (End-to-End Latency): X.XXX s
 */
std::string FormatPerfReport(const CosyVoice3Perf& perf);

// ============================================================================
// HmPerfMonitor - Performance Monitoring Class
// ============================================================================

/**
 * @brief Performance monitoring class for tracking TTS inference metrics.
 *
 * Thread-safe class that records performance metrics for each inference run
 * and provides aggregation and reporting capabilities.
 *
 * Usage:
 *   HmPerfMonitor monitor;
 *   monitor.Record(perf1);
 *   monitor.Record(perf2);
 *   monitor.PrintSummary();  // Prints aggregated report
 *
 * Reference: demo.py lines 91-131
 */
class HmPerfMonitor {
 public:
  // ========================================================================
  // Construction
  // ========================================================================

  /**
   * @brief Default constructor.
   */
  HmPerfMonitor();

  /**
   * @brief Destructor.
   */
  ~HmPerfMonitor();

  // Non-copyable (due to mutex)
  HmPerfMonitor(const HmPerfMonitor&) = delete;
  HmPerfMonitor& operator=(const HmPerfMonitor&) = delete;

  // Moveable
  HmPerfMonitor(HmPerfMonitor&& other) noexcept;
  HmPerfMonitor& operator=(HmPerfMonitor&& other) noexcept;

  // ========================================================================
  // Recording Operations
  // ========================================================================

  /**
   * @brief Record a performance entry.
   * @param perf Performance metrics to record.
   *
   * Thread-safe: Uses mutex to protect history_ vector.
   */
  void Record(const CosyVoice3Perf& perf);

  /**
   * @brief Get all recorded performance entries.
   * @return Vector of all recorded performance metrics.
   *
   * Thread-safe: Uses mutex to protect history_ vector.
   */
  std::vector<CosyVoice3Perf> GetAll() const;

  /**
   * @brief Get number of recorded entries.
   * @return Number of performance records.
   */
  size_t Size() const;

  /**
   * @brief Check if history is empty.
   * @return True if no records.
   */
  bool Empty() const;

  /**
   * @brief Clear all history.
   *
   * Thread-safe: Uses mutex to protect history_ vector.
   */
  void Clear();

  // ========================================================================
  // Aggregate Statistics
  // ========================================================================

  /**
   * @brief Get average performance metrics across all records.
   * @return CosyVoice3Perf with averaged values.
   *
   * Returns zeroed struct if history is empty.
   */
  CosyVoice3Perf GetAverage() const;

  /**
   * @brief Get minimum performance metrics across all records.
   * @return CosyVoice3Perf with minimum values.
   *
   * Returns zeroed struct if history is empty.
   * Note: For metrics like time, minimum means best performance.
   */
  CosyVoice3Perf GetMin() const;

  /**
   * @brief Get maximum performance metrics across all records.
   * @return CosyVoice3Perf with maximum values.
   *
   * Returns zeroed struct if history is empty.
   * Note: For metrics like time, maximum means worst performance.
   */
  CosyVoice3Perf GetMax() const;

  /**
   * @brief Get the most recent performance entry.
   * @return Last recorded performance metrics.
   *
   * Returns zeroed struct if history is empty.
   */
  CosyVoice3Perf GetLast() const;

  // ========================================================================
  // Reporting
  // ========================================================================

  /**
   * @brief Print summary report to stdout.
   *
   * Prints:
   * - Total inference count
   * - Individual performance reports (compact format)
   * - Aggregated statistics (average, min, max)
   *
   * Reference: demo.py lines 91-131
   */
  void PrintSummary() const;

  /**
   * @brief Print detailed report for a single entry.
   * @param index Index of the performance entry (0-based).
   */
  void PrintEntry(size_t index) const;

  /**
   * @brief Generate summary report as string.
   * @return Formatted summary report string.
   */
  std::string GetSummaryString() const;

  // ========================================================================
  // Statistics Helpers
  // ========================================================================

  /**
   * @brief Calculate average RTF across all records.
   * @return Average RTF value.
   */
  float GetAverageRtf() const;

  /**
   * @brief Calculate average LLM time across all records.
   * @return Average LLM total time in ms.
   */
  float GetAverageLlmMs() const;

  /**
   * @brief Calculate average tokens per second for prefill.
   * @return Average prefill speed in tokens/s.
   */
  float GetAveragePrefillSpeed() const;

  /**
   * @brief Calculate average tokens per second for decode.
   * @return Average decode speed in tokens/s.
   */
  float GetAverageDecodeSpeed() const;

 private:
  std::vector<CosyVoice3Perf> history_;  ///< Performance history
  mutable std::recursive_mutex
      mutex_;  ///< Thread-safe access (mutable for const methods)
};

}  // namespace houmo

#endif  // HM_PERF_MONITOR_H_