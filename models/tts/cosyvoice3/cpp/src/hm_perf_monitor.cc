/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_perf_monitor.cc
 * Description:
 *   Performance monitoring implementation for tracking TTS inference metrics.
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

#include "hm_perf_monitor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace houmo {

// ============================================================================
// Helper Functions for Performance Formatting
// ============================================================================

std::string FormatMs(float seconds, int digits) {
  float ms = ToMilliseconds(seconds);
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(digits) << ms << " ms";
  return ss.str();
}

std::string FormatSeconds(float seconds, int digits) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(digits) << seconds << " s";
  return ss.str();
}

std::string FormatTokensPerSec(float tokens, float seconds, int digits) {
  if (seconds <= 0.0f) {
    return "inf tokens/s";
  }
  float speed = tokens / seconds;
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(digits) << speed << " tokens/s";
  return ss.str();
}

std::string FormatPerfReport(const CosyVoice3Perf& perf) {
  std::ostringstream lines;

  // LLM Total Cost
  float llm_total_s = perf.llm_total_ms / 1000.0f;
  if (llm_total_s > 0.0f) {
    lines << "LLM Total Cost: " << FormatMs(llm_total_s) << "\n";
  }

  // LLM Prefill Speed
  float prefill_s = perf.prefill_ms / 1000.0f;
  if (prefill_s > 0.0f && perf.prefill_tokens > 0) {
    lines << "LLM Prefill Speed: "
          << FormatTokensPerSec(perf.prefill_tokens, prefill_s) << "\n";
  }

  // TTFT (Time to First Token)
  if (perf.ttft_ms >= 0.0f) {
    lines << "TTFT (Time to First Token): " << FormatMs(perf.ttft_ms / 1000.0f)
          << "\n";
  }

  // TPOT (Time Per Output Token) - actually output tokens per second
  float decode_s = perf.decode_ms / 1000.0f;
  if (decode_s > 0.0f && perf.decode_tokens > 0) {
    lines << "TPOT (Time Per Output Token): "
          << FormatTokensPerSec(perf.decode_tokens, decode_s) << "\n";
  }

  // TTS Total Cost (Flow + Vocoder)
  float tts_total_ms = perf.flow_total_ms + perf.vocoder_ms;
  if (tts_total_ms > 0.0f) {
    lines << "TTS Total Cost: " << FormatMs(tts_total_ms / 1000.0f) << "\n";
  }

  // TTS Real-Time Factor (RTF)
  if (perf.rtf > 0.0f) {
    lines << "TTS Real-Time Factor(RTF): " << std::fixed << std::setprecision(6)
          << perf.rtf << "\n";
    float generate_speed = 1.0f / perf.rtf;
    lines << "TTS Generate Speed: " << std::fixed << std::setprecision(2)
          << generate_speed << " x real-time\n";
  }

  // E2E Latency
  if (perf.e2e_latency_s > 0.0f) {
    lines << "E2E Latency (End-to-End Latency): "
          << FormatSeconds(perf.e2e_latency_s) << "\n";
  }

  return lines.str();
}

// ============================================================================
// HmPerfMonitor - Construction & Destruction
// ============================================================================

HmPerfMonitor::HmPerfMonitor() : history_(), mutex_() {}

HmPerfMonitor::~HmPerfMonitor() {
  // Nothing to clean up
}

HmPerfMonitor::HmPerfMonitor(HmPerfMonitor&& other) noexcept
    : history_(std::move(other.history_)), mutex_() {
  // Mutex cannot be moved, so we create a new one
  // The other's history is moved, but we don't lock since other is being
  // destructed
}

HmPerfMonitor& HmPerfMonitor::operator=(HmPerfMonitor&& other) noexcept {
  if (this != &other) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    history_ = std::move(other.history_);
  }
  return *this;
}

// ============================================================================
// HmPerfMonitor - Recording Operations
// ============================================================================

void HmPerfMonitor::Record(const CosyVoice3Perf& perf) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  history_.push_back(perf);
}

std::vector<CosyVoice3Perf> HmPerfMonitor::GetAll() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return history_;
}

size_t HmPerfMonitor::Size() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return history_.size();
}

bool HmPerfMonitor::Empty() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return history_.empty();
}

void HmPerfMonitor::Clear() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  history_.clear();
}

// ============================================================================
// HmPerfMonitor - Aggregate Statistics
// ============================================================================

CosyVoice3Perf HmPerfMonitor::GetAverage() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (history_.empty()) {
    return CosyVoice3Perf();
  }

  CosyVoice3Perf avg;
  size_t count = history_.size();

  // Sum all metrics
  for (const auto& perf : history_) {
    avg.llm_total_ms += perf.llm_total_ms;
    avg.prefill_ms += perf.prefill_ms;
    avg.prefill_tokens += perf.prefill_tokens;
    avg.ttft_ms += perf.ttft_ms;
    avg.decode_ms += perf.decode_ms;
    avg.decode_tokens += perf.decode_tokens;
    avg.speaker_emb_ms += perf.speaker_emb_ms;
    avg.speech_tokenizer_ms += perf.speech_tokenizer_ms;
    avg.flow_total_ms += perf.flow_total_ms;
    avg.flow_encoder_ms += perf.flow_encoder_ms;
    avg.flow_decoder_ms += perf.flow_decoder_ms;
    avg.vocoder_ms += perf.vocoder_ms;
    avg.e2e_latency_s += perf.e2e_latency_s;
    avg.audio_duration_s += perf.audio_duration_s;
    avg.rtf += perf.rtf;
  }

  // Compute averages
  avg.llm_total_ms /= count;
  avg.prefill_ms /= count;
  avg.prefill_tokens = static_cast<int>(avg.prefill_tokens / count);
  avg.ttft_ms /= count;
  avg.decode_ms /= count;
  avg.decode_tokens = static_cast<int>(avg.decode_tokens / count);
  avg.speaker_emb_ms /= count;
  avg.speech_tokenizer_ms /= count;
  avg.flow_total_ms /= count;
  avg.flow_encoder_ms /= count;
  avg.flow_decoder_ms /= count;
  avg.vocoder_ms /= count;
  avg.e2e_latency_s /= count;
  avg.audio_duration_s /= count;
  avg.rtf /= count;

  return avg;
}

CosyVoice3Perf HmPerfMonitor::GetMin() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (history_.empty()) {
    return CosyVoice3Perf();
  }

  CosyVoice3Perf min_perf;
  min_perf.llm_total_ms = std::numeric_limits<float>::max();
  min_perf.prefill_ms = std::numeric_limits<float>::max();
  min_perf.ttft_ms = std::numeric_limits<float>::max();
  min_perf.decode_ms = std::numeric_limits<float>::max();
  min_perf.speaker_emb_ms = std::numeric_limits<float>::max();
  min_perf.speech_tokenizer_ms = std::numeric_limits<float>::max();
  min_perf.flow_total_ms = std::numeric_limits<float>::max();
  min_perf.flow_encoder_ms = std::numeric_limits<float>::max();
  min_perf.flow_decoder_ms = std::numeric_limits<float>::max();
  min_perf.vocoder_ms = std::numeric_limits<float>::max();
  min_perf.e2e_latency_s = std::numeric_limits<float>::max();
  min_perf.rtf = std::numeric_limits<float>::max();

  for (const auto& perf : history_) {
    min_perf.llm_total_ms = std::min(min_perf.llm_total_ms, perf.llm_total_ms);
    min_perf.prefill_ms = std::min(min_perf.prefill_ms, perf.prefill_ms);
    min_perf.prefill_tokens =
        std::min(min_perf.prefill_tokens, perf.prefill_tokens);
    min_perf.ttft_ms = std::min(min_perf.ttft_ms, perf.ttft_ms);
    min_perf.decode_ms = std::min(min_perf.decode_ms, perf.decode_ms);
    min_perf.decode_tokens =
        std::min(min_perf.decode_tokens, perf.decode_tokens);
    min_perf.speaker_emb_ms =
        std::min(min_perf.speaker_emb_ms, perf.speaker_emb_ms);
    min_perf.speech_tokenizer_ms =
        std::min(min_perf.speech_tokenizer_ms, perf.speech_tokenizer_ms);
    min_perf.flow_total_ms =
        std::min(min_perf.flow_total_ms, perf.flow_total_ms);
    min_perf.flow_encoder_ms =
        std::min(min_perf.flow_encoder_ms, perf.flow_encoder_ms);
    min_perf.flow_decoder_ms =
        std::min(min_perf.flow_decoder_ms, perf.flow_decoder_ms);
    min_perf.vocoder_ms = std::min(min_perf.vocoder_ms, perf.vocoder_ms);
    min_perf.e2e_latency_s =
        std::min(min_perf.e2e_latency_s, perf.e2e_latency_s);
    min_perf.audio_duration_s =
        std::min(min_perf.audio_duration_s, perf.audio_duration_s);
    min_perf.rtf = std::min(min_perf.rtf, perf.rtf);
  }

  return min_perf;
}

CosyVoice3Perf HmPerfMonitor::GetMax() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (history_.empty()) {
    return CosyVoice3Perf();
  }

  CosyVoice3Perf max_perf;

  for (const auto& perf : history_) {
    max_perf.llm_total_ms = std::max(max_perf.llm_total_ms, perf.llm_total_ms);
    max_perf.prefill_ms = std::max(max_perf.prefill_ms, perf.prefill_ms);
    max_perf.prefill_tokens =
        std::max(max_perf.prefill_tokens, perf.prefill_tokens);
    max_perf.ttft_ms = std::max(max_perf.ttft_ms, perf.ttft_ms);
    max_perf.decode_ms = std::max(max_perf.decode_ms, perf.decode_ms);
    max_perf.decode_tokens =
        std::max(max_perf.decode_tokens, perf.decode_tokens);
    max_perf.speaker_emb_ms =
        std::max(max_perf.speaker_emb_ms, perf.speaker_emb_ms);
    max_perf.speech_tokenizer_ms =
        std::max(max_perf.speech_tokenizer_ms, perf.speech_tokenizer_ms);
    max_perf.flow_total_ms =
        std::max(max_perf.flow_total_ms, perf.flow_total_ms);
    max_perf.flow_encoder_ms =
        std::max(max_perf.flow_encoder_ms, perf.flow_encoder_ms);
    max_perf.flow_decoder_ms =
        std::max(max_perf.flow_decoder_ms, perf.flow_decoder_ms);
    max_perf.vocoder_ms = std::max(max_perf.vocoder_ms, perf.vocoder_ms);
    max_perf.e2e_latency_s =
        std::max(max_perf.e2e_latency_s, perf.e2e_latency_s);
    max_perf.audio_duration_s =
        std::max(max_perf.audio_duration_s, perf.audio_duration_s);
    max_perf.rtf = std::max(max_perf.rtf, perf.rtf);
  }

  return max_perf;
}

CosyVoice3Perf HmPerfMonitor::GetLast() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (history_.empty()) {
    return CosyVoice3Perf();
  }

  return history_.back();
}

// ============================================================================
// HmPerfMonitor - Reporting
// ============================================================================

void HmPerfMonitor::PrintSummary() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (history_.empty()) {
    std::cout << "\nNo performance reports collected.\n";
    return;
  }

  std::cout << "\n=== Performance Summary ===\n";
  std::cout << "Total inference runs: " << history_.size() << "\n\n";

  // Print individual reports
  for (size_t i = 0; i < history_.size(); ++i) {
    std::cout << "--- Inference " << (i + 1) << " ---\n";
    std::cout << FormatPerfReport(history_[i]);
  }

  // Print aggregated statistics
  std::cout << "\n--- Aggregated Statistics ---\n";

  CosyVoice3Perf avg = GetAverage();
  CosyVoice3Perf min_perf = GetMin();
  CosyVoice3Perf max_perf = GetMax();

  std::cout << "Average RTF: " << std::fixed << std::setprecision(6) << avg.rtf
            << " (min: " << min_perf.rtf << ", max: " << max_perf.rtf << ")\n";

  std::cout << "Average LLM time: " << FormatMs(avg.llm_total_ms / 1000.0f)
            << " (min: " << FormatMs(min_perf.llm_total_ms / 1000.0f)
            << ", max: " << FormatMs(max_perf.llm_total_ms / 1000.0f) << ")\n";

  std::cout << "Average Flow time: " << FormatMs(avg.flow_total_ms / 1000.0f)
            << " (min: " << FormatMs(min_perf.flow_total_ms / 1000.0f)
            << ", max: " << FormatMs(max_perf.flow_total_ms / 1000.0f) << ")\n";

  std::cout << "Average Vocoder time: " << FormatMs(avg.vocoder_ms / 1000.0f)
            << " (min: " << FormatMs(min_perf.vocoder_ms / 1000.0f)
            << ", max: " << FormatMs(max_perf.vocoder_ms / 1000.0f) << ")\n";

  float total_audio = 0.0f;
  for (const auto& perf : history_) {
    total_audio += perf.audio_duration_s;
  }
  std::cout << "Total audio generated: " << FormatSeconds(total_audio) << "\n";

  // Prefill and decode speeds
  if (avg.prefill_ms > 0.0f && avg.prefill_tokens > 0) {
    std::cout << "Average Prefill Speed: "
              << FormatTokensPerSec(avg.prefill_tokens,
                                    avg.prefill_ms / 1000.0f)
              << "\n";
  }
  if (avg.decode_ms > 0.0f && avg.decode_tokens > 0) {
    std::cout << "Average Decode Speed: "
              << FormatTokensPerSec(avg.decode_tokens, avg.decode_ms / 1000.0f)
              << "\n";
  }

  std::cout << "============================\n\n";
}

void HmPerfMonitor::PrintEntry(size_t index) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (index >= history_.size()) {
    std::cout << "Invalid index: " << index
              << " (history size: " << history_.size() << ")\n";
    return;
  }

  std::cout << "\n=== Performance Report [" << (index + 1) << "] ===\n";
  std::cout << FormatPerfReport(history_[index]);
  std::cout << "============================\n\n";
}

std::string HmPerfMonitor::GetSummaryString() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (history_.empty()) {
    return "No performance reports collected.\n";
  }

  std::ostringstream ss;
  ss << "=== Performance Summary ===\n";
  ss << "Total inference runs: " << history_.size() << "\n\n";

  CosyVoice3Perf avg = GetAverage();
  ss << FormatPerfReport(avg);

  return ss.str();
}

// ============================================================================
// HmPerfMonitor - Statistics Helpers
// ============================================================================

float HmPerfMonitor::GetAverageRtf() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (history_.empty()) {
    return 0.0f;
  }

  float total_rtf = 0.0f;
  for (const auto& perf : history_) {
    total_rtf += perf.rtf;
  }
  return total_rtf / history_.size();
}

float HmPerfMonitor::GetAverageLlmMs() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (history_.empty()) {
    return 0.0f;
  }

  float total_llm = 0.0f;
  for (const auto& perf : history_) {
    total_llm += perf.llm_total_ms;
  }
  return total_llm / history_.size();
}

float HmPerfMonitor::GetAveragePrefillSpeed() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (history_.empty()) {
    return 0.0f;
  }

  float total_speed = 0.0f;
  int valid_count = 0;
  for (const auto& perf : history_) {
    if (perf.prefill_ms > 0.0f && perf.prefill_tokens > 0) {
      total_speed += perf.prefill_tokens / (perf.prefill_ms / 1000.0f);
      valid_count++;
    }
  }
  return valid_count > 0 ? total_speed / valid_count : 0.0f;
}

float HmPerfMonitor::GetAverageDecodeSpeed() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (history_.empty()) {
    return 0.0f;
  }

  float total_speed = 0.0f;
  int valid_count = 0;
  for (const auto& perf : history_) {
    if (perf.decode_ms > 0.0f && perf.decode_tokens > 0) {
      total_speed += perf.decode_tokens / (perf.decode_ms / 1000.0f);
      valid_count++;
    }
  }
  return valid_count > 0 ? total_speed / valid_count : 0.0f;
}

}  // namespace houmo