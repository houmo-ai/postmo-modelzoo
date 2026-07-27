/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: perf_profiler.h
 * Description:
 *   Hierarchical performance profiler for per-inference timing statistics.
 *   Supports nested stages, scoped timers, and throughput metrics.
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

#pragma once

#include "base/houmo.h"

#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace houmo {

#if HOUOMO_ENABLE_PROFILING

// ============================================================================
// PerfProfiler - Per-inference statistics
// ============================================================================

class PerfProfiler {
 public:
  PerfProfiler() = default;
  ~PerfProfiler() = default;

  // ========== Timing interface ==========

  void start(const std::string& path);
  void stop(const std::string& path);

  class ScopedTimer {
   public:
    ScopedTimer(PerfProfiler& profiler, const std::string& path);
    ~ScopedTimer();
   private:
    PerfProfiler& profiler_;
    std::string path_;
  };

  ScopedTimer scope(const std::string& path);

  // ========== Statistics query ==========

  double get_time_ms(const std::string& path) const;
  int get_count(const std::string& path) const;
  double get_avg_time_ms(const std::string& path) const;

  // ========== Hierarchy query ==========

  std::vector<std::string> get_children(const std::string& path) const;
  bool has_stage(const std::string& path) const;

  // ========== Stage configuration ==========

  void set_root_stage(const std::string& stage) { root_stage_ = stage; }
  const std::string& root_stage() const { return root_stage_; }

  // ========== Token statistics ==========

  void set_input_tokens(int n);
  void add_output_token();
  int input_tokens() const;
  int output_tokens() const;

  // ========== Throughput metrics ==========

  void record_ttft();
  double e2e_ms() const;
  double ttft_ms() const;
  double prefill_tps() const;
  double decode_tps() const;
  double overall_tps() const;
  double avg_decode_latency_ms() const;

  // ========== Export interface ==========

  enum class OutputFormat {
    Tree,
    Table,
    Compact
  };

  // Print view structure
  struct PrintView {
    std::string name;
    double time_ms = 0;
    int count = 0;
    std::vector<std::pair<std::string, PrintView>> children;  // Preserve order
  };

  void print_summary(OutputFormat format = OutputFormat::Tree) const;
  PerfStats to_perf_stats() const;

  // Print tree structure
  void print_tree(const PrintView& view, const std::string& prefix = "") const;

  // ========== Reset ==========

  void reset();

 private:
  struct Node {
    double total_time_ms = 0;
    double self_time_ms = 0;
    int count = 0;
    double start_time = 0;
    std::map<std::string, Node> children;
    std::vector<std::string> child_order;  // Child insertion order
  };

  Node root_;
  std::string root_stage_ = "generate";
  int input_tokens_ = 0;
  int output_tokens_ = 0;
  double e2e_start_time_ = 0;
  double ttft_time_ = 0;

  Node* get_or_create_node(const std::string& path);
  const Node* find_node(const std::string& path) const;

  // Convert Node to PrintView
  PrintView node_to_print_view(const Node& node, const std::string& name) const;

  void print_table() const;
  void print_compact() const;
  static double now_ms();
};

#else  // HOUOMO_ENABLE_PROFILING == 0

// ============================================================================
// No-op implementation - zero overhead
// ============================================================================

class PerfProfiler {
 public:
  PerfProfiler() = default;
  ~PerfProfiler() = default;

  void start(const std::string&) {}
  void stop(const std::string&) {}

  class ScopedTimer {
   public:
    ScopedTimer(PerfProfiler&, const std::string&) {}
    ~ScopedTimer() {}
  };

  ScopedTimer scope(const std::string&) { return ScopedTimer(*this, ""); }

  double get_time_ms(const std::string&) const { return 0; }
  int get_count(const std::string&) const { return 0; }
  double get_avg_time_ms(const std::string&) const { return 0; }

  std::vector<std::string> get_children(const std::string&) const { return {}; }
  bool has_stage(const std::string&) const { return false; }

  void set_root_stage(const std::string&) {}
  std::string root_stage() const { return "generate"; }

  void set_input_tokens(int) {}
  void add_output_token() {}
  int input_tokens() const { return 0; }
  int output_tokens() const { return 0; }

  void record_ttft() {}
  double e2e_ms() const { return 0; }
  double ttft_ms() const { return 0; }
  double prefill_tps() const { return 0; }
  double decode_tps() const { return 0; }
  double overall_tps() const { return 0; }
  double avg_decode_latency_ms() const { return 0; }

  enum class OutputFormat { Tree, Table, Compact };

  void print_summary(OutputFormat = OutputFormat::Tree) const {}
  PerfStats to_perf_stats() const { return PerfStats{}; }

  void reset() {}
};

#endif  // HOUOMO_ENABLE_PROFILING

}  // namespace houmo
