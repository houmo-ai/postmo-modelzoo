/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: perf_profiler.cc
 * Description:
 *   Performance profiler implementation for timing inference stages.
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

#include <algorithm>
#include <iomanip>
#include <iostream>

namespace houmo {

#if HOUOMO_ENABLE_PROFILING

// ============================================================================
// PerfProfiler implementation
// ============================================================================

double PerfProfiler::now_ms() {
  auto now = std::chrono::high_resolution_clock::now();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      now.time_since_epoch());
  return static_cast<double>(ns.count()) / 1000000.0;
}

PerfProfiler::Node* PerfProfiler::get_or_create_node(const std::string& path) {
  Node* current = &root_;

  if (path.empty()) return current;

  size_t start = 0;
  size_t end = path.find('.');

  while (start < path.size()) {
    std::string segment;
    if (end == std::string::npos) {
      segment = path.substr(start);
      start = path.size();
    } else {
      segment = path.substr(start, end - start);
      start = end + 1;
      end = path.find('.', start);
    }

    if (!segment.empty()) {
      // Check if this is a newly created child node
      bool is_new =
          (current->children.find(segment) == current->children.end());

      // If newly created, record in child_order first
      if (is_new) {
        current->child_order.push_back(segment);
      }

      // Then move to the child node
      current = &current->children[segment];
    }
  }

  return current;
}

const PerfProfiler::Node* PerfProfiler::find_node(
    const std::string& path) const {
  const Node* current = &root_;

  if (path.empty()) return current;

  size_t start = 0;
  size_t end = path.find('.');

  while (start < path.size()) {
    std::string segment;
    if (end == std::string::npos) {
      segment = path.substr(start);
      start = path.size();
    } else {
      segment = path.substr(start, end - start);
      start = end + 1;
      end = path.find('.', start);
    }

    if (!segment.empty()) {
      auto it = current->children.find(segment);
      if (it == current->children.end()) {
        return nullptr;
      }
      current = &it->second;
    }
  }

  return current;
}

void PerfProfiler::start(const std::string& path) {
  Node* node = get_or_create_node(path);
  double now = now_ms();
  node->start_time = now;

  if (path == root_stage_) {
    e2e_start_time_ = now;
  }
}

void PerfProfiler::stop(const std::string& path) {
  double end_time = now_ms();
  Node* node = get_or_create_node(path);

  if (node->start_time > 0) {
    double elapsed = end_time - node->start_time;
    node->total_time_ms += elapsed;
    node->self_time_ms += elapsed;
    node->count++;
    node->start_time = 0;

    // Update total_time for all parent nodes (but not self_time)
    size_t pos = path.rfind('.');
    if (pos != std::string::npos) {
      std::string parent_path = path.substr(0, pos);
      Node* parent = get_or_create_node(parent_path);
      parent->total_time_ms += elapsed;
      // Subtract child node time from parent's self_time
      parent->self_time_ms -= elapsed;
    }
  }
}

PerfProfiler::ScopedTimer::ScopedTimer(PerfProfiler& profiler,
                                       const std::string& path)
    : profiler_(profiler), path_(path) {
  profiler_.start(path_);
}

PerfProfiler::ScopedTimer::~ScopedTimer() { profiler_.stop(path_); }

PerfProfiler::ScopedTimer PerfProfiler::scope(const std::string& path) {
  return ScopedTimer(*this, path);
}

double PerfProfiler::get_time_ms(const std::string& path) const {
  const Node* node = find_node(path);
  return node ? node->total_time_ms : 0;
}

int PerfProfiler::get_count(const std::string& path) const {
  const Node* node = find_node(path);
  return node ? node->count : 0;
}

double PerfProfiler::get_avg_time_ms(const std::string& path) const {
  int count = get_count(path);
  if (count == 0) return 0;
  return get_time_ms(path) / count;
}

std::vector<std::string> PerfProfiler::get_children(
    const std::string& path) const {
  const Node* node = find_node(path);
  if (!node) return {};

  std::vector<std::string> children;
  for (const auto& pair : node->children) {
    children.push_back(pair.first);
  }
  return children;
}

bool PerfProfiler::has_stage(const std::string& path) const {
  return find_node(path) != nullptr;
}

void PerfProfiler::set_input_tokens(int n) { input_tokens_ = n; }

void PerfProfiler::add_output_token() { output_tokens_++; }

int PerfProfiler::input_tokens() const { return input_tokens_; }

int PerfProfiler::output_tokens() const { return output_tokens_; }

void PerfProfiler::record_ttft() { ttft_time_ = now_ms() - e2e_start_time_; }

double PerfProfiler::e2e_ms() const { return get_time_ms(root_stage_); }

double PerfProfiler::ttft_ms() const { return ttft_time_; }

double PerfProfiler::prefill_tps() const {
  double prefill_time = get_time_ms(root_stage_ + ".prefill");
  if (prefill_time <= 0) return 0;
  return input_tokens_ / (prefill_time / 1000.0);
}

double PerfProfiler::decode_tps() const {
  double decode_time = get_time_ms(root_stage_ + ".decode");
  if (decode_time <= 0 || output_tokens_ <= 0) return 0;
  return output_tokens_ / (decode_time / 1000.0);
}

double PerfProfiler::overall_tps() const {
  double total = e2e_ms();
  if (total <= 0 || output_tokens_ <= 0) return 0;
  return output_tokens_ / (total / 1000.0);
}

double PerfProfiler::avg_decode_latency_ms() const {
  int decode_count = get_count(root_stage_ + ".decode.inference");
  if (decode_count <= 0) return 0;
  return get_time_ms(root_stage_ + ".decode") / decode_count;
}

void PerfProfiler::reset() {
  root_ = Node{};
  root_stage_ = "generate";
  input_tokens_ = 0;
  output_tokens_ = 0;
  e2e_start_time_ = 0;
  ttft_time_ = 0;
}

// ============================================================================
// Export implementation
// ============================================================================

PerfProfiler::PrintView PerfProfiler::node_to_print_view(
    const Node& node, const std::string& name) const {
  PrintView view;
  view.name = name;
  view.time_ms = node.total_time_ms;
  view.count = node.count;

  for (const auto& key : node.child_order) {
    const auto& it = node.children.find(key);
    if (it != node.children.end()) {
      view.children.push_back({key, node_to_print_view(it->second, key)});
    }
  }
  return view;
}

void PerfProfiler::print_tree(const PrintView& view,
                              const std::string& prefix) const {
  // Time numbers start at character position 30
  std::cout << prefix << view.name;

  if (view.children.empty()) {
    // Leaf node: print time and call count
    int current_pos = static_cast<int>(prefix.size() + view.name.size());
    int padding = 30 - current_pos;
    if (padding > 0) {
      std::cout << std::string(padding, ' ');
    }
    if (view.count > 0) {
      std::cout << std::fixed << std::setprecision(2) << view.time_ms << " ms";
      if (view.count > 1) {
        std::cout << " (" << view.count << " calls, avg "
                  << std::setprecision(2) << (view.time_ms / view.count)
                  << " ms)";
      }
    }
  }
  std::cout << "\n";

  // Recursively print child nodes
  for (const auto& [child_name, child_view] : view.children) {
    print_tree(child_view, prefix + "  ");
  }

  // If has child nodes, print total line
  if (!view.children.empty() && view.count > 0) {
    std::cout << prefix << "  total:";
    int current_pos =
        static_cast<int>(prefix.size()) + 8;  // "  total:" = 8 chars
    int padding = 30 - current_pos;
    if (padding > 0) {
      std::cout << std::string(padding, ' ');
    }
    std::cout << std::fixed << std::setprecision(2) << view.time_ms << " ms\n";
  }
}

void PerfProfiler::print_table() const {
  std::cout << "=== Performance Summary ===\n";
  std::cout << "Stage                       Time      Calls   Avg Time\n";
  std::cout << "─────────────────────────────────────────────────────────\n";

  std::function<void(const Node&, const std::string&)> print_node =
      [&](const Node& node, const std::string& path) {
        for (const auto& pair : node.children) {
          std::string full_path =
              path.empty() ? pair.first : path + "." + pair.first;
          const Node& child = pair.second;

          if (child.count > 0) {
            std::cout << std::left << std::setw(28) << full_path;
            std::cout << std::right << std::setw(8) << std::fixed
                      << std::setprecision(2) << child.total_time_ms << " ms";
            std::cout << std::setw(8) << child.count;
            std::cout << std::setw(10) << std::setprecision(2)
                      << (child.total_time_ms / child.count) << " ms\n";
          }

          print_node(child, full_path);
        }
      };

  print_node(root_, "");

  std::cout << "─────────────────────────────────────────────────────────\n";
  std::cout << "E2E:          " << std::fixed << std::setprecision(2)
            << e2e_ms() << " ms\n";
  std::cout << "TTFT:         " << ttft_ms() << " ms\n";
  std::cout << "Prefill TPS:  " << std::setprecision(1) << prefill_tps()
            << " tokens/s (" << input_tokens_ << " tokens)\n";
  std::cout << "Decode TPS:   " << decode_tps() << " tokens/s ("
            << output_tokens_ << " tokens)\n";
  std::cout << "Overall TPS:  " << overall_tps() << " tokens/s\n";
}

void PerfProfiler::print_compact() const {
  std::cout << "[Perf] " << root_stage_ << ": " << std::fixed
            << std::setprecision(2) << e2e_ms() << "ms\n";

  for (const auto& pair : root_.children) {
    const Node& child = pair.second;
    std::cout << "[Perf]   " << pair.first << ": " << child.total_time_ms
              << "ms";
    if (child.count > 1) {
      std::cout << " x" << child.count;
    }
    std::cout << "\n";
  }

  std::cout << "[Perf] E2E=" << e2e_ms() << "ms, TTFT=" << ttft_ms()
            << "ms, TPS=" << std::setprecision(1) << overall_tps() << "\n";
}

PerfStats PerfProfiler::to_perf_stats() const {
  PerfStats stats;
  stats.prefill_time_ms = get_time_ms(root_stage_ + ".prefill");
  stats.decode_time_ms = get_time_ms(root_stage_ + ".decode");
  stats.total_time_ms = e2e_ms();
  stats.ttft_ms = ttft_ms();
  stats.tps = overall_tps();
  stats.embedding_time_ms = get_time_ms(root_stage_ + ".vision");
  stats.n_input_tokens = input_tokens_;
  stats.n_output_tokens = output_tokens_;
  return stats;
}

void PerfProfiler::print_summary(OutputFormat format) const {
  switch (format) {
    case OutputFormat::Tree:
      std::cout << "───────────────────────────────────────────────────────────"
                   "──────────────────────\n";
      {
        auto it = root_.children.find(root_stage_);
        if (it != root_.children.end()) {
          print_tree(node_to_print_view(it->second, root_stage_), "");
        }
      }
      std::cout << "───────────────────────────────────────────────────────────"
                   "──────────────────────\n";
      std::cout << "E2E:                          " << std::fixed
                << std::setprecision(2) << e2e_ms() << " ms\n";
      std::cout << "TTFT:                         " << ttft_ms() << " ms\n";
      std::cout << "Prefill TPS:                  " << std::setprecision(1)
                << prefill_tps() << " tokens/s (" << input_tokens_
                << " tokens)\n";
      std::cout << "Decode TPS:                   " << decode_tps()
                << " tokens/s (" << output_tokens_ << " tokens)\n";
      std::cout << "Overall TPS:                  " << overall_tps()
                << " tokens/s\n";
      std::cout << "Avg Decode Latency:           " << std::setprecision(2)
                << avg_decode_latency_ms() << " ms/token\n";
      std::cout << "───────────────────────────────────────────────────────────"
                   "──────────────────────\n";
      break;
    case OutputFormat::Table:
      print_table();
      break;
    case OutputFormat::Compact:
      print_compact();
      break;
  }
}

#endif  // HOUOMO_ENABLE_PROFILING

}  // namespace houmo
