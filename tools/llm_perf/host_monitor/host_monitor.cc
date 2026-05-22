/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: host_monitor.cc
 * Description:
 *   host_monitor Implementation File - implements host memory monitoring
 * functionality for LLM performance testing.
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
#if defined(__linux__)
#include "host_monitor.h"

// ========== Memory Monitor Thread ==========
class HostMonitor::HostMonitorImpl {
 private:
  std::thread monitor_thread;    // monitoring thread
  std::atomic<bool> is_running;  // flag to control thread loop
  uint32_t interval;             // sampling interval (ms)
  std::mutex mtx_;
  std::condition_variable cv_;
  std::atomic<size_t> max_virtual_memory;   // track max virtual memory usage
  std::atomic<size_t> max_physical_memory;  // track max physical memory usage

  // main monitoring loop
  void monitorLoop() {
    auto next_time = std::chrono::steady_clock::now();
    while (is_running.load(std::memory_order_relaxed)) {
      HostMemoryInfo mem_info = getProcessHostMemoryInfo();

      // update max memory info
      size_t current_max_virtual =
          max_virtual_memory.load(std::memory_order_relaxed);
      if (mem_info.virtual_memory > current_max_virtual) {
        max_virtual_memory.store(mem_info.virtual_memory,
                                 std::memory_order_relaxed);
      }
      size_t current_max_physical =
          max_physical_memory.load(std::memory_order_relaxed);
      if (mem_info.physical_memory > current_max_physical) {
        max_physical_memory.store(mem_info.physical_memory,
                                  std::memory_order_relaxed);
      }

      next_time += std::chrono::milliseconds(interval);

      // wait for the next interval
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait_until(lock, next_time, [this]() {
        return !is_running.load(std::memory_order_relaxed);
      });
    }
  }

 public:
  HostMonitorImpl(uint32_t interval_ms = 1000)
      : is_running(false), interval(interval_ms) {
    max_physical_memory.store(0, std::memory_order_relaxed);
    max_virtual_memory.store(0, std::memory_order_relaxed);
  }

  ~HostMonitorImpl() {
    if (is_running.load(std::memory_order_relaxed)) {
      stop();
    }
  }

  // start monitoring thread
  void start() {
    if (!is_running.load(std::memory_order_relaxed)) {
      is_running.store(true, std::memory_order_relaxed);

      monitor_thread = std::thread(&HostMonitorImpl::monitorLoop, this);
      std::cout << "[HostMonitorImpl] Monitoring thread has started, interval: "
                << interval << "ms" << std::endl;
    } else {
      std::cout << "[HostMonitorImpl] Monitoring thread is already running"
                << std::endl;
    }
  }

  void stop() {
    if (is_running.load(std::memory_order_relaxed)) {
      is_running.store(false, std::memory_order_relaxed);
      cv_.notify_one();
      if (monitor_thread.joinable()) {
        monitor_thread.join();
      }
    } else {
      std::cout << "[HostHostMonitorImpl] Monitoring thread is not running"
                << std::endl;
    }
  }

  HostMemoryInfo getFinalMemoryInfo() {
    // Wait for monitoring thread to finish to ensure data consistency
    if (monitor_thread.joinable()) {
      monitor_thread.join();
    }

    // Prepare and return final memory info
    HostMemoryInfo final_info;
    final_info.virtual_memory =
        max_virtual_memory.load(std::memory_order_relaxed);
    final_info.physical_memory =
        max_physical_memory.load(std::memory_order_relaxed);

    return final_info;
  }

  HostMemoryInfo getCurrentMemoryInfo() {
    // Get current memory info without waiting for the monitoring thread to
    // finish
    return getProcessHostMemoryInfo();
  }

  HostMemoryInfo getMaxMemoryInfo() {
    // Get the maximum memory values tracked during monitoring
    HostMemoryInfo max_info;
    max_info.virtual_memory =
        max_virtual_memory.load(std::memory_order_relaxed);
    max_info.physical_memory =
        max_physical_memory.load(std::memory_order_relaxed);

    return max_info;
  }
};

HostMonitor::HostMonitor(uint32_t interval_ms)
    : impl_(std::make_unique<HostMonitorImpl>(interval_ms)) {}

HostMonitor::~HostMonitor() = default;

void HostMonitor::start() {
  if (impl_) {
    impl_->start();
  }
}

void HostMonitor::stop() {
  if (impl_) {
    impl_->stop();
  }
}

HostMemoryInfo HostMonitor::getFinalMemoryInfo() {
  if (impl_) {
    return impl_->getFinalMemoryInfo();
  }
  // Return empty struct if implementation is null
  HostMemoryInfo empty_info = {0, 0};
  return empty_info;
}

HostMemoryInfo HostMonitor::getCurrentMemoryInfo() {
  if (impl_) {
    return impl_->getCurrentMemoryInfo();
  }
  // Return empty struct if implementation is null
  HostMemoryInfo empty_info = {0, 0};
  return empty_info;
}

HostMemoryInfo HostMonitor::getMaxMemoryInfo() {
  if (impl_) {
    return impl_->getMaxMemoryInfo();
  }
  // Return empty struct if implementation is null
  HostMemoryInfo empty_info = {0, 0};
  return empty_info;
}

#endif  // Linux implementation