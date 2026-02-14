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
class MemoryMonitor::MemoryMonitorImpl {
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

      // wait for the next interval
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait_for(lock, std::chrono::milliseconds(interval), [this]() {
        return !is_running.load(std::memory_order_relaxed);
      });
    }
  }

 public:
  MemoryMonitorImpl(uint32_t interval_ms = 1000)
      : is_running(false), interval(interval_ms) {
    max_physical_memory.store(0, std::memory_order_relaxed);
    max_virtual_memory.store(0, std::memory_order_relaxed);
  }

  ~MemoryMonitorImpl() {
    if (is_running.load(std::memory_order_relaxed)) {
      stop();
    }
  }

  // start monitoring thread
  void start() {
    if (!is_running.load(std::memory_order_relaxed)) {
      is_running.store(true, std::memory_order_relaxed);

      monitor_thread = std::thread(&MemoryMonitorImpl::monitorLoop, this);
      std::cout
          << "[MemoryMonitorImpl] Monitoring thread has started, interval: "
          << interval << "ms" << std::endl;
    } else {
      std::cout << "[MemoryMonitorImpl] Monitoring thread is already running"
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
      std::cout << "[HostMemoryMonitorImpl] Monitoring thread is not running"
                << std::endl;
    }
    // print memory info
    std::cout << "\n[HostMemoryMonitorImpl] Memory Monitoring (" << interval
              << "ms interval)" << std::endl;
    std::cout << "  Virtual Memory: "
              << formatMemorySize(
                     max_virtual_memory.load(std::memory_order_relaxed))
              << std::endl;
    std::cout << "  Physical Memory: "
              << formatMemorySize(
                     max_physical_memory.load(std::memory_order_relaxed))
              << std::endl;
  }
};

MemoryMonitor::MemoryMonitor(uint32_t interval_ms)
    : impl_(new MemoryMonitorImpl(interval_ms)) {}

MemoryMonitor::~MemoryMonitor() {
  if (impl_ != nullptr) {
    delete impl_;
    impl_ = nullptr;
  }
}

void MemoryMonitor::start() {
  if (impl_ != nullptr) {
    impl_->start();
  }
}

void MemoryMonitor::stop() {
  if (impl_ != nullptr) {
    impl_->stop();
  }
}

#endif  // Linux implementation