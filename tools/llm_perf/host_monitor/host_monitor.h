/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: host_monitor.h
 * Description:
 *   host_monitor Header File - monitor host memory usage.
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
#ifndef __HOST_MONITOR_H__
#define __HOST_MONITOR_H__
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>  // Add memory header for unique_ptr
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

// host memory struct
struct HostMemoryInfo {
  size_t virtual_memory;   // virtual_memory (bytes)
  size_t physical_memory;  // physical_memory (bytes)
};

// Get Current Process Memory Info
static HostMemoryInfo getProcessHostMemoryInfo() {
  HostMemoryInfo info = {0};

  std::ifstream status_file("/proc/self/status");
  if (!status_file.is_open()) {
    std::cerr << "[HostMonitor] Unable to open /proc/self/status: "
              << strerror(errno) << std::endl;
    return info;
  }

  std::string line;
  while (std::getline(status_file, line)) {
    std::istringstream iss(line);
    std::string key;
    size_t value_kb;

    if (line.find("VmSize:") == 0) {
      iss >> key >> value_kb;
      info.virtual_memory = value_kb * 1024;
    } else if (line.find("VmRSS:") == 0) {
      iss >> key >> value_kb;
      info.physical_memory = value_kb * 1024;
    }
  }
  status_file.close();

  return info;
}

// format Memory Size (bytes -> KB/MB/GB)
static std::string formatMemorySize(size_t bytes) {
  const double KB = 1024.0;
  const double MB = KB * 1024;
  const double GB = MB * 1024;

  char buffer[32];
  if (bytes >= GB) {
    snprintf(buffer, sizeof(buffer), "%.2f GB", bytes / GB);
  } else if (bytes >= MB) {
    snprintf(buffer, sizeof(buffer), "%.2f MB", bytes / MB);
  } else if (bytes >= KB) {
    snprintf(buffer, sizeof(buffer), "%.2f KB", bytes / KB);
  } else {
    snprintf(buffer, sizeof(buffer), "%zu B", bytes);
  }
  return buffer;
}

// ========== Memory Monitor Thread ==========
class HostMonitor {
 private:
  class HostMonitorImpl;
  std::unique_ptr<HostMonitorImpl> impl_;

 public:
  HostMonitor(uint32_t interval_ms = 1000);
  ~HostMonitor();
  // start monitoring thread
  void start();
  void stop();

  // Get final memory info after stopping
  HostMemoryInfo getFinalMemoryInfo();
  
  // Get current memory info
  HostMemoryInfo getCurrentMemoryInfo();
  
  // Get max memory info during monitoring
  HostMemoryInfo getMaxMemoryInfo();
};

#endif
#endif  // Linux implementation