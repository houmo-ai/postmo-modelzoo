/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: device_monitor.h
 * Description:
 *   Device Monitor Header File - monitor multiple devices' temperature, power,
 * frequency and memory.
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

#ifndef DEVICE_MONITOR_H
#define DEVICE_MONITOR_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hm_sys.h"

struct DeviceStats;
class DeviceMonitor;

struct DeviceStats {
  int dev_id;
  float temperature;
  float power;
  float ipu_freq;
  hm_mem_info mem_info;
  uint32_t times;

  float temperature_max;
  float power_max;
  float ipu_freq_max;

  float temperature_min;
  float power_min;
  float ipu_freq_min;

  uint32_t mem_used_max;
  uint32_t mem_used_min;
  double mem_used_avg;

  double temperature_avg;
  double power_avg;
  double ipu_freq_avg;
};

class DeviceMonitor {
 private:
  std::unordered_map<int, DeviceStats> device_stats_map_;
  uint32_t interval_ms_;
  std::atomic<bool> running_{false};
  std::mutex mtx_;
  std::condition_variable cv_;

  std::mutex g_thread_mutex;
  std::thread g_thread_;
  std::vector<int> deviceIds;

 public:
  explicit DeviceMonitor(uint32_t interval_ms = 1000);
  ~DeviceMonitor();
  DeviceMonitor(const DeviceMonitor&) = delete;
  DeviceMonitor& operator=(const DeviceMonitor&) = delete;
  DeviceMonitor(DeviceMonitor&&) = delete;
  DeviceMonitor& operator=(DeviceMonitor&&) = delete;

  void start();

  void stop();

  void cleanup();

  std::unordered_map<int, DeviceStats> getDeviceStats();
  std::unordered_map<int, DeviceStats> getFinalDeviceStats();
  float getMaxTemperature();
};

#endif  // DEVICE_MONITOR_H