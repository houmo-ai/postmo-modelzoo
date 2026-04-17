/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: device_monitor.cc
 * Description:
 *   Device Monitor Implementation File - monitor multiple devices' temperature,
 * power, frequency and memory.
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

#include "device_monitor.h"

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
constexpr size_t LOG_FILE_MAX_SIZE = 1024 * 1024 * 100;  // 100MB per log file
constexpr size_t LOG_FILE_MAX_BACKUPS = 5;               // up to 5 backup files
constexpr const char* LOG_DIR = "./device_logs/";  // log file storage directory

// Global loggers map and mutex for thread safety
std::unordered_map<int, std::shared_ptr<spdlog::logger>> g_device_loggers;
std::mutex g_logger_mutex;

void create_log_dir() {
#ifdef _WIN32
#include <direct.h>
  if (_access(LOG_DIR, 0) == -1) {
    _mkdir(LOG_DIR);
  }
#else
  struct stat st = {0};
  if (stat(LOG_DIR, &st) == -1) {
    mkdir(LOG_DIR, 0755);
  }
#endif
}

/**
 * @brief Initialize a dedicated logger for specified device (thread-safe)
 * @param dev_index Device index
 * @return Pointer to device logger, returns nullptr on failure
 */
std::shared_ptr<spdlog::logger> init_device_logger(int dev_index) {
  std::lock_guard<std::mutex> lock(g_logger_mutex);

  // Check if already initialized to avoid duplicate creation
  auto it = g_device_loggers.find(dev_index);
  if (it != g_device_loggers.end()) {
    return it->second;
  }

  try {
    // 1. Construct log filename (e.g., device_logs/device_0.log)
    std::string log_filename =
        std::string(LOG_DIR) + "device_" + std::to_string(dev_index) + ".log";

    // 2. Create file sink (rotating by size, thread-safe)
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_filename, LOG_FILE_MAX_SIZE, LOG_FILE_MAX_BACKUPS);

    // 3. Configure log format (including device ID, timestamp, level, content)
    std::string log_pattern = "[%Y-%m-%d %H:%M:%S.%e] [DEVICE-" +
                              std::to_string(dev_index) + "] [%^%l%$] %v";
    file_sink->set_pattern(log_pattern);

    // 4. Create logger
    std::vector<spdlog::sink_ptr> sinks{file_sink};
    auto logger = std::make_shared<spdlog::logger>(
        "device_" + std::to_string(dev_index), sinks.begin(), sinks.end());

    // 6. Configure log level and flush policy
    logger->set_level(spdlog::level::info);
    logger->flush_on(
        spdlog::level::info);  // Flush immediately for info and above
    logger->set_error_handler([dev_index](const std::string& msg) {
      std::cerr << "DEVICE-" << dev_index << " Log error: " << msg << std::endl;
    });

    // 7. Store in global map
    g_device_loggers[dev_index] = logger;

    spdlog::info("DEVICE-{} Logger initialization complete, log file: {}",
                 dev_index, log_filename);
    return logger;
  } catch (const spdlog::spdlog_ex& ex) {
    std::cerr << "DEVICE-" << dev_index
              << " Logger initialization failed: " << ex.what() << std::endl;
    return nullptr;
  }
}

/**
 * @brief Get logger for specified device (automatic initialization)
 * @param dev_index Device index
 * @return Logger pointer
 */
inline std::shared_ptr<spdlog::logger> get_device_logger(int dev_index) {
  auto logger = init_device_logger(dev_index);
  if (!logger) {
    // Fall back to default logger
    return spdlog::default_logger();
  }
  return logger;
}

void cleanup_resources() {
  std::lock_guard<std::mutex> log_lock(g_logger_mutex);
  for (auto& [dev_idx, logger] : g_device_loggers) {
    logger->flush();
    spdlog::drop(logger->name());
  }
  g_device_loggers.clear();
  spdlog::shutdown();
}

// Device-specific log macros
#define DEVICE_LOG_INFO(dev_index, ...)         \
  do {                                          \
    auto logger = get_device_logger(dev_index); \
    if (logger) {                               \
      logger->info(__VA_ARGS__);                \
    }                                           \
  } while (0)
}  // namespace

DeviceMonitor::DeviceMonitor(uint32_t interval_ms) {
  this->interval_ms_ = interval_ms;
}

DeviceMonitor::~DeviceMonitor() = default;

namespace {
template <typename T>
void update_running_avg(double& avg, uint32_t count, T value) {
  avg = (static_cast<double>(value) + avg * (count - 1)) / count;
}
}  // namespace

void DeviceMonitor::start() {
#ifdef _MSC_VER
  HMODULE hDll = LoadLibraryA("libhal_xh2a.dll");
  typedef int (*HM_SYS_GET_DEVICE_INFO)(hm_device_info * info);
  HM_SYS_GET_DEVICE_INFO hm_sys_get_device_info = nullptr;
  hm_sys_get_device_info =
      (HM_SYS_GET_DEVICE_INFO)GetProcAddress(hDll, "hm_sys_get_device_info");
#endif
  hm_device_info dev_info = {0};
  int ret = hm_sys_get_device_info(&dev_info);
  if (ret <= 0 || dev_info.num_devices <= 0) {
    throw std::runtime_error(
        "Failed to get device info, No M50 devices finded!");
  } else {
    for (int id = 0; id < dev_info.num_devices; ++id) {
      deviceIds.push_back(dev_info.device_ids[id]);
    }
  }

  // Initialize statistics for all devices
  for (int dev_id : deviceIds) {
    DeviceStats stats{};
    memset(&stats, 0, sizeof(stats));
    stats.dev_id = dev_id;
    device_stats_map_[dev_id] = stats;
  }

  g_thread_ = std::thread([this]() {
#ifdef _MSC_VER
    HMODULE hDll = LoadLibraryA("libhal_xh2a.dll");
    typedef int (*HM_SYS_CHECK_DEVICE_INDEX)(int dev_index);
    typedef int (*HM_SYS_GET_TEMPERATURE)(int dev_id, float* temperature);
    typedef int (*HM_SYS_GET_BOARD_POWER)(int dev_id, float* power);
    typedef int (*HM_SYS_GET_IPU_FREQUENCY)(int dev_id, uint64_t* freq);
    typedef int (*HM_SYS_GET_MEM_INFO)(int dev_id,
                                       struct hm_mem_info* mem_info);
    HM_SYS_CHECK_DEVICE_INDEX hm_sys_check_device_index = nullptr;
    HM_SYS_GET_TEMPERATURE hm_sys_get_temperature = nullptr;
    HM_SYS_GET_BOARD_POWER hm_sys_get_board_power = nullptr;
    HM_SYS_GET_IPU_FREQUENCY hm_sys_get_ipu_frequency = nullptr;
    HM_SYS_GET_MEM_INFO hm_sys_get_mem_info = nullptr;
    hm_sys_check_device_index = (HM_SYS_CHECK_DEVICE_INDEX)GetProcAddress(
        hDll, "hm_sys_check_device_index");
    hm_sys_get_temperature =
        (HM_SYS_GET_TEMPERATURE)GetProcAddress(hDll, "hm_sys_get_temperature");
    hm_sys_get_board_power =
        (HM_SYS_GET_BOARD_POWER)GetProcAddress(hDll, "hm_sys_get_board_power");
    hm_sys_get_ipu_frequency = (HM_SYS_GET_IPU_FREQUENCY)GetProcAddress(
        hDll, "hm_sys_get_ipu_frequency");
    hm_sys_get_mem_info =
        (HM_SYS_GET_MEM_INFO)GetProcAddress(hDll, "hm_sys_get_mem_info");
#endif
    create_log_dir();
    if (!running_.load()) running_.store(true);

    while (running_.load()) {
      for (int dev_id : deviceIds) {
        float temperature = 0.0f;
        float power = 0.0f;
        float ipu_freq = 0.0f;
        hm_mem_info mem_info = {0};

        int ret = hm_sys_check_device_index(dev_id);
        if (ret != 0) {
          std::cerr << "Device " << dev_id << ": invalid index" << std::endl;
          continue;
        }

        hm_sys_get_temperature(dev_id, &temperature);
        hm_sys_get_board_power(dev_id, &power);
        uint64_t freq;
        hm_sys_get_ipu_frequency(dev_id, &freq);
        hm_sys_get_mem_info(dev_id, &mem_info);
        ipu_freq = static_cast<float>(freq) / 1000000.f;

        std::lock_guard<std::mutex> lock(mtx_);
        auto& device_stats = device_stats_map_[dev_id];
        device_stats.temperature = temperature;
        device_stats.power = power;
        device_stats.ipu_freq = ipu_freq;
        device_stats.mem_info = mem_info;

        // Update statistics
        if (device_stats.temperature_min == 0)
          device_stats.temperature_min = device_stats.temperature;
        if (device_stats.power_min == 0)
          device_stats.power_min = device_stats.power;
        if (device_stats.ipu_freq_min == 0)
          device_stats.ipu_freq_min = device_stats.ipu_freq;
        if (device_stats.mem_used_min == 0)
          device_stats.mem_used_min = device_stats.mem_info.mem_used;

        device_stats.times++;
        device_stats.temperature_max =
            (device_stats.temperature > device_stats.temperature_max)
                ? device_stats.temperature
                : device_stats.temperature_max;
        device_stats.temperature_min =
            (device_stats.temperature <= device_stats.temperature_min)
                ? device_stats.temperature
                : device_stats.temperature_min;
        device_stats.temperature_avg =
            (device_stats.temperature +
             device_stats.temperature_avg * (device_stats.times - 1)) /
            device_stats.times;
        device_stats.power_max = (device_stats.power > device_stats.power_max)
                                     ? device_stats.power
                                     : device_stats.power_max;
        device_stats.power_min = (device_stats.power < device_stats.power_min)
                                     ? device_stats.power
                                     : device_stats.power_min;
        device_stats.power_avg =
            (device_stats.power +
             device_stats.power_avg * (device_stats.times - 1)) /
            device_stats.times;
        device_stats.ipu_freq_max =
            (device_stats.ipu_freq > device_stats.ipu_freq_max)
                ? device_stats.ipu_freq
                : device_stats.ipu_freq_max;
        device_stats.ipu_freq_min =
            (device_stats.ipu_freq < device_stats.ipu_freq_min)
                ? device_stats.ipu_freq
                : device_stats.ipu_freq_min;
        device_stats.ipu_freq_avg =
            (device_stats.ipu_freq +
             device_stats.ipu_freq_avg * (device_stats.times - 1)) /
            device_stats.times;
        device_stats.mem_used_max =
            std::max(device_stats.mem_used_max, device_stats.mem_info.mem_used);
        device_stats.mem_used_min =
            std::min(device_stats.mem_used_min, device_stats.mem_info.mem_used);
        update_running_avg(device_stats.mem_used_avg, device_stats.times,
                           device_stats.mem_info.mem_used);

        DEVICE_LOG_INFO(device_stats.dev_id,
                        "Device_id: {:d}, Temperature: {:.2f}°C, BoardPower: "
                        "{:.2f}W, IPU Freq: {:.2f}MHz, Mem Used: {:d}MB, Mem "
                        "Avail: {:d}MB",
                        device_stats.dev_id, device_stats.temperature,
                        device_stats.power, device_stats.ipu_freq,
                        device_stats.mem_info.mem_used,
                        device_stats.mem_info.mem_avail);
      }

      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait_for(lock, std::chrono::milliseconds(interval_ms_),
                   [&]() { return !running_.load(std::memory_order_relaxed); });
    }
  });
}

void DeviceMonitor::stop() {
  running_.store(false, std::memory_order_relaxed);
  cv_.notify_one();
  if (g_thread_.joinable()) {
    g_thread_.join();
  }
}

void DeviceMonitor::cleanup() { cleanup_resources(); }

std::unordered_map<int, DeviceStats> DeviceMonitor::getDeviceStats() {
#ifdef _MSC_VER
  HMODULE hDll = LoadLibraryA("libhal_xh2a.dll");
  typedef int (*HM_SYS_GET_MEM_INFO)(int dev_id, struct hm_mem_info* mem_info);
  HM_SYS_GET_MEM_INFO hm_sys_get_mem_info = nullptr;
  hm_sys_get_mem_info =
      (HM_SYS_GET_MEM_INFO)GetProcAddress(hDll, "hm_sys_get_mem_info");
#endif
  std::lock_guard<std::mutex> lock(mtx_);
  for (int dev_id : deviceIds) {
    auto& device_stats = device_stats_map_[dev_id];
    hm_sys_get_mem_info(device_stats.dev_id, &device_stats.mem_info);
  }
  return device_stats_map_;
}

std::unordered_map<int, DeviceStats> DeviceMonitor::getFinalDeviceStats() {
  if (g_thread_.joinable()) {
    g_thread_.join();
  }

  // Return final statistics
  std::lock_guard<std::mutex> lock(mtx_);
  return device_stats_map_;
}

float DeviceMonitor::getMaxTemperature() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (device_stats_map_.empty()) {
    return -1.0f;  // Return -1 to indicate no device data
  }

  float max_temp = 0.0f;
  bool first = true;

  for (const auto& pair : device_stats_map_) {
    const DeviceStats& stats = pair.second;
    if (first || stats.temperature_max > max_temp) {
      max_temp = stats.temperature_max;
      first = false;
    }
  }

  return max_temp;
}