/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: devices_monitor.h
 * Description:
 *   devices_monitor Header File - monitor device 0 temperature and power and
 * frequency.
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
#include <signal.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <codecvt>
#include <filesystem>
#include <iostream>
#include <locale>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "hm_sys.h"

std::atomic<bool> g_running(true);
std::atomic<float> g_temperature(0);
std::atomic<uint64_t> g_times(0);

std::atomic<float> g_temperature_max(0);
std::atomic<float> g_temperature_min(0);
std::atomic<float> g_temperature_avg(0);

std::atomic<float> g_power_max(0);
std::atomic<float> g_power_min(0);
std::atomic<float> g_power_avg(0);

std::atomic<float> g_ipu_freq_max(0);
std::atomic<float> g_ipu_freq_min(0);
std::atomic<float> g_ipu_freq_avg(0);

std::atomic<uint32_t> g_mem_total(0);
std::atomic<uint32_t> g_mem_used(0);
std::atomic<uint32_t> g_mem_avail(0);

std::mutex mtx_;
std::condition_variable cv_;

std::mutex g_threads_mutex;
std::vector<std::thread> g_monitor_threads;

static std::unordered_map<int, std::shared_ptr<spdlog::logger>>
    g_device_loggers;
static std::mutex g_logger_mutex;

constexpr size_t LOG_FILE_MAX_SIZE = 1024 * 1024 * 100;  // 100MB per log file
constexpr size_t LOG_FILE_MAX_BACKUPS = 5;               // up to 5 backup files
constexpr const char* LOG_DIR =
    "./device_logs/";  // log file storage directory (must exist)

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

// Device-specific log macros (core: automatically bind device index)
#define DEVICE_LOG_INFO(dev_index, ...)         \
  do {                                          \
    auto logger = get_device_logger(dev_index); \
    if (logger) {                               \
      logger->info(__VA_ARGS__);                \
    }                                           \
  } while (0)

// Device monitoring information structure (example)
struct Hm_monitor_infos {
  int dev_id;
  float temperature;
  float power;
  float ipu_freq;
};

/**
 * @brief Device monitoring function (each thread calls this function with
 * different dev_index)
 * @param dev_index Device index
 */
void hm_device_monitor_info(int dev_index, uint32_t interval) {
#ifdef _MSC_VER
  HMODULE hDll = LoadLibraryA("libhal_xh2a.dll");
  typedef int (*HM_SYS_CHECK_DEVICE_INDEX)(int dev_index);
  typedef int (*HM_SYS_GET_TEMPERATURE)(int dev_id, float* temperature);
  typedef int (*HM_SYS_GET_BOARD_POWER)(int dev_id, float* power);
  typedef int (*HM_SYS_GET_IPU_FREQUENCY)(int dev_id, uint64_t* freq);
  HM_SYS_CHECK_DEVICE_INDEX hm_sys_check_device_index = nullptr;
  HM_SYS_GET_TEMPERATURE hm_sys_get_temperature = nullptr;
  HM_SYS_GET_BOARD_POWER hm_sys_get_board_power = nullptr;
  HM_SYS_GET_IPU_FREQUENCY hm_sys_get_ipu_frequency = nullptr;
  hm_sys_check_device_index = (HM_SYS_CHECK_DEVICE_INDEX)GetProcAddress(
      hDll, "hm_sys_check_device_index");
  hm_sys_get_temperature =
      (HM_SYS_GET_TEMPERATURE)GetProcAddress(hDll, "hm_sys_get_temperature");
  hm_sys_get_board_power =
      (HM_SYS_GET_BOARD_POWER)GetProcAddress(hDll, "hm_sys_get_board_power");
  hm_sys_get_ipu_frequency = (HM_SYS_GET_IPU_FREQUENCY)GetProcAddress(
      hDll, "hm_sys_get_ipu_frequency");
#endif
  while (g_running) {
    int ret = hm_sys_check_device_index(dev_index);
    if (ret != 0) {
      std::cerr << "Device " << dev_index << ": invalid index" << std::endl;
    }
    struct Hm_monitor_infos hm_infos;
    hm_infos.dev_id = dev_index;
    hm_sys_get_temperature(hm_infos.dev_id, &hm_infos.temperature);
    hm_sys_get_board_power(hm_infos.dev_id, &hm_infos.power);
    uint64_t freq;
    hm_sys_get_ipu_frequency(hm_infos.dev_id, &freq);
    hm_infos.ipu_freq = (float)(freq) / 1000000.f;
    // 3. Print device monitoring log (automatically written to corresponding
    // device log file)
    if (g_temperature_min.load() == 0)
      g_temperature_min.store(hm_infos.temperature);
    if (g_power_min.load() == 0) g_power_min.store(hm_infos.power);
    if (g_ipu_freq_min.load() == 0) g_ipu_freq_min.store(hm_infos.ipu_freq);
    g_temperature.store(hm_infos.temperature);
    g_times.store(g_times.load() + 1);
    g_temperature_max.store((hm_infos.temperature > g_temperature_max.load())
                                ? hm_infos.temperature
                                : g_temperature_max.load());
    g_temperature_min.store((hm_infos.temperature < g_temperature_min.load())
                                ? hm_infos.temperature
                                : g_temperature_min.load());
    g_temperature_avg.store((hm_infos.temperature +
                             g_temperature_avg.load() * (g_times.load() - 1)) /
                            g_times.load());
    g_power_max.store((hm_infos.power > g_power_max.load())
                          ? hm_infos.power
                          : g_power_max.load());
    g_power_min.store((hm_infos.power < g_power_min.load())
                          ? hm_infos.power
                          : g_power_min.load());
    g_power_avg.store(
        (hm_infos.power + g_power_avg.load() * (g_times.load() - 1)) /
        g_times.load());
    g_ipu_freq_max.store((hm_infos.ipu_freq > g_ipu_freq_max.load())
                             ? hm_infos.ipu_freq
                             : g_ipu_freq_max.load());
    g_ipu_freq_min.store((hm_infos.ipu_freq < g_ipu_freq_min.load())
                             ? hm_infos.ipu_freq
                             : g_ipu_freq_min.load());
    g_ipu_freq_avg.store(
        (hm_infos.ipu_freq + g_ipu_freq_avg.load() * (g_times.load() - 1)) /
        g_times.load());
    DEVICE_LOG_INFO(dev_index,
                    "Device_id: {:d}, Temperature: {:.2f}°C, BoardPower: "
                    "{:.2f}W, IPU Freq: "
                    "{:.2f}MHz",
                    hm_infos.dev_id, hm_infos.temperature, hm_infos.power,
                    hm_infos.ipu_freq);
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait_for(lock, std::chrono::milliseconds(interval),
                 [&]() { return !g_running.load(std::memory_order_relaxed); });
  }
}

/**
 * @brief Clean up all device loggers (called when program exits)
 */
void cleanup_resources() {
  std::lock_guard<std::mutex> lock(g_threads_mutex);
  for (auto& t : g_monitor_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  g_monitor_threads.clear();

  std::lock_guard<std::mutex> log_lock(g_logger_mutex);
  for (auto& [dev_idx, logger] : g_device_loggers) {
    logger->flush();
    spdlog::drop(logger->name());
  }
  g_device_loggers.clear();
  spdlog::shutdown();
}

void device_monitor(int dev_id, uint32_t interval) {
  create_log_dir();
  {
    std::lock_guard<std::mutex> lock(g_threads_mutex);
    g_monitor_threads.emplace_back(hm_device_monitor_info, dev_id, interval);
    std::cout << "Started monitor thread for device " << dev_id << std::endl;
  }

  for (auto& thread : g_monitor_threads) {
    thread.join();
  }
  cleanup_resources();
  return;
}

void stop_monitor(int dev_id) {
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "\n" << std::string(82, '=') << std::endl;
  std::cout << std::string(36, ' ') << "Device " << dev_id
            << " Monitor Summary Report " << std::endl;
  std::cout << std::string(82, '=') << std::endl;

  auto fmt = [](auto v, int prec, const char* unit) -> std::string {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v << unit;
    return o.str();
  };

  std::cout << std::left << std::setw(20) << "Temperature"
            << "|  " << std::left << std::setw(18)
            << fmt(g_temperature_min.load(), 2, "°C(Min)") << " |  "
            << std::left << std::setw(18)
            << fmt(g_temperature_max.load(), 2, "°C(Max)") << " |  "
            << std::left << std::setw(18)
            << fmt(g_temperature_avg.load(), 2, "°C(Avg)") << " |" << std::endl;

  std::cout << std::left << std::setw(20) << "Power"
            << "|  " << std::left << std::setw(18)
            << fmt(g_power_min.load(), 2, " W(Min)") << "|  " << std::left
            << std::setw(18) << fmt(g_power_max.load(), 2, " W(Max)") << "|  "
            << std::left << std::setw(18)
            << fmt(g_power_avg.load(), 2, " W(Avg)") << "|" << std::endl;

  std::cout << std::left << std::setw(20) << "IPU Freq"
            << "|  " << std::left << std::setw(18)
            << fmt(g_ipu_freq_min.load(), 2, " Mhz(Min)") << "|  " << std::left
            << std::setw(18) << fmt(g_ipu_freq_max.load(), 2, " Mhz(Max)")
            << "|  " << std::left << std::setw(18)
            << fmt(g_ipu_freq_avg.load(), 2, " Mhz(Avg)") << "|" << std::endl;

  std::cout << std::left << std::setw(20) << "Mem Info"
            << "|  " << std::left << std::setw(18)
            << (std::to_string(g_mem_total.load()) + " MB(Total)") << "|  "
            << std::left << std::setw(18)
            << (std::to_string(g_mem_used.load()) + " MB(Used)") << "|  "
            << std::left << std::setw(18)
            << (std::to_string(g_mem_avail.load()) + " MB(Avail)") << "|"
            << std::endl;

  g_running.store(false, std::memory_order_relaxed);
  cv_.notify_one();
}

void get_mem_info(int dev_id) {
#ifdef _MSC_VER
  HMODULE hDll = LoadLibraryA("libhal_xh2a.dll");
  typedef int (*HM_SYS_GET_MEM_INFO)(int dev_id, struct hm_mem_info* mem_info);
  HM_SYS_GET_MEM_INFO hm_sys_get_mem_info = nullptr;
  hm_sys_get_mem_info =
      (HM_SYS_GET_MEM_INFO)GetProcAddress(hDll, "hm_sys_get_mem_info");
#endif
  hm_mem_info dev_mem_info;
  if (hm_sys_get_mem_info(dev_id, &dev_mem_info) != 0) {
    std::cerr << "Failed to get memory info for device " << dev_id << std::endl;
    return;
  }

  g_mem_avail.store(dev_mem_info.mem_avail);
  g_mem_total.store(dev_mem_info.mem_total);
  g_mem_used.store(dev_mem_info.mem_used);

  return;
}
float get_temperature() { return g_temperature.load(); }

#endif  // DEVICE_MONITOR_HPP