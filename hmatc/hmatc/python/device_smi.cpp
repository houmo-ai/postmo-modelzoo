/* Copyright 2025 HOUMO AI
 *
 * File: device_smi.cpp
 * Description:
 *   Python API for device info getting.
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
#ifdef ENABLE_SMI
#include "hm_sys.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>
#include <sstream>
#include <string>
#include <thread>

using namespace std::chrono;
namespace py = pybind11;

#define COLOR_RED "\x1b[91;20m"
#define COLOR_GREEN "\x1b[92;20m"
#define COLOR_YELLOW "\x1b[93;20m"
#define COLOR_BLUE "\x1b[94;20m"
#define COLOR_MAGENT "\x1b[95;20m"
#define COLOR_CYAN "\x1b[96;20m"
#define COLOR_RESET "\x1b[0m"

class DeviceSmi {
public:
    DeviceSmi(int id = 0) {
        dev_id = id;
    }

    float get_ipu_use_rate() {
        if (dev_id >= 0) {
            return hm_sys_get_ipu_utili_rate(dev_id);
        }
        return 0;
    }

    float get_temperature() {
        float temperature = 0.f;
        if (dev_id >= 0) {
            hm_sys_get_temperature(dev_id, &temperature);
        }

        return temperature;
    }

    hm_mem_info get_mem_info() {
        struct hm_mem_info mem_info;
        if (dev_id >= 0) {
            hm_sys_get_mem_info(dev_id, &mem_info);
        }
        return mem_info;
    }

private:
    int dev_id = -1;
};

class DeviceMonitor {
public:
    DeviceMonitor(int dev_id, int interval = 100) {
        interval_ = (interval > 0) ? interval : 100;
        if (dev_id >= 0) {
            dev_id_ = dev_id;
        } else {
            throw std::runtime_error("Invalid device id");
        }
    }

    void start() {
        monitor_thread = std::move(std::thread([this]() {
            running.store(true, std::memory_order_relaxed);
            float ipu_use_rate_max = 0.f;
            float temperature_max = 0.f;
            float temperature = 0.f;
            struct hm_mem_info mem_info;
            uint32_t mem_used_max = 0.f;
            while (running.load()) {
                float ipu_use_rate = hm_sys_get_ipu_utili_rate(dev_id_);
                hm_sys_get_temperature(dev_id_, &temperature);
                hm_sys_get_mem_info(dev_id_, &mem_info);
                int mem_used = mem_info.mem_used;
                ipu_use_rate_max = (ipu_use_rate > ipu_use_rate_max) ? ipu_use_rate : ipu_use_rate_max;
                temperature_max = (temperature > temperature_max) ? temperature : temperature_max;
                mem_used_max = (mem_used > mem_used_max) ? mem_used : mem_used_max;
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait_for(lock, std::chrono::milliseconds(interval_),
                             [this]() { return !running.load(std::memory_order_relaxed); });
            }
            std::cout << "Device " << dev_id_ << ": "
                      << "Max IPU use rate: " << ipu_use_rate_max << "%, "
                      << "Max Temperature: " << temperature_max << "°C, "
                      << "Max Memory used: " << mem_used_max << "MB" << std::endl;
        }));
    }

    void stop() {
        if (dev_id_ < 0 || !running.load(std::memory_order_relaxed)) {
            return;
        }
        running.store(false, std::memory_order_relaxed);
        cv_.notify_one();
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
        std::cout << "Device " << dev_id_ << " monitor stopped!" << std::endl;
    }

private:
    int dev_id_ = -1;
    int interval_ = 100;
    std::atomic<bool> running{false};
    std::thread monitor_thread;
    std::mutex mtx_;
    std::condition_variable cv_;
};
/**
 * @brief Define Python module for device monitor.
 * Exposes the device monitor Apis to Python
 */
PYBIND11_MODULE(smi, m) {
    m.doc() = "Python bindings for houmo smi apis";
    py::class_<DeviceSmi>(m, "SmiInfo")
        .def(py::init<int>(), py::arg("device_id") = 0, "Initialize SmiInfo object with device ID")
        .def("get_ipu_use_rate", &DeviceSmi::get_ipu_use_rate, "Get IPU utilization rate of the device (return float)")
        .def("get_temperature", &DeviceSmi::get_temperature, "Get device temperature (return float)")
        .def("get_mem_info", &DeviceSmi::get_mem_info, "Get device memory info (return MemInfo object)");
    py::class_<DeviceMonitor>(m, "DeviceMonitor")
        .def(py::init<int, int>(), py::arg("device_id") = 0, py::arg("interval") = 100, "Initialize DeviceMonitor object with device ID")
        .def("start", &DeviceMonitor::start, "Start monitoring the device")
        .def("stop", &DeviceMonitor::stop, "Stop monitoring the device");
    py::class_<hm_mem_info>(m, "MemInfo")
        .def_readonly("total", &hm_mem_info::mem_total)
        .def_readonly("used", &hm_mem_info::mem_used)
        .def_readonly("free", &hm_mem_info::mem_avail);
}
#endif