/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: main.cc
 * Description:
 *   HM System Check Tool - Main application for checking system status,
 * hardware information, and performance testing for HOUMO AI devices.
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
#include "argparse.hpp"
#include "hm_sys.h"
#include "models.h"
#include "tcim/tcim_runtime.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <regex>
#include <thread>

using namespace std::chrono;

#define COLOR_RED "\x1b[91;20m"
#define COLOR_GREEN "\x1b[92;20m"
#define COLOR_YELLOW "\x1b[93;20m"
#define COLOR_BLUE "\x1b[94;20m"
#define COLOR_MAGENT "\x1b[95;20m"
#define COLOR_CYAN "\x1b[96;20m"
#define COLOR_RESET "\x1b[0m"

static const int INDENT = 2;       // Indentation spaces
static const int COL_PADDING = 4;  // Column padding
static std::string HOUMO_TARGET = "xh2";

typedef enum CheckStatus { PASS = 0, FAIL, WARN } checkStatus_t;

typedef struct CheckItem {
    std::string name;
    CheckStatus status;
    std::string message;
    std::string description;
} checkItem_t;

typedef struct HmVerInfo {
    std::string sdk_build_time;
    std::string sdk_version;
    std::string driver_version;
    std::string runtime_version;
    std::string runtime_build_time;
} hmVerInfo_t;

typedef struct HmDeviceInfo {
    int64_t device_id;
    std::string vendor;
    std::string serial_number;
    std::string device_name;
    int32_t core_count;
    float max_computing_power{0};  // in TOPS
    float computing_power{0};      // in TOPS
    std::string firmware_version;
    std::string ddr_size;  // in GB
    std::string dvfs_mode;
    std::string pcie_bandwidth{"unavailable"};  // in MB/s
    std::string pcie_bdf{"unavailable"};
    std::string temperature;                 // in C
    std::string avail_mem;                   // in MB
    std::string used_mem;                    // in MB
    std::string total_mem;                   // in MB
    std::string core_util_rate;              // in %
    std::vector<std::string> per_core_util;  // in %
    float avg_ipu_frequency{0};              // in MHz
    float max_ipu_frequency{1400.00};        // in MHz
    std::string board_power;                 // in W
    std::string ipu_voltage;                 // in mV
    float ddr_bandwidth_read{0};             // in MB/s
    float ddr_bandwidth_write{0};            // in MB/s
    float pcie_bandwidth_H2H{0};             // in GB/s
    float pcie_bandwidth_H2D{0};
    float pcie_bandwidth_D2H{0};
    float pcie_bandwidth_D2D{0};
} hmDeviceInfo_t;

typedef struct StatsInfo {
    int32_t idx{-1};
    int32_t repeat{0};
    int64_t start_timestamp{0};
    int64_t end_timestamp{0};
} statsInfo_t;

struct DeviceField {
    std::string key;
    std::string value;
};

std::vector<DeviceField> build_device_fields(const hmDeviceInfo_t &d) {
    std::vector<DeviceField> f;
    char buf[32];
    // f.push_back({"ID", std::to_string((long long)d.device_id)});
    f.push_back({"Name", d.device_name});
    f.push_back({"Vendor", d.vendor});
    f.push_back({"Serial", d.serial_number});
    f.push_back({"Firmware", d.firmware_version});
    memset(buf, 0, sizeof(buf));
    sprintf(buf, "%.2f MHz", d.avg_ipu_frequency);
    f.push_back({"Cur IPU Freq", std::string(buf)});
    f.push_back({"Core Count", std::to_string(d.core_count)});
    memset(buf, 0, sizeof(buf));
    sprintf(buf, "%.2f TOPS", d.max_computing_power);
    f.push_back({"Max Compute", std::string(buf)});
    // memset(buf, 0, sizeof(buf));
    // sprintf(buf, "%.2f TOPS", d.computing_power);
    // f.push_back({"Measured Compute", std::string(buf)});
    f.push_back({"DDR", d.ddr_size});
    f.push_back({"Mem Total", d.total_mem});
    f.push_back({"Mem Used", d.used_mem});
    f.push_back({"Mem Avail", d.avail_mem});
    f.push_back({"Core Util", d.core_util_rate});
    f.push_back({"PCIe BDF", d.pcie_bdf});
    f.push_back({"PCIe BW", d.pcie_bandwidth});
    f.push_back({"Temp", d.temperature});
    // f.push_back({"DDR Read", std::to_string(d.ddr_bandwidth_read) + "
    // GB/s"}); f.push_back({"DDR Write", std::to_string(d.ddr_bandwidth_write)
    // + " GB/s"});
    return f;
}

class CheckReport {
public:
    void add(const std::string &name, CheckStatus status,
             const std::string &msg = "", const std::string &desc = "") {
        items.push_back({name, status, msg, desc});
    }

    void print() const {
        printf("\n=== System Check Report ===\n");
        for (const auto &item : items) {
            std::string status_str;
            switch (item.status) {
            case CheckStatus::PASS:
                status_str = "[PASS]";
                status_str = COLOR_GREEN + status_str + COLOR_RESET;
                break;
            case CheckStatus::FAIL:
                status_str = "[FAIL]";
                status_str = COLOR_RED + status_str + COLOR_RESET;
                break;
            case CheckStatus::WARN:
                status_str = "[WARN]";
                status_str = COLOR_YELLOW + status_str + COLOR_RESET;
                break;
            }
            printf("  %-30s %-20s %-20s  %s\n", item.name.c_str(),
                   status_str.c_str(), item.message.c_str(),
                   item.description.c_str());
        }
        printf("===========================\n");
    }

    struct Summary {
        int pass{0};
        int warn{0};
        int fail{0};
    };

    Summary summary() const {
        Summary s;
        for (const auto &i : items) {
            switch (i.status) {
            case CheckStatus::PASS:
                s.pass++;
                break;
            case CheckStatus::WARN:
                s.warn++;
                break;
            case CheckStatus::FAIL:
                s.fail++;
                break;
            }
        }
        return s;
    }

    bool all_pass() const {
        for (auto &i : items)
            if (i.status == CheckStatus::FAIL)
                return false;
        return true;
    }

    bool has_fail() const {
        for (const auto &i : items)
            if (i.status == CheckStatus::FAIL)
                return true;
        return false;
    }

    bool has_warn() const {
        for (const auto &i : items)
            if (i.status == CheckStatus::WARN)
                return true;
        return false;
    }

private:
    std::vector<CheckItem> items;
};

static void print_devices_aligned(const std::vector<hmDeviceInfo_t> &devices) {
    const size_t N = devices.size();
    if (N == 0)
        return;

    // Generate fields for each device
    std::vector<std::vector<DeviceField>> all;
    for (auto &d : devices)
        all.push_back(build_device_fields(d));

    const size_t ROWS = all[0].size();

    // Calculate maximum width for each column (including indentation)
    std::vector<int> col_width(N, 0);
    for (size_t c = 0; c < N; ++c) {
        for (size_t r = 0; r < ROWS; ++r) {
            int len = INDENT + all[c][r].key.size() + 2 +
                      all[c][r].value.size();  // "  key: value"
            col_width[c] = std::max(col_width[c], len);
        }
    }

    printf("\n========== Devices ==========\n\n");

    // Print the header row
    for (size_t i = 0; i < N; ++i)
        printf("Device %zu:%-*s", devices[i].device_id,
               col_width[i] - (int)std::string("Device X:").size() +
                   COL_PADDING,
               "");
    printf("\n");

    // Print each field row
    for (size_t r = 0; r < ROWS; ++r) {
        for (size_t c = 0; c < N; ++c) {
            printf("%*s%-*s: %-*s", INDENT, "", (int)all[c][r].key.size(),
                   all[c][r].key.c_str(),
                   (int)(col_width[c] - INDENT - all[c][r].key.size() - 2),
                   all[c][r].value.c_str());
            printf("%*s", COL_PADDING, "");  // Column spacing
        }
        printf("\n");
    }

    printf("\n=============================\n");
}

// Remove leading and trailing spaces
static inline std::string trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    size_t e = s.find_last_not_of(" \t\r\n");
    return (b == std::string::npos) ? "" : s.substr(b, e - b + 1);
}

static std::string fmt_hex_or_unavail(int value) {
    if (value < 0)
        return "unavailable";
    char buf[32];
    sprintf(buf, "0x%x", value);
    return std::string(buf);
}

static std::string fmt_percent_or_unavail(float v) {
    if (v < 0.0f)
        return "unavailable";
    char buf[32];
    sprintf(buf, "%.2f %%", v);
    return std::string(buf);
}

static std::string fmt_mvolt_or_unavail(float v) {
    if (v < 0.0f)
        return "unavailable";
    char buf[32];
    sprintf(buf, "%.2f mV", v);
    return std::string(buf);
}

static std::string fmt_temp_or_unavail(float v) {
    if (v < 0.0f)
        return "unavailable";
    char buf[32];
    sprintf(buf, "%.2f C", v);
    return std::string(buf);
}

static void get_device_info(hmDeviceInfo_t &devInfo) {
    const int dev_id = devInfo.device_id;
    // Vendor
    devInfo.vendor = fmt_hex_or_unavail(hm_sys_get_vendor_id(dev_id));
    // SN
    char buf[128] = {0};
    devInfo.serial_number = hm_sys_get_device_sn(dev_id, buf, sizeof(buf)) == 0
                                ? buf
                                : "unavailable";

    // Device name
    memset(buf, 0, sizeof(buf));
    devInfo.device_name = hm_sys_get_device_name(dev_id, buf, sizeof(buf)) == 0
                              ? buf
                              : "unavailable";
    // core count
#ifndef _MSC_VER
    devInfo.core_count = std::max(0, hm_sys_get_core_count(dev_id));
#else
    // Windows does not support hm_sys_get_core_count yet.
    devInfo.core_count = 2;
#endif
    // Computing power
    devInfo.max_computing_power =
        std::max(0, hm_sys_get_computing_power(dev_id));

    memset(buf, 0, sizeof(buf));
    devInfo.firmware_version =
        hm_sys_get_device_version(dev_id, buf, sizeof(buf)) == 0
            ? (buf[0] == 'v') ? buf : ("v" + std::string(buf))
            : "unavailable";
    // DDR
    uint64_t ddr = 0;
    devInfo.ddr_size = (hm_sys_get_ddr_size(dev_id, &ddr) == 0)
                           ? std::to_string(ddr / (1024 * 1024)) + " MB"
                           : "unavailable";

    // Utilization
    devInfo.core_util_rate =
        fmt_percent_or_unavail(hm_sys_get_ipu_utili_rate(dev_id));

    devInfo.per_core_util.clear();
    for (int c = 0; c < devInfo.core_count; ++c) {
        float v = hm_sys_get_ipu_core_utili_rate(dev_id, c);
        devInfo.per_core_util.push_back(fmt_percent_or_unavail(v));
    }

    // Frequency
    uint64_t freq = 0;
    devInfo.avg_ipu_frequency =
        (hm_sys_get_ipu_frequency(dev_id, &freq) == 0) ? float(freq) / 1e6 : 0;

    // Voltage
    float voltage = 0.0f;
    devInfo.ipu_voltage = fmt_mvolt_or_unavail(
        (hm_sys_get_ipu_voltage(dev_id, &voltage) == 0) ? voltage : -1.f);

    struct hm_mem_info mem = {0};
    if (hm_sys_get_mem_info(dev_id, &mem) == 0) {
        devInfo.total_mem = std::to_string(mem.mem_total) + " MB";
        devInfo.used_mem = std::to_string(mem.mem_used) + " MB";
        devInfo.avail_mem = std::to_string(mem.mem_avail) + " MB";
    } else {
        devInfo.total_mem = devInfo.used_mem = devInfo.avail_mem =
            "unavailable";
    }

    // Temperature
    float temp = 0.0f;
    devInfo.temperature = fmt_temp_or_unavail(
        (hm_sys_get_temperature(dev_id, &temp) == 0) ? temp : -1.f);

#ifndef _MSC_VER
    // PCIe BDF
    memset(buf, 0, sizeof(buf));
    devInfo.pcie_bdf = (hm_sys_get_bdf(dev_id, buf, sizeof(buf)) == 0)
                           ? trim(buf)
                           : "unavailable";
    // PCIe bandwidth
    memset(buf, 0, sizeof(buf));
    devInfo.pcie_bandwidth =
        (hm_sys_get_bandwidth(dev_id, buf, sizeof(buf)) == 0) ? buf
                                                              : "unavailable";
#endif
}

static void get_runtime_info(hmVerInfo_t &hmVerInfo) {
    std::string text = tcim::GetVersion();
    std::istringstream iss(text);
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(iss, line)) {
        line = trim(line);
        if (line.rfind("- Build Time:", 0) == 0) {
            auto pos = line.find(":");
            if (pos != std::string::npos)
                line = trim(line.substr(pos + 1));
        }
        lines.emplace_back(line);
    }
    hmVerInfo.runtime_version = lines[0];
    hmVerInfo.runtime_build_time = lines[2];
}

/* Print SDK / driver / firmware information */
static void get_version_info(hmVerInfo_t &hmVerInfo) {
    char buf[128] = {0};
    hmVerInfo.sdk_build_time =
        hm_sys_get_buildtime(buf, sizeof(buf)) == 0 ? buf : "unavailable";

    memset(buf, 0, sizeof(buf));
    hmVerInfo.sdk_version =
        hm_sys_get_version(buf, sizeof(buf)) == 0 ? buf : "unavailable";

    memset(buf, 0, sizeof(buf));
    hmVerInfo.driver_version =
        hm_sys_get_driver_version(buf, sizeof(buf)) == 0 ? buf : "unavailable";

    get_runtime_info(hmVerInfo);
}

static int32_t SetInputData(tcim::Module &module,
                            std::map<std::string, tcim::Tensor> &inputs) {
    std::random_device rd;
    std::mt19937 rng(rd());
    for (int32_t i = 0; i < module.GetInputNum(); ++i) {
        auto name = module.GetInputName(i);
        auto info = module.GetInputInfo(name);
        auto tensor = tcim::Tensor::CreateHostTensor(info.AsContiguous());
        // fill tensor
        if (tcim::DataType::FLOAT16 == info.DataType()) {
            std::uniform_real_distribution<float> dist(0, 1.0f);
            auto fp32_tensor = tcim::Tensor::CreateHostTensor(
                info.AsContiguous().AsType(tcim::DataType::FLOAT32));
            auto *data = (float *)fp32_tensor.Data();
            for (int64_t j = 0; j < fp32_tensor.MemSize() / sizeof(float);
                 ++j) {
                data[j] = dist(rng);
            }
            fp32_tensor.CastTo(tensor);
        } else if (tcim::DataType::INT8 == info.DataType()) {
            std::uniform_int_distribution<int32_t> dist(-128, 127);
            auto *data = (int8_t *)tensor.Data();
            for (int64_t j = 0; j < tensor.MemSize(); ++j) {
                data[j] = dist(rng);
            }
        }
        module.SetInput(name, tensor);
    }
    return 0;
}

class Barrier {
public:
    explicit Barrier(unsigned count)
        : threshold(count), count(count), generation(0) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        unsigned gen = generation;

        if (--count == 0) {
            generation++;
            count = threshold;
            cv.notify_all();
        } else {
            cv.wait(lock, [this, gen] { return gen != generation; });
        }
    }

private:
    std::mutex mtx;
    std::condition_variable cv;
    const unsigned threshold;
    unsigned count;
    unsigned generation;
};

static void Infer(int32_t tid, std::string model_name, int32_t warmup,
                  int32_t rounds, statsInfo_t &stats,
                  tcim::Module::WeightManager &wm, Barrier &barrier) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100 * tid));
    auto option = tcim::Module::Option(wm);
    auto *model = get_model(model_name);
    printf("[INFO] Infer %d started, wramup: %d, rounds: %d, repeat: %d\n", tid,
           warmup, rounds, stats.repeat);
    auto module = tcim::Module::LoadFromMem(model->data, model->size, option);
    if (module.GetInitStatus() != tcim::Status::OK) {
        printf("%s[ERROR] Failed to load model: %s%s\n", COLOR_RED,
               model_name.c_str(), COLOR_RESET);
        return;
    }
    tcim::Stream stream(true);
    auto status = module.SetStream(stream);
    if (status != tcim::Status::OK) {
        printf("%s[ERROR] Failed to set stream, and error code: %d%s\n",
               COLOR_RED, status, COLOR_RESET);
        return;
    }

    std::map<std::string, tcim::Tensor> inputs;
    if (SetInputData(module, inputs) != 0) {
        printf("%s[ERROR] Failed to set input data%s\n", COLOR_RED,
               COLOR_RESET);
        return;
    }

    tcim::Module::RunOption run_option;
    bool sync = false;
    // warmup
    for (int32_t i = 0; i < warmup; ++i) {
        run_option.Rounds(1);
        status = module.Run(sync, run_option);
        if (status != tcim::Status::OK) {
            printf("%s[ERROR] Failed to run warmup, and error code: %d%s\n",
                   COLOR_RED, status, COLOR_RESET);
            return;
        }
    }
    status = module.Sync();
    if (status != tcim::Status::OK) {
        printf("%s[ERROR] Failed to sync warmup, and error code: %d%s\n",
               COLOR_RED, status, COLOR_RESET);
        return;
    }
    barrier.wait();
    stats.start_timestamp =
        duration_cast<microseconds>(system_clock::now().time_since_epoch())
            .count();
    for (int32_t i = 0; i < stats.repeat; ++i) {
        run_option.Rounds(rounds);
        status = module.Run(sync, run_option);
        if (status != tcim::Status::OK) {
            printf("%s[ERROR] Failed to run, and error code: %d%s\n", COLOR_RED,
                   status, COLOR_RESET);
            continue;
        }
    }
    status = module.Sync();
    if (status != tcim::Status::OK) {
        printf("%s[ERROR] Failed to sync, and error code: %d%s\n", COLOR_RED,
               status, COLOR_RESET);
        return;
    }
    stats.end_timestamp =
        duration_cast<microseconds>(system_clock::now().time_since_epoch())
            .count();
}

static std::string get_backend_name() {
    if (HOUMO_TARGET == "xh1") {
        return "Xh1HdiBackend";
    } else if (HOUMO_TARGET == "xh2") {
        return "Xh2HalBackend";
    } else {
        throw std::runtime_error("Invalid target");
    }
}

static float run_model(std::string model_name, int32_t device_id,
                       int32_t core_count, int32_t samples, int32_t rounds) {
    auto device_manager =
        tcim::DevManager::Create(device_id, get_backend_name());
    auto weight_manager =
        tcim::Module::WeightManager::CreateWeightManager(device_manager);
    int32_t warmup = 1;
    int32_t thread_num = core_count * 2;
    Barrier barrier(thread_num);
    int32_t repeat = samples / thread_num;
    int32_t mod = samples % thread_num;  // remainder
    std::vector<std::thread> threads;
    std::vector<statsInfo_t> statsInfos;
    statsInfos.resize(thread_num);
    for (int32_t i = 0; i < thread_num; ++i) {
        statsInfos[i].idx = i;
        statsInfos[i].repeat = repeat;
        if (i < mod)
            statsInfos[i].repeat += 1;
        threads.emplace_back(Infer, i, model_name, warmup, rounds,
                             std::ref(statsInfos[i]), std::ref(weight_manager),
                             std::ref(barrier));
    }
    for (auto &t : threads)
        t.join();
    int64_t min_timstamp = std::numeric_limits<int64_t>::max();
    int64_t max_timstamp = std::numeric_limits<int64_t>::min();
    for (int32_t i = 0; i < statsInfos.size(); ++i) {
        if (statsInfos[i].start_timestamp < min_timstamp) {
            min_timstamp = statsInfos[i].start_timestamp;
        }
        if (statsInfos[i].end_timestamp > max_timstamp) {
            max_timstamp = statsInfos[i].end_timestamp;
        }
    }
    float total_time = (max_timstamp - min_timstamp) / 1000.0f;  // ms
    return total_time;
}

typedef enum PCIE_TRANSFER_TYPE {
    H2H = 0,  // 0x00
    H2D = 1,  // 0x01
    D2H = 2,  // 0x10
    D2D = 3   // 0x11
} pcieTransferType_t;

static float pcie_theoretical_bw(const std::string &s) {
    std::regex r(R"(([\d\.]+)\s*GT/s.*?(\d+)\s*(lane|x))", std::regex::icase);

    std::smatch m;
    if (!std::regex_search(s, m, r)) {
        return -1.0f;  // parse failed
    }

    float gts = std::stof(m[1].str());
    int lanes = std::stoi(m[2].str());

    // choose encoding efficiency automatically
    float eff = 1.0f;
    if (gts <= 5.0f) {
        eff = 0.8f;  // PCIe 1.x / 2.x
    } else if (gts <= 32.0f) {
        eff = 128.0f / 130.0f;  // PCIe 3.x / 4.x / 5.x
    } else {
        eff = 242.0f / 256.0f;  // PCIe 6.0 PAM4 FLIT
    }

    // PCIe uses 1 bit per transfer (before encoding)
    float bw_GBps = gts * 1e9 * eff * lanes / 8.0f / 1e9;

    return bw_GBps;
}

static double pcie_transfer_bandwidth(int32_t warmup, int32_t loops,
                                      int32_t device_id, int32_t block_size,
                                      int32_t thread_num,
                                      pcieTransferType_t type) {

    std::vector<tcim::Buffer> from;
    std::vector<tcim::Buffer> to;
    double total_time = 0;
    for (int i = 0; i < thread_num; ++i) {
        auto src = !(type & 0b10)
                       ? tcim::Buffer::CreateHostBuffer(block_size)
                       : tcim::Buffer::CreateDeviceBuffer(block_size, device_id,
                                                          get_backend_name());
        auto dst = !(type & 0b01)
                       ? tcim::Buffer::CreateHostBuffer(block_size)
                       : tcim::Buffer::CreateDeviceBuffer(block_size, device_id,
                                                          get_backend_name());
        if (src.GetInitStatus() != tcim::Status::OK) {
            printf("%s[ERROR] Failed to create buffer, and error code: %d%s\n",
                   COLOR_RED, src.GetInitStatus(), COLOR_RESET);
            return -1;
        }
        if (dst.GetInitStatus() != tcim::Status::OK) {
            printf("%s[ERROR] Failed to create buffer, and error code: %d%s\n",
                   COLOR_RED, dst.GetInitStatus(), COLOR_RESET);
            return -1;
        }
        from.emplace_back(src);
        to.emplace_back(dst);
    }
    // printf("[INFO] Warmup...\n");
    for (int i = 0; i < thread_num; ++i) {
        for (int j = 0; j < warmup; ++j) {
            if (from[i].CopyTo(to[i]) != tcim::Status::OK) {
                printf("%s[ERROR] Failed to copy buffer during warmup, and "
                       "error code: %d%s\n",
                       COLOR_RED, from[i].GetInitStatus(), COLOR_RESET);
                return -1;
            }
        }
    }

    if (thread_num == 1) {
        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < loops; ++i) {
            from[0].CopyTo(to[0]);
        }
        auto t1 = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(t1 - t0).count();
        total_time = double(duration) / 1e6 / loops;  // s
    } else {
        std::vector<std::thread> threads;
        std::vector<double> durations;
        durations.resize(thread_num);
        Barrier barrier(thread_num);
        for (int t = 0; t < thread_num; ++t) {
            threads.push_back(std::thread([&, t]() {
                barrier.wait();
                auto t0 = high_resolution_clock::now();
                for (int i = 0; i < loops; ++i) {
                    if (from[t].CopyTo(to[t]) != tcim::Status::OK) {
                        printf("%s[ERROR] Failed to copy buffer during warmup, "
                               "and error code: %d%s\n",
                               COLOR_RED, from[i].GetInitStatus(), COLOR_RESET);
                        continue;
                    }
                }
                auto t1 = high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(t1 - t0).count();
                // printf("Thread %d transfer avg time: %.2f ms\n", t,
                // double(duration) / 1e3 / loops);
                durations[t] = double(duration) / 1e6;  // s
            }));
        }
        for (auto &t : threads) {
            t.join();
        }
        // Find the maximum duration
        total_time = *std::max_element(durations.begin(), durations.end());
    }
    return (double(block_size) * loops * thread_num / 1024 / 1024 / 1024) /
           total_time;  // GB/s
}

int main(int argc, char **argv) {
    argparse::ArgumentParser parser(
        "hm-version", "Show device and SDK/Libs version information");
    parser.add_argument("-r", "--repeat")
        .default_value(128)
        .help("Number of repeat iterations")
        .scan<'i', int32_t>();
    parser.add_argument("--verbose")
        .default_value(false)
        .implicit_value(true)
        .help("Print detailed device information");
    parser.add_argument("--target", "-t")
        .default_value(std::string("xh2"))
        .choices("xh2")
        .help("Target device");
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        return -1;
    }

    auto verbose = parser.get<bool>("verbose");
    auto repeat = parser.get<int32_t>("repeat");
    HOUMO_TARGET = parser.get<std::string>("target");

    struct hm_device_info info = {0};
    uint32_t ret = hm_sys_get_device_info(&info);
    uint32_t dev_count = info.num_devices ? info.num_devices : ret;
    if (dev_count == 0) {
        printf("Not found device\n");
        return 0;
    }
    printf("Found %u device(s)\n", dev_count);

    hmVerInfo_t hmVerInfo;
    get_version_info(hmVerInfo);

    std::vector<hmDeviceInfo_t> devices;
    devices.resize(dev_count);
    for (uint32_t i = 0; i < dev_count; ++i) {
        int32_t device_id = (int)info.device_ids[i];
        devices[i].device_id = device_id;
        get_device_info(devices[i]);
    }

    bool version_ok = true;
    if (hmVerInfo.driver_version == "unavailable" ||
        hmVerInfo.sdk_version == "unavailable" ||
        hmVerInfo.runtime_version == "unavailable") {
        version_ok = false;
    }
    if (hmVerInfo.sdk_version != hmVerInfo.driver_version ||
        hmVerInfo.sdk_version != hmVerInfo.runtime_version ||
        hmVerInfo.driver_version != hmVerInfo.runtime_version) {
        version_ok = false;
    }

    // Performance testing
    printf("===== Computing Power Test =====\n");
    std::string model_name = HOUMO_TARGET + "_compute";
    float elapsed_time = 0;
    int32_t rounds = 1;
    for (size_t i = 0; i < devices.size(); ++i) {
        auto &d = devices[i];
        elapsed_time = run_model(model_name, d.device_id, d.core_count, repeat,
                                 rounds);  // ms
        // Calculate TOPS (Trillions of Operations Per Second)
        auto *model = get_model(model_name);
        d.computing_power = model->num_ops * repeat * 1000.0f / elapsed_time;
        printf("Device %d Computing Power: %.2f TOPS\n", d.device_id,
               d.computing_power);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    printf("=============================\n\n");

    printf("===== DDR Bandwidth Test =====\n");
    // DDR read/write bandwidth test
    rounds = 100;
    for (size_t i = 0; i < devices.size(); ++i) {
        auto &d = devices[i];
        repeat = d.core_count * 4;
        model_name = HOUMO_TARGET + "_bandwidth_read";
        elapsed_time = run_model(model_name, d.device_id, d.core_count, repeat,
                                 rounds);  // ms
        auto *r_model = get_model(model_name);
        d.ddr_bandwidth_read =
            r_model->read_data_size * repeat * rounds * 1000.0f /
            (elapsed_time * 1024.0f * 1024.0f * 1024.0f);  // GB/s
        printf("Device %d DDR Read Bandwidth: %.2f GB/s\n", d.device_id,
               d.ddr_bandwidth_read);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        model_name = HOUMO_TARGET + "_bandwidth_write";
        elapsed_time = run_model(model_name, d.device_id, d.core_count, repeat,
                                 rounds);  // ms
        auto *w_model = get_model(model_name);
        d.ddr_bandwidth_write =
            w_model->write_data_size * repeat * rounds * 1000.0f /
            (elapsed_time * 1024.0f * 1024.0f * 1024.0f);  // GB/s
        printf("Device %d DDR Write Bandwidth: %.2f GB/s\n", d.device_id,
               d.ddr_bandwidth_write);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    printf("=============================\n\n");

    printf("===== PCIE Bandwidth Test =====\n");
    // PCIe transfer bandwidth test, block size 1MB, 8 threads
    int32_t block_size = 1 * 1024 * 1024;  // Bytes
    int32_t thread_num = 8;
    int32_t warmup = 1;
    int32_t loops = 100;
    float bandwidth = 0;
    for (size_t i = 0; i < devices.size(); ++i) {
        auto &d = devices[i];
        d.pcie_bandwidth_H2D =
            pcie_transfer_bandwidth(warmup, loops, d.device_id, block_size,
                                    thread_num, PCIE_TRANSFER_TYPE::H2D);
        printf("Device %d PCIE H2D Bandwidth: %.2f GB/s\n", d.device_id,
               d.pcie_bandwidth_H2D);
        d.pcie_bandwidth_D2H =
            pcie_transfer_bandwidth(warmup, loops, d.device_id, block_size,
                                    thread_num, PCIE_TRANSFER_TYPE::D2H);
        printf("Device %d PCIE D2H Bandwidth: %.2f GB/s\n", d.device_id,
               d.pcie_bandwidth_D2H);
    }
    // D2D
    printf("=============================\n\n");

    if (verbose) {
        // Print version details
        printf("===== Detail version info  =====\n\n");

        printf("Driver Info:\n");
        printf("  SDK Build Time       : %s\n",
               hmVerInfo.sdk_build_time.c_str());
        printf("  SDK Version          : %s\n", hmVerInfo.sdk_version.c_str());
        printf("  Device Driver Version: %s\n\n",
               hmVerInfo.driver_version.c_str());

        printf("TCIM Runtime Info:\n");
        printf("  Runtime Build Time   : %s\n",
               hmVerInfo.runtime_build_time.c_str());
        printf("  Runtime Version      : %s\n",
               hmVerInfo.runtime_version.c_str());

        printf("\n===========================================\n");

        print_devices_aligned(devices);
    }

    CheckReport report;
    report.add("Driver Version",
               hmVerInfo.driver_version != "unavailable" ? CheckStatus::PASS
                                                         : CheckStatus::FAIL,
               hmVerInfo.driver_version);
    report.add("SDK Version",
               hmVerInfo.driver_version != "unavailable" ? CheckStatus::PASS
                                                         : CheckStatus::FAIL,
               hmVerInfo.sdk_version);
    report.add("Runtime Version",
               hmVerInfo.runtime_version != "unavailable" ? CheckStatus::PASS
                                                          : CheckStatus::FAIL,
               hmVerInfo.runtime_version);
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto &d = devices[i];
        report.add("Device" + std::to_string(d.device_id) + " Firmware Version",
                   d.firmware_version != "unavailable" ? CheckStatus::PASS
                                                       : CheckStatus::FAIL,
                   d.firmware_version);
    }
    report.add("Version Consistency",
               version_ok ? CheckStatus::PASS : CheckStatus::WARN,
               version_ok ? "Match" : "Mismatch",
               "Maybe cause unknown problems.");
    char buf[32];
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto &d = devices[i];
        CheckStatus status = CheckStatus::PASS;
        memset(buf, 0, sizeof(buf));
        sprintf(buf, "%.2f MHz", d.max_ipu_frequency);
        std::string desc = "The maximum frequency of " + std::string(buf) +
                           " MHz is not reached";
        if (d.avg_ipu_frequency == 0) {
            status = CheckStatus::FAIL;
        } else if (d.avg_ipu_frequency != d.max_ipu_frequency) {
            status = CheckStatus::WARN;
        } else {
            desc = "";
        }
        memset(buf, 0, sizeof(buf));
        sprintf(buf, "%.2f MHz", d.avg_ipu_frequency);
        report.add("Device" + std::to_string(d.device_id) + " Cur IPU Freq",
                   status, std::string(buf), desc);
    }
    // Check if performance is below 75% of maximum performance
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto &d = devices[i];
        CheckStatus status = CheckStatus::PASS;
        memset(buf, 0, sizeof(buf));
        sprintf(buf, "%.2f TOPS", d.max_computing_power);
        std::string desc = "The computing power should not be less than 75 % "
                           "of the maximum computing power " +
                           std::string(buf);
        if (d.computing_power == 0) {
            status = CheckStatus::FAIL;
        } else if (d.computing_power < 0.75 * d.max_computing_power) {
            status = CheckStatus::WARN;
        } else {
            desc = "";
        }
        memset(buf, 0, sizeof(buf));
        sprintf(buf, "%.2f TOPS", d.computing_power);
        report.add("Device" + std::to_string(d.device_id) + " Measured Compute",
                   status, std::string(buf), desc);
    }
    // Check if DDR read and write bandwidths are within ±5% of 125GB and 120GB
    // respectively
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto &d = devices[i];
        CheckStatus status = CheckStatus::PASS;
        std::string desc =
            "The DDR read bandwidth should not be less than 125 GB/s * 0.95";
        if (d.ddr_bandwidth_read == 0) {
            status = CheckStatus::FAIL;
        } else if (d.ddr_bandwidth_read < 0.95 * 125) {
            status = CheckStatus::WARN;
        } else {
            desc = "";
        }
        memset(buf, 0, sizeof(buf));
        sprintf(buf, "%.2f GB/s", d.ddr_bandwidth_read);
        report.add("Device" + std::to_string(d.device_id) +
                       " Measured DDR Read",
                   status, std::string(buf), desc);
    }
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto &d = devices[i];
        CheckStatus status = CheckStatus::PASS;
        std::string desc =
            "The DDR write bandwidth should not be less than 120 GB/s * 0.95";
        if (d.ddr_bandwidth_write == 0) {
            status = CheckStatus::FAIL;
        } else if (d.ddr_bandwidth_write < 0.95 * 120) {
            status = CheckStatus::WARN;
        } else {
            desc = "";
        }
        memset(buf, 0, sizeof(buf));
        sprintf(buf, "%.2f GB/s", d.ddr_bandwidth_write);
        report.add("Device" + std::to_string(d.device_id) +
                       " Measured DDR Write",
                   status, std::string(buf), desc);
    }
    float threshold = 0.8;
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto &d = devices[i];
        CheckStatus status = CheckStatus::PASS;
        float pcie_theoretical_transfer_bw =
            pcie_theoretical_bw(d.pcie_bandwidth);
        memset(buf, 0, sizeof(buf));
        sprintf(buf, "%.2f GB/s * %.2f", pcie_theoretical_transfer_bw,
                threshold);
        std::string desc =
            "The PCIE transfer bandwidth should not be less than " +
            std::string(buf);
        if (d.pcie_bandwidth_H2D == 0) {
            status = CheckStatus::FAIL;
        } else if (d.pcie_bandwidth_H2D <
                   threshold * pcie_theoretical_transfer_bw) {
            status = CheckStatus::WARN;
        } else {
            desc = "";
        }
        memset(buf, 0, sizeof(buf));
        sprintf(buf, "%.2f GB/s", d.pcie_bandwidth_H2D);
        report.add("Device" + std::to_string(d.device_id) +
                       " Measured PCIe H2D",
                   status, std::string(buf), desc);

        // D2H
        memset(buf, 0, sizeof(buf));
        sprintf(buf, "%.2f GB/s * %.2f", pcie_theoretical_transfer_bw,
                threshold);
        desc = "The PCIE transfer bandwidth should not be less than " +
               std::string(buf);
        status = CheckStatus::PASS;
        if (d.pcie_bandwidth_D2H == 0) {
            status = CheckStatus::FAIL;
        } else if (d.pcie_bandwidth_D2H <
                   threshold * pcie_theoretical_transfer_bw) {
            status = CheckStatus::WARN;
        } else {
            desc = "";
        }
        memset(buf, 0, sizeof(buf));
        sprintf(buf, "%.2f GB/s", d.pcie_bandwidth_D2H);
        report.add("Device" + std::to_string(d.device_id) +
                       " Measured PCIe D2H",
                   status, std::string(buf), desc);
    }
    report.print();

    auto s = report.summary();

    printf("\n===== Check Summary =====\n");
    printf("  %sPASS%s : %d\n", COLOR_GREEN, COLOR_RESET, s.pass);
    printf("  %sWARN%s : %d\n", COLOR_YELLOW, COLOR_RESET, s.warn);
    printf("  %sFAIL%s : %d\n", COLOR_RED, COLOR_RESET, s.fail);
    printf("=========================\n\n");

    // if (!report.all_pass()) {
    //     printf("%s❌%s Some checks failed.\n");
    // } else {
    //     printf("%s✔%s All checks passed.\n", COLOR_GREEN, COLOR_RESET);
    // }
    return 0;
}
