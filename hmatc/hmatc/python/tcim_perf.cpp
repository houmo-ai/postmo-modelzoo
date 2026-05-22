/* Copyright 2025 HOUMO AI
 *
 * File: tcim_perf.cpp
 * Description:
 *   Performance test for TCIM
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
#include "nlohmann/json.hpp"
#include "tcim/tcim_runtime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>
#include <sstream>
#include <string>
#include <thread>

using namespace std::chrono;
namespace py = pybind11;

// ============================================================================
// Constants
// ============================================================================
constexpr float kScaleMax = 16.0f;
constexpr float kScaleMin = 1.0f / 32.0f;
constexpr int32_t kDefaultCoreNum = 2;

// ============================================================================
// Color codes for terminal output
// ============================================================================
constexpr char COLOR_RED[] = "\x1b[91;20m";
constexpr char COLOR_GREEN[] = "\x1b[92;20m";
constexpr char COLOR_YELLOW[] = "\x1b[93;20m";
constexpr char COLOR_BLUE[] = "\x1b[94;20m";
constexpr char COLOR_MAGENTA[] = "\x1b[95;20m";
constexpr char COLOR_CYAN[] = "\x1b[96;20m";
constexpr char COLOR_RESET[] = "\x1b[0m";

// ============================================================================
// Data structures
// ============================================================================

/**
 * @brief Performance metrics structure
 * Contains latency and throughput metrics for performance analysis
 */
struct PerfInfo {
    float input_avg_latency = 0.0f;     // Average input latency
    float input_max_latency = 0.0f;     // Maximum input latency
    float input_min_latency = 0.0f;     // Minimum input latency
    float input_tp99_latency = 0.0f;    // 99th percentile input latency
    float input_tp999_latency = 0.0f;   // 99.9th percentile input latency
    float infer_avg_latency = 0.0f;     // Average inference latency
    float infer_max_latency = 0.0f;     // Maximum inference latency
    float infer_min_latency = 0.0f;     // Minimum inference latency
    float infer_tp99_latency = 0.0f;    // 99th percentile inference latency
    float infer_tp999_latency = 0.0f;   // 99.9th percentile inference latency
    float output_avg_latency = 0.0f;    // Average output latency
    float output_max_latency = 0.0f;    // Maximum output latency
    float output_min_latency = 0.0f;    // Minimum output latency
    float output_tp99_latency = 0.0f;   // 99th percentile output latency
    float output_tp999_latency = 0.0f;  // 99.9th percentile output latency
    float e2e_avg_latency = 0.0f;       // Average end-to-end latency
    float e2e_max_latency = 0.0f;       // Maximum end-to-end latency
    float e2e_min_latency = 0.0f;       // Minimum end-to-end latency
    float e2e_tp99_latency = 0.0f;      // 99th percentile end-to-end latency
    float e2e_tp999_latency = 0.0f;     // 99.9th percentile end-to-end latency
    float avg_cost = 0.0f;              // Average cost per inference
    float qps = 0.0f;                   // Queries Per Second
};
using PerfInfo_t = PerfInfo;

/**
 * @brief Statistics tracking structure for each thread
 * Contains timing information for input, inference, and output operations
 */
struct StatsInfo {
    int32_t idx = -1;                     // Thread index
    int32_t repeat = 0;                   // Number of repeats
    int64_t start_timestamp = 0;          // Start timestamp
    int64_t end_timestamp = 0;            // End timestamp
    std::vector<float> set_input_times;   // Input setting times
    std::vector<float> infer_times;       // Inference times
    std::vector<float> get_output_times;  // Output getting times
    float total_set_input_time = 0.0f;    // Total input setting time
    float total_infer_time = 0.0f;        // Total inference time
    float total_get_output_time = 0.0f;   // Total output getting time
};
using StatsInfo_t = StatsInfo;

// ============================================================================
// Helper functions
// ============================================================================

/**
 * @brief Convert shape vector to string representation
 * @param shape Input shape vector
 * @return String representation of the shape
 */
static std::string ShapeToString(const std::vector<int64_t> &shape) {
    if (shape.empty()) {
        return "";
    }
    std::ostringstream result;
    result << shape[0];
    for (size_t j = 1; j < shape.size(); ++j) {
        result << ", " << shape[j];
    }
    return result.str();
}

/**
 * @brief Convert TCIM data type to string representation
 * @param dtype TCIM data type
 * @return String representation of the data type
 */
static std::string DataTypeToString(tcim::DataType dtype) {
    switch (dtype) {
    case tcim::INT8:
        return "INT8";
    case tcim::UINT8:
        return "UINT8";
    case tcim::INT16:
        return "INT16";
    case tcim::UINT16:
        return "UINT16";
    case tcim::INT32:
        return "INT32";
    case tcim::UINT32:
        return "UINT32";
    case tcim::FLOAT16:
        return "FLOAT16";
    case tcim::FLOAT32:
        return "FLOAT32";
    default:
        return "UNKNOWN";
    }
}

/**
 * @brief Convert TCIM data format to string representation
 * @param format TCIM data format
 * @return String representation of the data format
 */
static std::string FmtToString(tcim::DataFmt format) {
    switch (format) {
    case tcim::YUV420SP:
        return "YUV420SP";
    case tcim::YUV422SP:
        return "YUV422SP";
    case tcim::YUV444SP:
        return "YUV444SP";
    case tcim::ND:
        return "ND";
    default:
        return "Unknown";
    }
}

/**
 * @brief Calculate percentile value from a vector of latencies
 * @param latencies Vector of latency values
 * @param percentile Target percentile (e.g., 0.99 for 99th percentile)
 * @return Calculated percentile value
 */
static float calculate_percentile(const std::vector<float> &latencies, float percentile) {
    if (latencies.empty()) {
        return 0.0f;
    }
    if (latencies.size() == 1) {
        return latencies[0];
    }

    std::vector<float> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());

    int index = static_cast<int>(std::ceil(percentile * (sorted.size() - 1)));
    index = std::clamp(index, 0, static_cast<int>(sorted.size() - 1));
    return sorted[index];
}

/**
 * @brief Print statistics information and calculate performance metrics
 * @param statsInfos Vector of statistics information from all threads
 * @param samples Number of samples
 * @param rounds Number of rounds per sample
 * @param perfInfo Output performance information structure
 */
static void PrintStatsInfo(const std::vector<StatsInfo_t> &statsInfos,
                           int32_t samples, int32_t rounds, PerfInfo_t &perfInfo) {
    int64_t min_timestamp = std::numeric_limits<int64_t>::max();
    int64_t max_timestamp = std::numeric_limits<int64_t>::min();
    float infer_avg_time = 0.0f;
    float infer_max_time = std::numeric_limits<float>::min();
    float infer_min_time = std::numeric_limits<float>::max();
    float set_input_avg_time = 0.0f;
    float set_input_max_time = std::numeric_limits<float>::min();
    float set_input_min_time = std::numeric_limits<float>::max();
    float get_output_avg_time = 0.0f;
    float get_output_max_time = std::numeric_limits<float>::min();
    float get_output_min_time = std::numeric_limits<float>::max();
    float end2end_avg_time = 0.0f;
    float end2end_max_time = std::numeric_limits<float>::min();
    float end2end_min_time = std::numeric_limits<float>::max();

    std::vector<float> total_set_input_times;
    std::vector<float> total_infer_times;
    std::vector<float> total_get_output_times;
    std::vector<float> total_end2end_times;

    for (const auto &stats : statsInfos) {
        min_timestamp = std::min(min_timestamp, stats.start_timestamp);
        max_timestamp = std::max(max_timestamp, stats.end_timestamp);
        set_input_avg_time += stats.total_set_input_time;
        infer_avg_time += stats.total_infer_time;
        get_output_avg_time += stats.total_get_output_time;

        for (int32_t j = 0; j < stats.repeat; ++j) {
            const float set_input_time = stats.set_input_times[j];
            const float get_output_time = stats.get_output_times[j];
            const float infer_time = stats.infer_times[j] / rounds;
            const float end2end_time = set_input_time + infer_time + get_output_time;
            end2end_avg_time += end2end_time;

            set_input_max_time = std::max(set_input_max_time, set_input_time);
            set_input_min_time = std::min(set_input_min_time, set_input_time);
            infer_max_time = std::max(infer_max_time, infer_time);
            infer_min_time = std::min(infer_min_time, infer_time);
            get_output_max_time = std::max(get_output_max_time, get_output_time);
            get_output_min_time = std::min(get_output_min_time, get_output_time);
            end2end_max_time = std::max(end2end_max_time, end2end_time);
            end2end_min_time = std::min(end2end_min_time, end2end_time);

            total_set_input_times.emplace_back(set_input_time);
            total_infer_times.emplace_back(infer_time);
            total_get_output_times.emplace_back(get_output_time);
            total_end2end_times.emplace_back(end2end_time);
        }
    }

    const int64_t total_repeat = samples * rounds;
    const float total_time = (max_timestamp - min_timestamp) / 1000.0f;
    set_input_avg_time /= samples;
    infer_avg_time /= total_repeat;
    get_output_avg_time /= samples;
    end2end_avg_time /= samples;

    const float set_input_time_tp99 = calculate_percentile(total_set_input_times, 0.99f);
    const float set_input_time_tp999 = calculate_percentile(total_set_input_times, 0.999f);
    const float infer_time_tp99 = calculate_percentile(total_infer_times, 0.99f);
    const float infer_time_tp999 = calculate_percentile(total_infer_times, 0.999f);
    const float get_output_time_tp99 = calculate_percentile(total_get_output_times, 0.99f);
    const float get_output_time_tp999 = calculate_percentile(total_get_output_times, 0.999f);
    const float end2end_time_tp99 = calculate_percentile(total_end2end_times, 0.99f);
    const float end2end_time_tp999 = calculate_percentile(total_end2end_times, 0.999f);

    const float avg_cost = total_time / total_repeat;
    const float QPS = total_repeat / total_time * 1000;

    printf("%s[Latency] Inference  avg: %7.3f ms, max: %7.3f ms, min: %7.3f ms, tp99: %7.3f ms, tp999: %7.3f ms%s\n",
           COLOR_CYAN, infer_avg_time, infer_max_time, infer_min_time, infer_time_tp99, infer_time_tp999, COLOR_RESET);
    printf("%s[Latency] Input      avg: %7.3f ms, max: %7.3f ms, min: %7.3f ms, tp99: %7.3f ms, tp999: %7.3f ms%s\n",
           COLOR_CYAN, set_input_avg_time, set_input_max_time, set_input_min_time, set_input_time_tp99, set_input_time_tp999, COLOR_RESET);
    printf("%s[Latency] Output     avg: %7.3f ms, max: %7.3f ms, min: %7.3f ms, tp99: %7.3f ms, tp999: %7.3f ms%s\n",
           COLOR_CYAN, get_output_avg_time, get_output_max_time, get_output_min_time, get_output_time_tp99, get_output_time_tp999, COLOR_RESET);
    printf("%s[Latency] End2end    avg: %7.3f ms, max: %7.3f ms, min: %7.3f ms, tp99: %7.3f ms, tp999: %7.3f ms%s\n",
           COLOR_CYAN, end2end_avg_time, end2end_max_time, end2end_min_time, end2end_time_tp99, end2end_time_tp999, COLOR_RESET);
    printf("%s[Throughput] total: %.3f ms, avg: %.3f ms, repeat: %d, rounds: %d%s\n",
           COLOR_MAGENTA, total_time, avg_cost, samples, rounds, COLOR_RESET);
    printf("%s[Throughput] qps: %.3f %s\n", COLOR_MAGENTA, QPS, COLOR_RESET);

    perfInfo.input_avg_latency = set_input_avg_time;
    perfInfo.input_max_latency = set_input_max_time;
    perfInfo.input_min_latency = set_input_min_time;
    perfInfo.input_tp99_latency = set_input_time_tp99;
    perfInfo.input_tp999_latency = set_input_time_tp999;
    perfInfo.infer_avg_latency = infer_avg_time;
    perfInfo.infer_max_latency = infer_max_time;
    perfInfo.infer_min_latency = infer_min_time;
    perfInfo.infer_tp99_latency = infer_time_tp99;
    perfInfo.infer_tp999_latency = infer_time_tp999;
    perfInfo.output_avg_latency = get_output_avg_time;
    perfInfo.output_max_latency = get_output_max_time;
    perfInfo.output_min_latency = get_output_min_time;
    perfInfo.output_tp99_latency = get_output_time_tp99;
    perfInfo.output_tp999_latency = get_output_time_tp999;
    perfInfo.e2e_avg_latency = end2end_avg_time;
    perfInfo.e2e_max_latency = end2end_max_time;
    perfInfo.e2e_min_latency = end2end_min_time;
    perfInfo.e2e_tp99_latency = end2end_time_tp99;
    perfInfo.e2e_tp999_latency = end2end_time_tp999;
    perfInfo.avg_cost = avg_cost;
    perfInfo.qps = QPS;
}

/**
 * @brief Check if the provided device IDs are valid
 * @param devices Vector of device IDs to check
 * @return 0 if all devices are valid, -1 otherwise
 */
static int32_t CheckDevices(const std::vector<int32_t> &devices) {
    std::ostringstream devices_str;
    devices_str << "[";

    for (size_t i = 0; i < devices.size(); ++i) {
        const int32_t device_id = devices[i];
        if (device_id >= tcim::GetDeviceNum()) {
            printf("%s[ERROR] Invalid device id: %d %s\n", COLOR_RED, device_id, COLOR_RESET);
            return -1;
        }
        // Check for duplicate device IDs
        for (size_t j = i + 1; j < devices.size(); ++j) {
            if (device_id == devices[j]) {
                printf("%s[ERROR] Duplicate device id: %d %s\n", COLOR_RED, device_id, COLOR_RESET);
                return -1;
            }
        }
        devices_str << device_id;
        if (i != devices.size() - 1) {
            devices_str << ", ";
        }
    }
    devices_str << "]";
    printf("[INFO] Set Devices: %s\n", devices_str.str().c_str());
    return 0;
}

/**
 * @brief Print input and output information of the module
 * @param module TCIM module reference
 */
static void PrintInputOutputInfo(const tcim::Module &module) {
    const int32_t core_num = module.GetCoreNum();
    printf("[INFO] CoreNum: %d\n", core_num);

    const int32_t input_num = module.GetInputNum();
    printf("[INFO] InputNum: %d\n", input_num);
    for (int32_t i = 0; i < input_num; ++i) {
        const std::string name = module.GetInputName(i);
        const auto info = module.GetInputInfo(name);
        printf("[INFO] Input[%d] name: %s, shape: [%s], dtype: %s, fmt: %s, memSize: %d\n",
               i, name.c_str(), ShapeToString(info.Shape()).c_str(),
               DataTypeToString(info.DataType()).c_str(),
               FmtToString(info.Format()).c_str(), info.MemSize());
    }

    const int32_t output_num = module.GetOutputNum();
    printf("[INFO] OutputNum: %d\n", output_num);
    for (int32_t i = 0; i < output_num; ++i) {
        const std::string name = module.GetOutputName(i);
        const auto info = module.GetOutputInfo(name);
        printf("[INFO] Output[%d] name: %s, shape: [%s], dtype: %s, fmt: %s, memSize: %d\n",
               i, name.c_str(), ShapeToString(info.Shape()).c_str(),
               DataTypeToString(info.DataType()).c_str(),
               FmtToString(info.Format()).c_str(), info.MemSize());
    }
}

/**
 * @brief Set random input data for the module
 * @param module TCIM module reference
 * @param inputs Map to store input tensors
 * @return 0 on success, -1 on failure
 */
static int32_t SetInputData(tcim::Module &module, std::map<std::string, tcim::Tensor> &inputs) {
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int32_t i = 0; i < module.GetInputNum(); ++i) {
        const std::string name = module.GetInputName(i);
        const auto info = module.GetInputInfo(name);
        auto tensor = tcim::Tensor::CreateHostTensor(info.AsContiguous());
        const std::string kResizerCropPrefix = "resizer_crop_";

        if (tcim::DataType::FLOAT16 == info.DataType()) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            auto fp32_tensor = tcim::Tensor::CreateHostTensor(
                info.AsContiguous().AsType(tcim::DataType::FLOAT32));
            auto *data = reinterpret_cast<float *>(fp32_tensor.Data());
            for (int64_t j = 0; j < fp32_tensor.MemSize() / sizeof(float); ++j) {
                data[j] = dist(rng);
            }
            fp32_tensor.CastTo(tensor);
        } else if (tcim::DataType::INT8 == info.DataType()) {
            std::uniform_int_distribution<int32_t> dist(-128, 127);
            auto *data = reinterpret_cast<int8_t *>(tensor.Data());
            for (int64_t j = 0; j < tensor.MemSize(); ++j) {
                data[j] = static_cast<int8_t>(dist(rng));
            }
        } else if (tcim::DataType::INT32 == info.DataType() &&
                   info.Format() == tcim::DataFmt::ND &&
                   name.length() > kResizerCropPrefix.length() &&
                   name.substr(0, kResizerCropPrefix.length()) == kResizerCropPrefix) {
            const std::string custom_msg_str = module.GetCustomMsg();
            if (custom_msg_str.empty()) {
                printf("%s[ERROR] HM Model build without custom msg, not support yet. "
                       "Please build with Hmatc, and retry.%s\n",
                       COLOR_RED, COLOR_RESET);
                return -1;
            }

            const std::string image_name = name.substr(kResizerCropPrefix.length());
            const auto image_info = module.GetInputInfo(image_name);
            const auto custom_msg = nlohmann::json::parse(custom_msg_str);
            const auto &model_input_shape = custom_msg[image_name]["shape"];
            const auto &resizer_input_shape = image_info.Shape();
            const auto &dyn_info_shape = info.Shape();

            assert(model_input_shape.size() == 4);
            assert(resizer_input_shape.size() == 4);

            const auto MODEL_INPUT_H = model_input_shape[2];
            const auto MODEL_INPUT_W = model_input_shape[3];
            const auto RESIZER_INPUT_H = resizer_input_shape[2];
            const auto RESIZER_INPUT_W = resizer_input_shape[3];
            auto RESIZER_CROP_H = RESIZER_INPUT_H;
            auto RESIZER_CROP_W = RESIZER_INPUT_W;
            const float sh = static_cast<float>(MODEL_INPUT_H) / static_cast<float>(RESIZER_INPUT_H);
            const float sw = static_cast<float>(MODEL_INPUT_W) / static_cast<float>(RESIZER_INPUT_W);

            if (sh > kScaleMax || sh < kScaleMin) {
                RESIZER_CROP_H = static_cast<int32_t>(
                                     RESIZER_CROP_H * std::max(kScaleMin, std::min(kScaleMax, sw))) &
                                 ~1;
            }
            if (sw > kScaleMax || sw < kScaleMin) {
                RESIZER_CROP_W = static_cast<int32_t>(
                                     RESIZER_CROP_W * std::max(kScaleMin, std::min(kScaleMax, sh))) &
                                 ~1;
            }

            assert(dyn_info_shape.size() == 2 || dyn_info_shape.size() == 1);
            int32_t batch = 1;
            int32_t step = dyn_info_shape[0];
            if (dyn_info_shape.size() > 1) {
                batch = dyn_info_shape[0];
                step = dyn_info_shape[1];
            }

            auto *data = reinterpret_cast<int32_t *>(tensor.Data());
            for (int32_t k = 0; k < batch; ++k) {
                data[k * step + 0] = 0;
                data[k * step + 1] = 0;
                data[k * step + 2] = RESIZER_CROP_H;
                data[k * step + 3] = RESIZER_CROP_W;
                if (step == 10) {
                    data[k * step + 4] = MODEL_INPUT_H;
                    data[k * step + 5] = MODEL_INPUT_W;
                    data[k * step + 6] = 0;
                    data[k * step + 7] = 0;
                    data[k * step + 8] = 0;
                    data[k * step + 9] = 0;
                }
            }
        }
        inputs[name] = tensor;
    }
    return 0;
}

/**
 * @brief Perform inference in a separate thread
 * @param tid Thread ID
 * @param model_path Path to the model file
 * @param warmup Number of warmup iterations
 * @param rounds Number of rounds per sample
 * @param stats Statistics information reference
 * @param stream TCIM stream reference
 * @param option Module option reference
 * @param check_output Whether to check output consistency
 */
static void Infer(int32_t tid, const std::string &model_path, int32_t warmup, int32_t rounds,
                  StatsInfo_t &stats, tcim::Stream &stream, tcim::Module::Option &option,
                  bool check_output) {
    printf("[INFO] Infer %d started, warmup: %d, rounds: %d, repeat: %d\n",
           tid, warmup, rounds, stats.repeat);

    auto module = tcim::Module::LoadFromFile(model_path, option);
    if (module.GetInitStatus() != tcim::Status::OK) {
        printf("%s[ERROR] Failed to load model: %s%s\n", COLOR_RED, model_path.c_str(), COLOR_RESET);
        return;
    }

    std::map<std::string, tcim::Tensor> inputs;
    if (SetInputData(module, inputs) != 0) {
        printf("%s[ERROR] Failed to set input data%s\n", COLOR_RED, COLOR_RESET);
        return;
    }

    std::map<std::string, tcim::Tensor> outputs_ref;
    std::map<std::string, tcim::Tensor> outputs;
    for (int32_t i = 0; i < module.GetOutputNum(); ++i) {
        const std::string name = module.GetOutputName(i);
        const auto info = module.GetOutputInfo(name);
        auto tensor = tcim::Tensor::CreateHostTensor(info);
        outputs[name] = tensor;
        outputs_ref[name] = tensor.Clone();
    }

    const auto status = module.SetStream(stream);
    if (status != tcim::Status::OK) {
        printf("%s[ERROR] Failed to set stream, error code: %d%s\n", COLOR_RED, status, COLOR_RESET);
        return;
    }

    tcim::Module::RunOption run_option;
    constexpr bool kSync = false;

    for (int32_t i = 0; i < warmup + stats.repeat; ++i) {
        if (i == warmup) {
            stats.start_timestamp = duration_cast<microseconds>(
                                        system_clock::now().time_since_epoch())
                                        .count();
        }

        const auto t0 = high_resolution_clock::now();
        for (int32_t k = 0; k < module.GetInputNum(); ++k) {
            const std::string name = module.GetInputName(k);
            module.SetInput(name, inputs[name]);
        }
        const auto t1 = high_resolution_clock::now();

        run_option.Rounds(i < warmup ? 1 : rounds);
        const auto run_status = module.Run(kSync, run_option);
        if (run_status != tcim::Status::OK) {
            printf("%s[ERROR] Failed to run, error code: %d%s\n", COLOR_RED, run_status, COLOR_RESET);
            continue;
        }

        const auto sync_status = module.Sync();
        if (sync_status != tcim::Status::OK) {
            printf("%s[ERROR] Failed to sync, error code: %d%s\n", COLOR_RED, sync_status, COLOR_RESET);
            continue;
        }
        const auto t2 = high_resolution_clock::now();

        for (int32_t k = 0; k < module.GetOutputNum(); ++k) {
            const std::string name = module.GetOutputName(k);
            std::memset(outputs[name].Data(), 0, outputs[name].MemSize());
            module.GetOutput(name, outputs[name]);

            if (check_output) {
                if (i == warmup - 1) {
                    outputs[name].CopyTo(outputs_ref[name]);
                } else if (i >= warmup) {
                    if (std::memcmp(outputs[name].Data(), outputs_ref[name].Data(),
                                    outputs[name].MemSize()) != 0) {
                        printf("%s[ERROR] PID: %d, Iter: %5d, Output %s mismatch%s\n",
                               COLOR_RED, tid, i - warmup, name.c_str(), COLOR_RESET);
                    }
                }
            }
        }
        const auto t3 = high_resolution_clock::now();

        if (i < warmup) {
            continue;
        }
        if (i == warmup + stats.repeat - 1) {
            stats.end_timestamp = duration_cast<microseconds>(
                                      system_clock::now().time_since_epoch())
                                      .count();
        }

        const auto tp0 = duration_cast<microseconds>(t1 - t0);
        const auto tp1 = duration_cast<microseconds>(t2 - t1);
        const auto tp2 = duration_cast<microseconds>(t3 - t2);
        const float set_input_time = tp0.count() / 1000.0f;
        const float infer_time = tp1.count() / 1000.0f;
        const float get_output_time = tp2.count() / 1000.0f;

        stats.set_input_times.emplace_back(set_input_time);
        stats.infer_times.emplace_back(infer_time);
        stats.get_output_times.emplace_back(get_output_time);
        stats.total_set_input_time += set_input_time;
        stats.total_infer_time += infer_time;
        stats.total_get_output_time += get_output_time;
    }
    printf("[INFO] Infer %d done.\n", tid);
}

/**
 * @brief Run performance test with specified parameters
 * @param model_path Path to the model file
 * @param thread_num Number of threads to use
 * @param stream_num Number of streams to use
 * @param warmup Number of warmup iterations
 * @param samples Number of samples to process
 * @param rounds Number of rounds per sample
 * @param devices Vector of device IDs to use
 * @param check_output Whether to check output consistency
 * @param perf_info Output performance information structure
 * @return 0 on success, -1 on failure
 */
int32_t Run(const std::string &model_path, int32_t thread_num, int32_t stream_num,
            int32_t warmup, int32_t samples, int32_t rounds,
            const std::vector<int32_t> &devices, bool check_output, PerfInfo_t &perf_info) {
    printf("[INFO] %s\n", "TCIM Performance Test");
    printf("[INFO] TCIM Runtime Version: %s\n", tcim::GetVersion().c_str());

    // Handle ASIC platform override
    const char *platform = std::getenv("HDPL_PLATFORM");
    if (platform != nullptr && strcmp(platform, "ASIC") != 0) {
        thread_num = 1;
        stream_num = 1;
        samples = 1;
        warmup = 0;
        rounds = 1;
        printf("%s[WARNING] HDPL_PLATFORM is set to %s, parameters adjusted to: "
               "thread_num=1, stream_num=1, samples=1, warmup=0, rounds=1%s\n",
               COLOR_YELLOW, platform, COLOR_RESET);
    }

    // Handle core number environment variable
    const char *houmo_core_num = getenv("HOUMO_CORE_NUM");
    if (houmo_core_num == nullptr) {
        printf("%s[WARNING] HOUMO_CORE_NUM is not set, default set to %d%s\n",
               COLOR_YELLOW, kDefaultCoreNum, COLOR_RESET);
        houmo_core_num = "2";
    }
    if (stream_num <= 0) {
        stream_num = std::stoi(houmo_core_num) * 2;
    }

    printf("[INFO] Model path: %s\n", model_path.c_str());
    printf("[INFO] Warmup: %d\n", warmup);
    printf("[INFO] Rounds: %d\n", rounds);
    printf("[INFO] Repeat: %d\n", samples);
    printf("[INFO] Thread number: %d\n", thread_num);
    printf("[INFO] Stream number: %d\n", stream_num);

    // Validate target environment
    const std::string target = getenv("HOUMO_TARGET") ? getenv("HOUMO_TARGET") : "";
    if (target != "xh2") {
        printf("%s[ERROR] HOUMO_TARGET is invalid: %s (expected: xh2)%s\n", COLOR_RED, target.c_str(), COLOR_RESET);
        return -1;
    }

    const std::string backend_name = "Xh2HalBackend";
    printf("[INFO] Backend: %s\n", backend_name.c_str());

    if (devices.size() > static_cast<size_t>(tcim::GetDeviceNum())) {
        printf("%s[ERROR] Not enough devices%s\n", COLOR_RED, COLOR_RESET);
        return -1;
    }
    if (CheckDevices(devices) != 0) {
        return -1;
    }

    auto dev_manager = tcim::DevManager::Create(devices, backend_name);
    // Create weight manager
    auto weight_manager = tcim::Module::WeightManager::CreateWeightManager(dev_manager);
    auto option = tcim::Module::Option(weight_manager);

    // Load model to print info
    {
        auto module = tcim::Module::LoadFromFile(model_path, option);
        if (module.GetInitStatus() != tcim::Status::OK) {
            printf("%s[ERROR] Failed to load model: %s%s\n", COLOR_RED, model_path.c_str(), COLOR_RESET);
            return -1;
        }
        PrintInputOutputInfo(module);
    }

    // Create streams
    std::vector<tcim::Stream> streams;
    constexpr bool kAutoYield = true;
    for (int32_t i = 0; i < stream_num; ++i) {
        streams.emplace_back(tcim::Stream(kAutoYield));
    }

    // Create threads and stats
    std::vector<std::thread> threads;
    if (samples < thread_num) {
        printf("%s[WARNING] samples(%d) < thread_num(%d), adjust thread_num to %d%s\n",
               COLOR_YELLOW, samples, thread_num, samples, COLOR_RESET);
        thread_num = samples;
    }

    std::vector<StatsInfo_t> statsInfos(thread_num);
    const int32_t base_repeat = samples / thread_num;
    const int32_t remainder = samples % thread_num;

    for (int32_t i = 0; i < thread_num; ++i) {
        statsInfos[i].idx = i;
        statsInfos[i].repeat = base_repeat;
        if (i < remainder) {
            statsInfos[i].repeat += 1;
        }
        threads.emplace_back(Infer, i, model_path, warmup, rounds,
                             std::ref(statsInfos[i]), std::ref(streams[i % stream_num]),
                             std::ref(option), check_output);
    }

    for (auto &t : threads) {
        t.join();
    }

    PrintStatsInfo(statsInfos, samples, rounds, perf_info);
    return 0;
}

/**
 * @brief Main function to run model performance test
 * @param model_path Path to the model file
 * @param warmup_num Number of warmup iterations
 * @param sample_num Number of samples to process
 * @param loop_num Number of loops per sample
 * @param thread_num Number of threads to use
 * @param stream_num Number of streams to use (default 0)
 * @param check_output Whether to check output consistency (default false)
 * @param devices Vector of device IDs to use (default {0})
 * @return Performance information structure
 */
PerfInfo_t ModelRunner(
    const std::string &model_path,
    int32_t warmup_num,
    int32_t sample_num,
    int32_t loop_num,
    int32_t thread_num,
    int32_t stream_num = 0,
    bool check_output = false,
    std::vector<int32_t> devices = {0}) {

    PerfInfo_t perfInfo;
    if (Run(model_path, thread_num, stream_num, warmup_num, sample_num, loop_num,
            devices, check_output, perfInfo) != 0) {
        throw std::runtime_error("Failed to run model");
    }
    return perfInfo;
}

/**
 * @brief Define Python module for performance testing
 * Exposes the performance testing functionality to Python
 */
PYBIND11_MODULE(perf, m) {
    m.doc() = "Python bindings for houmo chip perf test";
    py::class_<PerfInfo_t>(m, "PerfInfo")
        .def(py::init<>())
        .def_readwrite("input_avg_latency", &PerfInfo_t::input_avg_latency)
        .def_readwrite("input_max_latency", &PerfInfo_t::input_max_latency)
        .def_readwrite("input_min_latency", &PerfInfo_t::input_min_latency)
        .def_readwrite("input_tp99_latency", &PerfInfo_t::input_tp99_latency)
        .def_readwrite("input_tp999_latency", &PerfInfo_t::input_tp999_latency)
        .def_readwrite("infer_avg_latency", &PerfInfo_t::infer_avg_latency)
        .def_readwrite("infer_max_latency", &PerfInfo_t::infer_max_latency)
        .def_readwrite("infer_min_latency", &PerfInfo_t::infer_min_latency)
        .def_readwrite("infer_tp99_latency", &PerfInfo_t::infer_tp99_latency)
        .def_readwrite("infer_tp999_latency", &PerfInfo_t::infer_tp999_latency)
        .def_readwrite("output_avg_latency", &PerfInfo_t::output_avg_latency)
        .def_readwrite("output_max_latency", &PerfInfo_t::output_max_latency)
        .def_readwrite("output_min_latency", &PerfInfo_t::output_min_latency)
        .def_readwrite("output_tp99_latency", &PerfInfo_t::output_tp99_latency)
        .def_readwrite("output_tp999_latency", &PerfInfo_t::output_tp999_latency)
        .def_readwrite("e2e_avg_latency", &PerfInfo_t::e2e_avg_latency)
        .def_readwrite("e2e_max_latency", &PerfInfo_t::e2e_max_latency)
        .def_readwrite("e2e_min_latency", &PerfInfo_t::e2e_min_latency)
        .def_readwrite("e2e_tp99_latency", &PerfInfo_t::e2e_tp99_latency)
        .def_readwrite("e2e_tp999_latency", &PerfInfo_t::e2e_tp999_latency)
        .def_readwrite("avg_cost", &PerfInfo_t::avg_cost)
        .def_readwrite("qps", &PerfInfo_t::qps);
    m.def("CModelRunner", &ModelRunner,
          py::arg("model_path"),
          py::arg("warmup_num"),
          py::arg("sample_num"),
          py::arg("loop_num"),
          py::arg("thread_num"),
          py::arg("stream_num") = 0,
          py::arg("check_output") = false,
          py::arg("devices") = std::vector<int32_t>{0});
}