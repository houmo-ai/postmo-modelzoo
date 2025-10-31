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

#define COLOR_RED "\x1b[91;20m"
#define COLOR_GREEN "\x1b[92;20m"
#define COLOR_YELLOW "\x1b[93;20m"
#define COLOR_BLUE "\x1b[94;20m"
#define COLOR_MAGENT "\x1b[95;20m"
#define COLOR_CYAN "\x1b[96;20m"
#define COLOR_RESET "\x1b[0m"

typedef struct PerfInfo {
    float input_avg_latency;
    float input_max_latency;
    float input_min_latency;
    float input_tp99_latency;
    float input_tp999_latency;
    float infer_avg_latency;
    float infer_max_latency;
    float infer_min_latency;
    float infer_tp99_latency;
    float infer_tp999_latency;
    float output_avg_latency;
    float output_max_latency;
    float output_min_latency;
    float output_tp99_latency;
    float output_tp999_latency;
    float e2e_avg_latency;
    float e2e_max_latency;
    float e2e_min_latency;
    float e2e_tp99_latency;
    float e2e_tp999_latency;
    float avg_cost;
    float qps;
} perfInfo_t;

typedef struct StatsInfo {
    int32_t idx{-1};
    int32_t repeat{0};
    int64_t start_timestamp{0};
    int64_t end_timestamp{0};
    std::vector<float> set_input_times;
    std::vector<float> infer_times;
    std::vector<float> get_output_times;
    float total_set_input_time{0};
    float total_infer_time{0};
    float total_get_output_time{0};
} statsInfo_t;

static std::string ShapeToString(const std::vector<int64_t> &shape) {
    std::string shape_str = std::to_string(shape[0]);
    for (int j = 1; j < shape.size(); ++j) {
        shape_str += ", " + std::to_string(shape[j]);
    }
    return shape_str;
}

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

static float calculate_percentile(const std::vector<float> &latencies, float percentile) {
    if (latencies.empty())
        return 0.0f;
    if (latencies.size() == 1)
        return latencies[0];
    std::vector<float> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    int index = static_cast<int>(std::ceil(percentile * (sorted.size() - 1)));
    index = std::min(index, static_cast<int>(sorted.size() - 1));
    index = std::max(index, 0);
    return sorted[index];
}

static void PrintStatsInfo(const std::vector<statsInfo_t> &statsInfos, int32_t samples, int32_t rounds, perfInfo_t &perfInfo) {
    int64_t min_timstamp = std::numeric_limits<int64_t>::max();
    int64_t max_timstamp = std::numeric_limits<int64_t>::min();
    float infer_avg_time{0};
    float infer_max_time = std::numeric_limits<float>::min();
    float infer_min_time = std::numeric_limits<float>::max();
    float set_input_avg_time{0};
    float set_input_max_time = std::numeric_limits<float>::min();
    float set_input_min_time = std::numeric_limits<float>::max();
    float get_output_avg_time{0};
    float get_output_max_time = std::numeric_limits<float>::min();
    float get_output_min_time = std::numeric_limits<float>::max();
    float end2end_avg_time{0};
    float end2end_max_time = std::numeric_limits<float>::min();
    float end2end_min_time = std::numeric_limits<float>::max();
    std::vector<float> total_set_input_times;
    std::vector<float> total_infer_times;
    std::vector<float> total_get_output_times;
    std::vector<float> total_end2end_times;
    for (int32_t i = 0; i < statsInfos.size(); ++i) {
        if (statsInfos[i].start_timestamp < min_timstamp) {
            min_timstamp = statsInfos[i].start_timestamp;
        }
        if (statsInfos[i].end_timestamp > max_timstamp) {
            max_timstamp = statsInfos[i].end_timestamp;
        }
        set_input_avg_time += statsInfos[i].total_set_input_time;
        infer_avg_time += statsInfos[i].total_infer_time;
        get_output_avg_time += statsInfos[i].total_get_output_time;
        for (int j = 0; j < statsInfos[i].repeat; ++j) {
            auto set_input_time = statsInfos[i].set_input_times[j];
            auto get_output_time = statsInfos[i].get_output_times[j];
            auto infer_time = statsInfos[i].infer_times[j] / rounds;
            auto end2end_time = set_input_time + infer_time + get_output_time;
            end2end_avg_time += end2end_time;
            if (set_input_time > set_input_max_time) {
                set_input_max_time = set_input_time;
            }
            if (set_input_time < set_input_min_time) {
                set_input_min_time = set_input_time;
            }
            if (infer_time > infer_max_time) {
                infer_max_time = infer_time;
            }
            if (infer_time < infer_min_time) {
                infer_min_time = infer_time;
            }
            if (get_output_time > get_output_max_time) {
                get_output_max_time = get_output_time;
            }
            if (get_output_time < get_output_min_time) {
                get_output_min_time = get_output_time;
            }
            if (end2end_time > end2end_max_time) {
                end2end_max_time = end2end_time;
            }
            if (end2end_time < end2end_min_time) {
                end2end_min_time = end2end_time;
            }
            total_set_input_times.emplace_back(set_input_time);
            total_infer_times.emplace_back(infer_time);
            total_get_output_times.emplace_back(get_output_time);
            total_end2end_times.emplace_back(end2end_time);
        }
    }
    int64_t total_repeat = samples * rounds;
    float total_time = (max_timstamp - min_timstamp) / 1000.0f;
    set_input_avg_time /= samples;
    infer_avg_time /= total_repeat;
    get_output_avg_time /= samples;
    end2end_avg_time /= samples;
    float set_input_time_tp99 = calculate_percentile(total_set_input_times, 0.99f);
    float set_input_time_tp999 = calculate_percentile(total_set_input_times, 0.999f);
    float infer_time_tp99 = calculate_percentile(total_infer_times, 0.99f);
    float infer_time_tp999 = calculate_percentile(total_infer_times, 0.999f);
    float get_output_time_tp99 = calculate_percentile(total_get_output_times, 0.99f);
    float get_output_time_tp999 = calculate_percentile(total_get_output_times, 0.999f);
    float end2end_time_tp99 = calculate_percentile(total_end2end_times, 0.99f);
    float end2end_time_tp999 = calculate_percentile(total_end2end_times, 0.999f);

    float avg_cost = total_time / total_repeat;
    float QPS = total_repeat / total_time * 1000;

    printf("%s[Latency] Inference  avg: %7.3f ms, max: %7.3f ms, min: %7.3f ms, tp99: %7.3f ms, tp999: %7.3f ms%s\n",
           COLOR_CYAN, infer_avg_time, infer_max_time, infer_min_time, infer_time_tp99, infer_time_tp999, COLOR_RESET);
    printf("%s[Latency] Input      avg: %7.3f ms, max: %7.3f ms, min: %7.3f ms, tp99: %7.3f ms, tp999: %7.3f ms%s\n",
           COLOR_CYAN, set_input_avg_time, set_input_max_time, set_input_min_time, set_input_time_tp99, set_input_time_tp999, COLOR_RESET);
    printf("%s[Latency] Output     avg: %7.3f ms, max: %7.3f ms, min: %7.3f ms, tp99: %7.3f ms, tp999: %7.3f ms%s\n",
           COLOR_CYAN, get_output_avg_time, get_output_max_time, get_output_min_time, get_output_time_tp99, get_output_time_tp999, COLOR_RESET);
    printf("%s[Latency] End2end    avg: %7.3f ms, max: %7.3f ms, min: %7.3f ms, tp99: %7.3f ms, tp999: %7.3f ms%s\n",
           COLOR_CYAN, end2end_avg_time, end2end_max_time, end2end_min_time, end2end_time_tp99, end2end_time_tp999, COLOR_RESET);
    printf("%s[Throughput] total: %.3f ms, avg: %.3f ms, repeat: %d, rounds: %d%s\n", COLOR_MAGENT, total_time, avg_cost, samples, rounds, COLOR_RESET);
    printf("%s[Throughput] qps: %.3f %s\n", COLOR_MAGENT, QPS, COLOR_RESET);

    perfInfo.input_avg_latency = set_input_avg_time;
    perfInfo.input_max_latency = set_input_max_time;
    perfInfo.input_min_latency = set_input_min_time;
    perfInfo.infer_avg_latency = infer_avg_time;
    perfInfo.infer_max_latency = infer_max_time;
    perfInfo.infer_min_latency = infer_min_time;
    perfInfo.output_avg_latency = get_output_avg_time;
    perfInfo.output_max_latency = get_output_max_time;
    perfInfo.output_min_latency = get_output_min_time;
    perfInfo.e2e_avg_latency = end2end_avg_time;
    perfInfo.e2e_max_latency = end2end_max_time;
    perfInfo.e2e_min_latency = end2end_min_time;
    perfInfo.avg_cost = avg_cost;
    perfInfo.qps = QPS;
}

static int32_t CheckDevices(const std::vector<int32_t> &devices) {
    std::string devices_str = "[";
    for (int32_t i = 0; i < devices.size(); ++i) {
        int32_t device_id = devices[i];
        if (device_id >= tcim::GetDeviceNum()) {
            printf("%s[ERROR] Invalid device id: %d %s\n", COLOR_RED, device_id, COLOR_RESET);
            return -1;
        }
        for (int32_t j = i + 1; j < devices.size(); ++j) {
            if (device_id == devices[j]) {
                printf("%s[ERROR] Duplicate device id: %d %s\n", COLOR_RED, device_id, COLOR_RESET);
                return -1;
            }
        }
        devices_str += std::to_string(device_id);
        if (i != devices.size() - 1) {
            devices_str += ", ";
        }
    }
    devices_str += "]";
    printf("[INFO] Set Devices: %s\n", devices_str.c_str());
    return 0;
}

static void PrintInputOutputInfo(const tcim::Module &module) {
    int32_t core_num = module.GetCoreNum();
    printf("[INFO] CoreNum: %d\n", core_num);
    auto input_num = module.GetInputNum();
    printf("[INFO] InputNum: %d\n", input_num);
    for (int32_t i = 0; i < input_num; ++i) {
        auto name = module.GetInputName(i);
        auto info = module.GetInputInfo(name);
        printf("[INFO] Input[%d] name: %s, shape: [%s], dtype: %s, fmt: %s, memSize: %d\n",
               i, name.c_str(), ShapeToString(info.Shape()).c_str(), DataTypeToString(info.DataType()).c_str(),
               FmtToString(info.Format()).c_str(), info.MemSize());
    }
    auto output_num = module.GetOutputNum();
    printf("[INFO] OutputNum: %d\n", output_num);
    for (int32_t i = 0; i < output_num; ++i) {
        auto name = module.GetOutputName(i);
        auto info = module.GetOutputInfo(name);
        printf("[INFO] Output[%d] name: %s, shape: [%s], dtype: %s, fmt: %s, memSize: %d\n",
               i, name.c_str(), ShapeToString(info.Shape()).c_str(), DataTypeToString(info.DataType()).c_str(),
               FmtToString(info.Format()).c_str(), info.MemSize());
    }
}

static int32_t SetInputData(tcim::Module &module, std::map<std::string, tcim::Tensor> &inputs) {
    std::random_device rd;
    std::mt19937 rng(rd());
    for (int32_t i = 0; i < module.GetInputNum(); ++i) {
        auto name = module.GetInputName(i);
        auto info = module.GetInputInfo(name);
        auto tensor = tcim::Tensor::CreateHostTensor(info);
        std::string prefix = "resizer_crop_";
        // fill tensor
        if (tcim::DataType::FLOAT16 == info.DataType()) {
            std::uniform_real_distribution<float> dist(0, 1.0f);
            auto fp32_tensor = tcim::Tensor::CreateHostTensor(info.AsContiguous().AsType(tcim::DataType::FLOAT32));
            auto *data = (float *)fp32_tensor.Data();
            for (int64_t j = 0; j < fp32_tensor.MemSize() / sizeof(float); ++j) {
                data[j] = dist(rng);
            }
            fp32_tensor.CastTo(tensor);
        } else if (tcim::DataType::INT8 == info.DataType()) {
            std::uniform_int_distribution<int32_t> dist(-128, 127);
            auto *data = (int8_t *)tensor.Data();
            for (int64_t j = 0; j < tensor.MemSize(); ++j) {
                data[j] = dist(rng);
            }
        } else if (tcim::DataType::INT32 == info.DataType() && info.Format() == tcim::DataFmt::ND &&
                   name.length() > prefix.length() && name.substr(0, prefix.length()) == prefix) {
            auto custom_msg_str = module.GetCustomMsg();
            // printf("[INFO] CustomMsg: %s\n", custom_msg_str.c_str());
            if (custom_msg_str.empty()) {
                printf("%s[ERROR] HM Model build without custom msg, not support yet. Please build with Hmatc, and retry.%s\n", COLOR_RED, COLOR_RESET);
                return -1;
            }
            auto image_name = name.substr(prefix.length());
            auto image_info = module.GetInputInfo(image_name);
            auto custom_msg = nlohmann::json::parse(custom_msg_str);
            auto &model_input_shape = custom_msg[image_name]["shape"];
            auto &reszier_input_shape = image_info.Shape();
            auto &dyn_info_shape = info.Shape();
            assert(model_input_shape.size() == 4);
            assert(reszier_input_shape.size() == 4);
            auto MODEL_INPUT_H = model_input_shape[2];
            auto MODEL_INPUT_W = model_input_shape[3];
            auto RESIZER_INPUT_H = reszier_input_shape[2];
            auto RESIZER_INPUT_W = reszier_input_shape[3];
            auto RESIZER_CROP_H = RESIZER_INPUT_H;
            auto RESIZER_CROP_W = RESIZER_INPUT_W;
            auto sh = float(MODEL_INPUT_H) / float(RESIZER_INPUT_H);
            auto sw = float(MODEL_INPUT_W) / float(RESIZER_INPUT_W);
            if (sh > 16.0f || sh < 1.0f / 32) {
                RESIZER_CROP_H = int32_t(RESIZER_CROP_H * std::max(1.0f / 32, std::min(16.0f, sw))) & ~1;
            }
            if (sw > 16.0f || sw < 1.0f / 32) {
                RESIZER_CROP_W = int32_t(RESIZER_CROP_W * std::max(1.0f / 32, std::min(16.0f, sh))) & ~1;
            }
            assert(dyn_info_shape.size() == 2 || dyn_info_shape.size() == 1);
            int32_t batch = 1;
            int32_t step = dyn_info_shape[0];
            if (dyn_info_shape.size() > 1) {
                batch = dyn_info_shape[0];
                step = dyn_info_shape[1];
            }
            auto *data = (int32_t *)tensor.Data();
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

static void Infer(int32_t tid, std::string model_path, int32_t warmup, int32_t rounds,
                  statsInfo_t &stats, tcim::Stream &stream, tcim::Module::Option &option, bool check_output) {
    printf("[INFO] Infer %d started, wramup: %d, rounds: %d, repeat: %d\n", tid, warmup, rounds, stats.repeat);
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
    std ::map<std::string, tcim::Tensor> outputs_ref;
    std::map<std::string, tcim::Tensor> outputs;
    for (int32_t i = 0; i < module.GetOutputNum(); ++i) {
        auto name = module.GetOutputName(i);
        auto info = module.GetOutputInfo(name);
        auto tensor = tcim::Tensor::CreateHostTensor(info);
        outputs[name] = tensor;
        outputs_ref[name] = tensor.Clone();
    }
    auto status = module.SetStream(stream);
    if (status != tcim::Status::OK) {
        printf("%s[ERROR] Failed to set stream, and error code: %d%s\n", COLOR_RED, status, COLOR_RESET);
        return;
    }
    tcim::Module::RunOption run_option;
    bool sync = false;
    for (int32_t i = 0; i < warmup + stats.repeat; ++i) {
        if (i == warmup) {
            stats.start_timestamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
        }
        auto t0 = high_resolution_clock::now();
        for (int32_t k = 0; k < module.GetInputNum(); ++k) {
            auto name = module.GetInputName(k);
            module.SetInput(name, inputs[name]);
        }
        auto t1 = high_resolution_clock::now();
        run_option.Rounds(i < warmup ? 1 : rounds);
        status = module.Run(sync, run_option);
        if (status != tcim::Status::OK) {
            printf("%s[ERROR] Failed to run, and error code: %d%s\n", COLOR_RED, status, COLOR_RESET);
            continue;
        }
        status = module.Sync();
        if (status != tcim::Status::OK) {
            printf("%s[ERROR] Failed to sync, and error code: %d%s\n", COLOR_RED, status, COLOR_RESET);
            continue;
        }
        auto t2 = high_resolution_clock::now();
        for (int32_t k = 0; k < module.GetOutputNum(); ++k) {
            auto name = module.GetOutputName(k);
            // set to zero
            memset(outputs[name].Data(), 0, outputs[name].MemSize());
            module.GetOutput(name, outputs[name]);
            if (check_output) {
                if (i == warmup - 1) {
                    outputs[name].CopyTo(outputs_ref[name]);
                } else if (i >= warmup) {
                    if (memcmp(outputs[name].Data(), outputs_ref[name].Data(), outputs[name].MemSize())) {
                        printf("%s[ERROR] PID: %d, Iter: %5d, Output %s mismatch%s\n",
                               COLOR_RED, tid, i - warmup, name.c_str(), COLOR_RESET);
                    }
                }
            }
        }
        auto t3 = high_resolution_clock::now();
        if (i < warmup)
            continue;
        if (i == warmup + stats.repeat - 1) {
            stats.end_timestamp = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
        }
        auto tp0 = duration_cast<microseconds>(t1 - t0);
        auto tp1 = duration_cast<microseconds>(t2 - t1);
        auto tp2 = duration_cast<microseconds>(t3 - t2);
        float set_input_time = tp0.count() / 1000.0f;
        float infer_time = tp1.count() / 1000.0f;
        float get_output_time = tp2.count() / 1000.0f;
        stats.set_input_times.emplace_back(set_input_time);
        stats.infer_times.emplace_back(infer_time);
        stats.get_output_times.emplace_back(get_output_time);
        stats.total_set_input_time += set_input_time;
        stats.total_infer_time += infer_time;
        stats.total_get_output_time += get_output_time;
    }
    printf("[INFO] Infer %d done.\n", tid);
}

int32_t Run(const std::string &model_path, const std::string &model_name, int32_t thread_num,
            int32_t stream_num, int32_t warmup, int32_t samples, int32_t rounds,
            std::vector<int32_t> &devices, bool check_output, perfInfo_t &perf_info) {
    printf("[INFO] %s\n", "TCIM Performance Test");
    printf("[INFO] TCIM Runtime Version: %s\n", tcim::GetVersion().c_str());

    if (auto platform = std::getenv("HDPL_PLATFORM")) {
        if (strcmp(platform, "ASIC")) {
            thread_num = 1;
            stream_num = 1;
            samples = 1;
            warmup = 0;
            rounds = 1;
            devices = {0};
            printf("%s[WARNING] HDPL_PLATFORM is set to %s, and thread_num, stream_num, samples, warmup, rounds, devices are set to 1, 1, 1, 0, 1, {0}%s\n",
                   COLOR_YELLOW, platform, COLOR_RESET);
        }
    }

    auto HOUMO_CORE_NUM = getenv("HOUMO_CORE_NUM");
    if (!HOUMO_CORE_NUM) {
        printf("%s[WARNING] HOUMO_CORE_NUM is not set, and default set to %d%s\n", COLOR_YELLOW, 2, COLOR_RESET);
        HOUMO_CORE_NUM = "2";
    }
    if (stream_num <= 0) {
        stream_num = std::stoi(HOUMO_CORE_NUM) * 2;
    }

    printf("[INFO] Model path: %s\n", model_path.c_str());
    printf("[INFO] Warmup: %d\n", warmup);
    printf("[INFO] Rounds: %d\n", rounds);
    printf("[INFO] Repeat: %d\n", samples);
    printf("[INFO] Thread number: %d\n", thread_num);
    printf("[INFO] Stream number: %d\n", stream_num);

    const std::string target = getenv("HOUMO_TARGET");
    if (target != "xh1" && target != "xh2") {
        printf("%s[ERROR] HOUMO_TARGET is invalid: %s%s\n", COLOR_RED, target.c_str(), COLOR_RESET);
        return -1;
    }
    std::string backend_name;
    if (target == "xh1") {
        backend_name = "Xh1HdiBackend";
    } else if (target == "xh2") {
        backend_name = "Xh2HalBackend";
    }
    printf("[INFO] Backend: %s\n", backend_name.c_str());
    if (devices.size() > tcim::GetDeviceNum()) {
        printf("%s[ERROR] Not enough devices%s\n", COLOR_RED, COLOR_RESET);
        return -1;
    };
    if (CheckDevices(devices) != 0)
        return -1;
    tcim::Module::WeightManager weight_manager;
    // auto device_manager = tcim::DevManager::Create(devices, backend_name);
    // weight_manager = tcim::Module::WeightManager::CreateWeightManager(device_manager);
    weight_manager = tcim::Module::WeightManager::CreateWeightManager(devices[0]);
    auto option = tcim::Module::Option(weight_manager);
    {
        auto module = tcim::Module::LoadFromFile(model_path, option);
        if (module.GetInitStatus() != tcim::Status::OK) {
            printf("%s[ERROR] Failed to load model: %s%s\n", COLOR_RED, model_path.c_str(), COLOR_RESET);
            return -1;
        }
        PrintInputOutputInfo(module);
    }
    std::vector<tcim::Stream> streams;
    bool auto_yield = true;
    for (int32_t i = 0; i < stream_num; ++i) {
        tcim::Stream stream(auto_yield);
        streams.emplace_back(std::move(stream));
    }
    std::vector<std::thread> threads;
    std::vector<statsInfo_t> statsInfos;
    statsInfos.resize(thread_num);
    int32_t repeat = samples / thread_num;
    int32_t mod = samples % thread_num;  // 余数
    for (int32_t i = 0; i < thread_num; ++i) {
        statsInfos[i].idx = i;
        statsInfos[i].repeat = repeat;
        if (i < mod)
            statsInfos[i].repeat += 1;
        threads.emplace_back(Infer, i, model_path, warmup, rounds, std::ref(statsInfos[i]),
                             std::ref(streams[i % stream_num]), std::ref(option), check_output);
    }
    for (auto &t : threads)
        t.join();
    PrintStatsInfo(statsInfos, samples, rounds, perf_info);
    return 0;
}

perfInfo_t ModelRunner(
    const std::string &model_path,
    int32_t warmup_num,
    int32_t sample_num,
    int32_t loop_num,
    int32_t thread_num,
    int32_t stream_num = 0,
    bool check_output = false,
    std::vector<int32_t> devices = {0}) {

    perfInfo_t perfInfo;
    if (0 != Run(model_path, "model_name", thread_num, stream_num, warmup_num, sample_num, loop_num,
                 devices, check_output, perfInfo)) {
        throw std::runtime_error("Failed to run model");
    }
    return perfInfo;
}

PYBIND11_MODULE(perf, m) {
    m.doc() = "Python bindings for huomo chip perf test";
    py::class_<perfInfo_t>(m, "PerfInfo")
        .def(py::init<>())
        .def_readwrite("input_avg_latency", &perfInfo_t::input_avg_latency)
        .def_readwrite("input_max_latency", &perfInfo_t::input_max_latency)
        .def_readwrite("input_min_latency", &perfInfo_t::input_min_latency)
        .def_readwrite("input_tp99_latency", &perfInfo_t::input_tp99_latency)
        .def_readwrite("input_tp999_latency", &perfInfo_t::input_tp999_latency)
        .def_readwrite("infer_avg_latency", &perfInfo_t::infer_avg_latency)
        .def_readwrite("infer_max_latency", &perfInfo_t::infer_max_latency)
        .def_readwrite("infer_min_latency", &perfInfo_t::infer_min_latency)
        .def_readwrite("infer_tp99_latency", &perfInfo_t::infer_tp99_latency)
        .def_readwrite("infer_tp999_latency", &perfInfo_t::infer_tp999_latency)
        .def_readwrite("output_avg_latency", &perfInfo_t::output_avg_latency)
        .def_readwrite("output_max_latency", &perfInfo_t::output_max_latency)
        .def_readwrite("output_min_latency", &perfInfo_t::output_min_latency)
        .def_readwrite("output_tp99_latency", &perfInfo_t::output_tp99_latency)
        .def_readwrite("output_tp999_latency", &perfInfo_t::output_tp999_latency)
        .def_readwrite("avg_cost", &perfInfo_t::avg_cost)
        .def_readwrite("qps", &perfInfo_t::qps);
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
