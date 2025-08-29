#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <queue>
#include <sstream>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#if (__GNUC__ < 8)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include "tcim/tcim_runtime.h"

#define MEM_CHECK 0  // enable memory check, it may effect the performance

#if MEM_CHECK
#include <sys/resource.h>
#endif

#define COLOR_RED "\x1b[91;20m"
#define COLOR_GREEN "\x1b[92;20m"
#define COLOR_YELLOW "\x1b[93;20m"
#define COLOR_BLUE "\x1b[94;20m"
#define COLOR_MAGENT "\x1b[95;20m"
#define COLOR_CYAN "\x1b[96;20m"
#define COLOR_RESET "\x1b[0m"

#define GET_TIME() std::chrono::system_clock::now()
#define GET_COST(start, end) \
    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()

using json = nlohmann::json;
namespace py = pybind11;

typedef struct {
    std::string model_path;
    tcim::Module::WeightManager weight_manager;
    int loop_num = 0;
    int sample_cnt = 0;
    int warm_up = 0;
    bool infer_only = false;
    bool is_result_check = true;
    uint32_t infer_max_cost = 0;
    uint32_t infer_total_cost = 0;
    uint32_t input_max_cost = 0;
    uint32_t input_total_cost = 0;
    uint32_t output_max_cost = 0;
    uint32_t output_total_cost = 0;
    uint32_t e2e_max_cost = 0;
    uint32_t e2e_total_cost = 0;
} ThreadInfo;

typedef struct {
    std::vector<tcim::Stream> streams;
    std::vector<int> counts;  // 用来保存各stream下计数
} StreamInfo;

typedef struct {
    uint64_t req_id;
    std::map<std::string, tcim::Tensor> data_in;
    std::map<std::string, tcim::Tensor> data_out;
    std::map<std::string, tcim::Tensor> ref_out;
} Task;

typedef struct {
    std::queue<Task> queue;
    std::mutex mutex;
    std::condition_variable cond;
    // std::map<std::string, tcim::TensorInfo> info_map;
} TaskQueue;

typedef struct {
    int32_t idx{-1};
    std::string name;
    std::vector<int64_t> shape;
    tcim::DataFmt fmt;
    bool is_pic{false};
    int32_t dyn_idx{-1};
} input_t;

/**
 * @brief whether the file exists
 *
 * @param file_path file path
 * @return true file exists
 * @return false file does not exist
 */
bool IsFileExists(std::string file_path) {
    std::ifstream f(file_path.c_str());
    return f.good();
}

int read_file(const char *fileName, char **fileData, int *fileLen) {
    FILE *file = fopen(fileName, "rb");
    if (file == NULL) {
        perror("open file failed\n");
        return -1;
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    *fileData = (char *)malloc(fileSize);
    if (*fileData == NULL) {
        printf("malloc fileData size:%ld failed\n", fileSize);
        fclose(file);
        return -1;
    }
    long readSize = fread(*fileData, 1, fileSize, file);
    if (readSize != fileSize) {
        printf("readSize(%ld) != fileSize(%ld), read %s failed!\n", readSize,
               fileSize, fileName);
        fclose(file);
        return -1;
    }
    *fileLen = fileSize;
    fclose(file);
    return 0;
}

int write_file(const char *fileName, char *fileData, int fileLen) {
    FILE *file = fopen(fileName, "wb");
    if (file == NULL) {
        perror("open file failed\n");
        return -1;
    }
    long writeSize = fwrite(fileData, 1, fileLen, file);
    if (writeSize != fileLen) {
        printf("writeSize(%ld) != fileLen(%d), write %s failed!\n", writeSize,
               fileLen, fileName);
        fclose(file);
        return -1;
    }
    fclose(file);
    return 0;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec) {
    out << "[";
    for (size_t idx = 0; idx < vec.size(); idx++) {
        if (idx != 0) {
            out << ", ";
        }
        out << vec[idx];
    }
    out << "]";
    return out;
}

class Barrier {
public:
    Barrier(int dest) : dest_(dest) {}

    void barrier() {
        std::unique_lock<std::mutex> lock(mtx_);
        count_++;
        cond0_.notify_all();
        cond_.wait(lock);
    }

    bool wait(int timeout = 0) {
        std::unique_lock<std::mutex> lock(mtx_);
        int time = 0;
        while (count_ < dest_) {
            if (timeout == 0) {
                cond0_.wait(lock);
            } else {
                lock.unlock();
                time += 10;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                lock.lock();
                if (time >= timeout) {
                    return false;
                }
            }
        }
        cond_.notify_all();
        return true;
    }

    void barrier_and_wait() {
        std::unique_lock<std::mutex> lock(mtx_);
        count_++;
        if (count_ < dest_) {
            cond_.wait(lock);
        } else {
            cond_.notify_all();
            cond0_.notify_all();
        }
    }

    void reset() {
        std::unique_lock<std::mutex> lock(mtx_);
        count_ = 0;
    }

protected:
    int count_ = 0;
    int dest_ = 0;
    std::condition_variable cond_, cond0_;
    std::mutex mtx_;
};

void SetAffinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    // sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np");
        exit(EXIT_FAILURE);
    }
}

std::string SanitizeName(const std::string &name) {
    std::string output = name;
    for (char &c : output) {
        if (c == '/' || c == '-' || c == '#') {
            c = '_';
        }
    }
    return output;
}

typedef struct {
    float input_avg_latency;
    float input_max_latency;
    float infer_avg_latency;
    float infer_max_latency;
    float output_avg_latency;
    float output_max_latency;
    float e2e_avg_latency;
    float e2e_max_latency;
    float avg_cost;
    float qps;
} perfInfo_t;

perfInfo_t ModelRunner(
    const std::string &model_path,
    int sample_num,
    int thread_num,
    int device_num,
    int loop_num,
    int warm_up) {

    if (auto platform = std::getenv("HDPL_PLATFORM")) {
        if (strcmp(platform, "ASIC")) {
            thread_num = 1;
            device_num = 1;
        }
    }

    std::cout << "model: " << model_path << std::endl;
    std::cout << "samples: " << sample_num << std::endl;
    std::cout << "loops: " << loop_num << std::endl;
    std::cout << "warmup: " << warm_up << std::endl;
    std::cout << "threads: " << thread_num << std::endl;
    std::cout << "devices: " << device_num << std::endl;

    if (sample_num < thread_num) {
        std::cout
            << COLOR_YELLOW
            << "[warn] the perf result may not be accurate while samples < threads"
            << COLOR_RESET << std::endl;
    }

    TaskQueue qin;
    TaskQueue qout;
    auto module = tcim::Module::LoadFromFile(model_path);
    if (!module) {
        std::cout << COLOR_RED << "[error] load model " << model_path
                  << " fail, exit..." << COLOR_RESET << std::endl;
        exit(-1);
    }

    // prepare input & output data
    std::map<std::string, input_t> inputs;
    std::map<std::string, tcim::Tensor> input_datas;
    int input_num = module.GetInputNum();
    std::cout << "Count of Input: " << input_num << std::endl;
    for (int idx = 0; idx < input_num; idx++) {
        auto input_name = module.GetInputName(idx);
        auto input_info = module.GetInputInfo(input_name).AsContiguous();
        auto tensor = tcim::Tensor::CreateHostTensor(input_info);
        std::cout << "Input[" << idx << "] name: " << input_name << ", " << input_info << std::endl;
        input_t input;
        input.name = input_name;
        input.shape = input_info.Shape();
        input.fmt = input_info.Format();
        input.idx = idx;
        if (input.fmt == tcim::DataFmt::YUV420SP || input.fmt == tcim::DataFmt::YUV422SP ||
            input.fmt == tcim::DataFmt::YUV444SP) {
            input.is_pic = true;
            inputs[input_name] = input;
        }
        input_datas.insert(std::pair<std::string, tcim::Tensor>(input_name, tensor));
    }

    if (inputs.size() > 0) {
        // 如果存在输入为pic，则必然存在resizer，必然有custom_msg可取
        std::string custom_msg_str = module.GetCustomMsg();
        json custom_msg = json::parse(custom_msg_str);
        for (auto &item : inputs) {
            std::string dyn_info_name = "resizer_crop_" + item.first;
            auto &raw_input_shape = custom_msg[item.first]["shape"];
            auto &shape = inputs[item.first].shape;
            assert(shape.size() == 4);
            assert(raw_input_shape.size() == 4);
            std::string dyn_info_str;
            for (int idx = 0; idx < input_num; ++idx) {
                auto input_name = module.GetInputName(idx);
                auto input_info = module.GetInputInfo(input_name).AsContiguous();
                if (dyn_info_name != input_name)
                    continue;
                auto dyn_shape = input_info.Shape();  // bs*10 or bs4
                auto tensor = input_datas[input_name];
                int32_t *data = (int32_t *)tensor.Data();
                assert(dyn_shape.size() == 2 || dyn_shape.size() == 1);
                int32_t batch = 1;
                int32_t step = 4;
                if (dyn_shape.size() == 2) {
                    batch = dyn_shape[0];
                    step = dyn_shape[1];
                }
                for (int n = 0; n < batch; ++n) {
                    data[n * step + 0] = 0;
                    data[n * step + 1] = 0;
                    data[n * step + 2] = shape[2];
                    data[n * step + 3] = shape[3];
                    dyn_info_str = std::to_string(data[n * step + 0]) + ", " +
                                   std::to_string(data[n * step + 1]) + ", " +
                                   std::to_string(data[n * step + 2]) + ", " +
                                   std::to_string(data[n * step + 3]);
                    if (step == 10) {
                        data[n * step + 4] = raw_input_shape[2];
                        data[n * step + 5] = raw_input_shape[3];
                        data[n * step + 6] = 0;
                        data[n * step + 7] = 0;
                        data[n * step + 8] = 0;
                        data[n * step + 9] = 0;
                        dyn_info_str += ", " +
                                        std::to_string(data[n * step + 4]) + ", " +
                                        std::to_string(data[n * step + 5]) + ", " +
                                        std::to_string(data[n * step + 6]) + ", " +
                                        std::to_string(data[n * step + 7]) + ", " +
                                        std::to_string(data[n * step + 8]) + ", " +
                                        std::to_string(data[n * step + 9]);
                    }
                    printf("[DynamicInfo] %s: idx: %d, info: %s\n", input_name.c_str(), n, dyn_info_str.c_str());
                }
            }
        }
    }

    for (int i = 0; i < sample_num; i++) {
        Task task;
        task.req_id = i;
        task.data_in = input_datas;
        qin.queue.push(task);
    }

    std::cout << "sample queue size is " << qin.queue.size() << std::endl;

    auto thread_func = [](int tid, int did, ThreadInfo &info,
                          StreamInfo &stream_info, TaskQueue &qin,
                          TaskQueue &qout, Barrier &barrier) {
        auto start = GET_TIME();
        auto end = GET_TIME();
        float cost = 0.0;
        // load model
        start = GET_TIME();
        std::unique_lock<std::mutex> lock_xx(qin.mutex);
        auto option = tcim::Module::Option(info.weight_manager);
        auto module = tcim::Module::LoadFromFile(info.model_path, option);
        lock_xx.unlock();
        end = GET_TIME();
        cost = GET_COST(start, end) / 1000.0 / info.warm_up;
        if (!module) {
            std::cerr << COLOR_RED << "Device " << did << " Thread " << tid
                      << " load model " << info.model_path << " fail." << COLOR_RESET
                      << std::endl;
            exit(-1);
        }
        std::cout << "Device " << did << " Thread " << tid << " " << info.model_path
                  << " model loaded. Cost " << cost << " ms." << std::endl;

        std::unique_lock<std::mutex> lock_in(qin.mutex);
        auto task = qin.queue.front();
        lock_in.unlock();
        for (auto &tensor : task.data_in) {
            module.SetInput(tensor.first, tensor.second);
        }
        // warm up
        start = GET_TIME();
        for (int i = 0; i < info.warm_up; i++) {
            module.Run(false);
        }
        module.Sync();
        end = GET_TIME();
        cost = GET_COST(start, end) / 1000.0 / info.warm_up;
        std::cout << "Device " << did << " Thread " << tid << " Warm Up "
                  << info.warm_up << " average cost " << cost << " ms."
                  << std::endl;

        // wait until all threads ready
        barrier.barrier();
        std::cout << "Device " << did << " Thread " << tid << " infer start..."
                  << std::endl;
        int count = 0;
        int stream_id = -1;

        while (true) {
            std::unique_lock<std::mutex> lock_in(qin.mutex);
            if (stream_id != -1) {
                stream_info.counts[stream_id]--;
            }
            if (qin.queue.empty()) {
                lock_in.unlock();
                break;
            }
            auto task = qin.queue.front();
            qin.queue.pop();
            stream_id = 0;
            for (int i = 1; i < stream_info.counts.size(); i++) {
                if (stream_info.counts[i] < stream_info.counts[stream_id]) {
                    stream_id = i;
                }
            }
            stream_info.counts[stream_id]++;
            lock_in.unlock();

            start = GET_TIME();
            for (auto &tensor : task.data_in) {
                module.SetInput(tensor.first, tensor.second);
            }
            auto input_end = GET_TIME();
            cost = GET_COST(start, input_end);
            info.input_total_cost += cost;
            if (info.input_max_cost < cost)
                info.input_max_cost = cost;
            tcim::Module::RunOption run_option;
            run_option.Rounds(info.loop_num);
            module.SetStream(stream_info.streams[stream_id]);
            module.Run(false, run_option);
            module.Sync();
            auto infer_end = GET_TIME();
            cost = GET_COST(input_end, infer_end);
            info.infer_total_cost += cost;
            if (info.infer_max_cost < cost)
                info.infer_max_cost = cost;
            end = GET_TIME();
            cost = GET_COST(infer_end, end);
            info.output_total_cost += cost;
            if (info.output_max_cost < cost)
                info.output_max_cost = cost;
            cost = GET_COST(start, end);
            info.e2e_total_cost += cost;
            if (info.e2e_max_cost < cost)
                info.e2e_max_cost = cost;
            count++;
        }
        info.sample_cnt = count;
        std::cout << "Device " << did << " Thread " << tid << " completed. "
                  << info.sample_cnt << " samples tested." << std::endl;
        barrier.barrier();
    };

    // create threads
    std::vector<std::thread> threads;
    Barrier barrier(thread_num * device_num);
    ThreadInfo thread_info[thread_num * device_num];
    StreamInfo stream_info;
    stream_info.counts.resize(4);
    stream_info.streams.resize(4);
    for (int did = 0; did < device_num; did++) {
        auto weight_manager = tcim::Module::WeightManager::CreateWeightManager(did);
        for (int tid = 0; tid < thread_num; tid++) {
            ThreadInfo *info = &thread_info[did * thread_num + tid];
            info->model_path = model_path;
            info->weight_manager = weight_manager;
            info->loop_num = loop_num;
            info->infer_only = true;
            info->is_result_check = false;
            info->warm_up = warm_up;
            int id = did * thread_num + tid;
            threads.push_back(std::thread(thread_func, id, did, std::ref(*info),
                                          std::ref(stream_info), std::ref(qin),
                                          std::ref(qout), std::ref(barrier)));
        }
    }
    barrier.wait();
    barrier.reset();
    auto start = GET_TIME();
#if MEM_CHECK
    do {
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == -1) {
            perror("getrusage");
            return -1;
        }
        auto cur = GET_TIME();
        auto cost = GET_COST(start, cur);
        printf("run %.3fs, rss %ldKB\n", cost / 1000000.0, usage.ru_maxrss);
    } while (!barrier.wait(1000));
#endif

    barrier.wait();
    auto end = GET_TIME();

    // wait all threads done
    for (auto &t : threads)
        t.join();

    uint32_t infer_max_cost = 0;
    uint32_t infer_total_cost = 0;
    uint32_t input_max_cost = 0;
    uint32_t input_total_cost = 0;
    uint32_t output_max_cost = 0;
    uint32_t output_total_cost = 0;
    uint32_t e2e_max_cost = 0;
    uint32_t e2e_total_cost = 0;

    for (int i = 0; i < thread_num * device_num; i++) {
        if (thread_info[i].infer_max_cost > infer_max_cost)
            infer_max_cost = thread_info[i].infer_max_cost;
        infer_total_cost += thread_info[i].infer_total_cost;
        if (thread_info[i].input_max_cost > input_max_cost)
            input_max_cost = thread_info[i].input_max_cost;
        input_total_cost += thread_info[i].input_total_cost;
        if (thread_info[i].output_max_cost > output_max_cost)
            output_max_cost = thread_info[i].output_max_cost;
        output_total_cost += thread_info[i].output_total_cost;
        if (thread_info[i].e2e_max_cost > e2e_max_cost)
            e2e_max_cost = thread_info[i].e2e_max_cost;
        e2e_total_cost += thread_info[i].e2e_total_cost;
    }

    int test_num = loop_num * sample_num;
    float infer_avg_latency = infer_total_cost / test_num / 1000.0;
    float infer_max_latency = infer_max_cost / 1000.0;
    float input_avg_latency = input_total_cost / test_num / 1000.0;
    float input_max_latency = input_max_cost / 1000.0;
    float output_avg_latency = output_total_cost / test_num / 1000.0;
    float output_max_latency = output_max_cost / 1000.0;
    float e2e_avg_latency = e2e_total_cost / test_num / 1000.0;
    float e2e_max_latency = e2e_max_cost / 1000.0;
    float total_cost = GET_COST(start, end) / 1000.0;
    float avg_cost = total_cost / test_num;
    float qps = (1000.0 / (total_cost / test_num));

    std::cout << COLOR_CYAN << std::fixed << std::setprecision(3)
              << "[latency] Inference "
              << "\tavg: " << std::setw(7) << infer_avg_latency << " ms,"
              << "\tmax: " << std::setw(7) << infer_max_latency << " ms"
              << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << std::fixed << std::setprecision(3)
              << "[latency] Input "
              << "\tavg: " << std::setw(7) << input_avg_latency << " ms,"
              << "\tmax: " << std::setw(7) << input_max_latency << " ms"
              << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << std::fixed << std::setprecision(3)
              << "[latency] Output "
              << "\tavg: " << std::setw(7) << output_avg_latency << " ms,"
              << "\tmax: " << std::setw(7) << output_max_latency << " ms"
              << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << std::fixed << std::setprecision(3)
              << "[latency] End2End "
              << "\tavg: " << std::setw(7) << e2e_avg_latency << " ms,"
              << "\tmax: " << std::setw(7) << e2e_max_latency << " ms"
              << COLOR_RESET << std::endl;
    std::cout << COLOR_MAGENT << std::fixed << std::setprecision(3)
              << "[Throughput] total: " << total_cost << " ms, "
              << "avg: " << avg_cost << " ms" << COLOR_RESET << std::endl;
    std::cout << COLOR_MAGENT << std::fixed << std::setprecision(3)
              << "[Throughput] qps: " << qps << COLOR_RESET << std::endl;

    perfInfo_t perfInfo;
    perfInfo.input_avg_latency = input_avg_latency;
    perfInfo.input_max_latency = input_max_latency;
    perfInfo.infer_avg_latency = infer_avg_latency;
    perfInfo.infer_max_latency = infer_max_latency;
    perfInfo.output_avg_latency = output_avg_latency;
    perfInfo.output_max_latency = output_max_latency;
    perfInfo.avg_cost = avg_cost;
    perfInfo.qps = qps;
    return perfInfo;
};

PYBIND11_MODULE(perf, m) {
    m.doc() = "Python bindings for huomo chip perf test";
    py::class_<perfInfo_t>(m, "PerfInfo")
        .def(py::init<>())
        .def_readwrite("input_avg_latency", &perfInfo_t::input_avg_latency)
        .def_readwrite("input_max_latency", &perfInfo_t::input_max_latency)
        .def_readwrite("infer_avg_latency", &perfInfo_t::infer_avg_latency)
        .def_readwrite("infer_max_latency", &perfInfo_t::infer_max_latency)
        .def_readwrite("output_avg_latency", &perfInfo_t::output_avg_latency)
        .def_readwrite("output_max_latency", &perfInfo_t::output_max_latency)
        .def_readwrite("avg_cost", &perfInfo_t::avg_cost)
        .def_readwrite("qps", &perfInfo_t::qps);
    m.def("CModelRunner", &ModelRunner);
}
