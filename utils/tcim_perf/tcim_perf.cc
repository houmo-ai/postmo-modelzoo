// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file main.cc
 */
#include <getopt.h>
#include <stdio.h>
#include <unistd.h>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

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

#define COLOR_RED     "\x1b[91;20m"
#define COLOR_GREEN   "\x1b[92;20m"
#define COLOR_YELLOW  "\x1b[93;20m"
#define COLOR_BLUE    "\x1b[94;20m"
#define COLOR_MAGENT  "\x1b[95;20m"
#define COLOR_CYAN    "\x1b[96;20m"
#define COLOR_RESET   "\x1b[0m"

#define GET_TIME() std::chrono::system_clock::now()
#define GET_COST(start, end) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()

struct CliArguments {
  std::string model_path;
  std::string data_path;
  std::string output_path;
  size_t batch = 1;
  size_t warm_up = 1;
  size_t threads = 1;
  size_t devices = 1;
  size_t loops = 1;
  size_t samples = 1;
  bool infer_only = false;
};


typedef struct {
  std::string model_path;
  tcim::Module::WeightManager weight_manager;
  int loop_num = 0;
  int sample_cnt = 0;
  int warm_up = 0;
  bool infer_only = false;
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

int read_file(const char *fileName, char **fileData, int *fileLen) 
{
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
    printf("readSize(%ld) != fileSize(%ld), read %s failed!\n", readSize, fileSize, fileName);
    fclose(file);
    return -1;
  }
  *fileLen = fileSize;
  fclose(file);
  return 0;
}

int write_file(const char *fileName, char *fileData, int fileLen) 
{
  FILE *file = fopen(fileName, "wb"); 
  if (file == NULL) {
    perror("open file failed\n");
    return -1;
  }
  long writeSize = fwrite(fileData, 1, fileLen, file);
  if (writeSize != fileLen) {
    printf("writeSize(%ld) != fileLen(%d), write %s failed!\n", writeSize, fileLen, fileName);
    fclose(file);
    return -1;
  }
  fclose(file);
  return 0;
}

/**
 * @brief Parse cmdline arguments to struct *arguments
 *
 * @param arguments pointer to output CliArguments struct
 * @param argc cmdline argument count
 * @param argv cmdline argument char*
 * @return true parse command line succeed
 * @return false parse command line failed
 */
bool ParseArgs(CliArguments *arguments, int argc, char *argv[]) {
  int option_idx = 0;
  struct option long_options[] = {
      {"help", 0, 0, 'h'},
      {"model", 1, 0, 'm'},
      {"data", 1, 0, 'd'},
      {"warm_up", 1, 0, 'w'},
      {"batch", 1, 0, 'b'},
      {"loops", 1, 0, 'l'},
      {"threads", 1, 0, 't'},
      {"devices", 1, 0, 'v'},
      {"samples", 1, 0, 's'},
      {"output", 1, 0, 'o'},
      {"infer_only", 1, 0, 'y'}
  };
  while (true) {
    int ch = getopt_long(argc, argv, "hm:d:w:b:t:v:l:s:o:y:", long_options, &option_idx);
    if (ch == -1) {
      break;
    }
    switch (ch) {
    case 'h':
      std::cout << "Usage: -h" << std::endl;
      break;
    case 'm':
      arguments->model_path = std::string(optarg);
      break;
    case 'd':
      arguments->data_path = std::string(optarg);
      break;
    case 'w':
      arguments->warm_up = atoi(optarg);
      break;
    case 'b':
      arguments->batch = atoi(optarg);
      break;
    case 't':
      arguments->threads = atoi(optarg);
      break;
    case 'v':
      arguments->devices = atoi(optarg);
      break;
    case 'l':
      arguments->loops = atoi(optarg);
      break;
    case 's':
      arguments->samples = atoi(optarg);
      break;
    case 'o':
      arguments->output_path = optarg;
      break;
    case 'y':
      arguments->infer_only = optarg;
      break;
    default:
      std::cerr << "Unsupported option: " << static_cast<char>(ch) << std::endl;
      return false;
    }
  }
  return true;
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
  Barrier(int dest): dest_(dest) {}

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
  //sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpuset);

  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
      perror("pthread_setaffinity_np");
      exit(EXIT_FAILURE);
  }
}

std::string SanitizeName(const std::string& name) {
  std::string output = name;
  for (char &c : output) {
    if (c == '/' || c == '-' || c == '#') {
      c = '_';
    }
  }
  return output;
}

int main(int argc, char *argv[]) {
  CliArguments arguments;
  ParseArgs(&arguments, argc, argv);

  std::string model_path = arguments.model_path;
  std::string data_path = arguments.data_path;
  std::string output_path = arguments.output_path;
  int batch = arguments.batch;
  int sample_num = arguments.samples;
  int thread_num = arguments.threads;
  int device_num = arguments.devices;
  int loop_num = arguments.loops;
  int warm_up = arguments.warm_up;
  bool infer_only = arguments.infer_only;
  bool is_result_check = true;

  if (infer_only) is_result_check = false;

  if (auto platform = std::getenv("HDPL_PLATFORM")) {
    if (strcmp(platform, "ASIC")) {
      thread_num = 1;
      device_num = 1;
    }
  } else {  // HDPL_PLATFORM not set
    thread_num = 1;
    device_num = 1;
  }

  std::cout << "model: " << model_path << std::endl;
  std::cout << "data: " << data_path << std::endl;
  std::cout << "samples: " << sample_num << std::endl;
  std::cout << "loops: " << loop_num << std::endl;
  std::cout << "warmup: " << warm_up << std::endl;
  std::cout << "batch: " << batch << std::endl;
  std::cout << "threads: " << thread_num << std::endl;
  std::cout << "devices: " << device_num << std::endl;
  std::cout << "infer_only: " << infer_only << std::endl;

  if (sample_num < thread_num) {
    std::cout << COLOR_YELLOW << "[warn] the perf result may not be accurate while samples < threads"
              << COLOR_RESET << std::endl;
  }

  TaskQueue qin;
  TaskQueue qout;

  // get module input & output info
  auto module = tcim::Module::LoadFromFile(model_path);
  if (!module) {
    std::cout << COLOR_RED << "[error] load model " << model_path << " fail, exit..."
              << COLOR_RESET << std::endl;
    exit(-1);
  }

  // prepare input & output data
  std::map<std::string, tcim::Tensor> input_datas;
  std::map<std::string, tcim::Tensor> output_golden;
  int input_num = module.GetInputNum();
  std::cout << "Count of Input: " << input_num << std::endl;
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name).AsContiguous();
    std::cout << "Input[" << idx << "] name: " << input_name << ", " << input_info << std::endl;
    auto data_file = data_path + "/" + input_name + ".bin";
    void* data_ptr = nullptr;
    int len = 0;
    auto tensor = tcim::Tensor::CreateHostTensor(input_info);
    if (fs::exists(data_file)) {
      if (read_file(data_file.c_str(), (char**)&data_ptr, &len)) {
        std::cout << COLOR_YELLOW << "[warn] Read input data file " << data_file
                  << " fail. Use random data and result check will be skipped."
                  << COLOR_RESET << std::endl;
        is_result_check = false;
      } else {
        memcpy(tensor.Data(), data_ptr, tensor.Info().Size());
        free(data_ptr);
      }
    } else {
      std::cout << COLOR_YELLOW << "[warn] Input data file " << data_file
                << " not exist. Use random data and result check will be skipped."
                << COLOR_RESET << std::endl;
      is_result_check = false;
    }
    input_datas.insert(std::pair<std::string, tcim::Tensor>(input_name, tensor));
  }

  int output_num = module.GetOutputNum();
  std::cout << "Count of Output: " << output_num << std::endl;
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name).AsContiguous();
    std::cout << "Output[" << output_name << "] " << output_info << std::endl;
    auto data_file = data_path + "/" + SanitizeName(output_name) + ".bin";
    void* data_ptr = nullptr;
    int len = 0;
    auto tensor = tcim::Tensor::CreateHostTensor(output_info);
    if (fs::exists(data_file)) {
      if (read_file(data_file.c_str(), (char**)&data_ptr, &len)) {
        std::cout << COLOR_YELLOW << "[warn] Read output data file " << data_file
                  << " fail. Result check will be skipped." << COLOR_RESET << std::endl;
        is_result_check = false;
      } else {
        memcpy(tensor.Data(), data_ptr, tensor.Info().Size());
        free(data_ptr);
      }
    } else {
      std::cout << COLOR_YELLOW << "[warn] Output data file " << data_file
                << " not exist. Result check will be skipped." << COLOR_RESET << std::endl;
      is_result_check = false;
    }
    output_golden.insert(std::pair<std::string, tcim::Tensor>(output_name, tensor));
  }

  for (int i = 0; i < sample_num; i++) {
    Task task;
    task.req_id = i;
    task.data_in = input_datas;
    task.ref_out = output_golden;
    std::map<std::string, tcim::Tensor> output_datas;
    for (auto& output : output_golden) {
      auto& info = output.second.Info();
      auto tensor = tcim::Tensor::CreateHostTensor(info);
      output_datas.insert(std::pair<std::string, tcim::Tensor>(output.first, tensor));
    }
    task.data_out = output_datas;
    qin.queue.push(task);
  }
  std::cout << "sample queue size is " << qin.queue.size() << std::endl;

  auto thread_func = [](int tid,
                        int did,
                        ThreadInfo& info,
                        TaskQueue& qin,
                        TaskQueue& qout,
                        Barrier& barrier) {
    auto start = GET_TIME();
    auto end = GET_TIME();
    float cost = 0.0;
    // load model
    start = GET_TIME();
    auto option = tcim::Module::Option(info.weight_manager);
    auto module = tcim::Module::LoadFromFile(info.model_path, option);
    end = GET_TIME();
    cost = GET_COST(start, end) / 1000.0 / info.warm_up;
    if (!module) {
      std::cerr << COLOR_RED << "Device " << did << " Thread " << tid << " load model "
                << info.model_path << " fail." << COLOR_RESET << std::endl;
      exit(-1);
    }
    std::cout << "Device " << did << " Thread " << tid << " " << info.model_path 
              << " model loaded. Cost " << cost << " ms." << std::endl;

    // create a stream and set to the module
    tcim::Stream stream;
    module.SetStream(stream);

    // warm up
    start = GET_TIME();
    for (int i = 0; i < info.warm_up; i++) {
      module.Run(false);
    }
    module.Sync();
    end = GET_TIME();
    cost = GET_COST(start, end) / 1000.0 / info.warm_up;
    std::cout << "Device " << did << " Thread " << tid << " Warm Up " << info.warm_up
              << " average cost " << cost << " ms." << std::endl;

    // wait until all threads ready
    barrier.barrier();
    std::cout << "Device " << did << " Thread " << tid << " infer start..." << std::endl;
    int count = 0;

    while (true) {
      std::unique_lock<std::mutex> lock_in(qin.mutex);
      if (qin.queue.empty()) {
        lock_in.unlock();
        break;
      }
      auto task = qin.queue.front();
      qin.queue.pop();
      lock_in.unlock();

      start = GET_TIME();
      if (!info.infer_only) {
        for (auto& tensor : task.data_in) {
          module.SetInput(tensor.first, tensor.second);
        }
      }
      auto input_end = GET_TIME();
      cost = GET_COST(start, input_end);
      info.input_total_cost += cost;
      if (info.input_max_cost < cost) info.input_max_cost = cost;
      tcim::Module::RunOption run_option;
      run_option.Rounds(info.loop_num);
      module.Run(false, run_option);
      module.Sync();
      auto infer_end = GET_TIME();
      cost = GET_COST(input_end, infer_end);
      info.infer_total_cost += cost;
      if (info.infer_max_cost < cost) info.infer_max_cost = cost;

      if (!info.infer_only) {
        for (auto& tensor : task.data_out) {
          module.GetOutput(tensor.first, tensor.second);
        }
      }
      end = GET_TIME();
      cost = GET_COST(infer_end, end);
      info.output_total_cost += cost;
      if (info.output_max_cost < cost) info.output_max_cost = cost;

      cost = GET_COST(start, end);
      info.e2e_total_cost += cost;
      if (info.e2e_max_cost < cost) info.e2e_max_cost = cost;

      if (!info.infer_only) {
        std::unique_lock<std::mutex> lock_out(qout.mutex);
        qout.queue.push(task);
        lock_out.unlock();
      }
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
  for (int did = 0; did < device_num; did++) {
    auto weight_manager = tcim::Module::WeightManager::CreateWeightManager(did);
    for (int tid = 0; tid < thread_num; tid++) {
      ThreadInfo* info = &thread_info[did * thread_num + tid];
      info->model_path = model_path;
      info->weight_manager = weight_manager;
      info->loop_num = loop_num;
      info->infer_only = infer_only;
      info->warm_up = warm_up;
      int id = did * thread_num + tid;
      threads.push_back(std::thread(thread_func, id, did, std::ref(*info), std::ref(qin),
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
  for (auto & t: threads) {
    t.join();
  }

  // check result
  if (is_result_check) {
    bool result = true;
    while (!qout.queue.empty()) {
      auto task = qout.queue.front();
      qout.queue.pop();

      for (auto& output : task.data_out) {
        auto data1 = (char*)output.second.Data();
        auto data2 = (char*)task.ref_out.at(output.first).Data();
        int len = output.second.Info().Size();
        if (memcmp(data1, data2, len)) {
          int err = 0;
          for (int i = 0; i < len; i++) {
            if (data1[i] != data2[i]) {
              err++;
            }
          }
          std::cout << COLOR_RED << "[error] req: " << task.req_id << ", output: "
                    << output.first << " result check failed (" << len - err << "/" << len << ")"
                    << COLOR_RESET << std::endl;
          std::stringstream ss;
          ss << output_path << "/result_" << task.req_id << '_' << output.first << ".bin";
          std::cout << "Save result to: " << ss.str() << std::endl;
          write_file(ss.str().c_str(), data1, len);
          result = false;
        }
      }
    }
    if (result) {
      std::cout << COLOR_GREEN << "Result check passed." << COLOR_RESET << std::endl;
    }
  } else {
    std::cout << COLOR_YELLOW << "[warn] Result check skipped." << COLOR_RESET << std::endl;
  }

  uint32_t infer_max_cost = 0;
  uint32_t infer_total_cost = 0;
  uint32_t input_max_cost = 0;
  uint32_t input_total_cost = 0;
  uint32_t output_max_cost = 0;
  uint32_t output_total_cost = 0;
  uint32_t e2e_max_cost = 0;
  uint32_t e2e_total_cost = 0;
  for (int i = 0; i < thread_num * device_num; i++) {
    if (thread_info[i].infer_max_cost > infer_max_cost) infer_max_cost = thread_info[i].infer_max_cost;
    infer_total_cost += thread_info[i].infer_total_cost;
    if (thread_info[i].input_max_cost > input_max_cost) input_max_cost = thread_info[i].input_max_cost;
    input_total_cost += thread_info[i].input_total_cost;
    if (thread_info[i].output_max_cost > output_max_cost) output_max_cost = thread_info[i].output_max_cost;
    output_total_cost += thread_info[i].output_total_cost;
    if (thread_info[i].e2e_max_cost > e2e_max_cost) e2e_max_cost = thread_info[i].e2e_max_cost;
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
  float qps = (1000.0 / (total_cost / test_num)) * batch;

  std::cout << COLOR_CYAN << std::fixed << std::setprecision(3) << "[latency] Inference "
            << "\tavg: " << std::setw(7) << infer_avg_latency << " ms,"
            << "\tmax: " << std::setw(7) << infer_max_latency << " ms"
            << COLOR_RESET  << std::endl;
  std::cout << COLOR_CYAN << std::fixed << std::setprecision(3) << "[latency] Input "
            << "\tavg: " << std::setw(7) << input_avg_latency << " ms,"
            << "\tmax: " << std::setw(7) << input_max_latency << " ms"
            << COLOR_RESET  << std::endl;
  std::cout << COLOR_CYAN << std::fixed << std::setprecision(3) << "[latency] Output "
            << "\tavg: " << std::setw(7) << output_avg_latency << " ms,"
            << "\tmax: " << std::setw(7) << output_max_latency << " ms"
            << COLOR_RESET  << std::endl;
  std::cout << COLOR_CYAN << std::fixed << std::setprecision(3) << "[latency] End2End "
            << "\tavg: " << std::setw(7) << e2e_avg_latency << " ms,"
            << "\tmax: " << std::setw(7) << e2e_max_latency << " ms"
            << COLOR_RESET << std::endl;
  std::cout << COLOR_MAGENT << std::fixed << std::setprecision(3) << "[Throughput] total: "
            << total_cost << " ms, "
            << "avg: " << avg_cost << " ms"
            << COLOR_RESET << std::endl;
  std::cout << COLOR_MAGENT << std::fixed << std::setprecision(3) << "[Throughput] qps: "
            << qps
            << COLOR_RESET << std::endl;

  if (output_path.size() != 0) {
    std::string output_file = output_path + "/hmperf.txt";
    std::cout << "Save result to: " << output_file << std::endl;
    std::fstream result_file(output_file.c_str(), std::ios::out);
    result_file << "batch: " << batch << std::endl;
    result_file << "thread_num: " << thread_num << std::endl;
    result_file << "shape: [";
    for (auto& input : input_datas) {
      result_file << input.second.Info().Shape() << ",";
    }
    result_file << "]" << std::endl;
    result_file << "loop_num: " << loop_num << std::endl;
    result_file << "sample_num: " << sample_num << std::endl;
    result_file << "avg_latency: " << infer_avg_latency << std::endl;
    result_file << "max_latency: " << infer_max_latency << std::endl;
    result_file << "qps: " << qps << std::endl;
    result_file.close();
  }

  return 0;
}
