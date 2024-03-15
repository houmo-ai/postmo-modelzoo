// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file main.cc
 */
#include <getopt.h>
#include <stdio.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "hdpl/hdpl_runtime_api.h"
#include "tcim/tcim_runtime.h"

#define GET_TIME() std::chrono::system_clock::now()
#define GET_COST(start, end) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()

struct CliArguments {
  std::string model_path;
  std::string data_path;
  std::string output;
  size_t batch = 1;
  size_t warm_up = 1;
  size_t threads = 1;
  size_t loops = 1;
  size_t samples = 1;
};

typedef struct {
  tcim::Module* module;
  int loop_num;
  uint32_t max_cost;
  uint32_t total_cost;
} ThreadInfo;

typedef struct {
  std::queue<std::map<std::string, tcim::Tensor>> queue;
  std::mutex mutex;
  std::condition_variable cond;
} QueueInfo;

std::mutex mutex;
std::condition_variable cond;

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
    printf("malloc fileData size:%ld fialed\n", fileSize);
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
      {"samples", 1, 0, 's'},
      {"output", 1, 0, 'o'}
  };
  while (true) {
    int ch = getopt_long(argc, argv, "hm:d:w:b:l:t:s:o:", long_options, &option_idx);
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
    case 'l':
      arguments->loops = atoi(optarg);
      break;
    case 's':
      arguments->samples = atoi(optarg);
      break;
    case 'o':
      arguments->output = optarg;
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
    cond_.wait(lock);
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    while (count_ < dest_) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      lock.lock();
    }
    cond_.notify_all();
  }

  void barrier_and_wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    count_++;
    while (count_ < dest_) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      lock.lock();
    }
  }
  
  void reset() {
    std::unique_lock<std::mutex> lock(mtx_);
    count_ = 0;
  }

 protected:
  int count_ = 0;
  int dest_ = 0;
  std::condition_variable cond_;
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

int main(int argc, char *argv[]) {
  CliArguments arguments;
  ParseArgs(&arguments, argc, argv);
  std::cout << "model: " << arguments.model_path << std::endl;
  std::cout << "data: " << arguments.data_path << std::endl;
  std::cout << "samples: " << arguments.samples << std::endl;
  std::cout << "loops: " << arguments.loops << std::endl;
  std::cout << "warmup: " << arguments.warm_up << std::endl;
  std::cout << "batch: " << arguments.batch << std::endl;
  std::cout << "threads: " << arguments.threads << std::endl;

  std::string model_path = arguments.model_path;
  std::string data_path = arguments.data_path;
  std::string output_file = arguments.output;
  int batch = arguments.batch;
  int sample_num = arguments.samples;
  int thread_num = arguments.threads;
  int loop_num = arguments.loops;
  int warm_up = arguments.warm_up;

  std::vector<std::thread> threads;
  std::vector<tcim::Module> modules;
  std::vector<std::map<std::string, tcim::Tensor>> input_datas;
  ThreadInfo thread_info[thread_num];

  char fileName[256];
  char *imageData = NULL;
  int fileLen = 0;
  // 640x384_422sp.yuv
  // read_file(arguments.data_path.c_str(), &imageData, &fileLen);
  // printf("file_len %d\n", fileLen);

  for (int i = 0; i < thread_num; i++) {
    // load model
    auto module = tcim::Module::LoadFromFile(model_path);
    if (!module) {
      std::cout << "load model " << model_path << " fail, exit..." << std::endl;
      exit(-1);
    }
    modules.push_back(std::move(module));
  }
  std::cout << "module loaded: " << model_path << std::endl;
  // auto size = modules[0].GetMemSize();
  // std::cout << "model memsize = " << size[0] << "," << size[1] << "," << size[2] << "," << size[3] << std::endl;

  //SetAffinity(5);

  QueueInfo queue_info;

  // prepare input data
  int input_num = modules[0].GetInputNum();
  std::cout << "Count of Input: " << input_num << std::endl;
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = modules[0].GetInputName(idx);
    tcim::TensorInfo input_info;
    modules[0].GetInputInfo(input_name, input_info, tcim::CPU);
    std::cout << "Input[" << idx << "] name: " << input_name << ", " << input_info << std::endl;
    for (int i = 0; i < thread_num; i++) {
      auto data = malloc(input_info.MemSize());
      tcim::Tensor input_data(input_info, data, input_info.MemSize());
      std::map<std::string, tcim::Tensor> tensor_map;
      tensor_map.insert(std::pair<std::string, tcim::Tensor>(input_name, input_data));
      input_datas.push_back(tensor_map);
    }
  }

  // prepare output data
  int output_num = modules[0].GetOutputNum();
  std::cout << "Count of Output: " << output_num << std::endl;
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = modules[0].GetOutputName(idx);
    tcim::TensorInfo output_info;
    modules[0].GetOutputInfo(output_name, output_info, tcim::CPU, true);
    auto data = malloc(output_info.MemSize());
    std::cout << "Output[" << idx << "] name: " << output_name << ", " << output_info << std::endl;
    tcim::Tensor output_data(output_info, data, output_info.MemSize());
#if SAVE_DATA
    snprintf(fileName, sizeof(fileName), "%s_output.bin", output_name.c_str());
    write_file(fileName, (char*)output_data->data, data_size);
#endif
  }

  for (int i = 0; i < thread_num; i++) {
    auto start = GET_TIME();
    modules[i].Run(false, warm_up);
    modules[i].Sync();
    auto end = GET_TIME();
    float cost = GET_COST(start, end) / 1000.0 / warm_up;
    std::cout << "Thread " << i << " Warm Up " << warm_up << " average cost " << cost << "ms" << std::endl;
  }

  for (int i = 0; i < sample_num; i++) {
    int j = i % thread_num;
    queue_info.queue.push(input_datas[j]);
  }
  std::cout << "request queue size is " << queue_info.queue.size() << std::endl;

  Barrier barrier(thread_num);

  auto thread_func = [](int tid, ThreadInfo& thread_info, QueueInfo& queue_info, Barrier& barrier) {
    // SetAffinity(t_id);
    barrier.barrier();
    printf("===> thread %d infer start...\n", tid);
    while (true) {
      std::unique_lock<std::mutex> lock(queue_info.mutex);
      if (queue_info.queue.empty()) {
        lock.unlock();
        break;
      }
      auto input_datas = queue_info.queue.front();
      queue_info.queue.pop();
      lock.unlock();
      void* stream;
      auto status = hdplStreamCreate(&stream);
      thread_info.module->SetStream(stream);
      for (auto& input_data : input_datas) {
        thread_info.module->SetInput(input_data.first, input_data.second);
      }
      auto start = GET_TIME();
      thread_info.module->Run(false, thread_info.loop_num);
      thread_info.module->Sync();
      hdplStreamDestroy(stream);
#if SAVE_DATA
      hdplStreamSynchronize(*stream);
      int tvm_output_count = module.GetOutputNum();
      for (int idx = 0; idx < tvm_output_count; idx++) {
        auto output_name = module.GetOutputNameByIndex(idx);
        tvm::runtime::NDArray output_data = module.GetOutputByName(output_name);
        // auto output_shape = output_data.Shape();
        // int data_size = 1;
        // for (size_t shape_idx = 0; shape_idx < output_shape.size(); shape_idx++) {
        //   data_size *= output_shape.data()[shape_idx];
        // }
        snprintf(fileName, sizeof(fileName), "thread_%d_run_%d_%s_output.bin", tid, i, output_name.c_str());
        write_file(fileName, (char*)output_data->data, datasize[idx]);
      }
#endif
      auto end = GET_TIME();
      auto cost = GET_COST(start, end);
      thread_info.total_cost += cost;
      if (thread_info.max_cost < cost) thread_info.max_cost = cost;
      printf("===> thread %d infer %d times cost %.3fms\n", tid, thread_info.loop_num, cost/1000.0);
    }
    // hdplStreamSynchronize(*stream);
    barrier.barrier();
  };

  // create threads
  for (int i = 0; i < thread_num; i++) {
    thread_info[i].module = &modules[i];
    thread_info[i].loop_num = loop_num;
    thread_info[i].max_cost = 0;
    thread_info[i].total_cost = 0;
    threads.push_back(std::thread(thread_func, i, std::ref(thread_info[i]), std::ref(queue_info), std::ref(barrier)));
  }

  barrier.wait();
  barrier.reset();
  auto start = GET_TIME();
  barrier.wait();
  auto end = GET_TIME();

  for (auto & t: threads) {
    t.join();
  }

  uint32_t max_cost = 0;
  uint32_t total_latency = 0;
  for (int i = 0; i < thread_num; i++) {
    if (thread_info[i].max_cost > max_cost) max_cost = thread_info[i].max_cost;
    total_latency += thread_info[i].total_cost;
  }

  int test_num = loop_num * sample_num;
  float avg_latency = total_latency / test_num / 1000.0;
  float max_latency = max_cost / loop_num / 1000.0;
  float total_cost = GET_COST(start, end) / 1000.0;
  float avg_cost = total_cost / test_num;
  float qps = (1000.0 / (total_cost / test_num)) * batch;

  std::cout << "\033[0;31mInference average latency: "
            << avg_latency << "ms" << "\033[0m" << std::endl;
  std::cout << "\033[0;31mInference max latency: "
            << max_latency << "ms" << "\033[0m" << std::endl;
  std::cout << "\033[0;31mInference Throughput total cost: "
            << total_cost << "ms" << "\033[0m" << std::endl;
  std::cout << "\033[0;31mInference Throughput average cost: "
            << avg_cost << "ms" << "\033[0m" << std::endl;
  std::cout << "\033[0;32mInference Throughput(QPS): "
            << qps << "\033[0m" << std::endl;

  if (output_file.size() != 0) {
    std::cout << "Save result to: " << output_file << std::endl;
    std::fstream result_file(output_file.c_str(), std::ios::out);
    result_file << "batch: " << batch << std::endl;
    result_file << "thread_num: " << thread_num << std::endl;
    result_file << "shape: [";
    for (auto it = input_datas[0].begin(); it != input_datas[0].end(); ++it) {
      result_file << it->second.Info().Shape() << ",";
    }
    result_file << "]" << std::endl;
    result_file << "loop_num: " << loop_num << std::endl;
    result_file << "sample_num: " << sample_num << std::endl;
    result_file << "avg_latency: " << avg_latency << std::endl;
    result_file << "max_latency: " << max_latency << std::endl;
    result_file << "qps: " << qps << std::endl;
    result_file.close();
  }

  return 0;
}
