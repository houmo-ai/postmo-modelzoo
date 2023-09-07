// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file main.cc
 */
#include <getopt.h>
#include <stdio.h>
#include <tvm/runtime/executor_info.h>
#include <tvm/runtime/hdpl/hdpl_runtime.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "hdpl/hdpl_runtime.h"
#include "get_time.h"

struct CliArguments {
  std::string model_path;
  size_t warm_up = 10;
  size_t iterations = 10;
  size_t thread = 1;
  size_t stream = 1;
  bool is_fused = false;
};

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
      {"warm_up", 1, 0, 'w'},
      {"iterations", 1, 0, 'i'},
      {"thread", 1, 0, 't'},
      {"stream", 1, 0, 's'},
  };
  while (true) {
    int ch = getopt_long(argc, argv, "hm:w:i:t:s:", long_options, &option_idx);
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
    case 'w':
      std::cout << "warm up: " << optarg << std::endl;
      arguments->warm_up = atoi(optarg);
      break;
    case 'i':
      std::cout << "iterations: " << optarg << std::endl;
      arguments->iterations = atoi(optarg);
      break;
    case 't':
      arguments->thread = atoi(optarg);
      break;
    case 's':
      arguments->stream = atoi(optarg);
      break;
    default:
      std::cerr << "Unsupported option: " << static_cast<char>(ch) << std::endl;
      return false;
    }
  }
  if (IsFileExists(arguments->model_path + "_fused_op.so")) {
    arguments->is_fused = true;
  } else {
    arguments->is_fused = false;
  }
  return true;
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

std::map<std::string, tvm::runtime::NDArray> user_datas;

void SetInput(tvm::hdpl::Module &module, std::vector<std::string> &input_names) {
  for (auto input_name : input_names) {
    // tvm::runtime::NDArray input_data = module.GetInputByName(input_name);
    // TODO: copy your data if you want to use a real one
    module.SetInput(input_name, user_datas[input_name]);
  }
}

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
  std::cout << "thread: " << arguments.thread << std::endl;
  std::cout << "stream: " << arguments.stream << std::endl;

  // multi_stream
  int stream_num = arguments.stream;
  int thread_num = arguments.thread;
  std::vector<hdplStream_t> streams;
  std::vector<std::thread> threads;
  std::vector<tvm::hdpl::Module> modules;
  std::vector<std::vector<std::string>> input_names(thread_num);
  size_t batch = 1;

  // create streams
  for (int i = 0; i < stream_num; i++) {
    hdplStream_t s;
    auto t0 = GetTime::Now();
    auto r = hdplStreamCreate(&s);
    CHECK(r == hdplSuccess);
    streams.push_back(s);
    auto t1 = GetTime::Now();
    printf("===> hdplStreamCreate %d:%x, cost %fms...\n", i, s, GetTime::DurationMs(t0, t1)); 
  }

  //SetAffinity(5);

  // create modules and set stream
  for (int i = 0; i < thread_num; i++) {
    auto t0 = GetTime::Now();
    modules.push_back(tvm::hdpl::LoadModelPackage(arguments.model_path));
    auto t1 = GetTime::Now();
    printf("===> module %x created on thread %d, cost %fms...\n", modules[i], i, GetTime::DurationMs(t0, t1));

    modules[i].SetStream(streams[i]);
    auto t2 = GetTime::Now();
    printf("===> module %x set to stream %x, cost %fms...\n", modules[i], streams[i], GetTime::DurationMs(t1, t2));

    int tvm_input_count = modules[i].GetInputNum();
    for (int idx = 0; idx < tvm_input_count; idx++) {
      if (!modules[i].IsParams(idx)) {
        std::string input_name = modules[i].GetInputNameByIndex(idx);
        input_names[i].push_back(input_name);
      }
    }
    std::cout << "Count of Input: " << input_names[i].size() << std::endl;
    for (auto input_name : input_names[i]) {
      tvm::runtime::NDArray input_data = modules[i].GetInputByName(input_name);
      auto data = tvm::runtime::NDArray::Empty(
        input_data.Shape(), input_data.DataType(), {kDLCPU, 0});
      user_datas[input_name] = data;

      std::cout << "Input " << input_name << ": (";
      auto input_shape = input_data.Shape();
      for (size_t shape_idx = 0; shape_idx < input_shape.size(); shape_idx++) {
        if (shape_idx != 0) {
          std::cout << ", ";
        } else {
          batch = input_shape.data()[0];
        }
        std::cout << input_shape.data()[shape_idx];
      }
      std::cout << "), " << input_data.DataType() << std::endl;
    }
    // modules[i].Run();
    int tvm_output_count = modules[i].GetOutputNum();
    std::cout << "Count of Output: " << tvm_output_count << std::endl;
    for (int idx = 0; idx < tvm_output_count; idx++) {
      auto output_name = modules[i].GetOutputNameByIndex(idx);
      tvm::runtime::NDArray output_data = modules[i].GetOutputByName(output_name);
      std::cout << "Output " << output_name << ": (";
      auto output_shape = output_data.Shape();
      for (size_t shape_idx = 0; shape_idx < output_shape.size(); shape_idx++) {
        if (shape_idx != 0) {
          std::cout << ", ";
        }
        std::cout << output_shape.data()[shape_idx];
      }
      std::cout << "), " << output_data.DataType() << std::endl;
    }
  }

  Barrier barrier(thread_num);
  // create threads
  for (int i = 0; i < thread_num; i++) {
    threads.push_back(
      std::thread([](int t_id, int loop, tvm::hdpl::Module &module, std::vector<std::string> &input_names,
        hdplStream_t &stream, Barrier &barrier) {
        // SetAffinity(t_id);
        barrier.barrier();
        printf("===> thread %d infer start...\n", t_id);
        auto start = GetTime::Now();
        for (int i = 0; i < loop; i++){
          auto t0 = GetTime::Now();
          SetInput(module, input_names);
          auto t1 = GetTime::Now();
          module.Run();
          auto t2 = GetTime::Now();
          auto result = module.GetOutput(0);
          auto t3 = GetTime::Now();
          auto d1 = GetTime::DurationMs(t0, t1);
          auto d2 = GetTime::DurationMs(t1, t2);
          auto d3 = GetTime::DurationMs(t2, t3);
          printf("===> thread %d module run %d times, setinput %.3fms, run %.3fms, getoutput %.3fms...\n", t_id, i+1, d1, d2, d3);
        }
        hdplStreamSynchronize(stream);
        auto finish = GetTime::Now();
        printf("===> thread %d infer %d times cost %.3fms\n", t_id, loop, GetTime::DurationMs(start, finish));
        barrier.barrier();
      },
      i,
      arguments.iterations,
      std::ref(modules[i]),
      std::ref(input_names[i]),
      std::ref(streams[i]),
      std::ref(barrier)));
  }

  barrier.wait();
  auto start = GetTime::Now();
  barrier.reset();
  barrier.wait();
  auto finish = GetTime::Now();

  for (auto & t: threads) {
    t.join();
  }
  
  for (int i = 0; i < stream_num; i++) {
    printf("===> hdplStreamDestroy stream %x...\n", streams[i]);
    hdplStreamDestroy(streams[i]);
  }
  printf("===> hdplStreamDestroy end...\n"); 

  auto total_time = GetTime::DurationMs(start, finish);

  std::cout << "\033[0;31mInference time cost total = " << (total_time) << "ms"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;31mInference time cost per frame = "
            << (total_time / arguments.iterations) << "ms"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Throughput(QPS): "
            << (1000.0 / (total_time / arguments.iterations)) * stream_num * batch << "fps"
            << "\033[0m" << std::endl;

  return 0;
}
