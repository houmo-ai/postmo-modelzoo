// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file hdpl_resnet50_run.cc
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

int SetInput(tvm::hdpl::Module &module) {
  int batch = 1;
  int tvm_input_count = module.GetInputNum();
  std::vector<int> input_idx_vec;
  for (int idx = 0; idx < tvm_input_count; idx++) {
    if (!module.IsParams(idx)) {
      input_idx_vec.push_back(idx);
    }
  }
  std::cout << "Count of Input: " << input_idx_vec.size() << std::endl;
  for (int input_idx : input_idx_vec) {
    std::string input_name = module.GetInputNameByIndex(input_idx);
    tvm::runtime::NDArray input_data = module.GetInputByName(input_name);
    // TODO: copy your data if you want to use a real one 
    module.SetInput(input_name, input_data);
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
  return batch;
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
  std::vector<std::vector<high_resolution_clock::time_point>> timers(thread_num);
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
  
  // create modules and set stream
  for (int i = 0; i < thread_num; i++) {
    auto t0 = GetTime::Now();
    modules.push_back(tvm::hdpl::LoadModelPackage(arguments.model_path));
    auto t1 = GetTime::Now();
    printf("===> module %x created on thread %d, cost %fms...\n", modules[i], i, GetTime::DurationMs(t0, t1)); 

    modules[i].SetStream(streams[i]);
    auto t2 = GetTime::Now();
    printf("===> module %x set to stream %x, cost %fms...\n", modules[i], streams[i], GetTime::DurationMs(t1, t2)); 
    batch = SetInput(modules[i]);
    auto t3 = GetTime::Now();
    printf("===> module %x set input data, cost %fms...\n", modules[i], GetTime::DurationMs(t2, t3)); 
  }

  Barrier barrier(thread_num);
  // create threads
  for (int i = 0; i < thread_num; i++) {
    threads.push_back(
      std::thread([](int t_id, int loop, tvm::hdpl::Module &module, hdplStream_t &stream, Barrier &barrier) {
        barrier.barrier();
        printf("===> thread %d infer start...\n", t_id);
        auto start = GetTime::Now();
        for (int i = 0; i < loop; i++){
          module.Run();
          //printf("===> thread %d module run %d times...\n", t_id, i+1);
        }
        hdplStreamSynchronize(stream);
        auto finish = GetTime::Now();
        printf("===> thread %d infer %d times cost %fms\n", t_id, loop, GetTime::DurationMs(start, finish)); 
        barrier.barrier();
    },
    i,
    arguments.iterations,
    std::ref(modules[i]),
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
