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
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "hdpl/hdpl_runtime.h"

struct CliArguments {
  std::string model_path;
  std::string output;
  size_t warm_up = 10;
  size_t iterations = 10;
  size_t threads = 1;
  size_t batch = 1;
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
      {"help", 0, 0, 'h'},       {"model", 1, 0, 'm'},  {"warm_up", 1, 0, 'w'},
      {"iterations", 1, 0, 'i'}, {"output", 1, 0, 'o'}, {"batch", 1, 0, 'b'},
      {"threads", 1, 0, 't'}};
  while (true) {
    int ch =
        getopt_long(argc, argv, "hm:w:i:o:b:t:", long_options, &option_idx);
    if (ch == -1) {
      break;
    }
    switch (ch) {
      case 'h':
        std::cout << "Usage: -h" << std::endl;
        return false;
        break;
      case 'm':
        std::cout << "Model: " << optarg << std::endl;
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
      case 'b':
        std::cout << "batch size: " << optarg << std::endl;
        arguments->batch = atoi(optarg);
        break;
      case 'o':
        std::cout << "output: " << optarg << std::endl;
        arguments->output = optarg;
        break;
      case 't':
        std::cout << "threads: " << optarg << std::endl;
        arguments->threads = atoi(optarg);
        break;
      default:
        std::cerr << "Unsupported option: " << static_cast<char>(ch)
                  << std::endl;
        return false;
    }
  }
  return true;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, std::vector<T> vec) {
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

int main(int argc, char *argv[]) {
  CliArguments arguments;
  if (!ParseArgs(&arguments, argc, argv)) {
    return -1;
  }
  std::cout << "Model: " << arguments.model_path << std::endl;
  tvm::hdpl::Module module =
      tvm::hdpl::LoadModelPackage(arguments.model_path, "aot");

  // create streams
  std::vector<hdplStream_t> streams;
  size_t device_count = 4;
  for (size_t i = 0; i < device_count; i++) {
    hdplStream_t s;
    if (hdplSuccess != hdplStreamCreate(&s)) {
      std::cerr << "Create " << i << "th stream failed." << std::endl;
      return -1;
    }
    streams.push_back(s);
  }

  module.SetStream(streams[0]);
  std::map<std::string, std::vector<size_t>> input_shapes;
  std::map<std::string, tvm::runtime::NDArray> input_datas;
  size_t batch = arguments.batch;
  int tvm_input_count = module.GetInputNum();
  std::cout << "Count of Input: " << tvm_input_count << std::endl;
  for (int input_idx = 0; input_idx < tvm_input_count; input_idx++) {
    std::string input_name = module.GetInputNameByIndex(input_idx);
    tvm::runtime::NDArray input_data = module.GetInputByName(input_name);
    std::vector<size_t> v_input_shape;
    std::cout << "Input " << input_name << ": (";
    auto input_shape = input_data.Shape();
    for (size_t shape_idx = 0; shape_idx < input_shape.size(); shape_idx++) {
      if (shape_idx != 0) {
        std::cout << ", ";
        v_input_shape.push_back(input_shape.data()[shape_idx]);
      } else {
        v_input_shape.push_back(batch);
      }
      std::cout << v_input_shape[shape_idx];
    }
    std::cout << "), " << input_data.DataType() << std::endl;
    input_shapes[input_name] = v_input_shape;
    input_datas[input_name] = input_data;
  }
  std::vector<tvm::hdpl::Module> modules;
  modules.push_back(module);
  for (size_t dev_id = 1; dev_id < device_count; dev_id++) {
    auto module1 = tvm::hdpl::LoadModelPackage(arguments.model_path, "aot");
    module1.SetStream(streams[dev_id]);
    modules.push_back(module1);
  }
  size_t eval_round = batch * arguments.iterations;
  auto start = std::chrono::system_clock::now();
  for (size_t img_idx = 0; img_idx < eval_round; img_idx++) {
    auto a_module = modules[img_idx % device_count];
    a_module.Run();
  }
  hdplDeviceSynchronize();
  auto finish = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  for (size_t i = 0; i < device_count; i++) {
    hdplStreamDestroy(streams[i]);
  }

  int64_t total_time = duration.count();
  std::cout << "\033[0;31mInference time cost total = " << total_time << "us"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;31mInference time cost per frame = " << std::fixed
            << std::setprecision(1) << 1.0 * total_time / eval_round << "us"
            << "\033[0m" << std::endl;
  float qps = 1.0e6 * eval_round / total_time;
  std::cout << "\033[0;32mAverage Throughput(QPS): " << std::fixed
            << std::setprecision(2) << qps << " qps"
            << "\033[0m" << std::endl;
  if (arguments.output.size() != 0) {
    std::cout << "Save result to: " << arguments.output << std::endl;
    std::fstream result_file(arguments.output.c_str(), std::ios::out);
    result_file << "{" << std::endl;
    result_file << "  \"batch\": " << batch << "," << std::endl;
    //    result_file << "  \"shape\": " << input_shapes << "," << std::endl;
    result_file << "  \"iterations\": " << eval_round << "," << std::endl;
    result_file << "  \"qps\": " << qps << std::endl;
    result_file << "}" << std::endl;
    result_file.close();
  }
  return 0;
}
