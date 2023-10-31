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

#include "hdpl/hdpl_runtime.h"

struct CliArguments {
  std::string model_path;
  size_t warm_up = 10;
  size_t iterations = 10;
  bool is_fused = false;
  bool host_loop = false;
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
      {"help", 0, 0, 'h'},      {"model", 1, 0, 'm'},
      {"warm_up", 1, 0, 'w'},   {"iterations", 1, 0, 'i'},
      {"host_loop", 0, 0, 'c'},
  };
  while (true) {
    int ch = getopt_long(argc, argv, "hm:w:i:c", long_options, &option_idx);
    if (ch == -1) {
      break;
    }
    switch (ch) {
      case 'h':
        std::cout << "Usage: -h" << std::endl;
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
      case 'c':
        std::cout << "cpu loop set " << std::endl;
        arguments->host_loop = true;
        break;
      default:
        std::cerr << "Unsupported option: " << static_cast<char>(ch)
                  << std::endl;
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

int main(int argc, char *argv[]) {
  CliArguments arguments;
  ParseArgs(&arguments, argc, argv);
  std::cout << "Model: " << arguments.model_path << std::endl;
  tvm::hdpl::Module module = tvm::hdpl::LoadModelPackage(arguments.model_path);
  int tvm_input_count = module.GetInputNum();
  std::vector<int> input_idx_vec;
  for (int idx = 0; idx < tvm_input_count; idx++) {
    if (!module.IsParams(idx)) {
      input_idx_vec.push_back(idx);
    }
  }
  std::cout << "Count of Input: " << input_idx_vec.size() << std::endl;
  size_t batch = 1;
  for (int input_idx : input_idx_vec) {
    std::string input_name = module.GetInputNameByIndex(input_idx);
    tvm::runtime::NDArray input_data = module.GetInputByName(input_name);
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
  size_t eval_round = arguments.iterations;
  auto start = std::chrono::system_clock::now();
  if (arguments.is_fused && !arguments.host_loop) {
    module.RunRounds(eval_round);
  } else {
    for (size_t idx = 0; idx < eval_round; idx++) {
      module.Run();
    }
  }
  hdplDeviceSynchronize();
  auto finish = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  int64_t total_time = duration.count();
  std::cout << "\033[0;31mInference time cost total = " << total_time << "us"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;31mInference time cost per frame = " << std::fixed
            << std::setprecision(1) << 1.0 * total_time / eval_round << "us"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Throughput(QPS): " << std::fixed
            << std::setprecision(2) << (1.0e6 * eval_round / total_time * batch)
            << "fps"
            << "\033[0m" << std::endl;
  return 0;
}
