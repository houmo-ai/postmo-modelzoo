// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file hdpl_resnet50_run.cc
 */
#include <tvm/runtime/executor_info.h>
#include <tvm/runtime/hdpl/hdpl_runtime.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "hdpl/hdpl_runtime.h"

// using namespace tvm;
// using namespace tvm::relay;

static tvm::hdpl::Module module =
    tvm::hdpl::LoadModelPackage("../compile_model/resnet50");

bool LoadLabelFile(std::unordered_map<std::string, int>* label_map,
                   std::vector<std::string>* file_names,
                   const std::string& label_file_path,
                   size_t max_count) {
  std::string line;
  std::ifstream inFile(label_file_path, std::ios::in);
  if (!inFile.is_open()) {
    LOG(ERROR) << "Error open label file: " << label_file_path;
    return false;
  }
  size_t count = 0;
  while (std::getline(inFile, line)) {
    if (count >= max_count) {
      break;
    }
    size_t pos = line.find(' ', 0);
    if (pos == line.length()) {
      LOG(ERROR) << "Incorrect line \"" << line << "\" in file " << label_file_path;
      return false;
    }
    std::string file_name = line.substr(0, pos);
    std::string label = line.substr(pos + 1, line.length() - pos - 1);
    try {
      (*label_map)[file_name] = std::stoi(label);
      file_names->push_back(file_name);
      count++;
    } catch (const std::invalid_argument& e) {
      LOG(ERROR) << "Incorrect line \"" << line << "\" in file " << label_file_path;
      return false;
    }
  }
  return true;
}

bool PreloadInputFile(std::unordered_map<std::string, tvm::runtime::NDArray>* input_map,
                      const std::vector<std::string>& input_files,
                      const std::string& input_file_root_path) {
  for (std::string input_file_name : input_files) {
    std::string input_file_path = input_file_root_path + "/" + input_file_name;
    std::ifstream inFile(input_file_path, std::ios::in | std::ios::binary);
    tvm::runtime::NDArray image = tvm::runtime::NDArray::Empty(
        {1, 224, 224, 3}, tvm::DataType::Int(8), {kDLCPU, 0});
    int file_size = 1 * 224 * 224 * 3;
    int offset = 0;
    while (inFile.read(reinterpret_cast<char*>(image->data) + offset, file_size)) {
      int readedBytes = inFile.gcount();
      offset += readedBytes;
    }
    if (offset != file_size) {
      LOG(ERROR) << "The size of file " << input_file_path << " is not expected. "
                 << offset << " vs " << file_size;
      return false;
    }
    (*input_map)[input_file_name] = image;
  }
  LOG(INFO) << "Loaded " << input_map->size() << " files.";
  return true;
}

void HDPLRuntime_SetParam(tvm::runtime::NDArray image) {
  module.SetInput("input.1", image);
}

void HDPLRuntime_Run(int n) {
  module.RunRounds(n);
  hdplDeviceSynchronize();
}

void HDPLRuntime_CheckResult(std::vector<int>* topk_result,
                             std::vector<int>* topk_values) {
  auto output = module.GetOutput(0);
  int n_ele = 1000;
  auto cpu_tensor = const_cast<DLTensor*>(output.operator->());
  std::vector<std::pair<int, int>> sort_pairs;
  for (int i = 0; i < n_ele; ++i) {
    sort_pairs.emplace_back((static_cast<int8_t*>(cpu_tensor->data))[i], i);
  }
  std::stable_sort(sort_pairs.begin(), sort_pairs.end(),
                   [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                     return a.first > b.first;
                   });
  const int topk = 5;
  for (int i = 0; i < topk; ++i) {
    topk_values->push_back(sort_pairs[i].first);
    topk_result->push_back(sort_pairs[i].second);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <label file path>"
              << " <input file root path>"
              << " <count>" << std::endl
              << "Example: " << argv[0] << " /nfsdata/datasets/imagenet/val_map.txt"
              << " ./preprocessed"
              << " 10" << std::endl;
    return 1;
  }
  std::unordered_map<std::string, int> file_lable_map;
  std::vector<std::string> file_names;
  size_t count = std::atoi(argv[3]);
  std::string label_file_path(argv[1]);
  std::string input_file_root_path(argv[2]);
  if (!LoadLabelFile(&file_lable_map, &file_names, label_file_path, count)) {
    LOG(ERROR) << "Load label file " << label_file_path << " failed";
    return 1;
  }

  int total_data_count = 0;
  int good_data_count = 0;
  int good5_data_count = 0;
  std::unordered_map<std::string, tvm::runtime::NDArray> input_map;
  PreloadInputFile(&input_map, file_names, input_file_root_path);
  int eval_round = 1;
  for (std::string& input_file_name : file_names) {
    HDPLRuntime_SetParam(input_map[input_file_name]);
    auto start = std::chrono::system_clock::now();
    HDPLRuntime_Run(eval_round);

    auto finish = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    auto total_time = (duration.count());
#if 0
    LOG(INFO) << "\033[0;31mResNet50 time cost total = "
              << (total_time) << "us" << "\033[0m";
    LOG(INFO) << "\033[0;31mResNet50 time cost per image = "
              << (total_time / eval_round) << "us" << "\033[0m";
    LOG(INFO) << "\033[0;32mAverage Throughput(QPS): "
              << (1000000 / (duration.count() / eval_round)) * 4 << "fps" << "\033[0m";
#endif
    int expected_label = file_lable_map[input_file_name];
    std::vector<int> top5;
    std::vector<int> top5_values;
    HDPLRuntime_CheckResult(&top5, &top5_values);
    if (top5[0] == expected_label) {
      good_data_count++;
    }
    if (std::find(top5.begin(), top5.end(), expected_label) != top5.end()) {
      good5_data_count++;
    }
    total_data_count++;
    LOG(INFO) << "Image: " << input_file_name << ", label: " << expected_label
              << ", predict: [" << top5[0] << "," << top5[1] << "," << top5[2] << ","
              << top5[3] << "," << top5[4] << "]"
              << ", values: [" << top5_values[0] << "," << top5_values[1] << ","
              << top5_values[2] << "," << top5_values[3] << "," << top5_values[4] << "]";
  }
  std::cout << "Accuracy:  top1: " << 100.0 * good_data_count / total_data_count
            << "%, top5: " << 100.0 * good5_data_count / total_data_count << "%"
            << std::endl;
  return 0;
}
