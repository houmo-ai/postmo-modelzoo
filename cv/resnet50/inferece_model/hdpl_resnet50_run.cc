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

static tvm::hdpl::Module module =
    tvm::hdpl::LoadModelPackage("../compile_model/resnet50");

bool LoadLabelFile(std::unordered_map<std::string, int> *label_map,
                   std::vector<std::string> *file_names,
                   const std::string &label_file_path, size_t max_count) {
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
      LOG(ERROR) << "Incorrect line \"" << line << "\" in file "
                 << label_file_path;
      return false;
    }
    std::string file_name = line.substr(0, pos);
    std::string label = line.substr(pos + 1, line.length() - pos - 1);
    try {
      (*label_map)[file_name] = std::stoi(label);
      file_names->push_back(file_name);
      count++;
    } catch (const std::invalid_argument &e) {
      LOG(ERROR) << "Incorrect line \"" << line << "\" in file "
                 << label_file_path;
      return false;
    }
  }
  return true;
}

bool PreloadInputFile(
    std::unordered_map<std::string, tvm::runtime::NDArray> *input_map,
    const std::vector<std::string> &input_files,
    const std::string &input_file_root_path) {
  for (std::string input_file_name : input_files) {
    std::string input_file_path = input_file_root_path + "/" + input_file_name;
    std::ifstream inFile(input_file_path, std::ios::in | std::ios::binary);
    tvm::runtime::NDArray image = tvm::runtime::NDArray::Empty(
        {1, 224, 224, 3}, tvm::DataType::Int(8), {kDLCPU, 0});
    int file_size = 1 * 224 * 224 * 3;
    int offset = 0;
    while (inFile.read(reinterpret_cast<char *>(image->data) + offset,
                       file_size)) {
      int readedBytes = inFile.gcount();
      offset += readedBytes;
    }
    if (offset != file_size) {
      LOG(ERROR) << "The size of file " << input_file_path
                 << " is not expected. " << offset << " vs " << file_size;
      return false;
    }
    (*input_map)[input_file_name] = image;
  }
  LOG(INFO) << "Loaded " << input_map->size() << " files.";
  return true;
}

void HDPLRuntime_SetParam(
    std::vector<std::string> file_names,
    std::unordered_map<std::string, tvm::runtime::NDArray> input_map,
    int batch) {
  std::cout << "batch :" << batch << std::endl;
  tvm::runtime::NDArray input_array = tvm::runtime::NDArray::Empty(
      {batch, 224, 224, 3}, tvm::DataType::Int(8), {kDLCPU, 0});
  size_t data_size = 1 * 224 * 224 * 3;
  for (size_t input_idx = 0; input_idx < file_names.size(); input_idx++) {
    std::string input_file_name = file_names[input_idx];
    tvm::runtime::NDArray input_part = input_map[input_file_name];
    memcpy(input_array->data + input_idx * data_size, input_part->data,
           data_size);
  }
  module.SetInput("input.1", input_array);
}

void HDPLRuntime_Run(int n) {
  module.RunRounds(n);
  hdplDeviceSynchronize();
}

void HDPLRuntime_CheckResult(std::vector<std::vector<int>> *topk_results,
                             std::vector<std::vector<int>> *topk_values,
                             const int topk, int batch) {
  auto output = module.GetOutput(0);
  int n_ele = 1000;
  auto cpu_tensor = const_cast<DLTensor *>(output.operator->());
  for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
    std::vector<std::pair<int, int>> sort_pairs;
    std::vector<int> topk_idx;
    std::vector<int> topk_val;
    for (int i = 0; i < n_ele; ++i) {
      sort_pairs.emplace_back(
          (static_cast<int8_t *>(cpu_tensor->data))[i + n_ele * batch_idx], i);
    }
    std::stable_sort(
        sort_pairs.begin(), sort_pairs.end(),
        [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
          return a.first > b.first;
        });
    for (int i = 0; i < topk; ++i) {
      topk_val.push_back(sort_pairs[i].first);
      topk_idx.push_back(sort_pairs[i].second);
    }
    topk_results->push_back(topk_idx);
    topk_values->push_back(topk_val);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <label file path>"
              << " <input file root path>"
              << " <count>" << std::endl
              << "Example: " << argv[0]
              << " /nfsdata/datasets/imagenet/val_map.txt"
              << " ./preprocessed"
              << " 10" << std::endl;
    return 1;
  }
  int batch = 16;
  std::unordered_map<std::string, int> file_label_map;
  std::vector<std::string> file_names;
  size_t count = std::atoi(argv[3]);
  std::string label_file_path(argv[1]);
  std::string input_file_root_path(argv[2]);
  if (!LoadLabelFile(&file_label_map, &file_names, label_file_path, count)) {
    LOG(ERROR) << "Load label file " << label_file_path << " failed";
    return 1;
  }

  int total_data_count = 0;
  int good_data_count = 0;
  int good5_data_count = 0;
  std::unordered_map<std::string, tvm::runtime::NDArray> input_map;
  PreloadInputFile(&input_map, file_names, input_file_root_path);
  int eval_round = 1;
  int data_count = 0;
  std::vector<std::string> current_inputs;
  for (std::string &input_file_name : file_names) {
    current_inputs.push_back(input_file_name);
    if (current_inputs.size() < batch) {
      continue;
    }
    HDPLRuntime_SetParam(current_inputs, input_map, batch);
    HDPLRuntime_Run(eval_round);

    std::vector<std::vector<int>> top5;
    std::vector<std::vector<int>> top5_values;
    HDPLRuntime_CheckResult(&top5, &top5_values, 5, batch);
    for (size_t input_idx = 0; input_idx < current_inputs.size(); input_idx++) {
      int expected_label = file_label_map[current_inputs[input_idx]];
      if (top5[input_idx][0] == expected_label) {
        good_data_count++;
      }
      if (std::find(top5[input_idx].begin(), top5[input_idx].end(),
                    expected_label) != top5[input_idx].end()) {
        good5_data_count++;
      }
      total_data_count++;
      LOG(INFO) << "Image: " << current_inputs[input_idx]
                << ", label: " << expected_label << ", predict: ["
                << top5[input_idx][0] << "," << top5[input_idx][1] << ","
                << top5[input_idx][2] << "," << top5[input_idx][3] << ","
                << top5[input_idx][4] << "]"
                << ", values: [" << top5_values[input_idx][0] << ","
                << top5_values[input_idx][1] << "," << top5_values[input_idx][2]
                << "," << top5_values[input_idx][3] << ","
                << top5_values[input_idx][4] << "]";
    }
    current_inputs.clear();
  }
  std::cout << "Accuracy:  top1: " << 100.0 * good_data_count / total_data_count
            << "%, top5: " << 100.0 * good5_data_count / total_data_count << "%"
            << std::endl;
  return 0;
}
