// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file hdpl_classify.cc
 */
#include <getopt.h>
#include <tvm/runtime/executor_info.h>
#include <tvm/runtime/hdpl/hdpl_runtime.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "hdpl/hdpl_runtime.h"

struct ClassificationArguments {
  std::string model_path;
  std::string label_file_path;
  std::string data_root_path;
  size_t count = 10;
};

/**
 * @brief Parse cmdline arguments to struct *arguments
 *
 * @param arguments pointer to output CliArguments struct
 * @param argc cmdline argument count
 * @param argv cmdline argument char*
 * @return true parse command line succeed
 * @return false parse command line failed
 */
bool ParseArgs(ClassificationArguments *arguments, int argc, char *argv[]);

/**
 * @brief Load imagenet label file
 *
 * @param label_map  loaded map of image and label, (image_file_name, label)
 * @param file_names loaded image file vector
 * @param label_file_path label file path
 * @param max_count the count of images to be loaded
 * @return true load succeed
 * @return false load failed
 */
bool LoadLabelFile(std::unordered_map<std::string, int> *label_map,
                   std::vector<std::string> *file_names,
                   const std::string &label_file_path, size_t max_count);

/**
 * @brief Load the preproccessed data file to input_map
 *
 * @param input_map output map of image name and preproccessed data
 * @param input_files data file vector
 * @param input_file_root_path preprocessed data root path
 * @return true loaded succeed
 * @return false loaded failed
 */
bool PreloadInputFile(
    std::unordered_map<std::string, tvm::runtime::NDArray> *input_map,
    const std::vector<std::string> &input_files,
    const std::string &input_file_root_path);

struct InputInfo {
  std::string input_name;
  std::vector<int64_t> input_shape;
  tvm::DataType input_dtype;
  size_t one_data_size;
};

/**
 * @brief Get the input name of model
 *
 * @param input_info output input information struct
 * @param module the houmo model pointer
 * @return true get input info succeed
 * @return false there is not only 1 input or no input
 */
bool HDPLRuntime_GetInputInfo(InputInfo *input_info, tvm::hdpl::Module *module);

/**
 * @brief Set input of the model, all data listed in file_names will be combined
 * to one vm::runtime::NDArray and set to tensor named input_name
 *
 * @param module the houmo model pointer
 * @param input_name the input name of the model
 * @param file_names the current input file name vector
 * @param input_map the image file name and data map
 * @param input_info the input info of the model
 */
void HDPLRuntime_SetParam(
    tvm::hdpl::Module *module, const std::string &input_name,
    const std::vector<std::string> &file_names,
    const std::unordered_map<std::string, tvm::runtime::NDArray> &input_map,
    const InputInfo &input_info);

/**
 * @brief post process of classification model
 *
 * @param topk_results topk result index vector
 * @param topk_values topk result value vector
 * @param module the houmo model pointer
 * @param topk the topk
 */
void HDPLRuntime_CheckResult(std::vector<std::vector<int>> *topk_results,
                             std::vector<std::vector<int>> *topk_values,
                             tvm::hdpl::Module *module, const int topk);

int main(int argc, char *argv[]) {
  ClassificationArguments arguments;
  if (!ParseArgs(&arguments, argc, argv)) {
    LOG(ERROR) << "Parse arguments failed";
    return -1;
  }
  tvm::hdpl::Module module = tvm::hdpl::LoadModelPackage(arguments.model_path);
  InputInfo input_info;
  if (!HDPLRuntime_GetInputInfo(&input_info, &module)) {
    LOG(ERROR) << "Get input info of the model failed";
    return -1;
  }
  int batch = input_info.input_shape[0];

  std::unordered_map<std::string, int> file_label_map;
  std::vector<std::string> file_names;
  if (!LoadLabelFile(&file_label_map, &file_names, arguments.label_file_path,
                     arguments.count)) {
    LOG(ERROR) << "Load label file " << arguments.label_file_path << " failed";
    return -1;
  }

  std::unordered_map<std::string, tvm::runtime::NDArray> input_map;
  if (!PreloadInputFile(&input_map, file_names, arguments.data_root_path)) {
    LOG(ERROR) << "Load data file failed";
    return -1;
  }

  int total_data_count = 0;
  int good_data_count = 0;
  int good5_data_count = 0;
  std::vector<std::string> current_inputs;
  for (size_t file_idx = 0; file_idx < file_names.size(); file_idx++) {
    std::string input_file_name = file_names[file_idx];
    current_inputs.push_back(input_file_name);
    // If it is not enough data to combine a batch and not the last data
    if (current_inputs.size() < batch && file_idx != file_names.size() - 1) {
      continue;
    }
    HDPLRuntime_SetParam(&module, input_info.input_name, current_inputs,
                         input_map, input_info);
    module.Run();
    hdplDeviceSynchronize();

    std::vector<std::vector<int>> top5;
    std::vector<std::vector<int>> top5_values;
    HDPLRuntime_CheckResult(&top5, &top5_values, &module, 5);
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

bool HDPLRuntime_GetInputInfo(InputInfo *input_info,
                              tvm::hdpl::Module *module) {
  int tvm_input_count = module->GetInputNum();
  bool found_input = false;
  for (int idx = 0; idx < tvm_input_count; idx++) {
    if (!module->IsParams(idx)) {
      if (found_input) {
        LOG(ERROR) << "Only one input supported in classification, but more "
                      "then one input found in the model. inputs: ("
                   << input_info->input_name << ", "
                   << module->GetInputNameByIndex(idx) << ")";
        return false;
      }
      input_info->input_name = module->GetInputNameByIndex(idx);
      found_input = true;
    }
  }
  if (!found_input) {
    LOG(ERROR) << "None input found in the model.";
    return false;
  }
  tvm::runtime::NDArray input_data =
      module->GetInputByName(input_info->input_name);
  auto input_shape = input_data.Shape();
  input_info->input_shape.clear();
  for (size_t idx = 0; idx < input_shape.size(); ++idx) {
    input_info->input_shape.push_back(input_shape[idx]);
  }
  input_info->input_dtype = input_data.DataType();
  size_t one_data_size = input_info->input_dtype.bytes();
  for (size_t dim_idx = 1; dim_idx < input_info->input_shape.size();
       dim_idx++) {
    one_data_size *= input_info->input_shape[dim_idx];
  }
  input_info->one_data_size = one_data_size;
  return true;
}

void HDPLRuntime_SetParam(
    tvm::hdpl::Module *module, const std::string &input_name,
    const std::vector<std::string> &file_names,
    const std::unordered_map<std::string, tvm::runtime::NDArray> &input_map,
    const InputInfo &input_info) {
  tvm::runtime::NDArray input_array = tvm::runtime::NDArray::Empty(
      input_info.input_shape, input_info.input_dtype, {kDLCPU, 0});
  size_t data_size = input_info.one_data_size;
  for (size_t input_idx = 0; input_idx < file_names.size(); input_idx++) {
    std::string input_file_name = file_names[input_idx];
    tvm::runtime::NDArray input_part = input_map.at(input_file_name);
    memcpy(input_array->data + input_idx * data_size, input_part->data,
           data_size);
  }
  module->SetInput(input_name, input_array);
}

void HDPLRuntime_CheckResult(std::vector<std::vector<int>> *topk_results,
                             std::vector<std::vector<int>> *topk_values,
                             tvm::hdpl::Module *module, const int topk) {
  auto output = module->GetOutput(0);
  auto output_shape = output.Shape();
  int batch = output_shape[0];
  int n_ele = 1;
  for (size_t dim_idx = 1; dim_idx < output_shape.size(); dim_idx++) {
    n_ele *= output_shape[dim_idx];
  }
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

bool ParseArgs(ClassificationArguments *arguments, int argc, char *argv[]) {
  int option_idx = 0;
  struct option long_options[] = {
      {"help", 0, 0, 'h'},      {"model", 1, 0, 'm'}, {"label", 1, 0, 'l'},
      {"data_root", 1, 0, 'd'}, {"count", 1, 0, 'c'},
  };
  while (true) {
    int ch = getopt_long(argc, argv, "hm:l:d:c:", long_options, &option_idx);
    if (ch == -1) {
      break;
    }
    switch (ch) {
    case 'h':
      std::cout << "Usage: -h" << std::endl;
      break;
    case 'm':
      std::cout << "Model path: " << optarg << std::endl;
      arguments->model_path = std::string(optarg);
      break;
    case 'l':
      std::cout << "Label path: " << optarg << std::endl;
      arguments->label_file_path = std::string(optarg);
      break;
    case 'd':
      std::cout << "Data root path: " << optarg << std::endl;
      arguments->data_root_path = std::string(optarg);
      break;
    case 'c':
      std::cout << "Count: " << optarg << std::endl;
      arguments->count = atoi(optarg);
      break;
    default:
      std::cerr << "Unsupported option: " << static_cast<char>(ch) << std::endl;
      return false;
    }
  }
  return true;
}
