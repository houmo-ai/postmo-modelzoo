/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: resnet50.cc
 * Description:
 *   ResNet50 Image Classification C++ Example.
 *   This file demonstrates how to use ResNet50 model for image classification
 *   tasks, including model loading, image preprocessing, inference execution,
 *   and result post-processing.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#if (__GNUC__ < 8 && !defined(_MSC_VER))
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "datasets/imagenet.hpp"
#include "logging.h"
#include "tcim/tcim_runtime.h"
#include "threads.hpp"
#include "utils.hpp"

/**
 * Get the top K maximum values and their index information
 * This function sorts the value-index pairs in descending order and prints the
 * top K results
 *
 * @param topk Number of top K elements to retrieve
 * @param sort_pairs Vector of pairs containing values and indices, where T is
 * the value type and int is the original index
 * @return Returns the original index corresponding to the maximum value
 */
template <typename T>
int get_topk(int topk, std::vector<std::pair<T, int>> sort_pairs) {
  // Sort pairs in descending order by value
  std::sort(sort_pairs.begin(), sort_pairs.end(),
            [](const std::pair<T, int>& a, const std::pair<T, int>& b) {
              return a.first > b.first;
            });

  // Print detailed information for top K elements, including index, confidence
  // and label
  for (int i = 0; i < topk; ++i) {
    LOG_INFO("top{}: Index={} Conf={}, Label=[{}]", i + 1, sort_pairs[i].second,
             sort_pairs[i].first, Imagenet::GetLabel(sort_pairs[i].second));
  }

  return sort_pairs[0].second;
}

int main() {
  LOG_INFO("===> resnet50 c++ example start...");
  const char* houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target =
      houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
  if (houmo_target != "xh2") {
    LOG_ERROR("Unsupported backend {}", houmo_target);
    exit(-1);
  }
  LOG_INFO("houmo_target:{}, tcim version: {}.", houmo_target,
           tcim::GetVersion());

  // 1. Load model
  std::string model_path;
  for (const auto& entry : fs::directory_iterator(fs::current_path())) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() == ".hmm") {
      model_path = entry.path().string();
      LOG_INFO("Found .hmm file: {}", model_path);
      break;
    }
  }

  if (model_path.empty() || !fs::exists(model_path)) {
    LOG_ERROR("No .hmm file found in {}", fs::current_path().string());
    exit(-1);
  }
  LOG_INFO("Load resnet50 model from file {}", model_path);

  // Load the model file
  auto module = tcim::Module::LoadFromFile(model_path);
  if (!module) {
    LOG_ERROR("Failed to load model {}.", model_path);
    exit(-1);
  }
  LOG_INFO("Model {} loaded.", model_path);

  // 2. Get input information
  std::map<std::string, tcim::Tensor> input_map;
  std::map<std::string, tcim::TensorInfo> input_map_f32;
  int input_num = module.GetInputNum();
  LOG_INFO("Count of Input: {}", input_num);

  // Iterate through all inputs and create tensors
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name).AsContiguous();
    LOG_INFO("Input[{}] info: {}", input_name, TensorInfo2Str(input_info));
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));

    auto input_info_f32 = input_info.AsType(tcim::DataType::FLOAT32);
    LOG_INFO("Input[{}] float32 info: {}", input_name,
             TensorInfo2Str(input_info_f32));
    input_map_f32.insert(
        std::pair<std::string, tcim::TensorInfo>(input_name, input_info_f32));
  }

  // 3. Input preprocessing
  std::string data_path = "../../data/snake.jpg";
  if (!fs::exists(data_path)) {
    LOG_ERROR("{} not exist.", data_path);
    exit(-1);
  }

  // Load and preprocess image
  cv::Mat img_rgb;
  img_rgb = cv::imread(data_path);

  cv::Mat img_norm;
  // Define normalization parameters for ImageNet
  const float mean[3] = {123.675f, 116.28f, 103.53f};
  const float std[3] = {58.395f, 57.12f, 57.375f};
  // Convert BGR to RGB, resize to 224x224 (standard for ResNet)
  cv::cvtColor(img_rgb, img_rgb, cv::COLOR_BGR2RGB);
  cv::resize(img_rgb, img_rgb, {224, 224});
  // Convert to float32 and normalize
  img_rgb.convertTo(img_norm, CV_32FC3);
  std::vector<cv::Mat> channels;
  cv::split(img_norm, channels);
  for (int i = 0; i < 3; ++i) {
    channels[i] = (channels[i] - mean[i]) / std[i];
  }
  // Convert from HWC (Height-Width-Channel) to CHW (Channel-Height-Width)
  for (auto& ch : channels) {
    ch = ch.reshape(1, 1);
  }
  cv::vconcat(channels, img_norm);

  // Calculate image size in bytes
  size_t img_bytes = img_norm.total() * img_norm.elemSize();
  LOG_INFO("img_bytes: {}", img_bytes);

  // Create tensor from preprocessed image data
  auto input_tensor_f32 =
      tcim::Tensor::CreateHostTensor(input_map_f32.at("input.1"), img_bytes,
                                     reinterpret_cast<void*>(img_norm.data));
  input_tensor_f32.CastTo(input_map.at("input.1"));

  // 4. Get output information
  std::map<std::string, tcim::Tensor> output_map;
  std::map<std::string, tcim::Tensor> output_map_f32;
  int output_num = module.GetOutputNum();
  LOG_INFO("Count of Output: {}", output_num);

  // Iterate through all outputs and create tensors
  for (int idx = 0; idx < output_num; idx++) {
    // Get the name and information of each output
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name).AsContiguous();
    LOG_INFO("Output[{}] info: {}", output_name, TensorInfo2Str(output_info));

    // Create host tensor for output
    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    output_map.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor));

    // Create float32 version of output tensor
    auto output_info_f32 = output_info.AsType(tcim::DataType::FLOAT32);
    auto output_tensor_f32 = tcim::Tensor::CreateHostTensor(output_info_f32);
    output_map_f32.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor_f32));
  }

  // 5. Set inputs to the model
  for (const auto& input : input_map) {
    module.SetInput(input.first, input.second);
  }

  // 6. run and sync
  module.Run();
  module.Sync();

  // 7. Get output
  for (auto& output : output_map) {
    module.GetOutput(output.first, output.second);
  }

  // 8. Postprocess, with no softmax
  int top1 = 0;
  const int topk = 5;

  // Process each output tensor
  for (auto& output : output_map) {
    // Get the float32 version of the output
    auto f32_opt = output_map_f32[output.first];
    output.second.CastTo(f32_opt);

    // Create vector to store value-index pairs
    std::vector<std::pair<float, int>> sort_pairs;
    // Populate the vector with 1000 pairs (for ImageNet's 1000 classes)
    for (int i = 0; i < 1000; ++i) {
      sort_pairs.emplace_back(static_cast<float*>(f32_opt.Data())[i], i);
    }
    // Get top K results
    top1 = get_topk(topk, sort_pairs);
  }

  // Verify the top-1 prediction (class 65 corresponds to snake in ImageNet)
  if (top1 != 65) {
    LOG_ERROR("top1 != 65");
    exit(-1);
  }

  LOG_INFO("<=== resnet50 c++ example completed.");
  return 0;
}
