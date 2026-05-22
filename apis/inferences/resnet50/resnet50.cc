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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

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

#include "imageproc.hpp"
#include "logging.h"
#include "tcim/tcim_runtime.h"
#include "threads.hpp"
#include "utils.hpp"

constexpr const char* kResizeCropInputName = "resizer_crop";
constexpr const char* kInputImagePath = "../../data/snake.jpg";
constexpr int kTargetHeight = 224;
constexpr int kTargetWidth = 224;

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

  // 1. Discover model file and input image.
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

  if (!fs::exists(kInputImagePath)) {
    LOG_ERROR("{} not exist.", kInputImagePath);
    exit(-1);
  }

  cv::Mat image_data = cv::imread(kInputImagePath);
  if (image_data.empty()) {
    LOG_ERROR("Failed to read image {}", kInputImagePath);
    exit(-1);
  }
  int img_height = image_data.rows;
  int img_width = image_data.cols;
  LOG_INFO("input image shape: [{} x {} x {}]", img_height, img_width,
           image_data.channels());

  // 2. Load model.
  LOG_INFO("Load resnet50 model from file {}", model_path);
  auto module = tcim::Module::LoadFromFile(model_path);
  if (!module) {
    LOG_ERROR("Failed to load model {}.", model_path);
    exit(-1);
  }
  LOG_INFO("Model {} loaded.", model_path);

  // 3. Query input information.
  // Find the image input canvas size and keep all input names for later
  // feeding.
  int target_height = kTargetHeight;
  int target_width = kTargetWidth;
  int max_img_height = 0;
  int max_img_width = 0;
  std::vector<std::string> input_names;
  const int input_num = static_cast<int>(module.GetInputNum());
  LOG_INFO("Model input info:");
  for (int idx = 0; idx < input_num; ++idx) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name).AsContiguous();
    input_names.emplace_back(input_name);
    LOG_INFO("  Input[{}] info: {}", input_name, TensorInfo2Str(input_info));
    if (input_name.find(kResizeCropInputName) == std::string::npos) {
      max_img_height = static_cast<int>(input_info.Shape().at(2));
      max_img_width = static_cast<int>(input_info.Shape().at(3));
    }
  }
  if (max_img_height <= 0 || max_img_width <= 0) {
    LOG_ERROR("Invalid model input shape: height={}, width={}", max_img_height,
              max_img_width);
    exit(-1);
  }

  // 4. Preprocess image.
  // Preserve the original valid image region in crop_height and crop_width.
  int crop_height = max_img_height;
  int crop_width = max_img_width;
  if (img_height < max_img_height && img_width <= max_img_width) {
    // Pad smaller images with zeros on the bottom and right sides.
    const int pad_bottom = max_img_height - img_height;
    const int pad_right = max_img_width - img_width;
    cv::copyMakeBorder(image_data, image_data, 0, pad_bottom, 0, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    crop_height = img_height;
    crop_width = img_width;
    LOG_INFO("pad input image to [{} x {} x {}], height={}, width={}",
             image_data.rows, image_data.cols, image_data.channels(),
             max_img_height, max_img_width);
  } else {
    // Resize images that exceed the supported canvas size.
    cv::resize(image_data, image_data, cv::Size(max_img_width, max_img_height));
    LOG_INFO("resize input image to height={}, width={}", max_img_height,
             max_img_width);
  }

  // Align crop size to even dimensions before feeding the dynamic crop input.
  crop_height -= crop_height % 2;
  crop_width -= crop_width % 2;
  if (crop_height <= 0 || crop_width <= 2 || crop_height % 2 != 0 ||
      crop_width % 2 != 0) {
    LOG_ERROR("crop_height and crop_width must be even, got {} and {}",
              crop_height, crop_width);
    exit(-1);
  }

  // Validate that the resize ratios stay within the accepted range.
  const float height_scale = static_cast<float>(target_height) / crop_height;
  const float width_scale = static_cast<float>(target_width) / crop_width;
  if (height_scale < 1.0f / 32.0f || height_scale > 16.0f) {
    LOG_ERROR("{} / img_height must be in [1/32, 16], got {}", target_height,
              height_scale);
    exit(-1);
  }
  if (width_scale < 1.0f / 32.0f || width_scale > 16.0f) {
    LOG_ERROR("{} / img_width must be in [1/32, 16], got {}", target_width,
              width_scale);
    exit(-1);
  }

  std::vector<int32_t> dyn_info = {
      0, 0, crop_height, crop_width, target_height, target_width, 0, 0, 0, 0};

  // 5. Feed image data and dynamic crop information into the model.
  // Keep owned backing buffers alive for tensors created directly from
  // pointers.
  std::map<std::string, std::vector<uint8_t>> input_buffers;
  std::map<std::string, tcim::Tensor> input_tensors;
  for (const auto& input_name : input_names) {
    auto input_info = module.GetInputInfo(input_name).AsContiguous();

    if (input_name.find(kResizeCropInputName) != std::string::npos) {
      // Feed the dynamic crop tensor, using dyn_info directly when sizes match.
      const size_t dyn_bytes = dyn_info.size() * sizeof(int32_t);
      if (dyn_bytes > input_info.MemSize()) {
        LOG_ERROR("dyn_info bytes {} exceed input tensor bytes {} for {}",
                  dyn_bytes, input_info.MemSize(), input_name);
        exit(-1);
      }

      if (dyn_bytes == input_info.MemSize()) {
        auto input_tensor = tcim::Tensor::CreateHostTensor(
            input_info, dyn_bytes, static_cast<void*>(dyn_info.data()));
        input_tensors.emplace(input_name, input_tensor);
      } else {
        auto& buffer = input_buffers[input_name];
        buffer.assign(input_info.MemSize(), 0);
        std::memcpy(buffer.data(), dyn_info.data(), dyn_bytes);
        auto input_tensor = tcim::Tensor::CreateHostTensor(
            input_info, buffer.size(), static_cast<void*>(buffer.data()));
        input_tensors.emplace(input_name, input_tensor);
      }
    } else {
      // Convert the padded or resized image into YUV420sp directly in the input
      // buffer.
      auto& buffer = input_buffers[input_name];
      buffer.assign(input_info.MemSize(), 0);
      const size_t image_bytes =
          ConvertBgrToYuv420sp(image_data, buffer.data(), buffer.size());
      if (image_bytes == 0) {
        LOG_ERROR("input image bytes exceed input tensor bytes {} for {}",
                  input_info.MemSize(), input_name);
        exit(-1);
      }
      auto input_tensor = tcim::Tensor::CreateHostTensor(
          input_info, buffer.size(), static_cast<void*>(buffer.data()));
      input_tensors.emplace(input_name, input_tensor);
    }

    if (module.SetInput(input_name, input_tensors.at(input_name)) !=
        tcim::Status::OK) {
      LOG_ERROR("Failed to set input {}", input_name);
      exit(-1);
    }
  }

  LOG_INFO("dyn_info: [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]", dyn_info[0],
           dyn_info[1], dyn_info[2], dyn_info[3], dyn_info[4], dyn_info[5],
           dyn_info[6], dyn_info[7], dyn_info[8], dyn_info[9]);

  // 6. Run and sync.
  module.Run();
  module.Sync();

  // 7. Read outputs and apply softmax postprocess.
  // Copy outputs to host, convert to float32, then report top-k predictions.
  int top1 = 0;
  const int topk = 5;
  const int output_num = static_cast<int>(module.GetOutputNum());
  for (int idx = 0; idx < output_num; ++idx) {
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name).AsContiguous();
    LOG_INFO("output[{}] info: {}", output_name, TensorInfo2Str(output_info));

    auto output_tensor = module.GetDevOutput(output_name)
                             .ToHost(true)
                             .AsType(tcim::DataType::FLOAT32, true);
    auto* output_ptr = static_cast<float*>(output_tensor.Data());
    const size_t output_count = GetElementCount(output_tensor.Info());
    std::vector<float> probs = Softmax(output_ptr, output_count);

    std::vector<std::pair<float, int>> sort_pairs;
    for (size_t i = 0; i < probs.size(); ++i) {
      sort_pairs.emplace_back(probs[i], static_cast<int>(i));
    }
    top1 = GetTopK(topk, sort_pairs);
  }

  if (top1 != 65) {
    LOG_ERROR("top1 != 65");
    exit(-1);
  }

  LOG_INFO("<=== resnet50 c++ example completed.");
  return 0;
}
