/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: resnet50_multistreams.cc
 * Description:
 *   ResNet50 Multi-Stream Image Classification C++ Example.
 *   This file demonstrates how to use ResNet50 model for image classification
 *   tasks with multi-threading support, including model
 *   loading, image preprocessing, inference execution, and result
 *   post-processing.
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
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#if defined(__clang__) || __GNUC__ >= 8 || defined(_MSC_VER)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#if !defined(_MSC_VER)
#include <getopt.h>
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

// Structure to hold command-line arguments
struct CliArguments {
  std::string model_path;  // Path to the model
  size_t thread_num = 1;   // Number of threads
  size_t sample_num = 1;   // Number of samples to process
  size_t device_num = 1;   // Number of devices
};

// Structure to hold task information for processing
typedef struct {
  // Shared host buffers for model inputs. Requests can reuse the same payloads.
  std::map<std::string, std::shared_ptr<void>> data_map;
  // Host output tensors kept directly to avoid an extra host-side memcpy.
  std::map<std::string, tcim::Tensor> tensor_map;
  uint64_t req_id;  // Request ID
} TaskInfo;

// Structure to hold task queue with synchronization primitives
typedef struct {
  std::queue<TaskInfo> queue;
  std::mutex mutex;
} TaskQueue;

// Function to parse command-line arguments
bool ParseArgs(CliArguments* arguments, int argc, char* argv[]) {
#if !defined(_MSC_VER)
  int option_idx = 0;
  // Define long options for command-line argument parsing
  struct option long_options[] = {
      {"help", 0, 0, 'h'},
      {"devices", 1, 0, 'n'},
      {"threads", 1, 0, 't'},
      {"samples", 1, 0, 's'},
  };

  // Parse command-line arguments
  while (true) {
    int ch = getopt_long(argc, argv, "hn:t:s:", long_options, &option_idx);
    if (ch == -1) {
      break;
    }
    switch (ch) {
      case 'h':
        std::cout << "Usage: -h" << std::endl;
        break;
      case 'n':
        arguments->device_num = atoi(optarg);
        break;
      case 't':
        arguments->thread_num = atoi(optarg);
        break;
      case 's':
        arguments->sample_num = atoi(optarg);
        break;
      default:
        std::cerr << "Unsupported option: " << static_cast<char>(ch)
                  << std::endl;
        return false;
    }
  }
#endif
  return true;
}

int main(int argc, char* argv[]) {
  LOG_INFO("===> resnet50_multistreams c++ example start...");

  // Check if running on correct hardware target (xh2)
  const char* houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target =
      houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
  if (houmo_target != "xh2") {
    LOG_ERROR("Unsupported backend {}", houmo_target);
    exit(-1);
  }
  LOG_INFO("houmo target: {}, tcim version: {}", houmo_target,
           tcim::GetVersion());

  // Set default parameters and override with environment variables if needed
  std::string default_model_path;
  for (const auto& entry : fs::directory_iterator(fs::current_path())) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() == ".hmm") {
      default_model_path = entry.path().string();
      LOG_INFO("Found .hmm file: {}", default_model_path);
      break;
    }
  }

  if (default_model_path.empty() || !fs::exists(default_model_path)) {
    LOG_ERROR("No .hmm file found in {}", fs::current_path().string());
    exit(-1);
  }
  LOG_INFO("Find resnet50 model file {}", default_model_path);

  CliArguments arguments;
  arguments.model_path = default_model_path;
  arguments.device_num = 1;
  arguments.thread_num = 4;
  arguments.sample_num = 10;
  if (auto platform = std::getenv("HDPL_PLATFORM")) {
    if (strcmp(platform, "ASIC")) {
      arguments.thread_num = 1;
    }
  } else {  // HDPL_PLATFORM not set
    arguments.thread_num = 1;
  }
  ParseArgs(&arguments, argc, argv);
  LOG_INFO("model: {}, devices: {}, threads: {}, samples: {}",
           arguments.model_path, arguments.device_num, arguments.thread_num,
           arguments.sample_num);

  // Verify model file exists
  std::string model_path = arguments.model_path;
  if (!fs::exists(model_path)) {
    LOG_ERROR("{} not exist.", model_path);
    exit(-1);
  }

  std::vector<std::thread> threads;
  TaskQueue qin;
  TaskQueue qout;

  // Define the thread function for processing tasks
  auto thread_func = [](int tid, int did, tcim::Module& module, TaskQueue& qin,
                        TaskQueue& qout, Barrier& barrier) {
    // Query input metadata for the current module instance.
    std::map<std::string, tcim::TensorInfo> input_infos;
    int input_num = module.GetInputNum();
    LOG_INFO("Count of Input: {}", input_num);
    for (int idx = 0; idx < input_num; idx++) {
      auto input_name = module.GetInputName(idx);
      auto input_info = module.GetInputInfo(input_name).AsContiguous();
      LOG_INFO("Input[{}] info: {}", input_name, TensorInfo2Str(input_info));
      input_infos[input_name] = input_info;
    }

    // Query output metadata for the current module instance.
    std::vector<std::string> output_names;
    int output_num = module.GetOutputNum();
    LOG_INFO("Count of Output: {}", output_num);
    for (int idx = 0; idx < output_num; idx++) {
      auto output_name = module.GetOutputName(idx);
      auto output_info = module.GetOutputInfo(output_name).AsContiguous();
      LOG_INFO("Output[{}] info: {}", output_name, TensorInfo2Str(output_info));
      output_names.push_back(output_name);
    }

    // Wait until all threads are ready
    barrier.barrier();
    LOG_INFO("thread {} on device {} infer start...", tid, did);
    int count = 0;

    // Main inference loop
    while (true) {
      // Get data from input queue
      std::unique_lock<std::mutex> lock_in(qin.mutex);
      if (qin.queue.empty()) {
        lock_in.unlock();
        break;
      }
      auto input_map = qin.queue.front().data_map;
      auto req_id = qin.queue.front().req_id;
      qin.queue.pop();
      lock_in.unlock();

      // Bind each input name to its shared host buffer for this request.
      for (const auto& info : input_infos) {
        auto data_it = input_map.find(info.first);
        if (data_it == input_map.end()) {
          LOG_ERROR("thread {} on device {} missing input {}", tid, did,
                    info.first);
          exit(-1);
        }

        auto input_tensor = tcim::Tensor::CreateHostTensor(
            info.second, info.second.MemSize(), data_it->second.get());
        if (module.SetInput(info.first, input_tensor) != tcim::Status::OK) {
          LOG_ERROR("thread {} on device {} failed to set input {}", tid, did,
                    info.first);
          exit(-1);
        }
      }

      // Execute inference
      module.Run();
      module.Sync();

      // Materialize outputs on host and keep the tensors directly in the queue.
      TaskInfo tinfo;
      tinfo.req_id = req_id;
      for (const auto& output_name : output_names) {
        auto output_tensor = module.GetDevOutput(output_name)
                                 .ToHost(true)
                                 .AsType(tcim::DataType::FLOAT32, true);
        tinfo.tensor_map.emplace(output_name, output_tensor);
      }
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(tinfo);
      lock_out.unlock();
      count++;
      LOG_INFO("thread {} on device {} run sample {} end.", tid, did, req_id);
      std::this_thread::yield();
    }

    LOG_INFO("thread {} on device {} completed. {} sampels tested.", tid, did,
             count);
  };

  // Load models for each device and thread
  std::map<int, std::vector<tcim::Module>> module_map;
  for (int did = 0; did < arguments.device_num; did++) {
    auto wm = tcim::Module::WeightManager::CreateWeightManager(did);
    tcim::Module::Option option(wm);
    std::vector<tcim::Module> dev_modules;
    for (int i = 0; i < arguments.thread_num; i++) {
      auto module = tcim::Module::LoadFromFile(model_path, option);
      if (!module) {
        LOG_ERROR("thread {} on device {} load model {} failed.", i, did,
                  model_path);
        exit(-1);
      }
      dev_modules.push_back(module);
      LOG_INFO("thread {} on device {} model {}.", i, did, model_path);
    }
    module_map[did] = dev_modules;
  }

  // Reuse the single-stream preprocessing flow to prepare shared inputs.
  auto& reference_module = module_map[0][0];
  int target_height = kTargetHeight;
  int target_width = kTargetWidth;
  int max_img_height = 0;
  int max_img_width = 0;
  std::vector<std::string> input_names;
  std::map<std::string, size_t> output_element_counts;

  LOG_INFO("Reference model input info:");
  const int input_num = static_cast<int>(reference_module.GetInputNum());
  for (int idx = 0; idx < input_num; ++idx) {
    auto input_name = reference_module.GetInputName(idx);
    auto input_info = reference_module.GetInputInfo(input_name).AsContiguous();
    input_names.push_back(input_name);
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

  const int output_num = static_cast<int>(reference_module.GetOutputNum());
  for (int idx = 0; idx < output_num; ++idx) {
    auto output_name = reference_module.GetOutputName(idx);
    auto output_info =
        reference_module.GetOutputInfo(output_name).AsContiguous();
    output_element_counts[output_name] = GetElementCount(output_info);
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

  int crop_height = max_img_height;
  int crop_width = max_img_width;
  // Pad smaller inputs to the model canvas and preserve the original crop area.
  if (img_height < max_img_height && img_width <= max_img_width) {
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
    // Resize larger inputs so the runtime always receives the expected canvas.
    cv::resize(image_data, image_data, cv::Size(max_img_width, max_img_height));
    LOG_INFO("resize input image to height={}, width={}", max_img_height,
             max_img_width);
  }

  // Align crop sizes to even values because YUV420 requires even dimensions.
  crop_height -= crop_height % 2;
  crop_width -= crop_width % 2;
  if (crop_height <= 0 || crop_width <= 2 || crop_height % 2 != 0 ||
      crop_width % 2 != 0) {
    LOG_ERROR("crop_height and crop_width must be even, got {} and {}",
              crop_height, crop_width);
    exit(-1);
  }

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

  // Dynamic resize-crop metadata follows the same layout as the Python example.
  std::vector<int32_t> dyn_info = {
      0, 0, crop_height, crop_width, target_height, target_width, 0, 0, 0, 0};

  // Build one shared payload map and reuse it for all queued requests.
  std::map<std::string, std::shared_ptr<void>> shared_inputs;
  for (const auto& input_name : input_names) {
    auto input_info = reference_module.GetInputInfo(input_name).AsContiguous();
    if (input_name.find(kResizeCropInputName) != std::string::npos) {
      const size_t dyn_bytes = dyn_info.size() * sizeof(int32_t);
      if (dyn_bytes > input_info.MemSize()) {
        LOG_ERROR("dyn_info bytes {} exceed input tensor bytes {} for {}",
                  dyn_bytes, input_info.MemSize(), input_name);
        exit(-1);
      }

      // Reuse dyn_info directly when tensor size matches, otherwise pad the
      // tail.
      if (dyn_bytes == input_info.MemSize()) {
        shared_inputs[input_name] = std::shared_ptr<void>(
            static_cast<void*>(dyn_info.data()), [](void*) {});
      } else {
        auto data = malloc(input_info.MemSize());
        if (data == nullptr) {
          LOG_ERROR("Failed to allocate dyn_info buffer for {}", input_name);
          exit(-1);
        }
        std::memset(data, 0, input_info.MemSize());
        std::memcpy(data, dyn_info.data(), dyn_bytes);
        shared_inputs[input_name] = std::shared_ptr<void>(data, free);
      }
    } else {
      // Allocate one reusable image buffer per input and pre-convert it once.
      auto data = malloc(input_info.MemSize());
      if (data == nullptr) {
        LOG_ERROR("Failed to allocate image buffer for {}", input_name);
        exit(-1);
      }
      std::memset(data, 0, input_info.MemSize());
      const size_t image_bytes = ConvertBgrToYuv420sp(
          image_data, reinterpret_cast<uint8_t*>(data), input_info.MemSize());
      if (image_bytes == 0) {
        LOG_ERROR("input image bytes exceed input tensor bytes {} for {}",
                  input_info.MemSize(), input_name);
        exit(-1);
      }
      shared_inputs[input_name] = std::shared_ptr<void>(data, free);
    }
  }

  // Enqueue repeated requests that all share the same preprocessed inputs.
  for (int i = 0; i < arguments.sample_num; i++) {
    TaskInfo tinfo;
    tinfo.req_id = i;
    tinfo.data_map = shared_inputs;
    qin.queue.push(tinfo);
  }
  LOG_INFO("sample queue size is {}", qin.queue.size());

  // Create and start threads for parallel processing
  Barrier barrier(arguments.thread_num * arguments.device_num);
  int tid = 0;
  for (int did = 0; did < arguments.device_num; did++) {
    for (int i = 0; i < arguments.thread_num; i++) {
      threads.push_back(std::thread(thread_func, tid, did,
                                    std::ref(module_map[did][i]), std::ref(qin),
                                    std::ref(qout), std::ref(barrier)));
      tid++;
    }
  }

  barrier.wait();

  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }

  // Process results and validate output
  while (!qout.queue.empty()) {
    auto output_map = qout.queue.front().tensor_map;
    auto req_id = qout.queue.front().req_id;
    qout.queue.pop();
    int top1 = 0;
    const int topk = 1;
    for (auto& output : output_map) {
      // Use the already-hosted tensor data directly for softmax and top-k.
      auto output_count = output_element_counts[output.first];
      std::vector<float> probs =
          Softmax(static_cast<float*>(output.second.Data()), output_count);

      std::vector<std::pair<float, int>> sort_pairs;
      for (size_t i = 0; i < probs.size(); ++i) {
        sort_pairs.emplace_back(probs[i], static_cast<int>(i));
      }
      top1 = GetTopK(topk, sort_pairs);
    }

    // Validate the result - snake image should have class ID 65
    if (top1 != 65) {
      LOG_ERROR("top1 != 65");
      exit(-1);
    }
  }

  LOG_INFO("<=== resnet50_multistreams c++ example completed.");
  return 0;
}