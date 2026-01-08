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

#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if (__GNUC__ < 8 && !defined(_MSC_VER))
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#if !defined(_MSC_VER)
#include <getopt.h>
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "datasets/imagenet.hpp"
#include "logging.h"
#include "tcim/tcim_runtime.h"
#include "threads.hpp"
#include "utils.hpp"

// Structure to hold command-line arguments
struct CliArguments {
  std::string model_path;  // Path to the model
  size_t thread_num = 1;   // Number of threads
  size_t sample_num = 1;   // Number of samples to process
  size_t device_num = 1;   // Number of devices
};

// Structure to hold task information for processing
typedef struct {
  std::map<std::string, std::shared_ptr<void>> data_map;
  uint64_t req_id;  // Request ID
} TaskInfo;

// Structure to hold task queue with synchronization primitives
typedef struct {
  std::queue<TaskInfo> queue;
  std::mutex mutex;
  std::condition_variable cond;
  std::map<std::string, tcim::TensorInfo> info_map;
  std::map<std::string, tcim::TensorInfo> info_map_f32;
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
    LOG_INFO("top{}: Index={}, Conf={}, Label=[{}]", (i + 1),
             sort_pairs[i].second, sort_pairs[i].first,
             Imagenet::GetLabel(sort_pairs[i].second));
  }

  return sort_pairs[0].second;
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
  std::string default_model_path = "./resnet50_xh2_b1_1core.hmm";
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

  // Prepare input image: load, resize, normalize
  std::string data_path = "../../data/snake.jpg";
  if (!fs::exists(data_path)) {
    LOG_ERROR("{} not exist.", data_path);
    exit(-1);
  }
  cv::Mat img_rgb;
  cv::Mat img_processed;
  size_t img_size = 0;
  img_rgb = cv::imread(data_path);

  cv::Mat img_norm;
  const float mean[3] = {123.675f, 116.28f, 103.53f};
  const float std[3] = {58.395f, 57.12f, 57.375f};
  cv::cvtColor(img_rgb, img_rgb, cv::COLOR_BGR2RGB);
  cv::resize(img_rgb, img_rgb, {224, 224});

  img_rgb.convertTo(img_norm, CV_32FC3);
  std::vector<cv::Mat> channels;
  cv::split(img_norm, channels);
  for (int i = 0; i < 3; ++i) {
    channels[i] = (channels[i] - mean[i]) / std[i];
  }
  // HWC --> CHW
  for (auto& ch : channels) {
    ch = ch.reshape(1, 1);
  }
  cv::vconcat(channels, img_processed);
  img_size = img_norm.total() * img_norm.elemSize();

  std::vector<std::thread> threads;
  // Prepare input and output queues with sample data
  TaskQueue qin;
  TaskQueue qout;
  for (int i = 0; i < arguments.sample_num; i++) {
    auto data = malloc(img_size);
    std::shared_ptr<void> data_ptr(data, free);
    memcpy(data, reinterpret_cast<void*>(img_processed.data), img_size);

    TaskInfo tinfo;
    tinfo.req_id = i;
    tinfo.data_map.insert(
        std::pair<std::string, std::shared_ptr<void>>("", data_ptr));
    qin.queue.push(tinfo);
  }
  LOG_INFO("sample queue size is {}", qin.queue.size());

  // Define the thread function for processing tasks
  auto thread_func = [](int tid, int did, tcim::Module& module, TaskQueue& qin,
                        TaskQueue& qout, Barrier& barrier) {
    // Initialize input tensor information
    int input_num = module.GetInputNum();
    LOG_INFO("Count of Input: {}", input_num);
    for (int idx = 0; idx < input_num; idx++) {
      auto input_name = module.GetInputName(idx);
      auto input_info = module.GetInputInfo(input_name).AsContiguous();
      LOG_INFO("Input[{}] info: {}", input_name, TensorInfo2Str(input_info));
      qin.info_map[input_name] = input_info;

      auto input_info_f32 = input_info.AsType(tcim::DataType::FLOAT32);
      qin.info_map_f32[input_name] = input_info_f32;
    }

    // Initialize output tensor information
    int output_num = module.GetOutputNum();
    LOG_INFO("Count of Output: {}", output_num);
    for (int idx = 0; idx < output_num; idx++) {
      auto output_name = module.GetOutputName(idx);
      auto output_info = module.GetOutputInfo(output_name).AsContiguous();
      LOG_INFO("Output[{}] info: {}", output_name, TensorInfo2Str(output_info));
      qout.info_map[output_name] = output_info;

      auto output_info_f32 = output_info.AsType(tcim::DataType::FLOAT32);
      qout.info_map_f32[output_name] = output_info_f32;
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

      // Set input tensors to the module
      for (auto& info : qin.info_map) {
        tcim::Tensor input_tensor;
        input_tensor = tcim::Tensor::CreateHostTensor(info.second);
        auto info_f32 = qin.info_map_f32[info.first];
        auto input_tensor_f32 = tcim::Tensor::CreateHostTensor(
            info_f32, info_f32.MemSize(), input_map[""].get());
        input_tensor_f32.CastTo(input_tensor);

        module.SetInput(info.first, input_tensor);
      }

      // Execute inference
      module.Run();
      module.Sync();

      // Get output tensors and add to output queue
      TaskInfo tinfo;
      tinfo.req_id = req_id;
      for (auto& info : qout.info_map) {
        tcim::Tensor output_tensor;
        void* data = nullptr;
        output_tensor = tcim::Tensor::CreateHostTensor(info.second);
        module.GetOutput(info.first, output_tensor);

        auto info_f32 = qout.info_map_f32[info.first];
        auto size = info_f32.MemSize();
        data = malloc(size);
        auto output_tensor_f32 =
            tcim::Tensor::CreateHostTensor(info_f32, size, data);
        output_tensor.CastTo(output_tensor_f32);

        std::shared_ptr<void> data_ptr(data, free);
        tinfo.data_map.insert(std::pair<std::string, std::shared_ptr<void>>(
            info.first, data_ptr));
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
    auto output_map = qout.queue.front().data_map;
    auto req_id = qout.queue.front().req_id;
    qout.queue.pop();
    int top1 = 0;
    const int topk = 1;
    for (auto& output : output_map) {
      std::vector<std::pair<float, int>> sort_pairs;
      for (int i = 0; i < 1000; ++i) {
        sort_pairs.emplace_back(static_cast<float*>(output.second.get())[i], i);
      }
      top1 = get_topk(topk, sort_pairs);
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