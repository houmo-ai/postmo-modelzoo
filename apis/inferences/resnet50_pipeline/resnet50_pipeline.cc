/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: resnet50_pipeline.cc
 * Description:
 *   ResNet50 Pipeline Image Classification C++ Example.
 *   This file demonstrates how to implement a pipeline for image classification
 *   using ResNet50 model with multi-threading support. The pipeline includes
 *   separate threads for host-to-device transfer, inference execution, and
 *   device-to-host transfer to optimize performance. The implementation
 *   includes model loading, image preprocessing, inference execution,
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

#include <iostream>
#include <sstream>
#include <string>
#include <thread>

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

/**
 * @struct CliArguments
 * @brief Structure to hold command-line arguments for the ResNet50 pipeline
 */
struct CliArguments {
  std::string model_path;  ///< Path to the model file
  size_t sample_num = 1;   ///< Number of samples to process
};

/**
 * @struct TaskInfo
 * @brief Structure to hold task information including input/output tensors,
 * request ID, and end flag
 */
typedef struct {
  std::map<std::string, tcim::Tensor> input_map;
  std::map<std::string, tcim::Tensor> output_map;
  uint64_t req_id;
  bool is_end = false;  ///< Flag indicating if this is an end marker
} TaskInfo;

/**
 * @struct TaskQueue
 * @brief Structure to represent a thread-safe queue for tasks with
 * synchronization primitives
 */
typedef struct {
  std::queue<TaskInfo> queue;    ///< Queue of task information
  std::mutex mutex;              ///< Mutex for thread synchronization
  std::condition_variable cond;  ///< Condition variable for signaling
  std::map<std::string, tcim::TensorInfo> input_info_map;
  std::map<std::string, tcim::TensorInfo> output_info_map;
} TaskQueue;

/**
 * @brief Parse command-line arguments for the ResNet50 pipeline
 *
 * @param arguments Pointer to CliArguments structure to store parsed arguments
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return true if parsing is successful, false otherwise
 */
bool ParseArgs(CliArguments* arguments, int argc, char* argv[]) {
#if !defined(_MSC_VER)
  int option_idx = 0;
  struct option long_options[] = {
      {"help", 0, 0, 'h'},
      {"samples", 1, 0, 's'},
  };

  while (true) {
    int ch = getopt_long(argc, argv, "hs:", long_options, &option_idx);
    if (ch == -1) {
      break;
    }
    switch (ch) {
      case 'h':
        LOG_INFO("Usage: resnet50_pipeline [options]");
        LOG_INFO("-s <sample_num> [default: 1]");
        LOG_INFO("-h: print help message");
        break;
      case 's':
        arguments->sample_num = atoi(optarg);
        break;
      default:
        LOG_ERROR("Unsupported option: {}", static_cast<char>(ch));
        return false;
    }
  }
#endif
  return true;
}

/**
 * @brief Get the top-k predictions from the model output
 *
 * @tparam T Type of the values in the pairs
 * @param topk Number of top predictions to return
 * @param sort_pairs Vector of value-index pairs to sort
 * @return Index of the top-1 prediction
 */
template <typename T>
int get_topk(int topk, std::vector<std::pair<T, int>> sort_pairs) {
  // Sort pairs in descending order based on the first element (confidence)
  std::sort(sort_pairs.begin(), sort_pairs.end(),
            [](const std::pair<T, int>& a, const std::pair<T, int>& b) {
              return a.first > b.first;
            });

  // Print the top-k predictions with their indices, confidence scores, and
  // labels
  for (int i = 0; i < topk; ++i) {
    LOG_INFO("top{}: Index={}, Conf={}, Label=[{}]", (i + 1),
             sort_pairs[i].second, sort_pairs[i].first,
             Imagenet::GetLabel(sort_pairs[i].second));
  }

  return sort_pairs[0].second;
}

/**
 * @brief Main function for the ResNet50 pipeline example
 *
 * This function demonstrates a pipeline approach to image classification using
 * ResNet50, with separate threads for host-to-device transfer, inference, and
 * device-to-host transfer.
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return 0 on success, non-zero on failure
 */
int main(int argc, char* argv[]) {
  LOG_INFO("===> resnet50_pipeline c++ example start...");
  const char* houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target =
      houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
  // Verify that the target platform is supported
  if (houmo_target != "xh2") {
    LOG_ERROR("Unsupported backend {}", houmo_target);
    exit(-1);
  }
  LOG_INFO("houmo target: {}, tcim version: {}", houmo_target,
           tcim::GetVersion());

  // Set default parameters
  std::string default_model_path = "./resnet50_xh2_b1_1core.hmm";
  CliArguments arguments;
  arguments.model_path = default_model_path;
  arguments.sample_num = 10;  // Default to 10 samples
  ParseArgs(&arguments, argc, argv);
  LOG_INFO("model: {}, samples: {}", arguments.model_path,
           arguments.sample_num);

  std::string model_path = arguments.model_path;
  // Verify that the model file exists
  if (!fs::exists(model_path)) {
    LOG_ERROR("{} not exist.", model_path);
    exit(-1);
  }

  // Load the model from file
  auto module = tcim::Module::LoadFromFile(model_path);
  if (!module) {
    LOG_ERROR("Failed to load model {}.", model_path);
    exit(-1);
  }
  LOG_INFO("model {} loaded.", model_path);

  // Create pipeline queues for different stages
  TaskQueue qh2d;    // Input queue for host-to-device transfer thread
  TaskQueue qinfer;  // Input queue for inference thread
  TaskQueue qd2h;    // Input queue for device-to-host transfer thread
  TaskQueue qout;    // Output queue from device-to-host transfer thread

  // Prepare input information by querying the module
  int input_num = module.GetInputNum();
  LOG_INFO("Count of Input: {}", input_num);
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name);
    LOG_INFO("Input[{}] info: {}", input_name, TensorInfo2Str(input_info));
    // Store input info in all queues for reference
    qh2d.input_info_map[input_name] = input_info;
    qinfer.input_info_map[input_name] = input_info;
    qd2h.input_info_map[input_name] = input_info;
  }

  // Prepare output information by querying the module
  int output_num = module.GetOutputNum();
  LOG_INFO("Count of Output: {}", output_num);
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name);
    LOG_INFO("Output[{}] info: {}", output_name, TensorInfo2Str(output_info));
    // Store output info in all queues for reference
    qh2d.output_info_map[output_name] = output_info;
    qinfer.output_info_map[output_name] = output_info;
    qd2h.output_info_map[output_name] = output_info;
  }

  // Preprocess input image
  std::string data_path = "../../data/snake.jpg";
  if (!fs::exists(data_path)) {
    LOG_ERROR("{} not exist.", data_path);
    exit(-1);
  }

  cv::Mat img_bgr;        // Input image in BGR format
  cv::Mat img_processed;  // Processed image ready for inference
  size_t img_size = 0;    // Size of the processed image in bytes

  // Load the input image
  img_bgr = cv::imread(data_path);

  cv::Mat img_rgb;   // Image in RGB format
  cv::Mat img_norm;  // Normalized image
  // Define normalization parameters (ImageNet mean and std values)
  const float mean[3] = {123.675f, 116.28f, 103.53f};
  const float std[3] = {58.395f, 57.12f, 57.375f};

  // Convert BGR to RGB and resize to required input size (224x224)
  cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
  cv::resize(img_rgb, img_rgb, {224, 224});

  // Convert to float and normalize
  img_rgb.convertTo(img_norm, CV_32FC3);
  std::vector<cv::Mat> channels;  // Vector to store individual channels
  cv::split(img_norm, channels);  // Split into separate RGB channels
  // Normalize each channel using mean and std values
  for (int i = 0; i < 3; ++i) {
    channels[i] = (channels[i] - mean[i]) / std[i];
  }
  // Convert from HWC (Height-Width-Channel) to CHW (Channel-Height-Width)
  // format
  for (auto& ch : channels) {
    ch = ch.reshape(1, 1);
  }
  cv::vconcat(channels, img_processed);
  // Calculate image size in bytes
  img_size = img_norm.total() * img_norm.elemSize();

  // Prepare input tensors for all samples
  auto& name = qh2d.input_info_map.begin()->first;
  auto info = qh2d.input_info_map.begin()->second.AsContiguous();
  for (int i = 0; i < arguments.sample_num; i++) {
    // Allocate memory for the image data
    auto data = malloc(img_size);
    std::shared_ptr<void> data_ptr(data, free);
    // Copy processed image data to allocated memory
    memcpy(data, reinterpret_cast<void*>(img_processed.data), img_size);

    TaskInfo task_info;
    task_info.req_id = i;
    tcim::Tensor tensor;

    // Convert float32 buffer to float16 buffer as required by the model
    auto info_f32 = info.AsType(tcim::DataType::FLOAT32);
    auto tensor_f32 = tcim::Tensor::CreateHostTensor(info_f32, img_size, data);
    tensor = tcim::Tensor::CreateHostTensor(info);
    tensor_f32.CastTo(tensor);

    task_info.input_map[name] = tensor;  // Store the tensor in the task info
    qh2d.queue.push(task_info);          // Add task to the host-to-device queue
  }
  LOG_INFO("sample queue size is {}.", qh2d.queue.size());

  // Add an end marker to the queue to signal completion
  TaskInfo task_info;
  task_info.is_end = true;
  qh2d.queue.push(task_info);

  // Define host-to-device (H2D) transfer thread function
  auto H2D = [](TaskQueue& qin, TaskQueue& qout) {
    LOG_INFO("thread H2D start...");
    // H2D transfer loop
    while (true) {
      // Get data from the input queue with synchronization
      std::unique_lock<std::mutex> lock_in(qin.mutex);
      while (qin.queue.empty()) {
        qin.cond.wait(lock_in);  // Wait if queue is empty
      }
      auto task_info = qin.queue.front();
      if (task_info.is_end) {
        // If end marker, pass it to the next queue and exit
        lock_in.unlock();
        std::unique_lock<std::mutex> lock_out(qout.mutex);
        qout.queue.push(task_info);
        qout.cond.notify_all();
        lock_out.unlock();
        break;
      }
      auto& input_map = task_info.input_map;
      auto& output_map = task_info.output_map;
      qin.queue.pop();  // Remove task from input queue
      lock_in.unlock();

      // Transfer each input tensor from host to device
      for (auto& input : input_map) {
        auto& name = input.first;
        auto& info = qin.input_info_map[name];
        // Create device tensor
        auto tensor = tcim::Tensor::CreateDeviceTensor(info);
        // Copy host tensor to device tensor
        input.second.CopyTo(tensor);
        // Update the map with device tensor
        input.second = tensor;
      }

      // Create device tensors for outputs
      for (auto& output_info : qin.output_info_map) {
        auto& name = output_info.first;
        auto& info = output_info.second;
        auto tensor = tcim::Tensor::CreateDeviceTensor(info);
        task_info.output_map[name] = tensor;
      }

      // Send the task to the inference queue
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(task_info);
      qout.cond.notify_all();
      lock_out.unlock();
    }
  };

  // Define device-to-host (D2H) transfer thread function
  auto D2H = [](TaskQueue& qin, TaskQueue& qout) {
    LOG_INFO("thread D2H start...");
    // D2H transfer loop
    while (true) {
      // Get data from the input queue with synchronization
      std::unique_lock<std::mutex> lock_in(qin.mutex);
      while (qin.queue.empty()) {
        qin.cond.wait(lock_in);  // Wait if queue is empty
      }
      auto task_info = qin.queue.front();
      if (task_info.is_end) {
        // If end marker, exit without passing it to the next queue
        lock_in.unlock();
        break;
      }

      auto& output_map = task_info.output_map;
      qin.queue.pop();  // Remove task from input queue
      lock_in.unlock();

      // Transfer each output tensor from device to host
      for (auto& output : output_map) {
        auto& name = output.first;
        auto info = qin.output_info_map[name].AsContiguous();
        // Create host tensor
        auto tensor = tcim::Tensor::CreateHostTensor(info);
        // Copy device tensor to host tensor
        output.second.CopyTo(tensor);
        // Update the map with host tensor
        output.second = tensor;
      }

      // Send the task to the output queue
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(task_info);
      qout.cond.notify_all();
      lock_out.unlock();
    }
  };

  // Define inference thread function
  auto Infer = [](tcim::Module& module, TaskQueue& qin, TaskQueue& qout) {
    LOG_INFO("thread infer start...");
    int count = 0;  // Counter for processed samples

    // Inference loop
    while (true) {
      // Get data from the input queue with synchronization
      std::unique_lock<std::mutex> lock_in(qin.mutex);
      while (qin.queue.empty()) {
        qin.cond.wait(lock_in);  // Wait if queue is empty
      }
      auto task_info = qin.queue.front();
      if (task_info.is_end) {
        // If end marker, pass it to the next queue and exit
        lock_in.unlock();
        std::unique_lock<std::mutex> lock_out(qout.mutex);
        qout.queue.push(task_info);
        qout.cond.notify_all();
        lock_out.unlock();
        break;
      }
      auto input_map = task_info.input_map;
      auto output_map = task_info.output_map;
      auto req_id = task_info.req_id;
      qin.queue.pop();  // Remove task from input queue
      lock_in.unlock();

      // Set input tensors for the module
      for (auto& input : input_map) {
        module.SetInput(input.first, input.second);
      }

      // Set output tensors for the module
      for (auto& output : output_map) {
        module.SetOutput(output.first, output.second);
      }

      // Execute the inference
      module.Run();
      module.Sync();  // Synchronize to ensure completion

      // send to D2H thread
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(task_info);
      qout.cond.notify_all();
      lock_out.unlock();

      count++;
      LOG_INFO("run sample {} end.", req_id);
    }

    LOG_INFO("infer thread completed, {} samples tested.", count);
  };

  // Create and start pipeline threads
  std::vector<std::thread> threads;
  threads.push_back(std::thread(H2D, std::ref(qh2d), std::ref(qinfer)));
  threads.push_back(
      std::thread(Infer, std::ref(module), std::ref(qinfer), std::ref(qd2h)));
  threads.push_back(std::thread(D2H, std::ref(qd2h), std::ref(qout)));

  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }

  // Post-process results and verify output
  while (!qout.queue.empty()) {
    auto output_map = qout.queue.front().output_map;
    auto req_id = qout.queue.front().req_id;
    qout.queue.pop();
    int top1 = 0;
    const int topk = 1;  // Get top-1 prediction

    for (auto& output : output_map) {
      // Convert output tensor to float32 for processing
      auto info_f32 = output.second.Info().AsType(tcim::DataType::FLOAT32);
      auto output_tensor_f32 = tcim::Tensor::CreateHostTensor(info_f32);
      output.second.CastTo(output_tensor_f32);

      std::vector<std::pair<float, int>> sort_pairs;
      // Create pairs of (confidence, index) for all 1000 ImageNet classes
      for (int i = 0; i < 1000; ++i) {
        sort_pairs.emplace_back(
            static_cast<float*>(output_tensor_f32.Data())[i], i);
      }
      top1 = get_topk(topk, sort_pairs);
    }

    // Verify result (65 corresponds to snake class in ImageNet)
    if (top1 != 65) {
      LOG_ERROR("top1 != 65");
      exit(-1);
    }
  }

  LOG_INFO("<=== resnet50_pipeline c++ example completed.");
  return 0;
}
