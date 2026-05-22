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

#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <queue>
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

#include "imageproc.hpp"
#include "logging.h"
#include "tcim/tcim_runtime.h"
#include "threads.hpp"
#include "utils.hpp"

constexpr const char* kResizeCropInputName = "resizer_crop";
constexpr const char* kInputImagePath = "../../data/snake.jpg";
constexpr int kTargetHeight = 224;
constexpr int kTargetWidth = 224;

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

  // Query input metadata and locate the image canvas expected by the model.
  std::vector<std::string> input_names;
  int max_img_height = 0;
  int max_img_width = 0;
  int input_num = module.GetInputNum();
  LOG_INFO("Count of Input: {}", input_num);
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name);
    input_names.push_back(input_name);
    LOG_INFO("Input[{}] info: {}", input_name, TensorInfo2Str(input_info));

    if (input_name.find(kResizeCropInputName) == std::string::npos) {
      max_img_height = static_cast<int>(input_info.Shape().at(2));
      max_img_width = static_cast<int>(input_info.Shape().at(3));
    }

    // Store input info in all queues for reference.
    qh2d.input_info_map[input_name] = input_info;
    qinfer.input_info_map[input_name] = input_info;
    qd2h.input_info_map[input_name] = input_info;
  }
  if (max_img_height <= 0 || max_img_width <= 0) {
    LOG_ERROR("Invalid model input shape: height={}, width={}", max_img_height,
              max_img_width);
    exit(-1);
  }

  // Query output metadata once and share it across all pipeline stages.
  int output_num = module.GetOutputNum();
  LOG_INFO("Count of Output: {}", output_num);
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name);
    LOG_INFO("Output[{}] info: {}", output_name, TensorInfo2Str(output_info));
    // Store output info in all queues for reference.
    qh2d.output_info_map[output_name] = output_info;
    qinfer.output_info_map[output_name] = output_info;
    qd2h.output_info_map[output_name] = output_info;
  }

  // Load and preprocess the input image with the same logic as resnet50.cc.
  const int target_height = kTargetHeight;
  const int target_width = kTargetWidth;
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

  // Preserve the original valid crop region when padding is needed.
  int crop_height = max_img_height;
  int crop_width = max_img_width;
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

  // Dynamic resize-crop metadata matches the single-stream example.
  std::vector<int32_t> dyn_info = {
      0, 0, crop_height, crop_width, target_height, target_width, 0, 0, 0, 0};
  LOG_INFO("dyn_info: [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]", dyn_info[0],
           dyn_info[1], dyn_info[2], dyn_info[3], dyn_info[4], dyn_info[5],
           dyn_info[6], dyn_info[7], dyn_info[8], dyn_info[9]);

  // Prepare one set of host tensors per request for all model inputs.
  for (int i = 0; i < arguments.sample_num; i++) {
    TaskInfo task_info;
    task_info.req_id = i;

    for (const auto& input_name : input_names) {
      auto info = qh2d.input_info_map.at(input_name).AsContiguous();
      auto tensor = tcim::Tensor::CreateHostTensor(info);
      std::memset(tensor.Data(), 0, info.MemSize());

      if (input_name.find(kResizeCropInputName) != std::string::npos) {
        const size_t dyn_bytes = dyn_info.size() * sizeof(int32_t);
        if (dyn_bytes > info.MemSize()) {
          LOG_ERROR("dyn_info bytes {} exceed input tensor bytes {} for {}",
                    dyn_bytes, info.MemSize(), input_name);
          exit(-1);
        }
        std::memcpy(tensor.Data(), dyn_info.data(), dyn_bytes);
      } else {
        const size_t image_bytes = ConvertBgrToYuv420sp(
            image_data, reinterpret_cast<uint8_t*>(tensor.Data()),
            info.MemSize());
        if (image_bytes == 0) {
          LOG_ERROR("input image bytes exceed input tensor bytes {} for {}",
                    info.MemSize(), input_name);
          exit(-1);
        }
      }

      task_info.input_map[input_name] = tensor;
    }

    // Push the request into the host-to-device stage.
    qh2d.queue.push(task_info);
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
        // Keep one host tensor per output and defer float32 conversion to the
        // final postprocess stage.
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

  // Post-process results with the same float32 + softmax flow as resnet50.cc.
  while (!qout.queue.empty()) {
    auto output_map = qout.queue.front().output_map;
    auto req_id = qout.queue.front().req_id;
    qout.queue.pop();
    int top1 = 0;
    const int topk = 1;

    for (auto& output : output_map) {
      // Convert the host output to float32 before applying softmax.
      auto info_f32 = output.second.Info().AsType(tcim::DataType::FLOAT32);
      auto output_tensor_f32 = tcim::Tensor::CreateHostTensor(info_f32);
      output.second.CastTo(output_tensor_f32);
      const size_t output_count = GetElementCount(output.second.Info());
      std::vector<float> probs =
          Softmax(static_cast<float*>(output_tensor_f32.Data()), output_count);

      std::vector<std::pair<float, int>> sort_pairs;
      // Rank scores using the softmax probabilities of the real output size.
      for (size_t i = 0; i < probs.size(); ++i) {
        sort_pairs.emplace_back(probs[i], static_cast<int>(i));
      }
      top1 = GetTopK(topk, sort_pairs);
    }

    // Verify the result. The bundled snake image should map to class 65.
    if (top1 != 65) {
      LOG_ERROR("top1 != 65");
      exit(-1);
    }
  }

  LOG_INFO("<=== resnet50_pipeline c++ example completed.");
  return 0;
}
