// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.

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
#include "imageproc.hpp"
#include "tcim/tcim_runtime.h"
#include "threads.hpp"
#include "utils.hpp"

struct CliArguments {
  std::string model_path;
  size_t sample_num = 1;
};

typedef struct {
  std::map<std::string, tcim::Tensor> input_map;
  std::map<std::string, tcim::Tensor> output_map;
  uint64_t req_id;
  bool is_end = false;
} TaskInfo;

typedef struct {
  std::queue<TaskInfo> queue;
  std::mutex mutex;
  std::condition_variable cond;
  std::map<std::string, tcim::TensorInfo> input_info_map;
  std::map<std::string, tcim::TensorInfo> output_info_map;
} TaskQueue;

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
        std::cout << "Usage: -h" << std::endl;
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

template <typename T>
int get_topk(int topk, std::vector<std::pair<T, int>> sort_pairs) {
  std::sort(sort_pairs.begin(), sort_pairs.end(),
            [](const std::pair<T, int>& a, const std::pair<T, int>& b) {
              return a.first > b.first;
            });

  for (int i = 0; i < topk; ++i) {
    std::cout << "top" << (i + 1) << ": Index=" << sort_pairs[i].second
              << " Conf=" << sort_pairs[i].first << ", Label=["
              << Imagenet::GetLabel(sort_pairs[i].second) << "]" << std::endl;
  }

  return sort_pairs[0].second;
}

int main(int argc, char* argv[]) {
  printf("\n===> resnet50_pipeline c++ example start...\n");
  const char* houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target =
      houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
  if (houmo_target != "xh2") {
    std::cerr << "Unsupported backend " << houmo_target << std::endl;
    exit(-1);
  }
  printf("tcim version: %s, houmo target: %s \n", tcim::GetVersion().c_str(),
         houmo_target.c_str());

  std::string default_model_path = "./resnet50_xh2_b1_1core.hmm";
  // set the parameters
  CliArguments arguments;
  arguments.model_path = default_model_path;
  arguments.sample_num = 10;
  ParseArgs(&arguments, argc, argv);
  std::cout << "model: " << arguments.model_path << std::endl;
  std::cout << "samples: " << arguments.sample_num << std::endl;

  std::string model_path = arguments.model_path;
  if (!fs::exists(model_path)) {
    std::cerr << model_path << "not exist." << std::endl;
    exit(-1);
  }
  // load model
  auto module = tcim::Module::LoadFromFile(model_path);
  if (!module) {
    std::cerr << "load model " << model_path << " fail." << std::endl;
    exit(-1);
  }
  printf("model %s loaded.\n", model_path.c_str());

  // pipeline queue
  TaskQueue qh2d;    // input queue of h2d copy thread
  TaskQueue qinfer;  // input queue of infer thread
  TaskQueue qd2h;    // input queue of d2h copy thread
  TaskQueue qout;    // output queue of d2h copy thread

  // prepare input
  int input_num = module.GetInputNum();
  std::cout << "Count of Input: " << input_num << std::endl;
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name);
    std::cout << "Input[" << input_name << "] " << input_info << std::endl;
    qh2d.input_info_map[input_name] = input_info;
    qinfer.input_info_map[input_name] = input_info;
    qd2h.input_info_map[input_name] = input_info;
  }

  // prepare output
  int output_num = module.GetOutputNum();
  std::cout << "Count of Output: " << output_num << std::endl;
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name);
    std::cout << "Output[" << output_name << "] " << output_info << std::endl;
    qh2d.output_info_map[output_name] = output_info;
    qinfer.output_info_map[output_name] = output_info;
    qd2h.output_info_map[output_name] = output_info;
  }

  // input preprocess
  std::string data_path = "../../data/snake.jpg";
  if (!fs::exists(data_path)) {
    std::cerr << data_path << "not exist." << std::endl;
    exit(-1);
  }

  cv::Mat img_bgr;
  cv::Mat img_processed;
  size_t img_size = 0;
  img_bgr = cv::imread(data_path);

  cv::Mat img_rgb;
  cv::Mat img_norm;
  const float mean[3] = {123.675f, 116.28f, 103.53f};
  const float std[3] = {58.395f, 57.12f, 57.375f};
  cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
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

  // prepare input
  auto& name = qh2d.input_info_map.begin()->first;
  auto info = qh2d.input_info_map.begin()->second.AsContiguous();
  for (int i = 0; i < arguments.sample_num; i++) {
    auto data = malloc(img_size);
    std::shared_ptr<void> data_ptr(data, free);
    memcpy(data, reinterpret_cast<void*>(img_processed.data), img_size);

    TaskInfo task_info;
    task_info.req_id = i;
    tcim::Tensor tensor;

    // convert f32 buffer to f16 buffer
    auto info_f32 = info.AsType(tcim::DataType::FLOAT32);
    auto tensor_f32 = tcim::Tensor::CreateHostTensor(info_f32, img_size, data);
    tensor = tcim::Tensor::CreateHostTensor(info);
    tensor_f32.CastTo(tensor);

    task_info.input_map[name] = tensor;
    qh2d.queue.push(task_info);
  }
  std::cout << "sample queue size is " << qh2d.queue.size() << std::endl;

  // send end point
  TaskInfo task_info;
  task_info.is_end = true;
  qh2d.queue.push(task_info);

  // define h2d thread
  auto H2D = [](TaskQueue& qin, TaskQueue& qout) {
    printf("thread H2D start...\n");
    // h2d loop
    while (true) {
      // get data from the task queue
      std::unique_lock<std::mutex> lock_in(qin.mutex);
      while (qin.queue.empty()) {
        qin.cond.wait(lock_in);
      }
      auto task_info = qin.queue.front();
      if (task_info.is_end) {
        lock_in.unlock();
        std::unique_lock<std::mutex> lock_out(qout.mutex);
        qout.queue.push(task_info);
        qout.cond.notify_all();
        lock_out.unlock();
        break;
      }
      auto& input_map = task_info.input_map;
      auto& output_map = task_info.output_map;
      qin.queue.pop();
      lock_in.unlock();

      for (auto& input : input_map) {
        auto& name = input.first;
        auto& info = qin.input_info_map[name];
        auto tensor = tcim::Tensor::CreateDeviceTensor(info);
        input.second.CopyTo(tensor);
        input.second = tensor;
      }

      for (auto& output_info : qin.output_info_map) {
        auto& name = output_info.first;
        auto& info = output_info.second;
        auto tensor = tcim::Tensor::CreateDeviceTensor(info);
        task_info.output_map[name] = tensor;
      }

      // send to infer thread
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(task_info);
      qout.cond.notify_all();
      lock_out.unlock();
    }
  };

  // define d2h thread
  auto D2H = [](TaskQueue& qin, TaskQueue& qout) {
    printf("thread D2H start...\n");
    // d2h loop
    while (true) {
      // get data from the task queue
      std::unique_lock<std::mutex> lock_in(qin.mutex);
      while (qin.queue.empty()) {
        qin.cond.wait(lock_in);
      }
      auto task_info = qin.queue.front();
      if (task_info.is_end) {
        lock_in.unlock();
        break;
      }

      auto& output_map = task_info.output_map;
      qin.queue.pop();
      lock_in.unlock();

      for (auto& output : output_map) {
        auto& name = output.first;
        auto info = qin.output_info_map[name].AsContiguous();
        auto tensor = tcim::Tensor::CreateHostTensor(info);
        output.second.CopyTo(tensor);
        output.second = tensor;
      }

      // send to main thread
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(task_info);
      qout.cond.notify_all();
      lock_out.unlock();
    }
  };

  // define infer thread
  auto Infer = [](tcim::Module& module, TaskQueue& qin, TaskQueue& qout) {
    printf("thread infer start...\n");
    int count = 0;

    // infer loop
    while (true) {
      // get data from the task queue
      std::unique_lock<std::mutex> lock_in(qin.mutex);
      while (qin.queue.empty()) {
        qin.cond.wait(lock_in);
      }
      auto task_info = qin.queue.front();
      if (task_info.is_end) {
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
      qin.queue.pop();
      lock_in.unlock();

      // set input to the module
      for (auto& input : input_map) {
        module.SetInput(input.first, input.second);
      }

      // set output to the module
      for (auto& output : output_map) {
        module.SetOutput(output.first, output.second);
      }

      // run and sync
      module.Run();
      module.Sync();

      // send to D2H thread
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(task_info);
      qout.cond.notify_all();
      lock_out.unlock();

      count++;
      printf("run sample %lld end.\n", req_id);
    }

    printf("infer thread completed. %d sampels tested.\n", count);
  };

  // create threads
  std::vector<std::thread> threads;
  threads.push_back(std::thread(H2D, std::ref(qh2d), std::ref(qinfer)));
  threads.push_back(
      std::thread(Infer, std::ref(module), std::ref(qinfer), std::ref(qd2h)));
  threads.push_back(std::thread(D2H, std::ref(qd2h), std::ref(qout)));

  // wait all threads done
  for (auto& t : threads) {
    t.join();
  }

  // postprocess without softmax, and check result
  while (!qout.queue.empty()) {
    auto output_map = qout.queue.front().output_map;
    auto req_id = qout.queue.front().req_id;
    qout.queue.pop();
    int top1 = 0;
    const int topk = 1;
    for (auto& output : output_map) {
      auto info_f32 = output.second.Info().AsType(tcim::DataType::FLOAT32);
      auto output_tensor_f32 = tcim::Tensor::CreateHostTensor(info_f32);
      output.second.CastTo(output_tensor_f32);

      std::vector<std::pair<float, int>> sort_pairs;
      for (int i = 0; i < 1000; ++i) {
        sort_pairs.emplace_back(
            static_cast<float*>(output_tensor_f32.Data())[i], i);
      }
      top1 = get_topk(topk, sort_pairs);
    }

    // check result, modify it when you change model or data
    if (top1 != 65) {
      std::cout << "top1 != 65" << std::endl;
      exit(-1);
    }
  }

  printf("<=== resnet50_pipeline c++ example completed.\n\n");
  return 0;
}
