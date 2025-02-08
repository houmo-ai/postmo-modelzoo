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

#include "tcim/tcim_runtime.h"

#include "imageproc.hpp"
#include "imagenet.hpp"
#include "threads.hpp"
#include "utils.hpp"


struct CliArguments {
  std::string model_path;
  size_t thread_num = 1;
  size_t sample_num = 1;
  size_t device_num = 1;
};


typedef struct {
  std::map<std::string, std::shared_ptr<void>> data_map;
  uint64_t req_id;
} TaskInfo;


typedef struct {
  std::queue<TaskInfo> queue;
  std::mutex mutex;
  std::condition_variable cond;
  std::map<std::string, tcim::TensorInfo> info_map;
} TaskQueue;


typedef struct {
  std::string model_path;
  tcim::Module::WeightManager wm;
} ThreadInfo;


bool ParseArgs(CliArguments *arguments, int argc, char *argv[]) {
#if !defined(_MSC_VER)
  int option_idx = 0;
  struct option long_options[] = {
      {"help", 0, 0, 'h'},
      {"devices", 1, 0, 'n'},
      {"threads", 1, 0, 't'},
      {"samples", 1, 0, 's'},
  };

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
      std::cerr << "Unsupported option: " << static_cast<char>(ch) << std::endl;
      return false;
    }
  }
#endif
  return true;
}


int main(int argc, char *argv[]) {
  printf("\n===> resnet50_multistreams c++ example start...\n");
  printf("tcim version: %s\n", tcim::GetVersion().c_str());

  // set the parameters
  CliArguments arguments;
  arguments.model_path = "resnet50.hmm";
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
  std::cout << "model: " << arguments.model_path << std::endl;
  std::cout << "devices: " << arguments.device_num << std::endl;
  std::cout << "threads: " << arguments.thread_num << std::endl;
  std::cout << "samples: " << arguments.sample_num << std::endl;

  std::string model_path = arguments.model_path;
  if (!fs::exists(model_path)) {
    std::cerr << model_path << "not exist." << std::endl;
    exit(-1);
  }
  std::vector<std::thread> threads;

  // 1. input preprocess
  std::string data_path = "../../data/snake.jpg";
  if (!fs::exists(data_path)) {
    std::cerr << data_path << "not exist." << std::endl;
    exit(-1);
  }
  cv::Mat img_rgb;
  cv::Mat img_yuv;
  img_rgb = cv::imread(data_path);
  ImageProc::BgrToRgb((int8_t *)(img_rgb.data), img_rgb.rows, img_rgb.cols);
  cv::resize(img_rgb, img_rgb, {224, 224});
  cv::cvtColor(img_rgb, img_yuv, cv::COLOR_RGB2YUV_I420);
  int size = 224 * 224 * 3;

  // 2. prepare input & output queue
  TaskQueue qin;
  TaskQueue qout;
  for (int i = 0; i < arguments.sample_num; i++) {
    auto data = malloc(size);
    std::shared_ptr<void> data_ptr(data, free);
    ImageProc::I420To420sp((uint8_t *)data, (uint8_t *)img_yuv.data, size);
    TaskInfo tinfo;
    tinfo.req_id = i;
    tinfo.data_map.insert(std::pair<std::string, std::shared_ptr<void>>("", data_ptr));
    qin.queue.push(tinfo);
  }
  std::cout << "sample queue size is " << qin.queue.size() << std::endl;

  // 3. define threads
  auto thread_func = [](int tid,
                        int did,
                        ThreadInfo& info,
                        TaskQueue& qin,
                        TaskQueue& qout,
                        Barrier& barrier) {
    // 3.1 load model
    tcim::Module::Option option(info.wm);
    auto module = tcim::Module::LoadFromFile(info.model_path, option);
    if (!module) {
      std::cerr << "thread " << tid << " on device " << did << " load model "
                << info.model_path << " fail." << std::endl;
      exit(-1);
    }
    printf("thread %d on device %d model %s loaded.\n", tid, did, info.model_path.c_str());

    // 3.2 create a stream and set to the module
    tcim::Stream stream(true);
    module.SetStream(stream);

    // 3.3 prepare input
    int input_num = module.GetInputNum();
    // std::cout << "Count of Input: " << input_num << std::endl;
    for (int idx = 0; idx < input_num; idx++) {
      auto input_name = module.GetInputName(idx);
      auto input_info = module.GetInputInfo(input_name).AsContiguous();
      std::cout << "Input[" << input_name << "] " << input_info << std::endl;
      qin.info_map[input_name] = input_info;
    }

    // 3.4 prepare output
    int output_num = module.GetOutputNum();
    // std::cout << "Count of Output: " << output_num << std::endl;
    for (int idx = 0; idx < output_num; idx++) {
      auto output_name = module.GetOutputName(idx);
      auto output_info = module.GetOutputInfo(output_name).AsContiguous();
      std::cout << "Output[" << output_name << "] " << output_info << std::endl;
      qout.info_map[output_name] = output_info;
    }

    // 3.5 wait until all threads ready
    barrier.barrier();
    printf("thread %d on device %d infer start...\n", tid, did);
    int count = 0;

    // 3.6 infer loop
    while (true) {
      // 3.6.1 get data from the task queue
      std::unique_lock<std::mutex> lock_in(qin.mutex);
      if (qin.queue.empty()) {
        lock_in.unlock();
        break;
      }
      auto input_map = qin.queue.front().data_map;
      auto req_id = qin.queue.front().req_id;
      qin.queue.pop();
      lock_in.unlock();

      // 3.6.2 set input to the module
      for (auto& info : qin.info_map) {
        auto size = info.second.MemSize();
        auto input_tensor = tcim::Tensor::CreateHostTensor(info.second, size, input_map[""].get());
        module.SetInput(info.first, input_tensor);
      }

      // 3.6.3 run and sync
      module.Run();
      module.Sync();

      // 3.6.4 get output and push to the output queue
      TaskInfo tinfo;
      tinfo.req_id = req_id;
      for (auto& info : qout.info_map) {
        auto size = info.second.MemSize();
        auto data = malloc(size);
        std::shared_ptr<void> data_ptr(data, free);
        auto output_tensor = tcim::Tensor::CreateHostTensor(info.second, size, data);
        module.GetOutput(info.first, output_tensor);
        tinfo.data_map.insert(std::pair<std::string, std::shared_ptr<void>>(info.first, data_ptr));
      }
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(tinfo);
      lock_out.unlock();
      count++;
      printf("thread %d on device %d run sample %lld end.\n", tid, did, req_id);
    }

    printf("thread %d on device %d completed. %d sampels tested.\n", tid, did, count);
  };

  // 4. create threads
  Barrier barrier(arguments.thread_num * arguments.device_num);
  auto tinfo = new ThreadInfo[arguments.device_num];
  int tid = 0;
  for (int did = 0; did < arguments.device_num; did++) {
    auto wm = tcim::Module::WeightManager::CreateWeightManager(did);
    tinfo[did].model_path = arguments.model_path;
    tinfo[did].wm = wm;

    for (int i = 0; i < arguments.thread_num; i++) {
      threads.push_back(std::thread(thread_func, tid, did, std::ref(tinfo[did]), std::ref(qin),
                                    std::ref(qout), std::ref(barrier)));
      tid++;
    }
  }

  barrier.wait();

  // 5. wait all threads done
  for (auto & t: threads) {
    t.join();
  }
  delete[] tinfo;

  // 6. postprocess without softmax, and check result
  while (!qout.queue.empty()) {
    auto output_map = qout.queue.front().data_map;
    auto req_id = qout.queue.front().req_id;
    qout.queue.pop();
    int top1 = 0;
    for (auto& output : output_map) {
      std::vector<std::pair<int8_t, int>> sort_pairs;
      for (int i = 0; i < 1000; ++i) {
        sort_pairs.emplace_back(static_cast<int8_t*>(output.second.get())[i], i);
      }
      std::sort(sort_pairs.begin(), sort_pairs.end(),
                [](const std::pair<int8_t, int>& a, const std::pair<int8_t, int>& b) {
                return a.first > b.first;
                });
      top1 = sort_pairs[0].second;
      std::cout << "sample " << req_id << " top1: Index="
                << top1 << " Conf=" << (int)(sort_pairs[0].first)
                << ", Label=[" << Imagenet::GetLabel(top1) << "]" << std::endl;
    }

    // check result, modify it when you change model or data
    if (top1 != 65) {
      std::cout << "top1 != 65" << std::endl;
      exit(-1);
    }
  }

  printf("<=== resnet50_multistreams c++ example completed.\n\n");
  return 0;
}
