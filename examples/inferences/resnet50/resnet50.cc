// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.

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

#include "tcim/tcim_runtime.h"

#include "datasets/imagenet.hpp"
#include "imageproc.hpp"
#include "threads.hpp"
#include "utils.hpp"


int main() {
  printf("\n===> resnet50 c++ example start...\n");
  printf("tcim version: %s\n", tcim::GetVersion().c_str());

  // 1. load model
  std::cout << "LoadFromFile resnet50" << std::endl;
  std::string model_path = "resnet50.hmm";
  if (!fs::exists(model_path)) {
    std::cerr << model_path << " not exist. you should run build.py in resnet50 example first." << std::endl;
    exit(-1);
  }
  auto module = tcim::Module::LoadFromFile(model_path);
  if (!module) {
    std::cerr << " load model " << model_path << " fail." << std::endl;
    exit(-1);
  }
  printf("model %s loaded.\n", model_path.c_str());

  // 2. get input info
  std::map<std::string, tcim::Tensor> input_map;
  int input_num = module.GetInputNum();
  std::cout << "Count of Input: " << input_num << std::endl;
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name).AsContiguous();
    std::cout << "Input[" << input_name << "] " << input_info << std::endl;
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    input_map.insert(std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
  }

  // 3. input preprocess
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
  int size = 224*224*3;
  ImageProc::I420To420sp((uint8_t *)input_map.at("input.1").Data(), (uint8_t *)img_yuv.data, size);

  // 4. get output info
  std::map<std::string, tcim::Tensor> output_map;
  int output_num = module.GetOutputNum();
  std::cout << "Count of Output: " << output_num << std::endl;
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name).AsContiguous();
    std::cout << "Output[" << output_name << "] " << output_info << std::endl;
    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    output_map.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
  }

  // 5. set input
  for (const auto& input : input_map) {
    module.SetInput(input.first, input.second);
  }

  // 6. run and sync
  module.Run();
  module.Sync();

  // 7. get output
  for (auto& output : output_map) {
    module.GetOutput(output.first, output.second);
  }

  // 8. postprocess, with no softmax
  int top1 = 0;
  for (auto& output : output_map) {
    std::vector<std::pair<int8_t, int>> sort_pairs;
    for (int i = 0; i < 1000; ++i) {
      sort_pairs.emplace_back(static_cast<int8_t*>(output.second.Data())[i], i);
    }
    std::sort(sort_pairs.begin(), sort_pairs.end(),
              [](const std::pair<int8_t, int>& a, const std::pair<int8_t, int>& b) {
              return a.first > b.first;
              });

    const int topk = 5;
    for (int i = 0; i < topk; ++i) {
      std::cout << "top" << (i + 1) << ": Index="
                << sort_pairs[i].second << " Conf=" << static_cast<int>(sort_pairs[i].first)
                << ", Label=[" << Imagenet::GetLabel(sort_pairs[i].second) << "]" << std::endl;
    }
    // check result, modify it when you change model or data
    if (sort_pairs[0].second != 65) {
      std::cout << "top1 != 65" << std::endl;
      exit(-1);
    }
  }

  printf("<=== resnet50 c++ example completed.\n\n");
  return 0;
}
