// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.

#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>

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


template <typename T>
int get_topk(int topk, std::vector<std::pair<T, int>> sort_pairs) {
  std::sort(sort_pairs.begin(), sort_pairs.end(),
            [](const std::pair<T, int>& a, const std::pair<T, int>& b) {
            return a.first > b.first;
            });

  for (int i = 0; i < topk; ++i) {
    std::cout << "top" << (i + 1) << ": Index="
              << sort_pairs[i].second << " Conf=" << sort_pairs[i].first
              << ", Label=[" << Imagenet::GetLabel(sort_pairs[i].second) << "]" << std::endl;
  }

  return sort_pairs[0].second;
}


int main() {
  printf("\n===> resnet50 c++ example start...\n");
  const char* houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target = houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
  if (houmo_target != "xh1") {
    std::cerr << "Not support houmo target:" << houmo_target << std::endl;
    exit(-1);
  }
  printf("tcim version: %s.\n", tcim::GetVersion().c_str());

  // 1. load model
  std::cout << "LoadFromFile resnet50" << std::endl;
  std::string model_path = "./resnet50.hmm";
  if (houmo_target == "xh2") {
    model_path =  "./resnet50_xh2_b1_1core.hmm";
  }
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
  std::map<std::string, tcim::TensorInfo> input_map_f32;
  int input_num = module.GetInputNum();
  std::cout << "Count of Input: " << input_num << std::endl;
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name).AsContiguous();
    auto input_info_f32 = input_info.AsType(tcim::DataType::FLOAT32);
    std::cout << "Input[" << input_name << "] " << input_info << std::endl;
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    input_map.insert(std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    if (houmo_target == "xh2") {
      std::cout << "Input f32[" << input_name << "] " << input_info_f32 << std::endl;
      input_map_f32.insert(std::pair<std::string, tcim::TensorInfo>(input_name, input_info_f32));
    }
  }

  // 3. input preprocess
  std::string data_path = "../../data/snake.jpg";
  if (!fs::exists(data_path)) {
    std::cerr << data_path << "not exist." << std::endl;
    exit(-1);
  }
  cv::Mat img_rgb;
  img_rgb = cv::imread(data_path);
  if (houmo_target == "xh1") {
    cv::Mat img_yuv;
    ImageProc::BgrToRgb((int8_t *)(img_rgb.data), img_rgb.rows, img_rgb.cols);
    cv::resize(img_rgb, img_rgb, {224, 224});
    cv::cvtColor(img_rgb, img_yuv, cv::COLOR_RGB2YUV_I420);
    int size = 224*224*3;
    ImageProc::I420To420sp((uint8_t *)input_map.at("input.1").Data(), (uint8_t *)img_yuv.data, size);
  } else if (houmo_target == "xh2") {
    cv::Mat img_norm;
    const float mean[3] = {123.675f, 116.28f, 103.53f};
    const float std[3] = {58.395f, 57.12f, 57.375f};
    cv::cvtColor(img_rgb, img_rgb, cv::COLOR_BGR2RGB);
    cv::resize(img_rgb, img_rgb, {224, 224});

    img_rgb.convertTo(img_norm, CV_32FC3);
    cv::Mat channels[3];
    cv::split(img_norm, channels);
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }
    cv::merge(channels, 3, img_norm);

    size_t img_bytes = img_norm.total() * img_norm.elemSize();
    auto input_tensor_f32 = tcim::Tensor::CreateHostTensor(
      input_map_f32.at("input.1"), img_bytes, static_cast<void*>(img_norm.data));
    input_tensor_f32.CastTo(input_map.at("input.1"));
  }

  // 4. get output info
  std::map<std::string, tcim::Tensor> output_map;
  std::map<std::string, tcim::Tensor> output_map_f32;
  int output_num = module.GetOutputNum();
  std::cout << "Count of Output: " << output_num << std::endl;
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name).AsContiguous();
    std::cout << "Output[" << output_name << "] " << output_info << std::endl;
    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    output_map.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
    if (houmo_target == "xh2") {
      auto output_info_f32 = output_info.AsType(tcim::DataType::FLOAT32);
      auto output_tensor_f32 = tcim::Tensor::CreateHostTensor(output_info_f32);
      output_map_f32.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor_f32));
    }
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
  const int topk = 5;
  if (houmo_target == "xh1") {
    for (auto& output : output_map) {
      std::vector<std::pair<int, int>> sort_pairs;
      for (int i = 0; i < 1000; ++i) {
        int8_t tmp_val = static_cast<int8_t*>(output.second.Data())[i];
        sort_pairs.emplace_back(static_cast<int>(tmp_val), i);
      }
      top1 = get_topk(topk, sort_pairs);
    }
  } else if (houmo_target == "xh2") {
    for (auto& output : output_map) {
      auto f32_opt = output_map_f32[output.first];
      output.second.CastTo(f32_opt);

      std::vector<std::pair<float, int>> sort_pairs;
      for (int i = 0; i < 1000; ++i) {
        sort_pairs.emplace_back(static_cast<float*>(f32_opt.Data())[i], i);
      }
      top1 = get_topk(topk, sort_pairs);
    }
  }
  // check result, modify it when you change model or data
  if (houmo_target == "xh1" && top1 != 65) {
    std::cout << "top1 != 65" << std::endl;
    exit(-1);
  }

  printf("<=== resnet50 c++ example completed.\n\n");
  return 0;
}
