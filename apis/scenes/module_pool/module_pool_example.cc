// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.

#include <cstdlib>
#include <future>
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

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "datasets/imagenet.hpp"
#include "imageproc.hpp"
#include "models/yolov5s.hpp"
#include "module_pool.hpp"
#include "tcim/tcim_runtime.h"
#include "threads.hpp"
#include "utils.hpp"

template <typename T>
int GetTopk(int topk, std::vector<std::pair<T, int>> sort_pairs) {
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

std::vector<uint8_t> readBinaryFile(const std::string& file_path) {
  std::ifstream file(file_path,
                     std::ios::in | std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Can not open file:" << file_path << std::endl;
    return {};  // 空文件
  }
  // get file size
  std::streamsize size = file.tellg();
  if (size <= 0) {
    std::cerr << "File content is null." << std::endl;
    return {};  // 空文件
  }

  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    std::cerr << "Failed to load file:" << file_path << std::endl;
    return {};  // 空文件
  }

  return buffer;
}

PooledModule* LoadModelFromFile(ModulePool* module_pool,
                                std::string model_path) {
  if (!fs::exists(model_path)) {
    std::cerr
        << model_path
        << " not exist. you should run build.py in resnet50 example first."
        << std::endl;
    return nullptr;
  }

  auto pooled_md = module_pool->Load(model_path);
  if (pooled_md == nullptr) {
    std::cerr << "load model " << model_path << " fail." << std::endl;
    return nullptr;
  }
  std::cout << "load model " << model_path << std::endl;
  return pooled_md;
}

PooledModule* LoadModelFromBuffer(ModulePool* module_pool,
                                  std::string model_name, void* model_data,
                                  int len) {
  if (model_data == nullptr or len <= 0) {
    std::cerr << "Invalid model data ptr or model length!" << std::endl;
    return nullptr;
  }

  auto pooled_md = module_pool->Load(model_name, model_data, len);
  if (pooled_md == nullptr) {
    std::cerr << "load model " << model_name << " fail." << std::endl;
    return nullptr;
  }
  std::cout << "load model " << model_name << " success." << std::endl;
  return pooled_md;
}

int ObserveInference(ModulePool* pool_ptr,
                     std::vector<PooledModule*> pooled_mds, int interval,
                     bool& stop_flag) {
  while (!stop_flag) {
    for (const auto& pool_md : pooled_mds) {
      pool_md->GetStats(true);
    }
    pool_ptr->GetStats(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(interval));
  }

  return 0;
}

int ExecuteClassifyModel(PooledModule* pooled_md, std::string data_path,
                         int task_num) {
  std::cout << "start execute model, thread id:" << std::this_thread::get_id()
            << std::endl;

  // 1. prepare input tensors
  std::map<std::string, tcim::Tensor> input_map;
  int input_num = pooled_md->GetInputNum();
  std::cout << "Count of Input: " << input_num << std::endl;
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = pooled_md->GetInputName(idx);
    auto input_info = pooled_md->GetInputInfo(input_name, true);
    std::cout << "Input[" << input_name << "] " << input_info << std::endl;
    tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
  }

  // 2. preprocess input image
  if (!fs::exists(data_path)) {
    std::cerr << data_path << "not exist." << std::endl;
    return -1;
  }
  cv::Mat img_rgb;
  img_rgb = cv::imread(data_path);
  cv::Mat img_yuv;
  ImageProc::BgrToRgb((int8_t*)(img_rgb.data), img_rgb.rows, img_rgb.cols);
  cv::resize(img_rgb, img_rgb, {224, 224});
  cv::cvtColor(img_rgb, img_yuv, cv::COLOR_RGB2YUV_I420);
  int size = 224 * 224 * 3;
  ImageProc::I420To420sp((uint8_t*)input_map.at("input.1").Data(),
                         (uint8_t*)img_yuv.data, size);

  // 3. prepare input tensors
  std::map<std::string, tcim::Tensor> output_map;
  int output_num = pooled_md->GetOutputNum();
  std::cout << "Count of Output: " << output_num << std::endl;
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = pooled_md->GetOutputName(idx);
    auto output_info = pooled_md->GetOutputInfo(output_name, true);
    std::cout << "Output[" << output_name << "] " << output_info << std::endl;
    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    output_map.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
  }

  while (task_num > 0) {
    // 4. inference
    auto ret = pooled_md->Infer(input_map, output_map);

    // 5. postprocess, with no softmax
    int top1 = 0;
    const int topk = 1;
    for (auto& output : output_map) {
      std::vector<std::pair<int, int>> sort_pairs;
      for (int i = 0; i < 1000; ++i) {
        int8_t tmp_val = static_cast<int8_t*>(output.second.Data())[i];
        sort_pairs.emplace_back(static_cast<int>(tmp_val), i);
      }
      top1 = GetTopk(topk, sort_pairs);
    }

    // 6. check result (modify it when you change model or data)
    if (top1 != 65) {
      std::cout << "top1 != 65" << std::endl;
      return -1;
    }

    task_num--;
  }

  return 0;
}

int ExecuteDetectModel(PooledModule* pooled_md, std::string data_path,
                       int task_num) {
  std::cout << "start execute yolov5s model, thread id:"
            << std::this_thread::get_id() << std::endl;

  // 1. prepare input tensors
  std::map<std::string, tcim::Tensor> input_map;
  int input_num = pooled_md->GetInputNum();
  std::cout << "Count of Input: " << input_num << std::endl;
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = pooled_md->GetInputName(idx);
    auto input_info = pooled_md->GetInputInfo(input_name, true);
    std::cout << "Input[" << input_name << "] " << input_info << std::endl;
    tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
  }

  // 2. preprocess input image
  if (!fs::exists(data_path)) {
    std::cerr << data_path << "not exist." << std::endl;
    return -1;
  }
  cv::Mat img_raw = cv::imread(data_path);
  cv::Mat img_rgb;
  cv::Mat img_yuv;
  img_rgb = letterbox(img_raw, 640, 640);
  ImageProc::BgrToRgb((int8_t*)(img_rgb.data), img_rgb.rows, img_rgb.cols);
  cv::cvtColor(img_rgb, img_yuv, cv::COLOR_RGB2YUV_I420);
  int size = 640 * 640 * 3;
  ImageProc::I420To420sp((uint8_t*)input_map.at("images").Data(),
                         (uint8_t*)img_yuv.data, size);
  auto it = input_map.find("dyn_info");
  if (it != input_map.end()) {
    int32_t dyn_info[10] = {0, 0, 1080, 1920, 360, 640, 140, 0, 140, 0};
    memcpy(it->second.Data(), dyn_info, 10 * sizeof(int32_t));
  }

  // 3. prepare output tensors
  std::map<std::string, tcim::Tensor> output_map;
  std::map<std::string, tcim::Tensor> output_map_f32;
  int output_num = pooled_md->GetOutputNum();
  std::cout << "Count of Output: " << output_num << std::endl;
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = pooled_md->GetOutputName(idx);
    auto output_info = pooled_md->GetOutputInfo(output_name, true);
    std::cout << "Output[" << output_name << "] " << output_info << std::endl;
    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    output_map.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
    auto output_info_f32 = output_info.AsType(tcim::DataType::FLOAT32);
    auto output_tensor_f32 = tcim::Tensor::CreateHostTensor(output_info_f32);
    output_map_f32.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor_f32));
  }

  while (task_num > 0) {
    // 4. inference
    auto ret = pooled_md->Infer(input_map, output_map);

    // 5. postprocess
    std::vector<DetectOutput> outputs;
    for (auto& output : output_map_f32) {
      output_map[output.first].CastTo(output.second);
      DetectOutput out;
      out.data = (float*)output.second.Data();
      auto& shape = output.second.Info().Shape();
      out.num_anchors = shape[1] * shape[2] * shape[3];
      out.stride = 640 / shape[2];
      outputs.emplace_back(out);
    }
    YoloV5 yolov5;
    auto detections = yolov5.postprocess(img_raw, outputs);

    // 6. print and draw
    std::cout << "detection thread id:" << std::this_thread::get_id()
              << ", detect num:" << detections.size() << std::endl;
    for (const auto& detection : detections) {
      printf("box[%d, %d, %d, %d], conf:%f, cls:%d\n", detection.box.x1,
             detection.box.y1, detection.box.x2, detection.box.y2,
             detection.conf, detection.cls);
    }

    // check result
    if (detections.size() != 17) {
      std::cerr << "[Error] detection result wrong, detect num != 17"
                << std::endl;
      return -1;
    }

    task_num--;
  }

  return 0;
}

int main() {
  std::cout << "===> module pool c++ example start..." << std::endl;
  const char* houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target =
      houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
  if (houmo_target != "xh1") {
    std::cerr << "Unsupported backend " << houmo_target << std::endl;
    exit(-1);
  }
  std::cout << "tcim version:" << tcim::GetVersion()
            << ", houmo target:" << houmo_target << std::endl;

  int core_num = 4;
  int md_num = 8;
  int inference_thread_num = 16;
  int infer_task_num = 500;
  auto pool_ptr = ModulePool::Init(core_num);

  // 1. load model
  // 1.1 prepare model file
  std::cout << "Load models: resnet50 & yolov5s" << std::endl;
  std::string resnet50_path = "./resnet50.hmm";
  std::string yolov5s_path = "./yolov5s.hmm";
  if (houmo_target == "xh2") {
    resnet50_path = "./resnet50_xh2_b1_1core.hmm";
    yolov5s_path = "./yolov5s_clip_xh2_b1_1core.hmm";
  }
  // 1.2 prepare model buffer
  std::string resnet50_name = "resnet50_b1_1core";
  std::string yolov5s_name = "yolov5s_b1_1core";
  std::vector<uint8_t> resnet50_buffer = readBinaryFile(resnet50_path);
  std::vector<uint8_t> yolov5s_buffer = readBinaryFile(yolov5s_path);
  // 1.3 load model from file
  std::vector<std::future<PooledModule*>> resnet50_load_res;
  std::vector<std::future<PooledModule*>> yolov5s_load_res;
  for (int i = 0; i < md_num; i++) {
    resnet50_load_res.emplace_back(std::async(
        std::launch::async, LoadModelFromFile, pool_ptr, resnet50_path));
    yolov5s_load_res.emplace_back(std::async(
        std::launch::async, LoadModelFromFile, pool_ptr, yolov5s_path));
  }
  // 1.4 load model from buffer
  for (int i = 0; i < md_num; i++) {
    resnet50_load_res.emplace_back(std::async(
        std::launch::async, LoadModelFromBuffer, pool_ptr, resnet50_name,
        resnet50_buffer.data(), resnet50_buffer.size()));
    yolov5s_load_res.emplace_back(
        std::async(std::launch::async, LoadModelFromBuffer, pool_ptr,
                   yolov5s_name, yolov5s_buffer.data(), yolov5s_buffer.size()));
  }
  std::vector<PooledModule*> resnet50_pooled_mds;
  std::vector<PooledModule*> yolov5s_pooled_mds;
  for (int i = 0; i < resnet50_load_res.size(); i++) {
    PooledModule* pooled_md = resnet50_load_res[i].get();
    if (pooled_md == nullptr) {
      continue;
    }
    resnet50_pooled_mds.emplace_back(pooled_md);
  }
  for (int i = 0; i < yolov5s_load_res.size(); i++) {
    PooledModule* pooled_md = yolov5s_load_res[i].get();
    if (pooled_md == nullptr) {
      continue;
    }
    yolov5s_pooled_mds.emplace_back(pooled_md);
  }
  if (resnet50_pooled_mds.size() != resnet50_load_res.size() or
      yolov5s_pooled_mds.size() != yolov5s_load_res.size()) {
    std::cerr << "Load models failed." << std::endl;
    exit(-1);
  }
  std::cout << "Get pooled modules, resnet50 num is "
            << resnet50_pooled_mds.size() << ", yolov5s num is "
            << yolov5s_pooled_mds.size() << std::endl;

  bool stop_flag = false;
  int interval = 1000;  // 1000ms
  std::vector<PooledModule*> observed_mds = {
      resnet50_pooled_mds[0], resnet50_pooled_mds.back(), yolov5s_pooled_mds[0],
      yolov5s_pooled_mds.back()};
  auto observer = std::async(std::launch::async, ObserveInference, pool_ptr,
                             observed_mds, interval, std::ref(stop_flag));

  // 2. execute inference
  std::vector<std::future<int>> resnet50_execute_res;
  std::vector<std::future<int>> yolov5s_execute_res;
  std::string resnet50_data_path = "../../data/snake.jpg";
  std::string yolov5s_data_path = "../../data/000000000139.jpg";
  for (int i = 0; i < inference_thread_num; i++) {
    int resnet50_idx = i % resnet50_pooled_mds.size();
    int yolov5s_idx = i % yolov5s_pooled_mds.size();
    resnet50_execute_res.emplace_back(std::async(
        std::launch::async, ExecuteClassifyModel,
        resnet50_pooled_mds[resnet50_idx], resnet50_data_path, infer_task_num));
    yolov5s_execute_res.emplace_back(std::async(
        std::launch::async, ExecuteDetectModel, yolov5s_pooled_mds[yolov5s_idx],
        yolov5s_data_path, infer_task_num));
  }
  bool res_flag = true;
  for (int i = 0; i < inference_thread_num; i++) {
    int tmp_resnet50_res = resnet50_execute_res[i].get();
    int tmp_yolov5s_res = yolov5s_execute_res[i].get();
    if (tmp_resnet50_res != 0) {
      res_flag = false;
      std::cout << "execute resnet50 thread " << i << " failed" << std::endl;
    }
    if (tmp_yolov5s_res != 0) {
      res_flag = false;
      std::cout << "execute yolov5s thread " << i << " failed" << std::endl;
    }
  }
  if (!res_flag) {
    std::cerr << "Execute models failed." << std::endl;
  }

  stop_flag = true;
  observer.get();

  for (auto& ptr : resnet50_pooled_mds) {
    delete ptr;
  }
  for (auto& ptr : yolov5s_pooled_mds) {
    delete ptr;
  }
  auto pool_stats = pool_ptr->GetStats(true);
  delete pool_ptr;

  printf("<=== module pool c++ example completed.\n\n");
  return 0;
}
