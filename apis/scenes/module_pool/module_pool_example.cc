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
    LOG_INFO("TOP {}: Index={}, Conf={}, Label=[{}]", (i + 1),
             sort_pairs[i].second, sort_pairs[i].first,
             Imagenet::GetLabel(sort_pairs[i].second));
  }

  return sort_pairs[0].second;
}

std::string TensorInfo2Str(const tcim::TensorInfo& tensor_info) {
  std::stringstream ss;
  ss << tensor_info;
  return ss.str();
}

std::vector<uint8_t> ReadBinaryFile(const std::string& file_path) {
  std::ifstream file(file_path,
                     std::ios::in | std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    LOG_ERROR("Can not open file {}.", file_path);
    return {};
  }
  // get file size
  std::streamsize size = file.tellg();
  if (size <= 0) {
    LOG_ERROR("File content is null.");
    return {};
  }

  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    LOG_ERROR("Failed to read model file {}.", file_path);
    return {};
  }

  return buffer;
}

PooledModule* LoadModelFromFile(ModulePool* module_pool,
                                std::string model_path,
                                tcim::Module::WeightManager& wm) {
  if (!fs::exists(model_path)) {
    LOG_ERROR("Model file {} doesn't exist.", model_path);
    return nullptr;
  }

  tcim::Module::Option option(wm);
  auto pooled_md = module_pool->Load(model_path, option);
  if (pooled_md == nullptr) {
    LOG_ERROR("Failed to load model {}.", model_path);
    return nullptr;
  }
  LOG_INFO("Load model {} and pool it into {}.", model_path,
           reinterpret_cast<void*>(pooled_md));

  return pooled_md;
}

PooledModule* LoadModelFromBuffer(ModulePool* module_pool,
                                  std::string model_name, void* model_data,
                                  int len, tcim::Module::WeightManager& wm) {
  if (model_data == nullptr or len <= 0) {
    LOG_ERROR("Invalid model data ptr or model length {}!", len);
    return nullptr;
  }

  tcim::Module::Option option(wm);
  auto pooled_md = module_pool->Load(model_name, model_data, len, option);
  if (pooled_md == nullptr) {
    LOG_ERROR("Failed to load model {} from buffer {}!", model_name,
              model_data);
    return nullptr;
  }
  LOG_INFO("Load model {} from buffer and pool it into {}.", model_name,
           reinterpret_cast<void*>(pooled_md));
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
  LOG_WARNING("===> Start to execute classify model, pooled module is ({}){}.",
              reinterpret_cast<void*>(pooled_md), pooled_md->GetPooledMdName());

  // 1. prepare input tensors
  std::map<std::string, tcim::Tensor> input_map;
  int input_num = pooled_md->GetInputNum();
  LOG_INFO("Count of Input: {}.", input_num);
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = pooled_md->GetInputName(idx);
    auto input_info = pooled_md->GetInputInfo(input_name, true);
    LOG_INFO("  Input[{}] {}.", input_name, TensorInfo2Str(input_info));
    tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
  }

  // 2. preprocess input image
  if (!fs::exists(data_path)) {
    LOG_ERROR("Input data {} doesn't exist.", data_path);
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
  std::map<std::string, tcim::Tensor> output_host_map;
  int output_num = pooled_md->GetOutputNum();
  LOG_INFO("Count of Output: {}.", output_num);
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = pooled_md->GetOutputName(idx);
    auto output_info = pooled_md->GetOutputInfo(output_name, false);
    LOG_INFO("  Output[{}] {}", output_name, TensorInfo2Str(output_info));
    // create device tensor to stroe inference results
    auto output_tensor = tcim::Tensor::CreateDeviceTensor(output_info);
    output_map.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
    // create host tensor to get inference results
    auto output_host_info = output_info.AsContiguous();
    auto output_host_tensor = tcim::Tensor::CreateHostTensor(output_host_info);
    output_host_map.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_host_tensor));
  }

  while (task_num > 0) {
    // 4. inference
    auto ret = pooled_md->Infer(input_map, output_map);

    // 5. postprocess, with no softmax
    int top1 = 0;
    const int topk = 1;
    for (auto& output : output_map) {
      output.second.CopyTo(output_host_map[output.first]);
      std::vector<std::pair<int, int>> sort_pairs;
      for (int i = 0; i < 1000; ++i) {
        int8_t tmp_val =
            static_cast<int8_t*>(output_host_map[output.first].Data())[i];
        sort_pairs.emplace_back(static_cast<int>(tmp_val), i);
      }
      top1 = GetTopk(topk, sort_pairs);
    }

    // 6. check result (modify it when you change model or data)
    if (top1 != 65) {
      LOG_ERROR("Top1 is {}, the expected value is 65.", top1);
      return -1;
    }

    task_num--;
  }

  LOG_WARNING("<=== End to execute classify model, pooled module is ({}){}.",
              reinterpret_cast<void*>(pooled_md), pooled_md->GetPooledMdName());
  return 0;
}

int ExecuteDetectModel(PooledModule* pooled_md, std::string data_path,
                       int task_num) {
  LOG_WARNING("===> Start to execute detection model, pooled module is ({}){}.",
              reinterpret_cast<void*>(pooled_md), pooled_md->GetPooledMdName());

  // 1. prepare input tensors
  std::map<std::string, tcim::Tensor> input_map;
  int input_num = pooled_md->GetInputNum();
  LOG_INFO("Count of Input: {}.", input_num);
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = pooled_md->GetInputName(idx);
    auto input_info = pooled_md->GetInputInfo(input_name, true);
    LOG_INFO("  Input[{}] {}", input_name, TensorInfo2Str(input_info));
    tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
  }

  // 2. preprocess input image
  if (!fs::exists(data_path)) {
    LOG_ERROR("Input data {} doesn't exist.", data_path);
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
  LOG_INFO("Count of Output: {}.", output_num);
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = pooled_md->GetOutputName(idx);
    auto output_info = pooled_md->GetOutputInfo(output_name, true);
    LOG_INFO("  Output[{}] {}", output_name, TensorInfo2Str(output_info));
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

    // 6. print
    LOG_INFO("detect num {}.", detections.size());
    for (const auto& detection : detections) {
      LOG_DEBUG("  box[{}, {}, {}, {}], conf:{}, cls:{}.", detection.box.x1,
                detection.box.y1, detection.box.x2, detection.box.y2,
                detection.conf, detection.cls);
    }

    // check result
    if (detections.size() != 17) {
      LOG_ERROR("detect num is {}, the expected value is 17.",
                detections.size());
      return -1;
    }

    task_num--;
  }

  LOG_WARNING("<=== End to execute detection model, pooled module is ({}){}.",
              reinterpret_cast<void*>(pooled_md), pooled_md->GetPooledMdName());
  return 0;
}

int main() {
  LOG_WARNING("===> module pool c++ example start...");
  const char* houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target =
      houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
  if (houmo_target != "xh1") {
    LOG_ERROR("Unsupported backend {}.", houmo_target);
    exit(-1);
  }
  LOG_INFO("tcim version:{}, houmo target:{}.", tcim::GetVersion(),
           houmo_target);

  int device_id = 0;

  int core_num = 4;
  int pooled_md_num = 8;
  int inference_thread_num = 16;
  int infer_task_num = 500;
  auto pool_ptr = ModulePool::Init(core_num);

  // 1. load model
  // 1.1 prepare model file
  std::string resnet50_path = "./resnet50.hmm";
  std::string yolov5s_path = "./yolov5s.hmm";
  // 1.2 prepare model buffer
  std::string resnet50_name = "resnet50_b1_1core";
  std::string yolov5s_name = "yolov5s_b1_1core";
  std::vector<uint8_t> resnet50_buffer = ReadBinaryFile(resnet50_path);
  std::vector<uint8_t> yolov5s_buffer = ReadBinaryFile(yolov5s_path);
  auto res50_wm = tcim::Module::WeightManager::CreateWeightManager(device_id);
  auto yolov5s_wm = tcim::Module::WeightManager::CreateWeightManager(device_id);
  // 1.3 load model from file
  std::vector<std::future<PooledModule*>> resnet50_load_res;
  std::vector<std::future<PooledModule*>> yolov5s_load_res;
  for (int i = 0; i < pooled_md_num; i++) {
    resnet50_load_res.emplace_back(std::async(
        std::launch::async, LoadModelFromFile, pool_ptr, resnet50_path, std::ref(res50_wm)));
    yolov5s_load_res.emplace_back(std::async(
        std::launch::async, LoadModelFromFile, pool_ptr, yolov5s_path, std::ref(yolov5s_wm)));
  }
  // 1.4 load model from buffer
  for (int i = 0; i < pooled_md_num; i++) {
    resnet50_load_res.emplace_back(std::async(
        std::launch::async, LoadModelFromBuffer, pool_ptr, resnet50_name,
        resnet50_buffer.data(), resnet50_buffer.size(), std::ref(res50_wm)));
    yolov5s_load_res.emplace_back(
        std::async(std::launch::async, LoadModelFromBuffer, pool_ptr,
                   yolov5s_name, yolov5s_buffer.data(), yolov5s_buffer.size(), std::ref(yolov5s_wm)));
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
    LOG_ERROR("Load models failed.");
    exit(-1);
  }
  LOG_INFO("Get pooled modules, resnet50 num is {}, yolov5s num is {}.",
           resnet50_pooled_mds.size(), yolov5s_pooled_mds.size());

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
      LOG_ERROR("Execute classify thread {} failed.", i);
    }
    if (tmp_yolov5s_res != 0) {
      res_flag = false;
      LOG_ERROR("Execute detection thread {} failed.", i);
    }
  }
  if (!res_flag) {
    LOG_ERROR("Execute models failed.");
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

  LOG_WARNING("<=== module pool c++ example completed.");
  return 0;
}
