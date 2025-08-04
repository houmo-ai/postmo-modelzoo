// Copyright (c) The Houmo.ai Authors. All rights reserved.

#include <cmath>
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

#include "datasets/imagenet.hpp"
#include "imageproc.hpp"
#include "log.hpp"
#include "models/resnet50.hpp"
#include "models/yolov5s.hpp"
#include "tcim/tcim_runtime.h"
#include "utils.hpp"

std::vector<std::vector<DetectResult>> detect(
    int batch_num, std::string model_path,
    const std::vector<cv::Mat> &cv_images) {
  LOG_INFO << "===> multibatch yolov5s example start...";
  std::vector<std::vector<DetectResult>> batch_detections;

  // 1. load model
  LOG_INFO << "Load yolov5s hmm model from file " << model_path;
  if (!fs::exists(model_path)) {
    LOG_ERROR << model_path << " doesn't exist, please check hmm model path.";
    return batch_detections;
  }
  auto module = tcim::Module::LoadFromFile(model_path);
  if (!module) {
    LOG_ERROR << "Load model " << model_path << " fail.";
    return batch_detections;
  }
  LOG_INFO << "Loaded yolov5s model " << model_path;

  YoloV5 yolov5;
  std::string img_str = "images";
  std::string resizer_crop_str = "resizer_crop_" + img_str;
  int32_t resizer_crop_info[10] = {0, 0, 1080, 1920, 360, 640, 140, 0, 140, 0};
  size_t resizer_crop_size = 10 * sizeof(int32_t);

  // 2. get input info
  std::map<std::string, tcim::Tensor> input_map;
  size_t input_img_width = 0;
  size_t input_img_height = 0;
  int input_num = module.GetInputNum();
  LOG_INFO << "Count of Input: " << input_num;
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name).AsContiguous();
    LOG_INFO << "Input[" << input_name << "] " << input_info;
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    if (input_name == img_str) {
      input_img_width = input_info.Shape()[3];
      input_img_height = input_info.Shape()[2];
    }
  }

  // 3. preprocess input images and then copy to input tensor
  size_t size = input_img_height * input_img_width * 3;
  for (int idx = 0; idx < batch_num; idx++) {
    cv::Mat img_yuv;
    cv::cvtColor(cv_images[idx], img_yuv, cv::COLOR_BGR2YUV_I420);
    ImageProc::I420To420sp(
        (uint8_t *)input_map.at(img_str).Data() + idx * (size / 2),
        (uint8_t *)img_yuv.data, size);
  }
  auto it = input_map.find(resizer_crop_str);
  if (it != input_map.end()) {
    for (int idx = 0; idx < batch_num; idx++) {
      memcpy(it->second.Data() + idx * resizer_crop_size, resizer_crop_info,
             resizer_crop_size);
    }
  }

  // 4. get output info
  std::map<std::string, tcim::Tensor> output_map;
  int output_num = module.GetOutputNum();
  LOG_INFO << "Count of Output: " << output_num;
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module.GetOutputName(idx);
    auto output_info =
        module.GetOutputInfo(output_name).AsContiguous().AsType(tcim::FLOAT32);
    LOG_INFO << "Output[" << output_name << "] " << output_info;
    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    output_map.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
  }

  // 5. set input
  for (const auto &input : input_map) {
    module.SetInput(input.first, input.second);
  }

  // 6. run and sync
  module.Run();
  module.Sync();

  // 7. get output
  for (auto &output : output_map) {
    auto output_tensor = module.GetOutput(output.first);
    output_tensor.CastTo(output.second);
  }

  // 8. postprocess
  for (int idx = 0; idx < batch_num; idx++) {
    std::vector<DetectOutput> batch_opt;
    for (auto &output : output_map) {
      DetectOutput out;
      auto shape = output.second.Info().Shape();
      auto mem_size = output.second.Info().MemSize();
      size_t tmp_size = mem_size / batch_num;
      out.data = (float *)(output.second.Data() + idx * tmp_size);
      out.num_anchors = shape[1] * shape[2] * shape[3];
      out.stride = 640 / shape[2];
      batch_opt.emplace_back(out);
    }

    auto detections = yolov5.postprocess(cv_images[idx], batch_opt);
    batch_detections.push_back(detections);
  }

  // 9. print and draw
  // create results folder
  std::string result_dir = "./demo_results";
  fs::path result_folder(result_dir);
  if (!fs::exists(result_folder)) {
    fs::create_directory(result_dir);
  }
  for (int idx = 0; idx < batch_detections.size(); idx++) {
    auto &detections = batch_detections[idx];
    LOG_INFO << "detection results, batch " << idx
             << ", detections num: " << detections.size();
    for (const auto &detection : detections) {
      printf("box[%d, %d, %d, %d], conf:%f, cls:%d\n", detection.box.x1,
             detection.box.y1, detection.box.x2, detection.box.y2,
             detection.conf, detection.cls);
      cv::rectangle(cv_images[idx],
                    cv::Point(detection.box.x1, detection.box.y1),
                    cv::Point(detection.box.x2, detection.box.y2),
                    cv::Scalar(0, 0, 255), 2);
    }

    std::string result_path =
        result_dir + "/yolov5s_output_batch" + std::to_string(idx) + ".jpg";
    cv::imwrite(result_path, cv_images[idx]);
    LOG_INFO << "detection results, batch " << idx << " saved to "
             << result_path;
  }

  LOG_INFO << "<=== multibatch yolov5s example end.";
  return batch_detections;
}

int classify(int batch_num, std::string model_path,
             const std::vector<cv::Mat> &cv_images,
             const std::vector<std::vector<DetectResult>> &batch_detections,
             int resize_type) {
  LOG_INFO << "===> multibatch resnet50 example start...";

  // 1. load model
  LOG_INFO << "Load resnet50 hmm model from file " << model_path;
  if (!fs::exists(model_path)) {
    LOG_ERROR << model_path << " doesn't exist, please check hmm model path.";
    return -1;
  }
  auto module = tcim::Module::LoadFromFile(model_path);
  if (!module) {
    LOG_ERROR << "Load model " << model_path << " fail.";
    return -2;
  }
  LOG_INFO << "Loaded resnet50 model " << model_path;

  Resnet50 resnet50;
  std::string img_str = "input.1";
  std::string resizer_crop_str = "resizer_crop_" + img_str;

  // 2. get input info
  std::map<std::string, tcim::Tensor> input_map;
  size_t input_img_width = 0;
  size_t input_img_height = 0;
  int input_num = module.GetInputNum();
  LOG_INFO << "Count of Input: " << input_num;
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name).AsContiguous();
    LOG_INFO << "Input[" << input_name << "] " << input_info;
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    if (input_name == img_str) {
      input_img_width = input_info.Shape()[3];
      input_img_height = input_info.Shape()[2];
    }
  }
  size_t resizer_crop_size =
      input_map[resizer_crop_str].Info().MemSize() / batch_num;

  // 3. get output info
  std::map<std::string, tcim::Tensor> output_map;
  int output_num = module.GetOutputNum();
  LOG_INFO << "Count of Output: " << output_num;
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module.GetOutputName(idx);
    auto output_info =
        module.GetOutputInfo(output_name).AsContiguous().AsType(tcim::FLOAT32);
    LOG_INFO << "Output[" << output_name << "] " << output_info;
    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    output_map.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
  }
  size_t opt_size = output_map.begin()->second.Info().MemSize() / batch_num;

  // iterate inputs
  for (int ipt_idx = 0; ipt_idx < batch_detections.size(); ipt_idx++) {
    auto &detections = batch_detections[ipt_idx];
    LOG_INFO << "batch " << ipt_idx
             << ", detections size:" << detections.size();

    // 4.1 preprocess input images and then copy to input tensor
    size_t size = input_img_width * input_img_height * 3;
    cv::Mat img_yuv;
    cv::cvtColor(cv_images[ipt_idx], img_yuv, cv::COLOR_BGR2YUV_I420);
    ImageProc::I420To420sp((uint8_t *)input_map.at(img_str).Data(),
                           (uint8_t *)img_yuv.data, size);

    std::vector<std::vector<int32_t>> resizer_crop_inputs;
    auto it = input_map.find(resizer_crop_str);
    if (it != input_map.end()) {
      // 4.2 calculate resizer_crop info and then copy to input tensor
      for (int det_idx = 0; det_idx < detections.size(); det_idx++) {
        // construct resizer_crop inputs
        std::vector<int32_t> resizer_crop_info(10);

        auto roi_h = TO_EVEN(detections[det_idx].box.h());
        auto roi_w = TO_EVEN(detections[det_idx].box.w());
        if (roi_h <= 0 || roi_w <= 0) {
          LOG_ERROR << "error: batch[" << ipt_idx << "][" << det_idx
                    << "] invalid roi, height is " << roi_h << ", width is "
                    << roi_w;
          continue;
        }
        // roi crop [y1, x1, h, w]
        resizer_crop_info[0] = TO_EVEN(detections[det_idx].box.y1);
        resizer_crop_info[1] = TO_EVEN(detections[det_idx].box.x1);
        resizer_crop_info[2] = roi_h;
        resizer_crop_info[3] = roi_w;

        if (resize_type == 0) {
          size_t target_h = resnet50.input_sizes_[1];
          size_t target_w = resnet50.input_sizes_[0];
          // calculate scaling factor
          float scale =
              std::min((1.0 * target_h / roi_h), (1.0 * target_w / roi_w));
          // check whether the scaling factor is valid
          if (scale > 16 || scale <= 1 / 32) {
            LOG_ERROR << "error: batch[" << ipt_idx << "][" << det_idx
                      << "] scale " << scale << " out of range(1/32, 16].";
            continue;
          }
          size_t scaled_h =
              TO_EVEN(static_cast<size_t>(std::round(roi_h * scale)));
          size_t scaled_w =
              TO_EVEN(static_cast<size_t>(std::round(roi_w * scale)));
          // resize [H, W]
          resizer_crop_info[4] = scaled_h;
          resizer_crop_info[5] = scaled_w;
          // pad [top, left, bottom, right]
          size_t pad_h = (target_h - scaled_h) >= 0 ? (target_h - scaled_h) : 0;
          size_t pad_w = (target_w - scaled_w) >= 0 ? (target_w - scaled_w) : 0;
          resizer_crop_info[6] = TO_EVEN(pad_h / 2);
          resizer_crop_info[7] = TO_EVEN(pad_w / 2);
          resizer_crop_info[8] = pad_h - resizer_crop_info[6];
          resizer_crop_info[9] = pad_w - resizer_crop_info[7];
          LOG_INFO << "scaling resize, roi_h:" << roi_h << ", roi_w:" << roi_w
                   << ", target_h:" << target_h << ", target_w:" << target_w
                   << ", scaling factor:" << scale << ", scaled_h:" << scaled_h
                   << ", scaled_w:" << scaled_w << ", pad_h:" << pad_h
                   << ", pad_w:" << pad_w;
        } else {
          // resize [H, W]
          resizer_crop_info[4] = resnet50.input_sizes_[1];
          resizer_crop_info[5] = resnet50.input_sizes_[0];
          // pad [top, left, bottom, right]
          resizer_crop_info[6] = 0;
          resizer_crop_info[7] = 0;
          resizer_crop_info[8] = 0;
          resizer_crop_info[9] = 0;
        }
        resizer_crop_inputs.push_back(resizer_crop_info);
      }
    }
    int resizer_crop_num = resizer_crop_inputs.size();

    std::vector<std::vector<ClassResult>> classify_results;
    int resizer_crop_idx = 0;
    while (true) {
      // each batch of data for inference must be valid, not support empty data
      auto it = input_map.find(resizer_crop_str);
      if (it != input_map.end()) {
        for (int idx = 0; idx < batch_num; idx++) {
          // use the first resizer_crop info to complete the batch data
          int tmp_idx =
              resizer_crop_idx < resizer_crop_num ? resizer_crop_idx : 0;
          memcpy(it->second.Data() + (idx * resizer_crop_size),
                 resizer_crop_inputs[tmp_idx].data(), resizer_crop_size);
          resizer_crop_idx++;
        }
      }

      // 5. set input
      for (const auto &input : input_map) {
        module.SetInput(input.first, input.second);
      }

      // 6. run and sync
      module.Run();
      module.Sync();

      // 7. get output
      for (auto &output : output_map) {
        auto output_tensor = module.GetOutput(output.first);
        output_tensor.CastTo(output.second);
      }

      // 8. postprocess
      for (int i = 0; i < batch_num; i++) {
        auto cls = resnet50.postprocess(
            static_cast<float *>(output_map.begin()->second.Data() +
                                 i * opt_size),
            1000);
        classify_results.emplace_back(cls);
      }

      if (resizer_crop_idx >= resizer_crop_num) {
        break;
      }
    }

    // 9. print
    for (int det_idx = 0; det_idx < detections.size(); det_idx++) {
      printf(
          "batch[%d] detection[%d] box[%d, %d, %d, %d], det[conf:%f, cls:%d], "
          "cls[id:%d], conf:%f, label:[%s]]\n",
          ipt_idx, det_idx, detections[det_idx].box.x1,
          detections[det_idx].box.y1, detections[det_idx].box.x2,
          detections[det_idx].box.y2, detections[det_idx].conf,
          detections[det_idx].cls, classify_results[det_idx][0].index,
          classify_results[det_idx][0].conf,
          Imagenet::GetLabel(classify_results[det_idx][0].index).c_str());
    }
  }

  LOG_INFO << "<=== multibatch resnet50 example end.";
  return 0;
}

int main(int argc, char *argv[]) {
  const char* houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target = houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
  if (houmo_target != "xh1") {
    std::cerr << "Not support houmo target:" << houmo_target << std::endl;
    exit(-1);
  }

  // 0: proportional scaling, 1: non-proportional scaling
  int resize_type = 0;
  if (argc == 2) {
    resize_type = std::stoi(argv[1]);
    LOG_INFO << "set resize type:" << resize_type;
  }
  LOG_INFO << "===> multibatch c++ example start, tcim version: "
           << tcim::GetVersion().c_str() << ", resize type:" << resize_type;

  std::string yolov5s_md_path = "yolov5s_xh1_b4_1core_O2.hmm";
  std::string resnet50_md_path = "resnet50_roi4_b1_xh1_O2.hmm";

  std::vector<std::string> input_images = {
      "./images_1920x1080/000000000139_1920x1080.jpg",
      "./images_1920x1080/000000000285_1920x1080.jpg",
      "./images_1920x1080/000000000632_1920x1080.jpg",
      "./images_1920x1080/000000000724_1920x1080.jpg"};
  int yolov5s_batch_num = 4;
  int resnet50_batch_num = 4;

  std::vector<cv::Mat> cv_raw_images;
  for (int i = 0; i < yolov5s_batch_num; i++) {
    auto data_path = input_images[(i % input_images.size())];
    if (!fs::exists(data_path)) {
      LOG_ERROR << data_path << " not exist.";
      exit(-1);
    }
    cv::Mat img_raw = cv::imread(data_path);
    if (img_raw.empty()) {
      LOG_ERROR << "Failed to read image using opencv.";
      return -1;
    }
    cv_raw_images.emplace_back(img_raw);
  }

  auto batch_detections =
      detect(yolov5s_batch_num, yolov5s_md_path, cv_raw_images);
  if (batch_detections.size() == 0) {
    LOG_ERROR << "Failed to detect images.";
    exit(-1);
  }

  int ret = classify(resnet50_batch_num, resnet50_md_path, cv_raw_images,
                     batch_detections, resize_type);
  if (ret != 0) {
    LOG_ERROR << "Failed to classify images.";
    exit(-1);
  }

  LOG_INFO << "<=== multibatch c++ example end.";
  return 0;
}
