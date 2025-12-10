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

#ifdef ENABLE_ORT
#include "onnxruntime_cxx_api.h"
#endif
#include "imageproc.hpp"
#include "tcim/tcim_runtime.h"
#include "threads.hpp"
#include "utils.hpp"

struct Box {
  int x1{0};
  int y1{0};
  int x2{0};
  int y2{0};

  Box() = default;

  Box(int _x1, int _y1, int _x2, int _y2) {
    x1 = _x1;
    y1 = _y1;
    x2 = _x2;
    y2 = _y2;
  }

  int w() const { return x2 - x1 + 1; }
  int h() const { return y2 - y1 + 1; }
  int x() const { return x1; }
  int y() const { return y1; }
  int cx() const { return (x1 + x2) / 2; }
  int cy() const { return (y1 + y2) / 2; }
};

typedef struct {
  float conf{0.0f};
  int cls{-1};       // class id
  std::string name;  // class name
  Box box;
  float mask[32]{};
} Detection;

typedef struct {
  float* data{nullptr};
  size_t data_size{0};
  int num_anchors{0};
  int stride{0};
} DetectOutput;

float bbox_overlap(const Box& vi, const Box& vo) {
  int xx1 = std::max(vi.x1, vo.x1);
  int yy1 = std::max(vi.y1, vo.y1);
  int xx2 = std::min(vi.x2, vo.x2);
  int yy2 = std::min(vi.y2, vo.y2);
  int w = std::max(0, xx2 - xx1);
  int h = std::max(0, yy2 - yy1);
  int area = w * h;
  float dist = float(area) / float((vi.x2 - vi.x1) * (vi.y2 - vi.y1) +
                                   (vo.y2 - vo.y1) * (vo.x2 - vo.x1) - area);
  return dist;
}

int non_max_suppression(std::vector<Detection>& detections,
                        const float iou_threshold) {
  // sort
  std::sort(detections.begin(), detections.end(),
            [](const Detection& d1, const Detection& d2) {
              return d1.conf > d2.conf;
            });

  // nms
  std::vector<Detection> keep_detections;
  bool* suppressed = new bool[detections.size()];
  memset(suppressed, 0, sizeof(bool) * detections.size());
  const int num_detections = detections.size();
  for (int i = 0; i < num_detections; ++i) {
    if (suppressed[i]) continue;
    keep_detections.emplace_back(detections[i]);
    for (int j = i + 1; j < num_detections; ++j) {
      if (suppressed[j]) continue;
      float iou = bbox_overlap(detections[i].box, detections[j].box);
      if (iou > iou_threshold) suppressed[j] = true;
    }
  }
  keep_detections.swap(detections);
  delete[] suppressed;

  return 0;
}

cv::Mat letterbox(cv::Mat& img, int height, int width) {
  float scale;
  int resize_rows;
  int resize_cols;
  if ((height * 1.0 / img.rows) < (width * 1.0 / img.cols)) {
    scale = height * 1.0 / img.rows;
  } else {
    scale = width * 1.0 / img.cols;
  }
  resize_cols = int(scale * img.cols);
  resize_rows = int(scale * img.rows);

  cv::resize(img, img, cv::Size(resize_cols, resize_rows));
  // Generate a gray image for letterbox using opencv
  int top = (height - resize_rows) / 2;
  int bot = (height - resize_rows + 1) / 2;
  int left = (width - resize_cols) / 2;
  int right = (width - resize_cols + 1) / 2;
  // Letterbox filling
  cv::Mat img_new;
  cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));

  return img_new;
}

class YoloV5 {
 public:
  // convert the coordinates of the archor box
  void convert_box_coords(float cx, float cy, float w, float h, Box& box) {
    float scale = (float)input_sizes_[0] / std::max(img_rows_, img_cols_);
    float pad_h = (input_sizes_[0] - img_rows_ * scale) * 0.5f;
    float pad_w = (input_sizes_[1] - img_cols_ * scale) * 0.5f;

    // scale coords
    int x1 = (int)((cx - w * 0.5f - pad_w) / scale);
    int y1 = (int)((cy - h * 0.5f - pad_h) / scale);
    int x2 = (int)((cx + w * 0.5f - pad_w) / scale);
    int y2 = (int)((cy + h * 0.5f - pad_h) / scale);

    // clip
    box.x1 = x1 < 0 ? 0 : x1;
    box.y1 = y1 < 0 ? 0 : y1;
    box.x2 = x2 >= img_cols_ ? img_cols_ - 1 : x2;
    box.y2 = y2 >= img_rows_ ? img_rows_ - 1 : y2;
  }

  // calculate the score of each anchor box through cpp code
  int calculate_detections(std::vector<DetectOutput> outputs,
                           std::vector<Detection>& detections) {
    static float anchors[18] = {10, 13, 16,  30,  33, 23,  30,  61,  62,
                                45, 59, 119, 116, 90, 156, 198, 373, 326};
    int anchor_num = 3;
    int anchor_group;

    for (auto& output : outputs) {
      int stride = output.stride;
      float* feat = output.data;
      int feat_w = input_sizes_[1] / stride;
      int feat_h = input_sizes_[0] / stride;
      if (stride == 8)
        anchor_group = 1;
      else if (stride == 16)
        anchor_group = 2;
      else if (stride == 32)
        anchor_group = 3;
      else {
        printf("[error] wrong stride: %d\n", stride);
        return -1;
      }
      for (int h = 0; h <= feat_h - 1; h++) {
        for (int w = 0; w <= feat_w - 1; w++) {
          for (int a = 0; a <= anchor_num - 1; a++) {
            // process class score
            int class_index = 0;
            float class_score = -1.0;
            for (int s = 0; s <= num_classes_ - 1; s++) {
              float score = feat[a * feat_w * feat_h * (num_classes_ + 5) +
                                 h * feat_w * (num_classes_ + 5) +
                                 w * (num_classes_ + 5) + s + 5];
              if (score < conf_threshold_) continue;
              if (score > class_score) {
                class_index = s;
                class_score = score;
              }
            }
            // process box score
            float box_score = feat[a * feat_w * feat_h * (num_classes_ + 5) +
                                   (h * feat_w) * (num_classes_ + 5) +
                                   w * (num_classes_ + 5) + 4];
            // calculate final confidence
            float final_score = box_score * class_score;

            // filter out boxes with low confidence
            if (final_score >= conf_threshold_) {
              int loc_idx = a * feat_h * feat_w * (num_classes_ + 5) +
                            h * feat_w * (num_classes_ + 5) +
                            w * (num_classes_ + 5);
              float dx = feat[loc_idx + 0];
              float dy = feat[loc_idx + 1];
              float dw = feat[loc_idx + 2];
              float dh = feat[loc_idx + 3];
              float pred_cx = (dx * 2.0f - 0.5f + w) * stride;
              float pred_cy = (dy * 2.0f - 0.5f + h) * stride;
              float anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0];
              float anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1];
              float pred_w = dw * dw * 4.0f * anchor_w;
              float pred_h = dh * dh * 4.0f * anchor_h;

              Box detect_box;
              convert_box_coords(pred_cx, pred_cy, pred_w, pred_h, detect_box);

              Detection detection;
              detection.box = detect_box;
              detection.cls = class_index;
              detection.conf = final_score;
              detections.emplace_back(detection);
            }
          }
        }
      }
    }
    return 0;
  }

#ifdef ENABLE_ORT
  // calculate the score of each anchor box through postprocess model inference
  int infer_detections(std::string model_path,
                       std::vector<DetectOutput> md_outputs,
                       std::vector<Detection>& detections) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "HM_Yolov5s_Example");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);  // set thread num
    // turn on graph optimization
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    // load postprocess model
    Ort::Session session(env, model_path.c_str(), session_options);

    // get input info
    size_t num_inputs = session.GetInputCount();
    auto input_names = session.GetInputNames();
    std::vector<const char*> input_names_chr;
    std::vector<std::vector<int64_t>> input_shapes(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
      input_names_chr.emplace_back(input_names[i].c_str());
      Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
      auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
      input_shapes[i] = input_tensor_info.GetShape();
    }
    // get output info
    size_t num_outputs = session.GetOutputCount();
    auto output_names = session.GetOutputNames();
    std::vector<const char*> output_names_chr;
    std::vector<size_t> output_element_cnt(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
      output_names_chr.emplace_back(output_names[i].c_str());
      Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(i);
      auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
      output_element_cnt[i] = output_tensor_info.GetElementCount();
    }

    // create input tensors
    std::vector<Ort::Value> input_tensors;
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (size_t i = 0; i < num_inputs; i++) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
          memory_info, md_outputs[i].data, md_outputs[i].data_size,
          input_shapes[i].data(), input_shapes[i].size()));
    }

    // execute postprocess inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names_chr.data(), input_tensors.data(),
        input_tensors.size(), output_names_chr.data(), output_names.size());
    if (output_tensors.size() == 0) {
      return -1;
    }

    // process detection results
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_element_cnt[0];

    // check all the archor boxes
    for (int i = 0; i < num_anchors_; ++i) {
      const float* box = output_data + i * (num_classes_ + 5);
      // get the box info
      float cx = box[0];
      float cy = box[1];
      float w = box[2];
      float h = box[3];
      float box_confidence = box[4];

      // find the class with the highest probability
      int class_id = 0;
      float class_score = 0.0f;
      for (int j = 0; j < num_classes_; ++j) {
        float prob = box[5 + j];
        if (prob > class_score) {
          class_score = prob;
          class_id = j;
        }
      }
      // calculate final confidence
      float confidence = box_confidence * class_score;

      // filter out boxes with low confidence
      if (confidence >= conf_threshold_) {
        Box detect_box;
        convert_box_coords(cx, cy, w, h, detect_box);

        Detection detection;
        detection.box = detect_box;
        detection.cls = class_id;
        detection.conf = confidence;
        detections.emplace_back(detection);
      }
    }
    return 0;
  }
#endif

  std::vector<Detection> postprocess(const cv::Mat& image,
                                     std::vector<DetectOutput> outputs,
                                     bool enable_ort) {
    int ret = -1;
    std::vector<Detection> detections;
    img_rows_ = image.rows;
    img_cols_ = image.cols;

    if (enable_ort) {
#ifdef ENABLE_ORT
      std::string model_path = "./yolov5s_640x640_postprocess.onnx";
      ret = infer_detections(model_path, outputs, detections);
#endif
      ;
    } else {
      ret = calculate_detections(outputs, detections);
    }

    if (ret == 0 && !detections.empty()) {
      non_max_suppression(detections, iou_threshold_);
    }
    return detections;
  }

 private:
  int min_wh_{0};
  int max_wh_{7680};
  float iou_threshold_{0.45f};
  float conf_threshold_{0.25f};
  const int input_sizes_[2] = {640, 640};  // wh
  const int num_anchors_{25200};
  const int num_classes_{80};
  int img_rows_{0};
  int img_cols_{0};
};

int main(int argc, char* argv[]) {
  printf("\n===> yolov5s c++ example start...\n");
  const char* houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target =
      houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
  if (houmo_target != "xh2") {
    std::cerr << "Unsupported backend " << houmo_target << std::endl;
    exit(-1);
  }

  bool enable_ort = false;
#ifdef ENABLE_ORT
  const char* ort_param = "--enable_ort";
  if (argc == 2 && std::strcmp(argv[1], ort_param) == 0) {
    enable_ort = true;
  }
#endif
  printf("tcim version: %s, houmo_target:%s, enable ort: %d.\n",
         tcim::GetVersion().c_str(), houmo_target.c_str(), enable_ort);

  // 1. load model
  std::cout << "LoadFromFile yolov5s" << std::endl;
  std::string model_path = "./yolov5s_clip_xh2_b1_1core.hmm";
  if (!fs::exists(model_path)) {
    std::cerr << model_path
              << " not exist. you should run build.py in yolov5s example first."
              << std::endl;
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
    std::cout << "Input[" << input_name << "] " << input_info << std::endl;
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));

    auto input_info_f32 = input_info.AsType(tcim::DataType::FLOAT32);
    std::cout << "Input f32[" << input_name << "] " << input_info_f32
              << std::endl;
    input_map_f32.insert(
        std::pair<std::string, tcim::TensorInfo>(input_name, input_info_f32));
  }

  // 3. input preprocess
  YoloV5 yolov5;
  std::string data_path = "../../data/000000000139.jpg";
  if (!fs::exists(data_path)) {
    std::cerr << data_path << " not exist." << std::endl;
    exit(-1);
  }

  cv::Mat img_raw = cv::imread(data_path);

  cv::Mat img_rgb;
  cv::Mat img_norm;
  img_rgb = letterbox(img_raw, 640, 640);
  ImageProc::BgrToRgb((int8_t*)(img_rgb.data), img_rgb.rows, img_rgb.cols);
  const float mean[3] = {0.0f, 0.0f, 0.0f};
  const float std[3] = {255.0f, 255.0f, 255.0f};
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
  cv::vconcat(channels, img_norm);

  size_t img_bytes = img_norm.total() * img_norm.elemSize();
  std::cout << "img_bytes: " << img_bytes << std::endl;
  auto input_tensor_f32 =
      tcim::Tensor::CreateHostTensor(input_map_f32.at("images"), img_bytes,
                                     reinterpret_cast<void*>(img_norm.data));
  input_tensor_f32.CastTo(input_map.at("images"));

  // 4. get output info
  std::map<std::string, tcim::Tensor> output_map;
  std::map<std::string, tcim::Tensor> output_map_f32;
  int output_num = module.GetOutputNum();
  std::cout << "Count of Output: " << output_num << std::endl;
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name)
                           .AsContiguous()
                           .AsType(tcim::DataType::FLOAT32);
    std::cout << "Output[" << output_name << "] " << output_info << std::endl;
    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    output_map.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
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
    auto output_tensor = module.GetOutput(output.first);
    output_tensor.CastTo(output.second);
  }

  // 8. postprocess
  std::vector<DetectOutput> outputs;
  for (auto& output : output_map) {
    DetectOutput out;
    out.data = (float*)output.second.Data();
    auto shape = output.second.Info().Shape();
    out.num_anchors = shape[1] * shape[2] * shape[3];
    out.data_size = shape[0] * out.num_anchors * shape[4];
    out.stride = 640 / shape[2];
    outputs.emplace_back(out);
  }

  auto detections = yolov5.postprocess(img_raw, outputs, enable_ort);

  // 9. print and draw
  printf("detect num: %d\n", (int)detections.size());
  for (const auto& detection : detections) {
    printf("box[%d, %d, %d, %d], conf:%f, cls:%d\n", detection.box.x1,
           detection.box.y1, detection.box.x2, detection.box.y2, detection.conf,
           detection.cls);
    cv::rectangle(img_raw, cv::Point(detection.box.x1, detection.box.y1),
                  cv::Point(detection.box.x2, detection.box.y2),
                  cv::Scalar(0, 0, 255), 2);
  }
  fs::path file_path(data_path);
  fs::path result_path("demo_results/cpp");
  if (!fs::exists(result_path)) {
    fs::create_directory("demo_results/cpp");
  }
  fs::path result_file = result_path / file_path.filename();
  cv::imwrite(result_file.string().c_str(), img_raw);
  printf("demo results saved to %s\n", result_file.string().c_str());

  // check result, modify it when you change model or data
  if (detections.size() != 16) {
    std::cout << "detect num != 16" << std::endl;
    exit(-1);
  }

  printf("<=== yolov5s c++ example completed.\n\n");
  return 0;
}
