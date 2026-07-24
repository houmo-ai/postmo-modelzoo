/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: yolov5s.cc
 * Description:
 *   YOLOv5 Object Detection C++ Example.
 *   This file implements an object detection application using the YOLOv5
 *   model. It includes image preprocessing, model inference using the TCIM
 *   runtime, and postprocessing. The implementation supports both native C++
 *   postprocessing and ONNX Runtime-based postprocessing for enhanced
 *   flexibility. The code handles multi-scale feature maps and applies
 *   appropriate anchor boxes for detection.
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(__clang__) || __GNUC__ >= 8 || defined(_MSC_VER)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#ifdef ENABLE_ORT
#include "onnxruntime_cxx_api.h"
#endif
#include "imageproc.hpp"
#include "logging.h"
#include "tcim/tcim_runtime.h"
#include "threads.hpp"
#include "utils.hpp"

constexpr const char *kResizeCropInputName = "resizer_crop";
constexpr const char *kInputImagePath = "../../data/000000000139.jpg";
constexpr int kTargetHeight = 640;
constexpr int kTargetWidth = 640;

/**
 * @struct Box
 * @brief Represents a bounding box with coordinates and utility functions
 */
struct Box {
  int x1{0};  ///< Top-left x coordinate
  int y1{0};  ///< Top-left y coordinate
  int x2{0};  ///< Bottom-right x coordinate
  int y2{0};  ///< Bottom-right y coordinate

  Box() = default;

  /**
   * @brief Constructor for Box with coordinates
   * @param _x1 Top-left x coordinate
   * @param _y1 Top-left y coordinate
   * @param _x2 Bottom-right x coordinate
   * @param _y2 Bottom-right y coordinate
   */
  Box(int _x1, int _y1, int _x2, int _y2) {
    x1 = _x1;
    y1 = _y1;
    x2 = _x2;
    y2 = _y2;
  }

  /**
   * @brief Calculate width of the box
   * @return Width of the box
   */
  int w() const { return x2 - x1 + 1; }

  /**
   * @brief Calculate height of the box
   * @return Height of the box
   */
  int h() const { return y2 - y1 + 1; }

  /**
   * @brief Get x coordinate of top-left corner
   * @return X coordinate of top-left corner
   */
  int x() const { return x1; }

  /**
   * @brief Get y coordinate of top-left corner
   * @return Y coordinate of top-left corner
   */
  int y() const { return y1; }

  /**
   * @brief Calculate center x coordinate
   * @return Center x coordinate
   */
  int cx() const { return (x1 + x2) / 2; }

  /**
   * @brief Calculate center y coordinate
   * @return Center y coordinate
   */
  int cy() const { return (y1 + y2) / 2; }
};

/**
 * @struct Detection
 * @brief Represents a detection result with confidence, class, and bounding box
 */
typedef struct {
  float conf{0.0f};  ///< Confidence score
  int cls{-1};       ///< Class ID
  std::string name;  ///< Class name
  Box box;           ///< Bounding box coordinates
  float mask[32]{};  ///< Mask data (if applicable)
} Detection;

/**
 * @struct DetectOutput
 * @brief Structure to hold detection output information
 */
typedef struct {
  float *data{nullptr};  ///< Pointer to output data
  size_t data_size{0};   ///< Size of the output data
  int num_anchors{0};    ///< Number of anchors
  int stride{0};         ///< Stride value for the feature map
} DetectOutput;

/**
 * @brief Calculate the Intersection over Union (IoU) of two bounding boxes
 * @param vi First bounding box
 * @param vo Second bounding box
 * @return IoU value between the two boxes
 */
float bbox_overlap(const Box &vi, const Box &vo) {
  // Calculate intersection coordinates
  int xx1 = std::max(vi.x1, vo.x1);
  int yy1 = std::max(vi.y1, vo.y1);
  int xx2 = std::min(vi.x2, vo.x2);
  int yy2 = std::min(vi.y2, vo.y2);

  // Calculate intersection area
  int w = std::max(0, xx2 - xx1);
  int h = std::max(0, yy2 - yy1);
  int area = w * h;

  // Calculate IoU
  float dist = float(area) / float((vi.x2 - vi.x1) * (vi.y2 - vi.y1) +
                                   (vo.y2 - vo.y1) * (vo.x2 - vo.x1) - area);
  return dist;
}

/**
 * @brief Perform Non-Maximum Suppression (NMS) to remove duplicate detections
 * @param detections Vector of detections to process
 * @param iou_threshold IoU threshold for suppression
 * @return 0 on success
 */
int non_max_suppression(std::vector<Detection> &detections,
                        const float iou_threshold) {
  // Sort detections by confidence in descending order
  std::sort(detections.begin(), detections.end(),
            [](const Detection &d1, const Detection &d2) {
              return d1.conf > d2.conf;
            });

  // Apply NMS algorithm
  std::vector<Detection> keep_detections;
  bool *suppressed = new bool[detections.size()];
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

/**
 * @brief Resize image with letterbox padding to maintain aspect ratio
 * @param img Input image to resize
 * @param height Target height
 * @param width Target width
 * @return Resized image with letterbox padding
 */
cv::Mat letterbox(cv::Mat &img, int height, int width) {
  float scale;
  int resize_rows;
  int resize_cols;

  // Calculate scale factor to maintain aspect ratio
  if ((height * 1.0 / img.rows) < (width * 1.0 / img.cols)) {
    scale = height * 1.0 / img.rows;
  } else {
    scale = width * 1.0 / img.cols;
  }

  // Calculate new dimensions
  resize_cols = int(scale * img.cols);
  resize_rows = int(scale * img.rows);

  // Resize image
  cv::resize(img, img, cv::Size(resize_cols, resize_rows));

  // Calculate padding values
  int top = (height - resize_rows) / 2;
  int bot = (height - resize_rows + 1) / 2;
  int left = (width - resize_cols) / 2;
  int right = (width - resize_cols + 1) / 2;

  // Apply letterbox padding with gray color (114, 114, 114)
  cv::Mat img_new;
  cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));

  return img_new;
}

/**
 * @class YoloV5
 * @brief Class for YOLOv5 object detection with preprocessing and
 * postprocessing
 */
class YoloV5 {
 public:
  // Store the exact resize/pad values encoded in dyn_info so postprocessing can
  // map model-space boxes back to the original image with the same geometry.
  void set_resizer_params(float scale_h, float scale_w, float pad_h,
                          float pad_w) {
    resize_scale_h_ = scale_h;
    resize_scale_w_ = scale_w;
    pad_h_ = pad_h;
    pad_w_ = pad_w;
  }

  /**
   * @brief Convert anchor box coordinates to image coordinates
   * @param cx Center x coordinate
   * @param cy Center y coordinate
   * @param w Width of the box
   * @param h Height of the box
   * @param box Output box structure
   */
  void convert_box_coords(float cx, float cy, float w, float h, Box &box) {
    // Undo the hardware-resizer letterbox transform: remove top/left padding,
    // then divide by the actual H/W scales after even-size rounding.
    int x1 = (int)((cx - w * 0.5f - pad_w_) / resize_scale_w_);
    int y1 = (int)((cy - h * 0.5f - pad_h_) / resize_scale_h_);
    int x2 = (int)((cx + w * 0.5f - pad_w_) / resize_scale_w_);
    int y2 = (int)((cy + h * 0.5f - pad_h_) / resize_scale_h_);

    // Clip coordinates to image boundaries
    box.x1 = x1 < 0 ? 0 : x1;
    box.y1 = y1 < 0 ? 0 : y1;
    box.x2 = x2 >= img_cols_ ? img_cols_ - 1 : x2;
    box.y2 = y2 >= img_rows_ ? img_rows_ - 1 : y2;
  }

  /**
   * @brief Calculate detections from model outputs
   * @param outputs Vector of detection outputs from the model
   * @param detections Vector to store calculated detections
   * @return 0 on success
   */
  int calculate_detections(std::vector<DetectOutput> outputs,
                           std::vector<Detection> &detections) {
    // YOLOv5 anchor values for different scales
    static float anchors[18] = {10, 13, 16,  30,  33, 23,  30,  61,  62,
                                45, 59, 119, 116, 90, 156, 198, 373, 326};
    // Number of anchors per scale
    int anchor_num = 3;
    int anchor_group;

    for (auto &output : outputs) {
      int stride = output.stride;
      float *feat = output.data;
      int feat_w = input_sizes_[1] / stride;
      int feat_h = input_sizes_[0] / stride;

      // Determine which anchor group to use based on stride
      if (stride == 8)
        anchor_group = 1;
      else if (stride == 16)
        anchor_group = 2;
      else if (stride == 32)
        anchor_group = 3;
      else {
        LOG_ERROR("Wrong stride: {}.", stride);
        return -1;
      }

      // Iterate through feature map positions
      for (int h = 0; h <= feat_h - 1; h++) {
        for (int w = 0; w <= feat_w - 1; w++) {
          // Iterate through anchors at each position
          for (int a = 0; a <= anchor_num - 1; a++) {
            // Process class scores
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

            // Process box score
            float box_score = feat[a * feat_w * feat_h * (num_classes_ + 5) +
                                   (h * feat_w) * (num_classes_ + 5) +
                                   w * (num_classes_ + 5) + 4];

            // Calculate final confidence
            float final_score = box_score * class_score;

            // Filter out boxes with low confidence
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
  /**
   * @brief Calculate detections using ONNX Runtime postprocessing model
   * @param model_path Path to the ONNX postprocessing model
   * @param md_outputs Model detection outputs
   * @param detections Vector to store calculated detections
   * @return 0 on success
   */
  int infer_detections(std::string model_path,
                       std::vector<DetectOutput> md_outputs,
                       std::vector<Detection> &detections) {
    // Create ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "HM_Yolov5s_Example");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);  // Set thread number
    // Enable graph optimization
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Load postprocessing model
    Ort::Session session(env, model_path.c_str(), session_options);

    // Get input information
    size_t num_inputs = session.GetInputCount();
    auto input_names = session.GetInputNames();
    std::vector<const char *> input_names_chr;
    std::vector<std::vector<int64_t>> input_shapes(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
      input_names_chr.emplace_back(input_names[i].c_str());
      Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
      auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
      input_shapes[i] = input_tensor_info.GetShape();
    }

    // Get output information
    size_t num_outputs = session.GetOutputCount();
    auto output_names = session.GetOutputNames();
    std::vector<const char *> output_names_chr;
    std::vector<size_t> output_element_cnt(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
      output_names_chr.emplace_back(output_names[i].c_str());
      Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(i);
      auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
      output_element_cnt[i] = output_tensor_info.GetElementCount();
    }

    // Create input tensors
    std::vector<Ort::Value> input_tensors;
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (size_t i = 0; i < num_inputs; i++) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
          memory_info, md_outputs[i].data, md_outputs[i].data_size,
          input_shapes[i].data(), input_shapes[i].size()));
    }

    // Execute postprocessing inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names_chr.data(), input_tensors.data(),
        input_tensors.size(), output_names_chr.data(), output_names.size());
    if (output_tensors.size() == 0) {
      return -1;
    }

    // Process detection results
    float *output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_element_cnt[0];

    // Check all anchor boxes
    for (int i = 0; i < num_anchors_; ++i) {
      const float *box = output_data + i * (num_classes_ + 5);
      // Get box information
      float cx = box[0];
      float cy = box[1];
      float w = box[2];
      float h = box[3];
      float box_confidence = box[4];

      // Find the class with the highest probability
      int class_id = 0;
      float class_score = 0.0f;
      for (int j = 0; j < num_classes_; ++j) {
        float prob = box[5 + j];
        if (prob > class_score) {
          class_score = prob;
          class_id = j;
        }
      }
      // Calculate final confidence
      float confidence = box_confidence * class_score;

      // Filter out boxes with low confidence
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

  /**
   * @brief Postprocess model outputs to get final detections
   * @param image Input image for coordinate conversion
   * @param outputs Vector of model outputs
   * @param enable_ort Whether to use ONNX Runtime for postprocessing
   * @return Vector of final detections after NMS
   */
  std::vector<Detection> postprocess(const cv::Mat &image,
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
  int min_wh_{0};                ///< Minimum width/height for detection
  int max_wh_{7680};             ///< Maximum width/height for detection
  float iou_threshold_{0.45f};   ///< IoU threshold for NMS
  float conf_threshold_{0.25f};  ///< Confidence threshold for detection
  const int input_sizes_[2] = {640, 640};  ///< Input dimensions (width, height)
  const int num_anchors_{25200};           ///< Total number of anchors
  const int num_classes_{80};              ///< Number of object classes
  int img_rows_{0};                        ///< Input image rows
  int img_cols_{0};                        ///< Input image columns
  float resize_scale_h_{1.0f};             ///< Resizer H scale used by dyn_info
  float resize_scale_w_{1.0f};             ///< Resizer W scale used by dyn_info
  float pad_h_{0.0f};                      ///< Top padding used by dyn_info
  float pad_w_{0.0f};                      ///< Left padding used by dyn_info
};

/**
 * @brief Main function for YOLOv5 object detection example
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return 0 on success, non-zero on failure
 */
int main(int argc, char *argv[]) {
  LOG_INFO("======> yolov5s c++ example start...");
  const char *houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target =
      houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";

  // Verify that the target platform is supported
  if (houmo_target != "xh2") {
    LOG_ERROR("Unsupported backend {}.", houmo_target);
    exit(-1);
  }

  bool enable_ort = false;
#ifdef ENABLE_ORT
  const char *ort_param = "--enable_ort";
  if (argc == 2 && std::strcmp(argv[1], ort_param) == 0) {
    enable_ort = true;
  }
#endif
  LOG_INFO("houmo_target: {}, enable ort: {}, tcim version: {}.", houmo_target,
           enable_ort, tcim::GetVersion());

  // 1. Load model from file
  std::string model_path;
  for (const auto &entry : fs::directory_iterator(fs::current_path())) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() == ".hmm") {
      model_path = entry.path().string();
      LOG_INFO("Found .hmm file: {}", model_path);
      break;
    }
  }

  if (model_path.empty() || !fs::exists(model_path)) {
    LOG_ERROR("No .hmm file found in {}", fs::current_path().string());
    exit(-1);
  }
  LOG_INFO("Load yolov5s model from file {}", model_path);
  if (!fs::exists(model_path)) {
    LOG_ERROR("Model file {} not exists.", model_path);
    exit(-1);
  }
  auto module = tcim::Module::LoadFromFile(model_path);
  if (!module) {
    LOG_ERROR("Load model {} fail.", model_path);
    exit(-1);
  }
  LOG_INFO("Model {} loaded.", model_path);

  // 2. Get input information.
  int max_img_height = 0;
  int max_img_width = 0;
  std::vector<std::string> input_names;
  int input_num = module.GetInputNum();
  LOG_INFO("Count of Input: {}", input_num);
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name).AsContiguous();
    LOG_INFO("Input[{}] info: {}", input_name, TensorInfo2Str(input_info));
    input_names.emplace_back(input_name);
    // The non-resizer input is the raw YUV image canvas consumed by the
    // hardware resizer. Its H/W define the maximum image size we can upload.
    if (input_name.find(kResizeCropInputName) == std::string::npos) {
      auto shape = input_info.Shape();
      max_img_height = static_cast<int>(shape.at(2));
      max_img_width = static_cast<int>(shape.at(3));
    }
  }
  if (max_img_height <= 0 || max_img_width <= 0) {
    LOG_ERROR("Invalid model input shape: height={}, width={}", max_img_height,
              max_img_width);
    exit(-1);
  }

  // 3. Preprocess input image
  YoloV5 yolov5;
  std::string data_path = kInputImagePath;
  if (!fs::exists(data_path)) {
    LOG_ERROR("{} not exist.", data_path);
    exit(-1);
  }

  cv::Mat img_raw = cv::imread(data_path);
  if (img_raw.empty()) {
    LOG_ERROR("Failed to read image {}", data_path);
    exit(-1);
  }
  cv::Mat image_data = img_raw.clone();
  const int img_height = image_data.rows;
  const int img_width = image_data.cols;
  LOG_INFO("input image shape: [{} x {} x {}]", img_height, img_width,
           image_data.channels());

  int crop_height = max_img_height;
  int crop_width = max_img_width;
  if (img_height < max_img_height && img_width <= max_img_width) {
    // Pad smaller images to the upload canvas. The crop size still records
    // only the original valid image region, so padded pixels are not resized
    // into the model input.
    const int pad_bottom = max_img_height - img_height;
    const int pad_right = max_img_width - img_width;
    cv::copyMakeBorder(image_data, image_data, 0, pad_bottom, 0, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    crop_height = img_height;
    crop_width = img_width;
    LOG_INFO("pad input image to [{} x {} x {}], height={}, width={}",
             image_data.rows, image_data.cols, image_data.channels(),
             max_img_height, max_img_width);
  } else {
    // If the source image exceeds the upload canvas, resize it first and let
    // dyn_info describe the full resized canvas as the crop region.
    cv::resize(image_data, image_data, cv::Size(max_img_width, max_img_height));
    LOG_INFO("resize input image to height={}, width={}", max_img_height,
             max_img_width);
  }

  // The hardware resizer consumes YUV420 input, so crop dimensions must be
  // even. This mirrors the TO_EVEN behavior used by the shared C++ helper.
  crop_height -= crop_height % 2;
  crop_width -= crop_width % 2;
  if (crop_height <= 0 || crop_width <= 2 || crop_height % 2 != 0 ||
      crop_width % 2 != 0) {
    LOG_ERROR("crop_height and crop_width must be even, got {} and {}",
              crop_height, crop_width);
    exit(-1);
  }
  if (kTargetHeight <= 0 || kTargetWidth <= 0 || kTargetHeight % 2 != 0 ||
      kTargetWidth % 2 != 0) {
    LOG_ERROR(
        "target height and width must be positive even values, got {} and {}",
        kTargetHeight, kTargetWidth);
    exit(-1);
  }

  // YOLOv5 uses letterbox preprocessing: resize the crop region with one
  // shared scale, then pad the remaining area to the fixed 640x640 canvas.
  const float scale = std::min(static_cast<float>(kTargetHeight) / crop_height,
                               static_cast<float>(kTargetWidth) / crop_width);
  if (scale < 1.0f / 32.0f || scale > 16.0f) {
    LOG_ERROR("resize scale must be in [1/32, 16], got {}", scale);
    exit(-1);
  }

  // The resizer requires even output sizes. Rounding down after round() keeps
  // the size close to the ideal scaled value while satisfying that constraint.
  int resizer_height =
      static_cast<int>(std::round(static_cast<float>(crop_height) * scale)) &
      ~1;
  int resizer_width =
      static_cast<int>(std::round(static_cast<float>(crop_width) * scale)) & ~1;
  resizer_height = std::min(resizer_height, kTargetHeight);
  resizer_width = std::min(resizer_width, kTargetWidth);
  if (resizer_height <= 0 || resizer_width <= 0) {
    LOG_ERROR("resizer height and width must be positive, got {} and {}",
              resizer_height, resizer_width);
    exit(-1);
  }

  const int pad_height = kTargetHeight - resizer_height;
  const int pad_width = kTargetWidth - resizer_width;
  const int pad_h_top = (pad_height / 2) & ~1;
  const int pad_w_left = (pad_width / 2) & ~1;
  const int pad_h_bottom = pad_height - pad_h_top;
  const int pad_w_right = pad_width - pad_w_left;

  // dyn_info layout:
  // [crop_offset_h, crop_offset_w, crop_h, crop_w,
  //  resize_h, resize_w, pad_top, pad_left, pad_bottom, pad_right]
  // Nonzero padding tells the runtime to use proportional resize + padding
  // instead of stretching directly to kTargetHeight x kTargetWidth.
  std::vector<int32_t> dyn_info = {0,
                                   0,
                                   crop_height,
                                   crop_width,
                                   resizer_height,
                                   resizer_width,
                                   pad_h_top,
                                   pad_w_left,
                                   pad_h_bottom,
                                   pad_w_right};
  yolov5.set_resizer_params(static_cast<float>(resizer_height) / crop_height,
                            static_cast<float>(resizer_width) / crop_width,
                            static_cast<float>(pad_h_top),
                            static_cast<float>(pad_w_left));
  LOG_INFO("dyn_info: [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]", dyn_info[0],
           dyn_info[1], dyn_info[2], dyn_info[3], dyn_info[4], dyn_info[5],
           dyn_info[6], dyn_info[7], dyn_info[8], dyn_info[9]);

  std::map<std::string, std::vector<uint8_t>> input_buffers;
  std::map<std::string, tcim::Tensor> input_tensors;
  for (const auto &input_name : input_names) {
    auto input_info = module.GetInputInfo(input_name).AsContiguous();
    // The compiled model has two input kinds: the int32 dyn_info tensor for
    // the hardware resizer and the uploaded YUV image canvas.
    if (input_name.find(kResizeCropInputName) != std::string::npos) {
      const size_t dyn_bytes = dyn_info.size() * sizeof(int32_t);
      if (dyn_bytes > input_info.MemSize()) {
        LOG_ERROR("dyn_info bytes {} exceed input tensor bytes {} for {}",
                  dyn_bytes, input_info.MemSize(), input_name);
        exit(-1);
      }

      if (dyn_bytes == input_info.MemSize()) {
        auto input_tensor = tcim::Tensor::CreateHostTensor(
            input_info, dyn_bytes, static_cast<void *>(dyn_info.data()));
        input_tensors.emplace(input_name, input_tensor);
      } else {
        auto &buffer = input_buffers[input_name];
        buffer.assign(input_info.MemSize(), 0);
        std::memcpy(buffer.data(), dyn_info.data(), dyn_bytes);
        auto input_tensor = tcim::Tensor::CreateHostTensor(
            input_info, buffer.size(), static_cast<void *>(buffer.data()));
        input_tensors.emplace(input_name, input_tensor);
      }
    } else {
      auto &buffer = input_buffers[input_name];
      buffer.assign(input_info.MemSize(), 0);
      // The image tensor remains uint8 YUV420sp; resize, crop, padding, and
      // model-specific normalization are handled inside the compiled graph.
      const size_t image_bytes =
          ConvertBgrToYuv420sp(image_data, buffer.data(), buffer.size());
      if (image_bytes == 0) {
        LOG_ERROR("input image bytes exceed input tensor bytes {} for {}",
                  input_info.MemSize(), input_name);
        exit(-1);
      }
      auto input_tensor = tcim::Tensor::CreateHostTensor(
          input_info, buffer.size(), static_cast<void *>(buffer.data()));
      input_tensors.emplace(input_name, input_tensor);
    }
  }

  // 4. Get output information and create output tensors
  std::map<std::string, tcim::Tensor> output_map;
  std::map<std::string, tcim::Tensor> output_map_f32;
  int output_num = module.GetOutputNum();
  LOG_INFO("Count of Output: {}", output_num);
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name)
                           .AsContiguous()
                           .AsType(tcim::DataType::FLOAT32);
    LOG_INFO("Output[{}] info: {}", output_name, TensorInfo2Str(output_info));
    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    output_map.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
  }

  // 5. Set input tensors to the module
  for (const auto &input : input_tensors) {
    if (module.SetInput(input.first, input.second) != tcim::Status::OK) {
      LOG_ERROR("Failed to set input {}", input.first);
      exit(-1);
    }
  }

  // 6. Run inference and synchronize
  module.Run();
  module.Sync();

  // 7. Get output tensors from the module
  for (auto &output : output_map) {
    auto output_tensor = module.GetOutput(output.first);
    output_tensor.CastTo(output.second);
  }

  // 8. Postprocess outputs to get detections
  std::vector<DetectOutput> outputs;
  for (auto &output : output_map) {
    DetectOutput out;
    out.data = (float *)output.second.Data();
    auto shape = output.second.Info().Shape();
    out.num_anchors = shape[1] * shape[2] * shape[3];
    out.data_size = shape[0] * out.num_anchors * shape[4];
    out.stride = 640 / shape[2];
    outputs.emplace_back(out);
  }

  auto detections = yolov5.postprocess(img_raw, outputs, enable_ort);

  // 9. Print and draw detection results
  LOG_INFO("detect num: {}.", detections.size());
  for (const auto &detection : detections) {
    LOG_INFO("box[{}, {}, {}, {}], conf: {}, cls: {}.", detection.box.x1,
             detection.box.y1, detection.box.x2, detection.box.y2,
             detection.conf, detection.cls);
    cv::rectangle(img_raw, cv::Point(detection.box.x1, detection.box.y1),
                  cv::Point(detection.box.x2, detection.box.y2),
                  cv::Scalar(0, 0, 255), 2);
  }

  // Save detection results to file
  fs::path file_path(data_path);
  fs::path result_path("demo_results/cpp");
  if (!fs::exists(result_path)) {
    fs::create_directory("demo_results/cpp");
  }
  fs::path result_file = result_path / file_path.filename();
  cv::imwrite(result_file.string().c_str(), img_raw);
  LOG_INFO("demo results saved to {}.", result_file.string());

  // Verify result (modify when changing model or data)
  if (detections.size() != 16 && detections.size() != 15 &&
      detections.size() != 17 && detections.size() != 18) {
    LOG_ERROR("detect num != 15, 16, 17, 18");
    exit(-1);
  }

  LOG_INFO("<=== yolov5s c++ example completed.");
  return 0;
}
