/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: HmImageProcessor.cc
 * Description:
 *   Image processing implementation for Qwen3-VL model using OpenCV.
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

#include "HmImageProcessor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

HmImageProcessor::HmImageProcessor(int target_width, int target_height,
                                   bool use_v1)
    : target_width_(target_width),
      target_height_(target_height),
      use_v1_(use_v1) {}

ProcessedImage HmImageProcessor::LoadAndProcess(const std::string &image_path) {
  // Load image using OpenCV
  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Failed to load image: " << image_path << std::endl;
    // Return a gray image as fallback
    ProcessedImage fallback;
    fallback.width = target_width_;
    fallback.height = target_height_;
    fallback.channels = 3;
    fallback.data.resize(target_width_ * target_height_ * 3, 114);
    return fallback;
  }

  // Convert BGR to RGB
  cv::Mat img_rgb;
  cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

  ProcessedImage processed;
  processed.width = img_rgb.cols;
  processed.height = img_rgb.rows;
  processed.channels = 3;
  processed.data.assign(img_rgb.data,
                        img_rgb.data + img_rgb.total() * img_rgb.elemSize());

  if (use_v1_) {
    return ResizeAndPadV1(processed);
  } else {
    return ResizeV2(processed);
  }
}

std::vector<ProcessedImage> HmImageProcessor::LoadAndProcessBatch(
    const std::vector<std::string> &image_paths) {
  std::vector<ProcessedImage> images;
  images.reserve(image_paths.size());
  for (const auto &path : image_paths) {
    images.emplace_back(LoadAndProcess(path));
  }

  return images;
}

std::vector<half_float::half> HmImageProcessor::ToHalfTensor(
    const ProcessedImage &image) {
  // Create OpenCV Mat from RGB image data
  cv::Mat img(image.height, image.width, CV_8UC3,
              const_cast<uint8_t *>(image.data.data()));

  // Convert to float (no YUV conversion, no 1.0/255.0 normalization)
  // This matches Python's _hm_preprocess which keeps raw RGB values
  cv::Mat img_float;
  img.convertTo(img_float, CV_32F);

  // Split into R, G, B channels
  std::vector<cv::Mat> channels;
  cv::split(img_float, channels);

  // Combine channels into NCHW format [1, 3, H, W]
  size_t num_pixels = image.width * image.height;
  // We need to expand HWC to 2HWC and convert to C2HW layout.
  // The layout will be [Channels, Temporal, Height, Width] -> [3, 2, H, W]
  int num_frames = 2;
  std::vector<half_float::half> tensor(3 * num_frames * num_pixels);

  // Convert each channel to half precision and store in C T H W format
  for (int c = 0; c < 3; c++) {
    for (int t = 0; t < num_frames; t++) {
      for (size_t i = 0; i < num_pixels; i++) {
        tensor[(c * num_frames + t) * num_pixels + i] =
            half_float::half(channels[c].at<float>(i));
      }
    }
  }

  return tensor;
}

ProcessedImage HmImageProcessor::ResizeAndPadV1(const ProcessedImage &image) {
  // Create OpenCV Mat from image data
  cv::Mat img(image.height, image.width, CV_8UC3,
              const_cast<uint8_t *>(image.data.data()));

  // Calculate scale to fit within target dimensions while preserving aspect
  // ratio
  float scale_w = static_cast<float>(target_width_) / image.width;
  float scale_h = static_cast<float>(target_height_) / image.height;
  float scale = std::min(scale_w, scale_h);

  int new_width = static_cast<int>(image.width * scale);
  int new_height = static_cast<int>(image.height * scale);

  // Resize image
  cv::Mat resized;
  if (scale < 1.0f) {
    cv::resize(img, resized, cv::Size(new_width, new_height), 0, 0,
               cv::INTER_CUBIC);
  } else {
    cv::resize(img, resized, cv::Size(new_width, new_height), 0, 0,
               cv::INTER_AREA);
  }

  // Create padded image with gray background
  cv::Mat padded(target_height_, target_width_, CV_8UC3,
                 cv::Scalar(114, 114, 114));

  // Copy resized image to top-left corner
  cv::Rect roi(0, 0, new_width, new_height);
  resized.copyTo(padded(roi));

  ProcessedImage result;
  result.width = target_width_;
  result.height = target_height_;
  result.channels = 3;
  result.data.assign(padded.data,
                     padded.data + padded.total() * padded.elemSize());

  return result;
}

ProcessedImage HmImageProcessor::ResizeV2(const ProcessedImage &image) {
  // Create OpenCV Mat from image data
  cv::Mat img(image.height, image.width, CV_8UC3,
              const_cast<uint8_t *>(image.data.data()));

  // Directly resize to target size
  cv::Mat resized;
  if (image.width > target_width_ || image.height > target_height_) {
    cv::resize(img, resized, cv::Size(target_width_, target_height_), 0, 0,
               cv::INTER_CUBIC);
  } else {
    cv::resize(img, resized, cv::Size(target_width_, target_height_), 0, 0,
               cv::INTER_AREA);
  }

  ProcessedImage result;
  result.width = target_width_;
  result.height = target_height_;
  result.channels = 3;
  result.data.assign(resized.data,
                     resized.data + resized.total() * resized.elemSize());

  return result;
}