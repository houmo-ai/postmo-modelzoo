/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: HmImageProcessor.h
 * Description:
 *   Image processing interface for Qwen3-VL model.
 *   Handles image loading, resizing, padding, and YUV conversion.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __HMIMAGE_PROCESSOR_H__
#define __HMIMAGE_PROCESSOR_H__

#include <cstdint>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "base/houmo.h"

/**
 * @brief Structure representing image dimensions
 */
struct ImageDims {
  int width;
  int height;
  int channels;
};

/**
 * @brief Structure representing a processed image
 */
struct ProcessedImage {
  std::vector<uint8_t> data;  // Raw pixel data (RGB format)
  int width;                  // Image width
  int height;                 // Image height
  int channels;               // Number of channels (3 for RGB)

  ProcessedImage() : width(0), height(0), channels(3) {}
};

/**
 * @brief Class for processing images for Qwen3-VL model
 *
 * This class handles:
 * 1. Loading images from file paths
 * 2. Resizing with aspect ratio preservation (v1) or direct resize (v2)
 * 3. Padding to target dimensions
 * 4. RGB to YUV conversion
 */
class HmImageProcessor {
 public:
  /**
   * @brief Constructor
   * @param target_width Target width for processed images
   * @param target_height Target height for processed images
   * @param use_v1 Whether to use v1 mode (aspect ratio preservation) or v2
   * (direct resize)
   */
  HmImageProcessor(int target_width = 448, int target_height = 448,
                   bool use_v1 = true);

  HmImageProcessor(const HmImageProcessor &it) = delete;
  HmImageProcessor &operator=(const HmImageProcessor &it) = delete;
  HmImageProcessor(HmImageProcessor &&it) noexcept = default;
  HmImageProcessor &operator=(HmImageProcessor &&it) noexcept = default;
  ~HmImageProcessor() = default;

  /**
   * @brief Load and process an image from file
   * @param image_path Path to the image file
   * @return ProcessedImage Structure containing processed image data
   */
  ProcessedImage LoadAndProcess(const std::string &image_path);

  /**
   * @brief Load and process multiple images from file paths
   * @param image_paths Vector of image file paths
   * @return Vector of processed images
   */
  std::vector<ProcessedImage> LoadAndProcessBatch(
      const std::vector<std::string> &image_paths);

  /**
   * @brief Convert processed image to float16 precision tensor (NCHW format,
   * YUV color space)
   * @param image Processed image
   * @return Vector of float16 values in NCHW format (YUV444)
   */
  std::vector<float16> ToFP16Tensor(const ProcessedImage &image);

  /**
   * @brief Get target dimensions
   * @return ImageDims structure with target width, height, and channels
   */
  ImageDims GetTargetDims() const { return {target_width_, target_height_, 3}; }

 private:
  int target_width_;
  int target_height_;
  bool use_v1_;  // true: preserve aspect ratio and pad, false: direct resize

  /**
   * @brief Resize image with aspect ratio preservation and pad to target size
   */
  ProcessedImage ResizeAndPadV1(const ProcessedImage &image);

  /**
   * @brief Directly resize image to target size
   */
  ProcessedImage ResizeV2(const ProcessedImage &image);
};

#endif  // __HMIMAGE_PROCESSOR_H__