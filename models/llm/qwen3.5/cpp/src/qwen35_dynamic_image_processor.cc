/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen35_dynamic_image_processor.cc
 * Description:
 *   Qwen3.5 dynamic vision image processing implementation.
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
#include "qwen35_dynamic_image_processor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "nlohmann/json.hpp"

namespace houmo::qwen35 {
namespace {

int RoundByFactor(int value, int factor) {
  return static_cast<int>(std::round(static_cast<double>(value) / factor)) * factor;
}

int FloorByFactor(double value, int factor) {
  return static_cast<int>(std::floor(value / factor)) * factor;
}

int CeilByFactor(double value, int factor) {
  return static_cast<int>(std::ceil(value / factor)) * factor;
}

uint8_t ResizePixel(
    const DynamicImageProcessor::RgbImage& image,
    int x,
    int y,
    int channel,
    double sx,
    double sy) {
  constexpr double a = -0.5;
  auto cubic = [](double value) {
    value = std::abs(value);
    if (value <= 1.0) return (a + 2.0) * value * value * value - (a + 3.0) * value * value + 1.0;
    if (value < 2.0) return a * value * value * value - 5.0 * a * value * value + 8.0 * a * value - 4.0 * a;
    return 0.0;
  };
  const double src_x = (x + 0.5) * sx - 0.5;
  const double src_y = (y + 0.5) * sy - 0.5;
  const int base_x = static_cast<int>(std::floor(src_x));
  const int base_y = static_cast<int>(std::floor(src_y));
  double sum = 0.0;
  double weight_sum = 0.0;
  for (int ky = -1; ky <= 2; ++ky) {
    const int py = std::clamp(base_y + ky, 0, image.height - 1);
    const double wy = cubic(src_y - (base_y + ky));
    for (int kx = -1; kx <= 2; ++kx) {
      const int px = std::clamp(base_x + kx, 0, image.width - 1);
      const double weight = wy * cubic(src_x - (base_x + kx));
      sum += image.data[(static_cast<size_t>(py) * image.width + px) * 3 + channel] * weight;
      weight_sum += weight;
    }
  }
  return static_cast<uint8_t>(std::clamp(
      std::round(weight_sum == 0.0 ? 0.0 : sum / weight_sum), 0.0, 255.0));
}

}  // namespace

DynamicImageProcessor::DynamicImageProcessor(
    int patch_size,
    int temporal_patch_size,
    int merge_size,
    int min_pixels,
    int max_pixels,
    const std::string& preprocessor_config_path)
    : patch_size_(patch_size),
      temporal_patch_size_(temporal_patch_size),
      merge_size_(merge_size),
      min_pixels_(min_pixels),
      max_pixels_(max_pixels) {
  if (patch_size_ <= 0 || temporal_patch_size_ <= 0 || merge_size_ <= 0 ||
      min_pixels_ <= 0 || max_pixels_ < min_pixels_) {
    throw std::invalid_argument("invalid Qwen3.5 dynamic image processor configuration");
  }
  try {
    if (preprocessor_config_path.empty()) {
      throw std::runtime_error("preprocessor_config.json path is empty");
    }
    std::ifstream stream(preprocessor_config_path);
    if (!stream.is_open()) {
      throw std::runtime_error("failed to open " + preprocessor_config_path);
    }
    const nlohmann::json config = nlohmann::json::parse(stream);
    if (!config.contains("image_mean") || !config.contains("image_std")) {
      throw std::runtime_error("image_mean or image_std is missing");
    }
    const auto mean = config.at("image_mean").get<std::vector<float>>();
    const auto std = config.at("image_std").get<std::vector<float>>();
    if (mean.size() != 3 || std.size() != 3 ||
        std[0] == 0.0f || std[1] == 0.0f || std[2] == 0.0f) {
      throw std::runtime_error("image_mean/image_std is invalid");
    }
    std::copy(mean.begin(), mean.end(), mean_.begin());
    std::copy(std.begin(), std.end(), std_.begin());
  } catch (const std::exception& error) {
    std::cerr << "Warning: " << error.what()
              << "; using default image_mean/image_std" << std::endl;
  }
}

std::pair<int, int> DynamicImageProcessor::SmartResize(
    int height, int width, int factor, int min_pixels, int max_pixels) {
  if (height <= 0 || width <= 0 || factor <= 0 || min_pixels <= 0 ||
      max_pixels < min_pixels) {
    throw std::invalid_argument("invalid smart resize arguments");
  }
  int resized_height = std::max(factor, RoundByFactor(height, factor));
  int resized_width = std::max(factor, RoundByFactor(width, factor));
  if (static_cast<int64_t>(resized_height) * resized_width > max_pixels) {
    const double beta = std::sqrt(
        static_cast<double>(height) * width / max_pixels);
    resized_height = std::max(factor, FloorByFactor(height / beta, factor));
    resized_width = std::max(factor, FloorByFactor(width / beta, factor));
  } else if (static_cast<int64_t>(resized_height) * resized_width < min_pixels) {
    const double beta = std::sqrt(
        static_cast<double>(min_pixels) / (height * static_cast<double>(width)));
    resized_height = CeilByFactor(height * beta, factor);
    resized_width = CeilByFactor(width * beta, factor);
  }
  return {resized_height, resized_width};
}

DynamicImageProcessor::RgbImage DynamicImageProcessor::LoadRgb(
    const std::string& image_path) const {
  int width = 0;
  int height = 0;
  int channels = 0;
  unsigned char* raw = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
  if (raw == nullptr) {
    throw std::runtime_error("failed to load image: " + image_path);
  }
  RgbImage image;
  image.width = width;
  image.height = height;
  image.data.assign(raw, raw + static_cast<size_t>(width) * height * 3);
  stbi_image_free(raw);
  return image;
}

DynamicImageProcessor::RgbImage DynamicImageProcessor::Resize(
    const RgbImage& image, int height, int width) const {
  RgbImage output;
  output.width = width;
  output.height = height;
  output.data.resize(static_cast<size_t>(width) * height * 3);
  const double sx = static_cast<double>(image.width) / width;
  const double sy = static_cast<double>(image.height) / height;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        output.data[(static_cast<size_t>(y) * width + x) * 3 + c] =
            ResizePixel(image, x, y, c, sx, sy);
      }
    }
  }
  return output;
}

std::vector<float> DynamicImageProcessor::Normalize(const RgbImage& image) const {
  const size_t pixels = static_cast<size_t>(image.width) * image.height;
  std::vector<float> chw(3 * pixels);
  for (int c = 0; c < 3; ++c) {
    for (size_t i = 0; i < pixels; ++i) {
      chw[static_cast<size_t>(c) * pixels + i] =
          (static_cast<float>(image.data[i * 3 + c]) / 255.0f - mean_[c]) / std_[c];
    }
  }
  return chw;
}

std::vector<float16> DynamicImageProcessor::Patchify(
    const std::vector<float>& chw, int height, int width) const {
  const int grid_h = height / patch_size_;
  const int grid_w = width / patch_size_;
  if (height % (patch_size_ * merge_size_) != 0 ||
      width % (patch_size_ * merge_size_) != 0) {
    throw std::runtime_error("dynamic image size is not divisible by patch*merge");
  }
  const int patch_dim = 3 * temporal_patch_size_ * patch_size_ * patch_size_;
  const int patches = grid_h * grid_w;
  std::vector<float16> output(static_cast<size_t>(patches) * patch_dim);
  for (int patch_id = 0; patch_id < patches; ++patch_id) {
    FillPatch(chw, height, width, patch_id, grid_w, patch_dim, output);
  }
  return output;
}

void DynamicImageProcessor::FillPatch(
    const std::vector<float>& chw,
    int height,
    int width,
    int patch_id,
    int grid_w,
    int patch_dim,
    std::vector<float16>& output) const {
  const int patches_per_block = merge_size_ * merge_size_;
  const int block_id = patch_id / patches_per_block;
  const int local_id = patch_id % patches_per_block;
  const int blocks_w = grid_w / merge_size_;
  const int py = (block_id / blocks_w) * merge_size_ + local_id / merge_size_;
  const int px = (block_id % blocks_w) * merge_size_ + local_id % merge_size_;
  const size_t pixels = static_cast<size_t>(height) * width;
  const size_t base = static_cast<size_t>(patch_id) * patch_dim;
  int offset = 0;
  for (int c = 0; c < 3; ++c) {
    for (int t = 0; t < temporal_patch_size_; ++t) {
      (void)t;
      for (int iy = 0; iy < patch_size_; ++iy) {
        for (int ix = 0; ix < patch_size_; ++ix) {
          const int y = py * patch_size_ + iy;
          const int x = px * patch_size_ + ix;
          output[base + offset++] = static_cast<float16>(
              chw[static_cast<size_t>(c) * pixels + static_cast<size_t>(y) * width + x]);
        }
      }
    }
  }
}

DynamicImageResult DynamicImageProcessor::LoadAndProcess(
    const std::string& image_path) const {
  const RgbImage source = LoadRgb(image_path);
  const int factor = patch_size_ * merge_size_;
  const auto [height, width] = SmartResize(
      source.height, source.width, factor, min_pixels_, max_pixels_);
  const RgbImage resized = Resize(source, height, width);
  const std::vector<float> chw = Normalize(resized);
  DynamicImageResult result;
  result.grid_t = 1;
  result.grid_h = height / patch_size_;
  result.grid_w = width / patch_size_;
  result.patch_dim = 3 * temporal_patch_size_ * patch_size_ * patch_size_;
  result.pixel_values = Patchify(chw, height, width);
  return result;
}

}  // namespace houmo::qwen35
