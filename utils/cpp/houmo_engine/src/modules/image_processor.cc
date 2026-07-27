/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: HmImageProcessor.cc
 * Description:
 *   Image processing for Qwen3-VL / Qwen3.5.
 *   Image decode: stb_image (Public Domain / MIT-0).
 *   Resize: pure C++ area/bicubic helpers adapted from llama.cpp vision /
 *   multimodal (mtmd) resize routines.
 *
 * Portions of the image resize implementation are adapted from llama.cpp:
 *   https://github.com/ggml-org/llama.cpp
 *
 * llama.cpp is licensed under the MIT License:
 *   Copyright (c) 2023-2026 The ggml authors
 *
 * stb_image is Public Domain / MIT-0:
 *   https://github.com/nothings/stb (3rdparty/stb/stb_image.h)
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
 * SPDX-License-Identifier: Apache-2.0 AND MIT
 */

#include "modules/image_processor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

namespace {

ProcessedImage MakeFallback(int target_w, int target_h) {
  ProcessedImage fallback;
  fallback.width = target_w;
  fallback.height = target_h;
  fallback.channels = 3;
  fallback.data.assign(static_cast<size_t>(target_w) * target_h * 3, 114);
  return fallback;
}

// Area (box) downsample — adapted from llama.cpp / mtmd vision resize.
bool AreaResizeRgb(const uint8_t *src, int src_w, int src_h, uint8_t *dst,
                   int dst_w, int dst_h) {
  const float scale_x = static_cast<float>(src_w) / dst_w;
  const float scale_y = static_cast<float>(src_h) / dst_h;
  auto sample = [&](int x, int y, int c) -> float {
    x = std::clamp(x, 0, src_w - 1);
    y = std::clamp(y, 0, src_h - 1);
    return static_cast<float>(src[(y * src_w + x) * 3 + c]);
  };

  for (int dy = 0; dy < dst_h; ++dy) {
    float sy1 = dy * scale_y;
    float sy2 = sy1 + scale_y;
    int sy_start = static_cast<int>(std::floor(sy1));
    int sy_end = static_cast<int>(std::ceil(sy2));
    for (int dx = 0; dx < dst_w; ++dx) {
      float sx1 = dx * scale_x;
      float sx2 = sx1 + scale_x;
      int sx_start = static_cast<int>(std::floor(sx1));
      int sx_end = static_cast<int>(std::ceil(sx2));
      for (int c = 0; c < 3; ++c) {
        float sum = 0.f;
        float area = 0.f;
        for (int sy = sy_start; sy < sy_end; ++sy) {
          float y1 = std::max(sy1, static_cast<float>(sy));
          float y2 = std::min(sy2, static_cast<float>(sy + 1));
          float wy = y2 - y1;
          if (wy <= 0.f) {
            continue;
          }
          for (int sx = sx_start; sx < sx_end; ++sx) {
            float x1 = std::max(sx1, static_cast<float>(sx));
            float x2 = std::min(sx2, static_cast<float>(sx + 1));
            float wx = x2 - x1;
            if (wx <= 0.f) {
              continue;
            }
            float w = wx * wy;
            sum += sample(sx, sy, c) * w;
            area += w;
          }
        }
        float val = (area > 0.f) ? (sum / area) : 0.f;
        dst[(dy * dst_w + dx) * 3 + c] =
            static_cast<uint8_t>(std::clamp(std::round(val), 0.f, 255.f));
      }
    }
  }
  return true;
}

// Bicubic (Catmull-Rom a=-0.5) — adapted from llama.cpp / mtmd vision resize.
bool BicubicResizeRgb(const uint8_t *src, int src_w, int src_h, uint8_t *dst,
                      int dst_w, int dst_h) {
  const float scale_x = static_cast<float>(src_w) / dst_w;
  const float scale_y = static_cast<float>(src_h) / dst_h;
  const float a = -0.5f;
  auto cubic_weight = [a](float x) -> float {
    x = std::fabs(x);
    if (x <= 1.f) {
      return (a + 2.f) * x * x * x - (a + 3.f) * x * x + 1.f;
    }
    if (x < 2.f) {
      return a * x * x * x - 5.f * a * x * x + 8.f * a * x - 4.f * a;
    }
    return 0.f;
  };
  auto sample = [&](int px, int py, int c) -> float {
    px = std::clamp(px, 0, src_w - 1);
    py = std::clamp(py, 0, src_h - 1);
    return static_cast<float>(src[(py * src_w + px) * 3 + c]);
  };

  for (int j = 0; j < dst_h; ++j) {
    float gy = (j + 0.5f) * scale_y - 0.5f;
    int y_int = static_cast<int>(std::floor(gy));
    float dy = gy - y_int;
    for (int i = 0; i < dst_w; ++i) {
      float gx = (i + 0.5f) * scale_x - 0.5f;
      int x_int = static_cast<int>(std::floor(gx));
      float dx = gx - x_int;
      for (int c = 0; c < 3; ++c) {
        double sum = 0.0;
        double wsum = 0.0;
        for (int m = -1; m <= 2; ++m) {
          float wy = cubic_weight(static_cast<float>(m) - dy);
          for (int n = -1; n <= 2; ++n) {
            float wx = cubic_weight(static_cast<float>(n) - dx);
            float w = wx * wy;
            sum += sample(x_int + n, y_int + m, c) * w;
            wsum += w;
          }
        }
        float v = (wsum != 0.0) ? static_cast<float>(sum / wsum) : 0.f;
        dst[(j * dst_w + i) * 3 + c] =
            static_cast<uint8_t>(std::clamp(std::round(v), 0.f, 255.f));
      }
    }
  }
  return true;
}

// V1: scale < 1 → bicubic; scale >= 1 → area.
ProcessedImage ResizeAndPadV1Rgb(const ProcessedImage &image, int target_w,
                                 int target_h) {
  float scale_w = static_cast<float>(target_w) / image.width;
  float scale_h = static_cast<float>(target_h) / image.height;
  float scale = std::min(scale_w, scale_h);
  int new_w = static_cast<int>(image.width * scale);
  int new_h = static_cast<int>(image.height * scale);
  if (new_w < 1) {
    new_w = 1;
  }
  if (new_h < 1) {
    new_h = 1;
  }

  std::vector<uint8_t> resized(static_cast<size_t>(new_w) * new_h * 3);
  if (scale < 1.0f) {
    BicubicResizeRgb(image.data.data(), image.width, image.height,
                     resized.data(), new_w, new_h);
  } else {
    AreaResizeRgb(image.data.data(), image.width, image.height, resized.data(),
                  new_w, new_h);
  }

  ProcessedImage out;
  out.width = target_w;
  out.height = target_h;
  out.channels = 3;
  out.data.assign(static_cast<size_t>(target_w) * target_h * 3, 114);
  for (int y = 0; y < new_h; ++y) {
    for (int x = 0; x < new_w; ++x) {
      for (int c = 0; c < 3; ++c) {
        out.data[static_cast<size_t>((y * target_w + x) * 3 + c)] =
            resized[static_cast<size_t>((y * new_w + x) * 3 + c)];
      }
    }
  }
  return out;
}

// V2: src larger on either dim → bicubic; else area.
ProcessedImage ResizeV2Rgb(const ProcessedImage &image, int target_w,
                           int target_h) {
  std::vector<uint8_t> resized(static_cast<size_t>(target_w) * target_h * 3);
  if (image.width > target_w || image.height > target_h) {
    BicubicResizeRgb(image.data.data(), image.width, image.height,
                     resized.data(), target_w, target_h);
  } else {
    AreaResizeRgb(image.data.data(), image.width, image.height, resized.data(),
                  target_w, target_h);
  }
  ProcessedImage out;
  out.width = target_w;
  out.height = target_h;
  out.channels = 3;
  out.data = std::move(resized);
  return out;
}

}  // namespace

HmImageProcessor::HmImageProcessor(int target_width, int target_height,
                                   bool use_v1)
    : target_width_(target_width),
      target_height_(target_height),
      use_v1_(use_v1) {}

ProcessedImage HmImageProcessor::LoadAndProcess(const std::string &image_path) {
  ProcessedImage processed;
  processed.channels = 3;

int w = 0;
  int h = 0;
  int c = 0;
  unsigned char *data = stbi_load(image_path.c_str(), &w, &h, &c, 3);
  if (!data) {
    std::cerr << "Failed to load image: " << image_path << std::endl;
    return MakeFallback(target_width_, target_height_);
  }
  processed.width = w;
  processed.height = h;
  processed.data.assign(data, data + static_cast<size_t>(w) * h * 3);
  stbi_image_free(data);

  if (use_v1_) {
    return ResizeAndPadV1(processed);
  }
  return ResizeV2(processed);
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

std::vector<float16> HmImageProcessor::ToFP16Tensor(
    const ProcessedImage &image) {
  const size_t num_pixels = static_cast<size_t>(image.width) * image.height;
  constexpr int num_frames = 2;
  std::vector<float16> tensor(3 * num_frames * num_pixels);
  // HWC RGB uint8 → [C, T, H, W] float16, raw 0..255, T duplicated
  for (int c = 0; c < 3; ++c) {
    for (int t = 0; t < num_frames; ++t) {
      for (size_t i = 0; i < num_pixels; ++i) {
        const float v = static_cast<float>(image.data[i * 3 + c]);
        tensor[(c * num_frames + t) * num_pixels + i] = float16(v);
      }
    }
  }
  return tensor;
}

ProcessedImage HmImageProcessor::ResizeAndPadV1(const ProcessedImage &image) {
  return ResizeAndPadV1Rgb(image, target_width_, target_height_);
}

ProcessedImage HmImageProcessor::ResizeV2(const ProcessedImage &image) {
  return ResizeV2Rgb(image, target_width_, target_height_);
}
