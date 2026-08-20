/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen35_dynamic_image_processor.h
 * Description:
 *   Qwen3.5 dynamic vision image decoding, resizing, normalization, and
 *   merge-major patch construction without changing the shared processor.
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
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "base/houmo.h"

namespace houmo::qwen35 {

/** Processed image patches and their unmerged patch grid. */
struct DynamicImageResult {
  std::vector<float16> pixel_values;
  int grid_t = 1;
  int grid_h = 0;
  int grid_w = 0;
  int patch_dim = 0;
};

class DynamicImageProcessor {
 public:
  struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data;
  };

  /** Construct a processor using model patch and pixel-budget settings. */
  DynamicImageProcessor(
      int patch_size = 16,
      int temporal_patch_size = 2,
      int merge_size = 2,
      int min_pixels = 65536,
      int max_pixels = 1572864,
      const std::string& preprocessor_config_path = {});

  /** Decode, resize, normalize, and patchify one RGB image. */
  DynamicImageResult LoadAndProcess(const std::string& image_path) const;

  static std::pair<int, int> SmartResize(
      int height, int width, int factor, int min_pixels, int max_pixels);

 private:
  RgbImage LoadRgb(const std::string& image_path) const;
  RgbImage Resize(const RgbImage& image, int height, int width) const;
  std::vector<float> Normalize(const RgbImage& image) const;
  std::vector<float16> Patchify(
      const std::vector<float>& chw,
      int height,
      int width) const;
  void FillPatch(
      const std::vector<float>& chw,
      int height,
      int width,
      int patch_id,
      int grid_w,
      int patch_dim,
      std::vector<float16>& output) const;

  int patch_size_;
  int temporal_patch_size_;
  int merge_size_;
  int min_pixels_;
  int max_pixels_;
  std::array<float, 3> mean_ = {0.48145466f, 0.4578275f, 0.40821073f};
  std::array<float, 3> std_ = {0.26862954f, 0.26130258f, 0.27577711f};
};

}  // namespace houmo::qwen35
