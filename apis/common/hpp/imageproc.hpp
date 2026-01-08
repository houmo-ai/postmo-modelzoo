/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: imageproc.hpp
 * Description:
 *   Image processing utility functions for color space conversion and format
 *   transformation.
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

#ifndef __APIS_COMMON_HPP_IMAGEPROC_HPP__
#define __APIS_COMMON_HPP_IMAGEPROC_HPP__

#include <iostream>
#include <sstream>
#include <string>

/**
 * @brief Image processing utility class containing static methods for color
 * space conversion
 */
class ImageProc {
 public:
  /**
   * @brief Convert image data from BGR to RGB color space
   *
   * @param src Pointer to the source image data array in BGR format
   * @param h Height of the image in pixels
   * @param w Width of the image in pixels
   */
  static void BgrToRgb(int8_t *src, int h, int w) {
    for (int i = 0; i < h * w * 3; i += 3) {
      int tmp;
      if (i % 3 == 0) {
        tmp = src[i];
        src[i] = src[i + 2];
        src[i + 2] = tmp;
      }
    }
  }

  /**
   * @brief Convert image data from I420 format to 420sp format
   *
   * @param src Pointer to the destination buffer for 420sp format data
   * @param dst Pointer to the source buffer containing I420 format data
   * @param size Total size of the source image data in bytes
   */
  static void I420To420sp(uint8_t *src, uint8_t *dst, int size) {
    size = size / 2;
    for (int i = 0; i < size / 3 * 2; i++) {
      src[i] = dst[i];
    }
    for (int i = 0; i < size / 6; i++) {
      src[size / 3 * 2 + i * 2] = dst[size / 6 * 4 + i];
      src[size / 3 * 2 + i * 2 + 1] = dst[size / 6 * 5 + i];
    }
  }
};

#endif  // __APIS_COMMON_HPP_IMAGEPROC_HPP__
