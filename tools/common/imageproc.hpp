/*
 * Copyright (c) 2022 HOUMO AI
 *
 * File: imageproc.hpp
 * Description:
 *   Image Processing Utilities Header File - Defines the ImageProc class for
 * image format conversion operations.
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
#ifndef IMAGEPROC_HPP
#define IMAGEPROC_HPP
#include <unistd.h>

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

class ImageProc {
 public:
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

#endif  // IMAGEPROC_HPP