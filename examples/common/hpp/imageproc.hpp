// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file imageproc.hpp
 */

#include <iostream>
#include <sstream>
#include <string>


class ImageProc {

 public:
  static void BgrToRgb(int8_t *src, int h, int w) {
    for (int i=0; i<h*w*3; i+=3) {
      int tmp;
      if (i%3 == 0) {
        tmp = src[i];
        src[i] = src[i+2];
        src[i+2] = tmp;
      }
    }
  }

  static void I420To420sp(uint8_t *src, uint8_t *dst, int size) {
    size = size/2;
    for (int i=0;i<size/3*2;i++){
      src[i] = dst[i];
    }
    for (int i=0;i<size/6;i++) {
      src[size/3*2+i*2] = dst[size/6*4+i];
      src[size/3*2+i*2+1] = dst[size/6*5+i];
    }
  }
};

