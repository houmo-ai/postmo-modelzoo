/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: image_processor_test.cc
 * Description:
 *   Unit tests for HmImageProcessor (stb / pure-C++ backend).
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

#include "modules/image_processor.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

// 8x6 RGB PNG generated at plan time; written to cwd so tests need no asset files
// (repo .gitignore ignores *.png).
std::string WriteTempRgbPng() {
  static const unsigned char kPng[] = {
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x06,
      0x08, 0x02, 0x00, 0x00, 0x00, 0x71, 0x67, 0x48, 0xac, 0x00, 0x00, 0x00,
      0x63, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x0d, 0xc9, 0x21, 0x01, 0x00,
      0x41, 0x08, 0x45, 0x41, 0x92, 0xa0, 0x09, 0xb1, 0x21, 0xd0, 0x9b, 0xe4,
      0xcb, 0x97, 0x02, 0x4d, 0x88, 0x0d, 0x81, 0x26, 0xd1, 0xdd, 0xd8, 0x31,
      0x33, 0xdc, 0x38, 0xc6, 0x35, 0x64, 0x94, 0xf1, 0x8c, 0x35, 0xcc, 0x02,
      0x0f, 0x4e, 0x70, 0x03, 0x05, 0x15, 0xbc, 0x60, 0xe3, 0x8f, 0xc4, 0x93,
      0x93, 0xdc, 0x44, 0x49, 0x25, 0x2f, 0xd9, 0xfc, 0x43, 0xb8, 0x38, 0xe2,
      0x0a, 0x89, 0x12, 0x4f, 0xac, 0xfe, 0x68, 0xbc, 0x39, 0xcd, 0x6d, 0xd4,
      0x54, 0xf3, 0x9a, 0xed, 0x3f, 0x06, 0x1f, 0xce, 0x70, 0x07, 0x0d, 0x35,
      0xbc, 0x61, 0x87, 0x0f, 0xcc, 0x7f, 0x3e, 0x71, 0x8f, 0xb9, 0xa0, 0x71,
      0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
  };
  const char *path = "hm_image_processor_test.png";
  std::ofstream ofs(path, std::ios::binary);
  ofs.write(reinterpret_cast<const char *>(kPng),
            static_cast<std::streamsize>(sizeof(kPng)));
  ofs.close();
  return path;
}

class ImageProcessorFixture : public ::testing::Test {
 protected:
  void SetUp() override { fixture_path_ = WriteTempRgbPng(); }
  void TearDown() override { std::remove(fixture_path_.c_str()); }
  std::string fixture_path_;
};

}  // namespace

TEST_F(ImageProcessorFixture, LoadAndProcessV2Shape) {
  HmImageProcessor proc(64, 48, /*use_v1=*/false);
  auto img = proc.LoadAndProcess(fixture_path_);
  EXPECT_EQ(img.width, 64);
  EXPECT_EQ(img.height, 48);
  EXPECT_EQ(img.channels, 3);
  EXPECT_EQ(img.data.size(), static_cast<size_t>(64 * 48 * 3));
}

TEST_F(ImageProcessorFixture, LoadAndProcessV1Shape) {
  HmImageProcessor proc(64, 48, /*use_v1=*/true);
  auto img = proc.LoadAndProcess(fixture_path_);
  EXPECT_EQ(img.width, 64);
  EXPECT_EQ(img.height, 48);
  EXPECT_EQ(img.channels, 3);
  EXPECT_EQ(img.data.size(), static_cast<size_t>(64 * 48 * 3));
}

TEST(ImageProcessorTest, ToFP16TensorLayoutCTHW) {
  ProcessedImage img;
  img.width = 2;
  img.height = 2;
  img.channels = 3;
  // HWC: pixel00=(1,2,3), pixel01=(4,5,6), pixel10=(7,8,9), pixel11=(10,11,12)
  img.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  HmImageProcessor proc(2, 2, false);
  auto t = proc.ToFP16Tensor(img);
  ASSERT_EQ(t.size(), 24u);
  const size_t num_pixels = 4;
  EXPECT_FLOAT_EQ(static_cast<float>(t[0 * 2 * num_pixels + 0 * num_pixels + 0]),
                  1.f);
  EXPECT_FLOAT_EQ(static_cast<float>(t[0 * 2 * num_pixels + 1 * num_pixels + 0]),
                  1.f);
  EXPECT_FLOAT_EQ(static_cast<float>(t[1 * 2 * num_pixels + 0 * num_pixels + 0]),
                  2.f);
  EXPECT_FLOAT_EQ(static_cast<float>(t[2 * 2 * num_pixels + 0 * num_pixels + 3]),
                  12.f);
}

TEST(ImageProcessorTest, LoadMissingFallsBackToGray114) {
  HmImageProcessor proc(16, 16, false);
  auto img = proc.LoadAndProcess("/nonexistent/path/no_such_image.png");
  EXPECT_EQ(img.width, 16);
  EXPECT_EQ(img.height, 16);
  ASSERT_EQ(img.data.size(), 16u * 16u * 3u);
  EXPECT_EQ(img.data[0], 114);
  EXPECT_EQ(img.data[1], 114);
  EXPECT_EQ(img.data[2], 114);
}

TEST_F(ImageProcessorFixture, BatchLoadCount) {
  HmImageProcessor proc(32, 32, false);
  auto batch = proc.LoadAndProcessBatch({fixture_path_, fixture_path_});
  ASSERT_EQ(batch.size(), 2u);
  EXPECT_EQ(batch[0].width, 32);
  EXPECT_EQ(batch[1].height, 32);
}

TEST(ImageProcessorTest, GetTargetDims) {
  HmImageProcessor proc(448, 336, true);
  auto dims = proc.GetTargetDims();
  EXPECT_EQ(dims.width, 448);
  EXPECT_EQ(dims.height, 336);
  EXPECT_EQ(dims.channels, 3);
}
