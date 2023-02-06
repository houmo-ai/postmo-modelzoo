// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file hdpl_attr_head_run.cc
 */
#include <unistd.h>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <tvm/runtime/executor_info.h>
#include <tvm/runtime/hdpl/hdpl_runtime.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

using namespace tvm;
using namespace tvm::relay;

void NCHW2NHWC(int8_t* input, int8_t* output, int n, int h, int w, int c) {
  for (int d = 0; d < n; d++) {
    for (int i = 0; i < c; i++) {
      for (int j = 0; j < h; j++) {
        for (int k = 0; k < w; k++) {
          int in_stride = d * h * w * c + i * h * w + j * w + k;
          int out_stride = d * h * w * c + j * w * c + k * c + i;
          output[out_stride] = input[in_stride];
        }
      }
    }
  }
}
void ReadDataFromFile(const std::string file_name, int8_t* data, uint32_t num) {
  DLOG(INFO) << "ReadDataFromFile, file_name = " << file_name;
  const char* filename = file_name.c_str();
  FILE* fp = fopen(filename, "r");
  for (uint32_t i = 0; i < num; ++i) {
    int x;
    int res = fscanf(fp, "%d", &x);
    if (res < 0) {
      DLOG(INFO) << "Length of file is: " << i << ", less than length parameter: " << num;
      fclose(fp);
      return;
    }
    *(data + i) = x;
  }
  fclose(fp);
}

void RunGolden(runtime::NDArray output, std::string file) {
  file = "attr3d_head/hmquant_attr3d_head_golden_txt/" + file + "_npy_output.txt";
  auto tensor = const_cast<DLTensor*>(output.operator->());
  int size = 1;
  for (int i = 0; i < tensor->ndim; ++i) {
    size *= tensor->shape[i];
  }
  std::vector<int8_t> host_out(size);
  ReadDataFromFile(file, host_out.data(), size);
  for (int i = 0; i < size; i++) {
    assert(host_out[i] == ((int8_t*)tensor->data)[i]);
  }
}
void HDPLRuntime() {
  runtime::NDArray feature = runtime::NDArray::Empty({1, 512, 8, 8}, DataType::Int(8), {kDLCPU, 0});
  auto pA = static_cast<int8_t*>(feature->data);
  int size = 512 * 8 * 8;
  std::vector<int8_t> nchw_data(size);
  ReadDataFromFile("attr3d_head/hmquant_attr3d_head_input.txt", nchw_data.data(), size);
  NCHW2NHWC(nchw_data.data(), pA, 1, 8, 8, 512);

  auto module = tvm::hdpl::LoadModelPackage("./libattrhead");
  module.SetInput("roi_feature_3d", feature);
  module.Run();
  RunGolden(module.GetOutput(0), "pred_dim");
  RunGolden(module.GetOutput(1), "pred_ori");
  RunGolden(module.GetOutput(2), "pred_dist");
  RunGolden(module.GetOutput(3), "pred_proj_center");
  RunGolden(module.GetOutput(4), "pred_proj_front");
  RunGolden(module.GetOutput(5), "pred_proj_back");
  RunGolden(module.GetOutput(6), "pred_heading_fb_score");
  RunGolden(module.GetOutput(7), "pred_heading_lr_score");
  RunGolden(module.GetOutput(8), "pred_onroad_prob");
  RunGolden(module.GetOutput(9), "pred_occ_prob");
  RunGolden(module.GetOutput(10), "pred_trc_prob");
}
int main() {
  HDPLRuntime();
  return 0;
}
