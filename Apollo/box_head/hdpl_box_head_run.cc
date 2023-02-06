// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file hdpl_box_head_run.cc
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
void RunGolden(runtime::NDArray output, std::string path, std::string file) {
  file = path + "hmquant_box_head_golden_txt/" + file + "_npy_output.txt";
  auto tensor = const_cast<DLTensor*>(output.operator->());
  int size = 1;
  for (int i = 0; i < tensor->ndim; ++i) {
    size *= tensor->shape[i];
  }
  std::vector<int8_t> host_out(size);
  ReadDataFromFile(file, host_out.data(), size);
  for (int i = 0; i < size; i++) {
    // printf("i:%d\n value:%d\n", i, ((int8_t *)tensor->data)[i]);
    assert(host_out[i] == ((int8_t*)tensor->data)[i]);
  }
}
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
void HDPLRuntimeWithoutSoftmax() {
  runtime::NDArray feature = runtime::NDArray::Empty({1, 512, 8, 8}, DataType::Int(8), {kDLCPU, 0});
  auto pA = static_cast<int8_t*>(feature->data);
  int size = 512 * 8 * 8;
  std::vector<int8_t> nchw_data(size);
  ReadDataFromFile("box_head_without_softmax/hmquant_box_head_input.txt", nchw_data.data(), size);
  NCHW2NHWC(nchw_data.data(), pA, 1, 8, 8, 512);

  auto module = tvm::hdpl::LoadModelPackage("./libboxhead_without_softmax");
  module.SetInput("roi_feature_2d", feature);
  module.Run();
  RunGolden(module.GetOutput(0), "box_head_without_softmax/", "23");
  RunGolden(module.GetOutput(1), "box_head_without_softmax/", "proposal_deltas");
}
void HDPLRuntimeWithSoftmax() {
  runtime::NDArray feature = runtime::NDArray::Empty({1, 512, 8, 8}, DataType::Int(8), {kDLCPU, 0});
  auto pA = static_cast<int8_t*>(feature->data);
  int size = 512 * 8 * 8;
  std::vector<int8_t> nchw_data(size);
  ReadDataFromFile("box_head/hmquant_box_head_input.txt", nchw_data.data(), size);
  NCHW2NHWC(nchw_data.data(), pA, 1, 8, 8, 512);

  auto module = tvm::hdpl::LoadModelPackage("./libboxhead");
  module.SetInput("roi_feature_2d", feature);
  module.Run();
  RunGolden(module.GetOutput(0), "box_head/", "scores");
  RunGolden(module.GetOutput(1), "box_head/", "proposal_deltas");
}
int main() {
  std::string module = "libboxhead.so";
  if(access(module.c_str(), F_OK) != -1) {
  HDPLRuntimeWithSoftmax();
  }
  module = "libboxhead_without_softmax.so";
  if(access(module.c_str(), F_OK) != -1) {
  HDPLRuntimeWithoutSoftmax();
  }
  return 0;
}
