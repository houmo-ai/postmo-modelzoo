// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file hdpl_backbone_run.cc
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
void NHWC2NCHW(int8_t* input, int8_t* output, int n, int h, int w, int c) {
  for (int d = 0; d < n; d++) {
    for (int i = 0; i < c; i++) {
      for (int j = 0; j < h; j++) {
        for (int k = 0; k < w; k++) {
          int in_stride = d * h * w * c + j * w * c + k * c + i;
          int out_stride = d * h * w * c + i * h * w + j * w + k;
          output[out_stride] = input[in_stride];
        }
      }
    }
  }
}
void RunGolden(runtime::NDArray output, std::string file) {
  file = "backbone/" + file + "_npy_output.txt";
  auto tensor = const_cast<DLTensor*>(output.operator->());
  int size = 1;
  for (int i = 0; i < tensor->ndim; ++i) {
    size *= tensor->shape[i];
  }
  std::vector<int8_t> host_out(size);
  ReadDataFromFile(file, host_out.data(), size);
  int8_t* output_ = new int8_t[size];
  NHWC2NCHW(host_out.data(), output_, tensor->shape[0], tensor->shape[2], tensor->shape[3],
            tensor->shape[1]);
  for (int i = 0; i < size; i++) {
    assert(output_[i] == ((int8_t*)tensor->data)[i]);
  }
  delete[] output_;
}
void HDPLRuntime() {
  runtime::NDArray feature =
      runtime::NDArray::Empty({1, 3, 360, 360}, DataType::Int(8), {kDLCPU, 0});
  auto pA = static_cast<int8_t*>(feature->data);
  int size = 360 * 360 * 3;
  ReadDataFromFile("backbone/backbone_input.txt", pA, size);

  auto module = tvm::hdpl::LoadModelPackage("./libbackbone");
  module.SetInput("data", feature);
  module.Run();
  RunGolden(module.GetOutput(0), "block_9");
  RunGolden(module.GetOutput(1), "pred_objectness");
  RunGolden(module.GetOutput(2), "pred_anchor_deltas");
}
int main() {
  HDPLRuntime();
  return 0;
}
