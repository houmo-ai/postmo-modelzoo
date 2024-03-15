
// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file hdpl_resnet50_run.cc
 */
#include <getopt.h>
#include <stdio.h>
#include <tvm/runtime/executor_info.h>
#include <tvm/runtime/hdpl/hdpl_runtime.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>
#include <memory>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include "hdpl/hdpl_runtime.h"

#define SAVE_DATA 0

struct CliArguments {
  std::string model_path;
  std::string data_path;
  size_t warm_up = 10;
  size_t iterations = 1;
  size_t loops = 100;
  size_t threads = 5;
};

int read_file(const char *fileName, char **fileData, int *fileLen) 
{
  FILE *file = fopen(fileName, "rb"); 
  if (file == NULL) {
    perror("open file failed\n");
    return -1;
  }

  fseek(file, 0, SEEK_END);
  long fileSize = ftell(file);
  fseek(file, 0, SEEK_SET);

  *fileData = (char *)malloc(fileSize);
  if (*fileData == NULL) {
    printf("malloc fileData size:%ld fialed\n", fileSize);
    fclose(file);
    return -1;
  }
  long readSize = fread(*fileData, 1, fileSize, file);
  if (readSize != fileSize) {
    printf("readSize(%ld) != fileSize(%ld), read %s failed!\n", readSize, fileSize, fileName);
    fclose(file);
    return -1;
  }
  *fileLen = fileSize;
  fclose(file);
  return 0;
}

int write_file(const char *fileName, char *fileData, int fileLen) 
{
  FILE *file = fopen(fileName, "wb"); 
  if (file == NULL) {
    perror("open file failed\n");
    return -1;
  }
  long writeSize = fwrite(fileData, 1, fileLen, file);
  if (writeSize != fileLen) {
    printf("writeSize(%ld) != fileLen(%d), write %s failed!\n", writeSize, fileLen, fileName);
    fclose(file);
    return -1;
  }
  fclose(file);
  return 0;
}

/**
 * @brief Parse cmdline arguments to struct *arguments
 *
 * @param arguments pointer to output CliArguments struct
 * @param argc cmdline argument count
 * @param argv cmdline argument char*
 * @return true parse command line succeed
 * @return false parse command line failed
 */
bool ParseArgs(CliArguments *arguments, int argc, char *argv[]) {
  int option_idx = 0;
  struct option long_options[] = {
      {"help", 0, 0, 'h'},
      {"model", 1, 0, 'm'},
      {"data", 1, 0, 'd'},
      {"warm_up", 1, 0, 'w'},
      {"iterations", 1, 0, 'i'},
      {"loops", 1, 0, 'l'},
      {"threads", 1, 0, 't'},
  };
  while (true) {
    int ch = getopt_long(argc, argv, "hm:d:w:i:l:t:", long_options, &option_idx);
    if (ch == -1) {
      break;
    }
    switch (ch) {
    case 'h':
      std::cout << "Usage: -h" << std::endl;
      break;
    case 'm':
      std::cout << "model: " << optarg << std::endl;
      arguments->model_path = std::string(optarg);
      break;
    case 'd':
      std::cout << "data: " << optarg << std::endl;
      arguments->data_path = std::string(optarg);
      break;
    case 'w':
      std::cout << "warm up: " << optarg << std::endl;
      arguments->warm_up = atoi(optarg);
      break;
    case 'i':
      std::cout << "iterations: " << optarg << std::endl;
      arguments->iterations = atoi(optarg);
      break;
    case 'l':
      std::cout << "loop: " << optarg << std::endl;
      arguments->loops = atoi(optarg);
      break;
    case 't':
      std::cout << "threads: " << optarg << std::endl;
      arguments->threads = atoi(optarg);
      break;
    default:
      std::cerr << "Unsupported option: " << static_cast<char>(ch) << std::endl;
      return false;
    }
  }
  return true;
}
bool createHdplStream(std::shared_ptr<hdplStream_t> &stream) {
  auto ptr = new hdplStream_t;
  if (ptr) {
    if (hdplStreamCreate(ptr) == hdplSuccess) {
      stream.reset(ptr, [](hdplStream_t *p) {
        hdplStreamSynchronize(*p);
        hdplStreamDestroy(*p);
        delete p;
      });
      return true;
    }
    delete ptr;
  }
  return false;
}

int main(int argc, char *argv[]) {

  CliArguments arguments;
  ParseArgs(&arguments, argc, argv);
  int loop = arguments.loops;
  int interloop = arguments.iterations;
  
  // Create Stream and Module
  std::vector<tvm::hdpl::Module> modules;
  std::vector<std::thread> threads;
  std::vector<int64_t> total_times;
  total_times.resize(arguments.threads);
  std::vector<std::string> input_names;
  std::vector<std::vector<tvm::runtime::NDArray>> output_datas(arguments.threads);
  std::vector<std::vector<int>> output_data_size(arguments.threads);
  size_t batch = 1;
  char fileName[256];
  char *imageData = NULL;
  int fileLen = 0;
  // 640x384_422sp.yuv
  read_file(arguments.data_path.c_str(), &imageData, &fileLen);
  printf("file_len %d\n", fileLen);
  std::vector<std::shared_ptr<tvm::runtime::NDArray>> input_datas;

  for (int i = 0; i < arguments.threads; i++) {
    std::cout << "Thread " << i << " model: " << arguments.model_path << std::endl;
    // load model
    tvm::hdpl::Module module = tvm::hdpl::LoadModelPackage(arguments.model_path, "aot");
    modules.push_back(std::move(module));
    auto size = module.GetGlobalMemSize();
    std::cout << "model memsize = " << size << std::endl;
  }
    
  // get input info && set input
  int tvm_input_count = modules[0].GetInputNum();
  std::cout << "Count of Input: " << tvm_input_count << std::endl;
  for (int idx = 0; idx < tvm_input_count; idx++) {
    std::string input_name = modules[0].GetInputNameByIndex(idx);
    tvm::runtime::NDArray data = modules[0].GetInputByName(input_name);
    std::shared_ptr<tvm::runtime::NDArray> input_data(new tvm::runtime::NDArray);
    *input_data = tvm::runtime::NDArray::Empty(
      data.Shape(), data.DataType(), {kDLCPU, 0});
    memcpy((*input_data)->data, imageData, fileLen);
    for (int tid = 0; tid < arguments.threads; tid++) {
      modules[tid].SetInput(input_name, *input_data);
    }
    input_datas.push_back(input_data);
    std::cout << "Input " << input_name << ": (";
    auto input_shape = input_data->Shape();
    for (size_t shape_idx = 0; shape_idx < input_shape.size(); shape_idx++) {
      if (shape_idx != 0) {
        std::cout << ", ";
      } else {
        batch = input_shape.data()[0];
      }
      std::cout << input_shape.data()[shape_idx];
    }
    std::cout << "), " << input_data->DataType() << std::endl;
  }
  
  // run
  modules[0].Run();
  hdplDeviceSynchronize();
    
  // get output info && set output
  int tvm_output_count = modules[0].GetOutputNum();
  std::cout << "Count of Output: " << tvm_output_count << std::endl;
  for (int idx = 0; idx < tvm_output_count; idx++) {
    auto output_name = modules[0].GetOutputNameByIndex(idx);
    tvm::runtime::NDArray output_data = modules[0].GetOutputByName(output_name);
    auto output_shape = output_data.Shape();
    int data_size = 1;
    std::cout << "Output " << output_name << ": (";
    for (size_t shape_idx = 0; shape_idx < output_shape.size(); shape_idx++) {
      if (shape_idx != 0) {
        std::cout << ", ";
      }
      std::cout << output_data.Shape().data()[shape_idx];
      data_size *= output_data.Shape().data()[shape_idx];
    }
    for (int tid = 0; tid < arguments.threads; tid++) {
      output_data_size[tid].push_back(data_size);
    }
    std::cout << "), " << output_data.DataType() << std::endl;
#if SAVE_DATA
    snprintf(fileName, sizeof(fileName), "%s_output.bin", output_name.c_str());
    write_file(fileName, (char*)output_data->data, data_size);
#endif
  }
  
  auto main_loop = [](int t_id, int loop, int interloop, tvm::hdpl::Module &module, std::vector<int> &datasize, int64_t &total_time) {
    auto start = std::chrono::system_clock::now();
    char fileName[256];
    for (int i = 0; i < loop; i++) {
      std::shared_ptr<hdplStream_t> stream;
      createHdplStream(stream);
      module.SetStream(*stream);
      module.Run(interloop);
      hdplStreamSynchronize(*stream);
#if SAVE_DATA
      int tvm_output_count = module.GetOutputNum();
      for (int idx = 0; idx < tvm_output_count; idx++) {
        auto output_name = module.GetOutputNameByIndex(idx);
        tvm::runtime::NDArray output_data = module.GetOutputByName(output_name);
        // auto output_shape = output_data.Shape();
        // int data_size = 1;
        // for (size_t shape_idx = 0; shape_idx < output_shape.size(); shape_idx++) {
        //   data_size *= output_shape.data()[shape_idx];
        // }
        snprintf(fileName, sizeof(fileName), "thread_%d_run_%d_%s_output.bin", t_id, i, output_name.c_str());
        write_file(fileName, (char*)output_data->data, datasize[idx]);
      }
#endif
    }
    auto finish = std::chrono::system_clock::now();
    auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    total_time = duration.count();
  };
#if 0
  for (int i = 0; i < modules.size(); i++) {
    main_loop(i, loop, interloop, modules[i], output_data_size[i], total_times[i]);
  }
#else
  for (int i = 0; i < modules.size(); i++) {
    threads.push_back(
        std::thread(main_loop, i, loop, interloop, std::ref(modules[i]), std::ref(output_data_size[i]), std::ref(total_times[i]))
        );
  }
  for (auto &t : threads) {
    t.join();
  }
#endif

  int64_t total_time = 0;
  int eval_loop = arguments.loops * arguments.iterations;
  for (auto t : total_times) {
      std::cout << "\033[0;31mInference time cost total = " << (t) << "us"
                << "\033[0m" << std::endl;
      std::cout << "\033[0;31mInference time cost per frame = "
                << (t / eval_loop) << "us"
                << "\033[0m" << std::endl;
      total_time += t;
  }
  std::cout << "\033[0;32mAverage Throughput(QPS): "
            << ((1000000.0 / (total_time / eval_loop)) * arguments.threads)
            << "fps"
            << "\033[0m" << std::endl;

}
