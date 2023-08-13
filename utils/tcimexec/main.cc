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

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if HAL
#include "hm800_hal.h"
#endif

#include "hdpl/hdpl_runtime.h"

struct CliArguments {
  std::string model_path;
  size_t warm_up = 10;
  size_t iterations = 10;
  bool is_fused = false;
  bool monitor_power = false;
};

/**
 * @brief whether the file exists
 *
 * @param file_path file path
 * @return true file exists
 * @return false file does not exist
 */
bool IsFileExists(std::string file_path) {
  std::ifstream f(file_path.c_str());
  return f.good();
}

#if HAL
void MonitorPower(std::vector<hm800_power_data> *data_vec, bool *stop_flag,
                  bool *started_flag);
void DumpPowerInfo(const std::vector<struct hm800_power_data> &data_vec);
#endif

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
      {"help", 0, 0, 'h'},       {"model", 1, 0, 'm'}, {"warm_up", 1, 0, 'w'},
      {"iterations", 1, 0, 'i'}, {"power", 0, 0, 'p'},
  };
  while (true) {
    int ch = getopt_long(argc, argv, "hm:w:i:p", long_options, &option_idx);
    if (ch == -1) {
      break;
    }
    switch (ch) {
      case 'h':
        std::cout << "Usage: -h" << std::endl;
        break;
      case 'm':
        std::cout << "Model: " << optarg << std::endl;
        arguments->model_path = std::string(optarg);
        break;
      case 'w':
        std::cout << "warm up: " << optarg << std::endl;
        arguments->warm_up = atoi(optarg);
        break;
      case 'i':
        std::cout << "iterations: " << optarg << std::endl;
        arguments->iterations = atoi(optarg);
        break;
      case 'p':
        std::cout << "enable monitor power" << std::endl;
        arguments->monitor_power = true;
        break;
      default:
        std::cerr << "Unsupported option: " << static_cast<char>(ch)
                  << std::endl;
        return false;
    }
  }
  if (IsFileExists(arguments->model_path + "_fused_op.so")) {
    arguments->is_fused = true;
  } else {
    arguments->is_fused = false;
  }
  return true;
}

int main(int argc, char *argv[]) {
  CliArguments arguments;
  ParseArgs(&arguments, argc, argv);
  std::cout << "Model: " << arguments.model_path << std::endl;
  tvm::hdpl::Module module = tvm::hdpl::LoadModelPackage(arguments.model_path);
  int tvm_input_count = module.GetInputNum();
  std::vector<int> input_idx_vec;
  for (int idx = 0; idx < tvm_input_count; idx++) {
    if (!module.IsParams(idx)) {
      input_idx_vec.push_back(idx);
    }
  }
  std::cout << "Count of Input: " << input_idx_vec.size() << std::endl;
  size_t batch = 1;
  for (int input_idx : input_idx_vec) {
    std::string input_name = module.GetInputNameByIndex(input_idx);
    tvm::runtime::NDArray input_data = module.GetInputByName(input_name);
    module.SetInput(input_name, input_data);
    std::cout << "Input " << input_name << ": (";
    auto input_shape = input_data.Shape();
    for (size_t shape_idx = 0; shape_idx < input_shape.size(); shape_idx++) {
      if (shape_idx != 0) {
        std::cout << ", ";
      } else {
        batch = input_shape.data()[0];
      }
      std::cout << input_shape.data()[shape_idx];
    }
    std::cout << "), " << input_data.DataType() << std::endl;
  }
  size_t eval_round = arguments.iterations;

#if HAL
  std::vector<struct hm800_power_data> power_data_vec;
  bool stop_flag = false;
  bool started_flag = false;
  std::vector<std::thread> threads;
  if (arguments.monitor_power) {
    std::thread monitor_th(
        [&] { MonitorPower(&power_data_vec, &stop_flag, &started_flag); });
    threads.push_back(std::move(monitor_th));
  }
  while (!started_flag) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
#endif
  auto start = std::chrono::system_clock::now();
  if (arguments.is_fused) {
    module.RunRounds(eval_round);
  } else {
    eval_round = 1;
    module.Run();
  }
  hdplDeviceSynchronize();
  auto finish = std::chrono::system_clock::now();

#if HAL
  stop_flag = true;
  for (auto &th : threads) {
    th.join();
  }
#endif
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  int64_t total_time = duration.count();
  std::cout << "\033[0;31mInference time cost total = " << total_time << "us"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;31mInference time cost per frame = " << std::fixed
            << std::setprecision(1) << 1.0 * total_time / eval_round << "us"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Throughput(QPS): " << std::fixed
            << std::setprecision(2) << (1.0e6 * eval_round / total_time * batch)
            << "fps"
            << "\033[0m" << std::endl;
#if HAL
  LOG(DEBUG) << "Power data point: " << power_data_vec.size();
  DumpPowerInfo(power_data_vec);
#endif
  return 0;
}

#if HAL
void MonitorPower(std::vector<struct hm800_power_data> *data_vec,
                  bool *stop_flag, bool *started_flag) {
  uint64_t vir_fd = 0;
  int32_t hmcl_ret = 0;
  hmcl_ret = hm800_usr_cmd_open_2(&vir_fd);
  if (hmcl_ret != 0) {
    LOG(ERROR) << "hm800_usr_cmd_open_2 failed with " << hmcl_ret;
    return;
  }
  struct hm800_power_data power_data;
  *started_flag = true;
  while (!(*stop_flag)) {
    hmcl_ret = hm800_get_power_2(vir_fd, &power_data);
    if (hmcl_ret == 0) {
      data_vec->push_back(power_data);
    } else {
      LOG(ERROR) << "hm800_get_power_2 failed with " << hmcl_ret;
      break;
    }
  }
  hm800_usr_cmd_close_2(vir_fd);
}

void DumpPowerInfo(const std::vector<struct hm800_power_data> &data_vec) {
  size_t data_length = data_vec.size();
  size_t effective_length = ceil(0.9 * data_length);
  size_t start_idx = (data_length - effective_length) / 2;
  struct hm800_power_data *sum_data =
      (struct hm800_power_data *)calloc(1, sizeof(struct hm800_power_data));
  for (size_t idx = start_idx; idx < effective_length + start_idx; idx++) {
    sum_data->ai_core_module_power += data_vec[idx].ai_core_module_power;
    sum_data->igital_core_module_power +=
        data_vec[idx].igital_core_module_power;
    sum_data->gddr_module_power += data_vec[idx].gddr_module_power;
    sum_data->noc_top_module_power += data_vec[idx].noc_top_module_power;
    sum_data->pcie_module_power += data_vec[idx].pcie_module_power;
    sum_data->video_codec_module_power +=
        data_vec[idx].video_codec_module_power;
    sum_data->jpeg_codec_module_power += data_vec[idx].jpeg_codec_module_power;
    sum_data->io_module_power += data_vec[idx].io_module_power;
    sum_data->low_speed_efuse_module_power +=
        data_vec[idx].low_speed_efuse_module_power;
    sum_data->all_modules_power += data_vec[idx].all_modules_power;
  }
  std::cout << "\033[0;32mAverage total consumption: "
            << 1. * sum_data->all_modules_power / effective_length << "W"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;32mAverage AiCore consumption: "
            << 1. * sum_data->ai_core_module_power / effective_length << "W"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Digital consumption: "
            << 1. * sum_data->igital_core_module_power / effective_length << "W"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;32mAverage GDDR consumption: "
            << 1. * sum_data->gddr_module_power / effective_length << "W"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;32mAverage NOC&TOP consumption: " << std::fixed
            << 1. * sum_data->noc_top_module_power / effective_length << "W"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;32mAverage PCIE consumption: "
            << 1. * sum_data->pcie_module_power / effective_length << "W"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Video Codec consumption: "
            << 1. * sum_data->video_codec_module_power / effective_length << "W"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Jpeg Codec consumption: "
            << 1. * sum_data->jpeg_codec_module_power / effective_length << "W"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;32mAverage IO consumption: "
            << 1. * sum_data->io_module_power / effective_length << "W"
            << "\033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Lower Speed Efuse consumption: "
            << 1. * sum_data->low_speed_efuse_module_power / effective_length
            << "W"
            << "\033[0m" << std::endl;
  free(sum_data);
}
#endif
