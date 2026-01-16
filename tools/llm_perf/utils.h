/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: utils.h
 * Description:
 *   Utility Functions Header File - Defines various utility functions for LLM
 * performance testing including argument parsing, path validation, performance
 * metrics display, and random vector generation.
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
#ifndef __UTILS_H__
#define __UTILS_H__

#include <algorithm>
#include <cctype>
#include <chrono>
#include <codecvt>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "perf_tracker/inference_perf_tracker.h"

#ifdef XH2A_HM_SYS
#ifdef __cplusplus
extern "C" {
#endif
#include "hm_sys.h"
#ifdef __cplusplus
}
#endif
#endif

#define TOKEN_ID_MAX 150000

#define COLOR_RED "\x1b[91;20m"
#define COLOR_GREEN "\x1b[92;20m"
#define COLOR_YELLOW "\x1b[93;20m"
#define COLOR_BLUE "\x1b[94;20m"
#define COLOR_MAGENT "\x1b[95;20m"
#define COLOR_CYAN "\x1b[96;20m"
#define COLOR_RESET "\x1b[0m"

namespace fs = std::filesystem;
using json = nlohmann::json;

typedef enum { PERFCMD = 0, PERFJSON, PERFINVAILD } PerfConfigType;

static void HelpUsage(char* argv[]) {
  std::cout
      << "Usage: " << argv[0]
      << " --key value [options...]\n\n"
         "Options:\n"
         "  -c, --config    FILE      use json file to start llm_perf, "
         "cat template config.json for more message\n"
         "Or:\n"
         "  --prefill       FILE      prefill model file\n"
         "  --decode        FILE      decode model file\n"
         "  --visual        FILE      visual model file, only vllm perf need\n"
         "  --embedding     FILE      embedding weight file\n"
         "  --input         NUM       number of input tokens\n"
         "  --stop          NUM       number of tokens to generate\n"
         "  --ndevices      NUM       device count\n"
         "  --loop          NUM       loop test rounds\n"
         "  --batch         NUM       if multibatch model only xh2 support!\n"
         "  --no_warm_up              disable warm up!\n"
         "  --LazyMode                enable lazy mode!\n"
         "  -h, --help                show help message\n";
}

static std::unordered_map<std::string, std::string> parse_json(const json& j) {
  std::unordered_map<std::string, std::string> args;
  for (auto& [key, val] : j.items()) {
    args[key] = val.is_string() ? val.get<std::string>() : val.dump();
  }

  return args;
}

static PerfConfigType ParsePerfRunType(int argc, char* argv[]) {
  if (argc == 1) {
    return PerfConfigType::PERFINVAILD;
  }

  if (argc == 2) {
    std::string arg = argv[1];
    if (arg == "-h" || arg == "--help") {
      return PerfConfigType::PERFINVAILD;
    }
  }

  if (argc == 3) {
    const std::string arg = argv[1];
    if (arg == "-c" || arg == "--config") {
      return PerfConfigType::PERFJSON;
    }
  }

  return PerfConfigType::PERFCMD;
}

/**
 * Parse command line arguments in --key value format
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return Parsed parameter mapping (key: parameter name, value: parameter
 * value)
 */
static std::unordered_map<std::string, std::string> parse_args(int argc,
                                                               char* argv[]) {
  std::unordered_map<std::string, std::string> args;
  std::set<std::string> flags = {"no_warm_up", "LazyMode"};

  // Check for invalid argument combinations at the beginning
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[1];  // Note: This should probably be argv[i]
    if (arg == "-c" || arg == "--config" || arg == "-h" || arg == "--help") {
      std::cerr << "Invalid args!" << std::endl;
      HelpUsage(argv);
      std::exit(0);
    }
  }

  // Parse arguments in --key value format
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) == "--") {
      std::string key = arg.substr(2);  // Extract key by removing leading "--"

      // Handle flag arguments that don't require a value
      if (flags.find(key) != flags.end()) {
        args[key] = "";  // Set empty value for flag
        continue;
      }
      // Check if there's a value following the key
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for argument: " + arg);
      }

      // Get the value and increment index to skip it in the next iteration
      std::string value = argv[++i];
      args[key] = value;
    } else {
      // Invalid argument format - must start with --
      throw std::invalid_argument("Invalid argument format: " + arg +
                                  " (use --key value)");
    }
  }

  return args;
}

/**
 * Validate if a path exists (file or directory)
 * @param args Map containing command line arguments
 * @param arg_name Name of the argument to validate (used for error messages)
 * @return Normalized path if it exists
 */
static fs::path validate_path(
    std::unordered_map<std::string, std::string>& args,
    const std::string& arg_name) {
  fs::path path;
  if (args.find(arg_name) != args.end()) {
    if (args[arg_name].empty()) {
      throw std::invalid_argument("Missing " + arg_name + " value (use --" +
                                  arg_name + " <value>).");
    }
    std::string path_str = args[arg_name];
    // Create path object supporting Unicode paths (cross-platform)
    path = fs::u8path(path_str);

    if (!fs::exists(path)) {
      throw std::invalid_argument(arg_name +
                                  " path does not exist: " + path.u8string());
    }
  } else {
    throw std::invalid_argument("Missing arg : " + arg_name + ", (use --" +
                                arg_name + " to set arg).");
  }
  return path;
}

static int validate_setting(std::unordered_map<std::string, std::string>& args,
                            const std::string& arg_name) {
  int value;
  if (args.find(arg_name) != args.end()) {
    if (args[arg_name].empty()) {
      throw std::invalid_argument("Missing " + arg_name + " value (use --" +
                                  arg_name + " <value>).");
    }

    value = stoi(args[arg_name]);
    if (value <= 0) {
      throw std::invalid_argument("Invalid " + arg_name + " value (use --" +
                                  arg_name + " <value> to set valid value).");
    }
  } else {
    throw std::invalid_argument("Missing arg : " + arg_name + ", (use --" +
                                arg_name + " to set arg).");
  }
  return value;
}

struct PerfInfos {
  uint32_t input_tokens;
  uint32_t stop_tokens;
  float prefill_time;
  float decode_time;
  float embedding_time;
  float vit_time;
  float ttft;
  float t_total;  // E2E Latency
  uint32_t decode_count;
};

static void ShowPerfInformation(PerfInfos llm_perf_datas) {
  std::ostringstream os;
  os << "\n-------------------  Performance Summary  --------------------\n";
  os << std::left << std::setfill(' ');
  os << std::setw(30) << "Metric" << std::setw(30) << "Value" << '\n';
  os << std::string(62, '-') << '\n';

  auto token = [&](const std::string& name, auto val,
                   const std::string& unit = "") {
    os << std::setw(50) << name << std::setw(30) << val << unit << '\n';
  };

  auto fmt = [](auto v, int prec, const char* unit) -> std::string {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v << unit;
    return o.str();
  };

  token("Prefill Time", fmt(llm_perf_datas.prefill_time, 2, " ms"));
  token("Decode Time", fmt(llm_perf_datas.decode_time, 2, " ms"));
  if (abs(llm_perf_datas.vit_time - 0) > 1e-10) {
    token("Vision Time", fmt(llm_perf_datas.vit_time, 2, " ms"));
  }
  token("Prefill Speed", fmt(llm_perf_datas.input_tokens /
                                 (llm_perf_datas.prefill_time * 0.001f),
                             2, " tokens/s"));
  token("Decode Speed",
        fmt(llm_perf_datas.decode_count / (llm_perf_datas.decode_time * 0.001f),
            2, " tokens/s"));
  token("TTFT (Time to First Token)", fmt(llm_perf_datas.ttft, 2, " ms"));
  token("TPOT (Time Per Output Token)",
        fmt(llm_perf_datas.decode_time / llm_perf_datas.decode_count, 2,
            " ms/token"));
  token("E2E Latency (End-to-End Latency)",
        fmt(llm_perf_datas.t_total * 0.001f, 2, " seconds"));
  token(
      "E2E TPS (End-to-End Tokens Per Second)",
      fmt((llm_perf_datas.decode_count + 1) / (llm_perf_datas.t_total * 0.001f),
          2, " tokens/s"));
  token("Embedding Time", fmt(llm_perf_datas.embedding_time, 2, " ms"));
  os << "--------------------------------------------------------------\n";
  std::cout << os.str();
}

/**
 * Generate a vector of random integers within a specified length
 * Used for creating random token IDs for testing purposes
 * @param len Length of the vector to generate
 * @return Vector of random integers in the range [0, TOKEN_ID_MAX]
 */
static std::vector<int> generateRandomVector(int len) {
  std::vector<int> result;
  if (len <= 0) {
    return result;  // Handle invalid length (return empty vector)
  }

  // Use current time as random seed to ensure different sequences each run
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed);  // Use Mersenne Twister random number engine

  // Define random number range: [0, TOKEN_ID_MAX]
  std::uniform_int_distribution<int> distribution(0, TOKEN_ID_MAX);

  // Fill the vector
  result.reserve(len);  // Pre-allocate memory for efficiency
  for (int i = 0; i < len; ++i) {
    result.push_back(distribution(generator));
  }

  return result;
}

/**
 * Compute argmax using Eigen tensor library
 * Finds the index of the element with the maximum value in the tensor
 * @param ptr Pointer to the beginning of the array/data
 * @param n Number of elements in the array
 * @return Index of the maximum value
 */
template <typename T>
static int eigen_argmax(const T* ptr, std::size_t n) {
  using Eigen::Tensor;
  using Eigen::TensorMap;

  // Create a tensor map from the raw pointer with specified size
  TensorMap<Tensor<const T, 1>> tm(static_cast<const T*>(ptr), n);

  // Compute the argmax operation to get the index of maximum value
  Eigen::Tensor<Eigen::Index, 0> t = tm.argmax();
  Eigen::Index idx = t(0);

  // Return the index as an integer
  return static_cast<int>(idx);
}

#ifdef XH2A_HM_SYS
static inline int GetDevMemInfo(std::map<int, hm_mem_info>& dev_mem_info) {
  hm_device_info dev_info = {0};
  int ret = hm_sys_get_device_info(&dev_info);
  if (ret <= 0 || dev_info.num_devices <= 0) {
    std::cerr << "Not found online devices, ret is " << ret << std::endl;
    return -1;
  }

  std::cout << "Online device num: " << dev_info.num_devices << std::endl;
  for (int i = 0; i < dev_info.num_devices; i++) {
    int device_id = dev_info.device_ids[i];
    dev_mem_info[device_id] = {0};
    ret = hm_sys_get_mem_info(device_id, &dev_mem_info[device_id]);
    if (ret != 0) {
      std::cerr << "Failed to get memory info of device " << device_id
                << ", ret is " << ret << std::endl;
      return ret;
    }
    auto mem_info = dev_mem_info[device_id];
    std::cout << "Online device id: " << device_id
              << ", mem_total: " << mem_info.mem_total
              << ", mem_used: " << mem_info.mem_used
              << ", mem_avail: " << mem_info.mem_avail << std::endl;
  }

  return ret;
}
#endif

class HmllmInferBase {
 public:
  HmllmInferBase() = default;
  virtual ~HmllmInferBase() = default;
  virtual PerfInfos perf_llm(const uint32_t input_tokens_len,
                             const uint32_t stop_tokens_len) = 0;
  virtual std::shared_ptr<InferencePerformanceTracker> get_perf_tracker() = 0;
};

#endif  // __UTILS_H__