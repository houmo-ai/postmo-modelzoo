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
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "perf_tracker/inference_perf_tracker.h"

#define TOKEN_ID_MAX 150000

#define COLOR_RED "\x1b[91;20m"
#define COLOR_GREEN "\x1b[92;20m"
#define COLOR_YELLOW "\x1b[93;20m"
#define COLOR_BLUE "\x1b[94;20m"
#define COLOR_MAGENT "\x1b[95;20m"
#define COLOR_CYAN "\x1b[96;20m"
#define COLOR_RESET "\x1b[0m"

namespace fs = std::filesystem;

typedef enum { PERFCMD = 0, PERFYAML, PERFINVAILD } PerfConfigType;

static void HelpUsage(char* argv[]) {
  std::cout << "llm_perf - A tool for LLM and VLM performance tests with "
               "flexible configuration options.\n\n";
  std::cout << "Release Time : " << __DATE__ << " " << __TIME__ << "\n\n";
  std::cout
      << "Usage: " << argv[0]
      << " --key value [options...]\n\n"
         "Options:\n"
         "  -c, --config      FILE      use yaml file to start llm_perf, "
         "cat template perf_config.yaml for more message\n"
         "Or:\n"
         "  --prefill         FILE      Prefill model file (required).\n"
         "  --decode          FILE      Decode model file (required).\n"
         "  --visual          FILE      Visual model file (optional, only "
         "for VLM).\n"
         "  --embedding       FILE      Embedding weight file (.bin, "
         "required).\n"
         "  --input           NUM[,NUM...] Number of input tokens, supports "
         "comma-separated groups (range: 1-max_context_length).\n"
         "  --output          NUM[,NUM...] Number of tokens to generate, "
         "supports comma-separated groups and must match input group count "
         "(range: 1-(max_context_length-input)).\n"
         "  --model_name      TEXT      Model name used in dump/log output "
         "(optional, command-line only).\n"
         "  --devices         NUM[,NUM...]      Device ids for init "
         "dev_manager .\n"
         "  --loop            NUM       Loop test rounds (range: 1-1000000).\n"
         "  --batch           NUM       Batch size (range: 1-batch_num, only "
         "xh2 "
         "supported for multi-batch).\n"
         "  --no_warm_up                Disable warm-up (flag, no value "
         "required).\n"
         "  --warm_up_input   NUM       set warm_up input tokens when "
         "warm_up enabled(optional, if not set, default is equal to input).\n"
         "  --warm_up_output  NUM       set warm_up decode times when "
         "warm_up enabled(optional, if not set, default is equal to output).\n"
         "  --LazyMode                  Enable lazy mode (flag, no value "
         "required).\n"
         "  --interval        NUM       Sampling interval in milliseconds for "
         "device/host monitoring (range: 100-60000).\n"
         "  --skip_perf                 skip prefill and decode performance "
         "test\n"
         "  --dump_file       FILE      Dump perf result to file (optional).\n"
         "  -h, --help                  Show this help message.\n\n"
         "Examples:\n"
         "  "
      << argv[0]
      << " --prefill prefill.hmm --decode decode.hmm --embedding embed.bin "
         "--input 256 --output 100\n"
         "  "
      << argv[0]
      << " --model_name qwen3_8b --prefill prefill.hmm --decode decode.hmm "
         "--embedding embed.bin "
         "--input 256,512 --output 100,200\n"
         "  "
      << argv[0] << " -c perf_config.yaml\n";
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
      return PerfConfigType::PERFYAML;
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
  std::set<std::string> flags = {"no_warm_up", "LazyMode", "skip_perf"};

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
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

static std::vector<int> validate_multi_setting(
    std::unordered_map<std::string, std::string>& args,
    const std::string& arg_name) {
  if (args.find(arg_name) == args.end()) {
    throw std::invalid_argument("Missing arg : " + arg_name + ", (use --" +
                                arg_name + " to set arg).");
  }

  if (args[arg_name].empty()) {
    throw std::invalid_argument("Missing " + arg_name + " value (use --" +
                                arg_name + " <value>).");
  }

  std::vector<int> values;
  std::stringstream ss(args[arg_name]);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(std::remove_if(item.begin(), item.end(),
                              [](unsigned char c) { return std::isspace(c); }),
               item.end());
    if (item.empty()) {
      throw std::invalid_argument(
          "Invalid " + arg_name +
          " value, empty item in comma-separated list.");
    }
    int value = stoi(item);
    if ((arg_name == "devices" && value < 0) ||
        (arg_name != "devices" && value <= 0)) {
      throw std::invalid_argument("Invalid " + arg_name + " value (use --" +
                                  arg_name + " <value> to set valid value).");
    }
    values.push_back(value);
  }

  if (values.empty()) {
    throw std::invalid_argument("Invalid " + arg_name +
                                " value, no valid item found.");
  }

  return values;
}

static std::string format_int_list(const std::vector<int>& values) {
  std::ostringstream oss;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << values[i];
  }
  return oss.str();
}

struct PerfInfos {
  uint32_t input_tokens;
  uint32_t stop_tokens;
  uint32_t decode_count;
};

/**
 * Generate a vector of random integers within a specified length
 * Used for creating random token IDs for testing purposes
 * @param len Length of the vector to generate
 * @param max_value Maximum value for the random integers (exclusive)
 * @return Vector of random integers in the range [0, max_value)
 */
static std::vector<int> generateRandomVector(
    int len, const int max_value = TOKEN_ID_MAX) {
  std::vector<int> result;
  if (len <= 0) {
    return result;  // Handle invalid length (return empty vector)
  }

  // Use current time as random seed to ensure different sequences each run
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed);  // Use Mersenne Twister random number engine

  // Define random number range: [0, max_value)
  std::uniform_int_distribution<int> distribution(0, max_value - 1);

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

class HmllmInferBase {
 public:
  HmllmInferBase() = default;
  virtual ~HmllmInferBase() = default;
  virtual void perf_llm(const uint32_t input_tokens_len,
                        const uint32_t stop_tokens_len) = 0;
  virtual std::shared_ptr<InferencePerformanceTracker> get_perf_tracker() = 0;
};

typedef struct perf_settings {
  struct PerfCase {
    int input_tokens_len;
    int stop_tokens_len;
  };

  std::string model_name;
  std::string prefill_path;
  std::string decode_path;
  std::string visual_path;
  std::string embedding_path;
  std::vector<PerfCase> perf_cases;
  int input_tokens_len;
  int stop_tokens_len;
  std::vector<int> devices;
  int batch_size;
  int loop_count;
  bool warm_up;
  uint32_t warm_up_input;
  uint32_t warm_up_output;
  bool LazyMode;
  bool skip_perf;
  uint32_t interval_ms;
  int perf_case_index = 1;
  int perf_case_total = 1;
} PerfSettings;

// host memory struct
struct HostMemoryInfo {
  size_t virtual_memory;   // virtual_memory (bytes)
  size_t physical_memory;  // physical_memory (bytes)
};
inline std::string format_double(double value, int precision = 2) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);
  oss << value;
  return oss.str();
}

struct DeviceCtcInfo {
  int dev_id;    // device ID
  int group_id;  // group ID
  int chip_id;   // chip ID
};
#endif  // __UTILS_H__
