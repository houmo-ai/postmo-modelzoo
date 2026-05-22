/*
 * Copyright (c) 2025 HOUMO AI
/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: qwen3.cc
 * Description:
 *   Main application file for Qwen3 inference - Implements the main function
for running Qwen3 model inference.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <codecvt>
#include <filesystem>
#include <locale>
#include <string>
#include <vector>

#include "HmQwenInfer.h"
#include "Hmtokenizer.h"
#include "tcim/tcim_dev_ctrl.h"
#include "tcim/tcim_runtime.h"
#ifdef _MSC_VER
#include <Windows.h>
#endif

int main(int argc, char *argv[]) {
#ifdef _MSC_VER
  SetConsoleOutputCP(
      CP_UTF8);           // Set console output code page to UTF-8 for Windows
  SetConsoleCP(CP_UTF8);  // Set console input code page to UTF-8 for Windows
#endif

  // Paths for model files, tokenizer, and embedding weights
  std::string prefillModelPath, decodeModelPath, tokenizerJsonPath,
      embeddingWeightPath;
  bool enablePowerDemo = false;
  std::vector<std::string> positionalArgs;

  // Handle command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "power_opt") {
      enablePowerDemo = true;
    } else {
      positionalArgs.push_back(arg);
    }
  }

  if (positionalArgs.empty()) {
    // Default paths if no arguments are provided
    prefillModelPath = "qwen3-8b_prefill.hmm";
    decodeModelPath = "qwen3-8b_decode.hmm";
    tokenizerJsonPath = "qwen3-8b/tokenizer.json";
    embeddingWeightPath = "hmquant/quant_embedding.bin";
  } else if (positionalArgs.size() == 4) {
    // Use paths provided via command line arguments
    prefillModelPath = positionalArgs[0];
    decodeModelPath = positionalArgs[1];
    tokenizerJsonPath = positionalArgs[2];
    embeddingWeightPath = positionalArgs[3];
  } else {
    // Print usage information if invalid number of arguments
    std::cerr << "Usage:\n"
              << "  <1> : ./${demo_name}\n"
              << "  <2> : ./${demo_name} [power_opt]\n"
              << "  <3> : ./${demo_name} <prefillModelPath> <decodeModelPath> "
                 "<tokenizerJsonPath> <embeddingWeightPath> [power_opt]\n"
              << "       power_opt can be placed at any position" << std::endl;
    return -1;
  }

  // Check if all required files exist
  if (!std::filesystem::exists(prefillModelPath) ||
      !std::filesystem::exists(decodeModelPath) ||
      !std::filesystem::exists(tokenizerJsonPath) ||
      !std::filesystem::exists(embeddingWeightPath)) {
    std::cerr << "Usage:\n"
              << "  <1> : ./${demo_name}\n"
              << "  <2> : ./${demo_name} [power_opt]\n"
              << "  <3> : ./${demo_name} <prefillModelPath> <decodeModelPath> "
                 "<tokenizerJsonPath> <embeddingWeightPath> [power_opt]\n"
              << "       power_opt can be placed at any position" << std::endl;
    std::cerr << "Please check that all files exist!" << std::endl;
    return -2;
  }

  // Check and validate the HOUMO_TARGET environment variable
  const char *houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target =
      (houmo_target_env != nullptr) ? std::string(houmo_target_env) : "houmo";

  // Only xh2 backend is supported
  if (houmo_target != "xh2") {
    std::cerr << "Unsupported backend: " << houmo_target << std::endl;
    exit(-1);
  } else {
    // Print backend and tcim version information
    std::cout << "Backend: " << houmo_target << std::endl;
    printf("tcim version: %s, houmo_target: %s.\n", tcim::GetVersion().c_str(),
           houmo_target.c_str());
  }

  // Initialize the Qwen3 inference engine
  std::unique_ptr<HmQwenInfer> qwen3Infer =
      std::make_unique<HmQwenInfer>(prefillModelPath, decodeModelPath,
                                    tokenizerJsonPath, embeddingWeightPath);
  qwen3Infer->Chat("请介绍一下存算一体技术的优势");

  if (enablePowerDemo) {
    auto dev = tcim::dev_ctrl::HalDeviceFactory::Create("Xh2aHalBackend");

    auto pre = dev->RegisterPreResetCallback(0, [&](int id) {
      if (qwen3Infer) {
        qwen3Infer.reset();
      }
    });
    auto post = dev->RegisterPostResetCallback(0, [&](int id) {
      qwen3Infer =
          std::make_unique<HmQwenInfer>(prefillModelPath, decodeModelPath,
                                        tokenizerJsonPath, embeddingWeightPath);
    });

    auto device_num = tcim::GetDeviceNum();

    for (auto i = 0; i < device_num; ++i) {
      uint64_t freq;
      float util_rate = 0.0f;
      struct tcim::dev_ctrl::MemInfo mem_info;
      memset(&mem_info, 0, sizeof(mem_info));

      if (dev->GetMemInfo(i, &mem_info) == tcim::Status::OK) {
        printf(
            "Device %d: Current DDR memory usage: total %u MB, used %u MB, "
            "available %u MB\n",
            i, mem_info.mem_total, mem_info.mem_used, mem_info.mem_avail);
      }
      if (dev->GetIpuUtilRate(i, &util_rate) == tcim::Status::OK) {
        printf("Device %d: Current IPU utilization rate: %.2f%%\n", i,
               util_rate);
      }
      if (dev->GetIpuFrequency(i, &freq) == tcim::Status::OK) {
        printf("Device %d: Current IPU frequency: %lu Hz\n", i, freq);
      }
    }

    tcim::dev_ctrl::DvfsMode mode;
    dev->GetDvfsMode(0, &mode);
    dev->SetDvfsMode(0, tcim::dev_ctrl::DvfsMode::kOnDemand);
    dev->IpuReset(0);  // （pre-callback → hw reset → post-callback）

    dev->UnregisterResetCallback(pre);
    dev->UnregisterResetCallback(post);
  }

  // Clean up the inference engine
  qwen3Infer.reset();

  return 0;
}