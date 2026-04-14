/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: demo.cc
 * Description:
 *   Main application for Qwen3-VL C++ inference demo.
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

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "HmQwenVLInfer.h"
#include "SamplingManager.h"
#include "tcim/tcim_runtime.h"

#ifdef _MSC_VER
#include <Windows.h>
#endif

void printUsage(const char *program_name) {
#ifdef _MSC_VER
  SetConsoleOutputCP(
      CP_UTF8);           // Set console output code page to UTF-8 for Windows
  SetConsoleCP(CP_UTF8);  // Set console input code page to UTF-8 for Windows
#endif
  std::cout << "Usage:" << std::endl;
  std::cout << "  " << program_name << " [options]" << std::endl;
  std::cout << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --image <path>           Path to image file (can be "
               "specified multiple times)"
            << std::endl;
  std::cout << "  --prompt <text>          User prompt text (default: "
               "\"请描述图片内容。\")"
            << std::endl;
  std::cout << "  --visual <path>          Path to visual model (default: "
               "output/xh2/qwen3-vl_visual.hmm)"
            << std::endl;
  std::cout << "  --prefill <path>         Path to prefill model (default: "
               "output/xh2/qwen3-vl_prefill.hmm)"
            << std::endl;
  std::cout << "  --decode <path>          Path to decode model (default: "
               "output/xh2/qwen3-vl_decode.hmm)"
            << std::endl;
  std::cout << "  --tokenizer <path>       Path to tokenizer.json (default: "
               "qwen3-vl/tokenizer.json)"
            << std::endl;
  std::cout << "  --embedding <path>       Path to embedding weights (default: "
               "output/xh2/hmquant/quant_embedding.bin)"
            << std::endl;
  std::cout << "  --repetition-penalty <f> Repetition penalty (default: 1.5)"
            << std::endl;
  std::cout << "  --temperature <f>        Sampling temperature (default: 1.0)"
            << std::endl;
  std::cout
      << "  --top-k <n>              Top-k sampling (default: -1, disabled)"
      << std::endl;
  std::cout << "  --top-p <f>              Top-p sampling (default: 1.0)"
            << std::endl;
  std::cout << "  -h, --help               Show this help message" << std::endl;
}

int main(int argc, char *argv[]) {
#ifdef _MSC_VER
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif

  // Default paths
  std::string visual_model_path = "output/xh2/qwen3-vl_visual.hmm";
  std::string prefill_model_path = "output/xh2/qwen3-vl_prefill.hmm";
  std::string decode_model_path = "output/xh2/qwen3-vl_decode.hmm";
  std::string tokenizer_path = "qwen3-vl/tokenizer.json";
  std::string embedding_path = "output/xh2/hmquant/quant_embedding.bin";
  std::string prompt = "请分析输入内容并简洁作答。";
  std::vector<std::string> image_paths;
  float repetition_penalty = 1.5f;
  float temperature = 1.0f;
  int top_k = -1;
  float top_p = 1.0f;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "--image" && i + 1 < argc) {
      image_paths.push_back(argv[++i]);
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--visual" && i + 1 < argc) {
      visual_model_path = argv[++i];
    } else if (arg == "--prefill" && i + 1 < argc) {
      prefill_model_path = argv[++i];
    } else if (arg == "--decode" && i + 1 < argc) {
      decode_model_path = argv[++i];
    } else if (arg == "--tokenizer" && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (arg == "--embedding" && i + 1 < argc) {
      embedding_path = argv[++i];
    } else if (arg == "--repetition-penalty" && i + 1 < argc) {
      repetition_penalty = std::stof(argv[++i]);
    } else if (arg == "--temperature" && i + 1 < argc) {
      temperature = std::stof(argv[++i]);
    } else if (arg == "--top-k" && i + 1 < argc) {
      top_k = std::stoi(argv[++i]);
    } else if (arg == "--top-p" && i + 1 < argc) {
      top_p = std::stof(argv[++i]);
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      printUsage(argv[0]);
      return -1;
    }
  }

  // Check environment
  const char *houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target =
      (houmo_target_env != nullptr) ? std::string(houmo_target_env) : "xh2";

  if (houmo_target != "xh2") {
    std::cerr << "Unsupported backend: " << houmo_target << std::endl;
    return -1;
  }

  std::cout << "Backend: " << houmo_target << std::endl;
  std::cout << "TCIM version: " << tcim::GetVersion() << std::endl;

  // Check if model files exist
  if (!std::filesystem::exists(visual_model_path)) {
    std::cerr << "Visual model not found: " << visual_model_path << std::endl;
    return -2;
  }
  if (!std::filesystem::exists(prefill_model_path)) {
    std::cerr << "Prefill model not found: " << prefill_model_path << std::endl;
    return -2;
  }
  if (!std::filesystem::exists(decode_model_path)) {
    std::cerr << "Decode model not found: " << decode_model_path << std::endl;
    return -2;
  }
  if (!std::filesystem::exists(tokenizer_path)) {
    std::cerr << "Tokenizer not found: " << tokenizer_path << std::endl;
    return -2;
  }
  if (!std::filesystem::exists(embedding_path)) {
    std::cerr << "Embedding weights not found: " << embedding_path << std::endl;
    return -2;
  }

  // Create sampling manager
  SamplingManager sampling_manager(temperature, top_k, top_p,
                                   repetition_penalty);

  try {
    // Initialize inference engine
    std::cout << "Initializing Qwen3-VL inference engine..." << std::endl;
    std::unique_ptr<HmQwenVLInfer> infer(new HmQwenVLInfer(
        visual_model_path, prefill_model_path, decode_model_path,
        tokenizer_path, embedding_path, sampling_manager));

    // Run chat
    std::string response = infer->Chat(image_paths, prompt);

    // Cleanup
    infer.reset();

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -3;
  }

  return 0;
}
