/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: demo.cc
 * Description:
 *   Main application for Qwen3-VL C++ inference demo.
 *   Supports single-shot and interactive chat modes.
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

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "HmQwenVLInfer.h"
#include "SamplingManager.h"
#include "tcim/tcim_runtime.h"

const std::string DEFAULT_IMAGE_PROMPT = "描述图片内容";

// Helper function to trim whitespace
std::string trim(const std::string &str) {
  size_t start = str.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return "";
  size_t end = str.find_last_not_of(" \t\n\r");
  return str.substr(start, end - start + 1);
}

// Helper function to parse user input in interactive mode
// Format: "image_path1 image_path2 -- prompt text" or just "prompt text"
void ParseInteractiveInput(const std::string &user_input,
                           std::vector<std::string> &image_paths,
                           std::string &prompt) {
  image_paths.clear();

  // Find the separator "--"
  size_t sep_pos = user_input.find("--");

  if (sep_pos != std::string::npos) {
    // Has separator: parse image paths and prompt
    std::string media_part = trim(user_input.substr(0, sep_pos));
    prompt = trim(user_input.substr(sep_pos + 2));

    if (!media_part.empty()) {
      // Split by space for multiple images
      std::istringstream iss(media_part);
      std::string path;
      while (iss >> path) {
        image_paths.push_back(path);
      }
    }
  } else {
    // No separator: treat entire input as prompt (text-only)
    prompt = trim(user_input);
  }
}

void printUsage(const char *program_name) {
  std::cout << "Usage:" << std::endl;
  std::cout << "  " << program_name << " [options]" << std::endl;
  std::cout << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --image <paths...>        Paths to image files (can specify "
               "multiple paths after --image)"
            << std::endl;
  std::cout << "  --prompt <text>          User prompt text (default: "
               "\""
            << DEFAULT_IMAGE_PROMPT << "\")" << std::endl;
  std::cout << "  --visual <path>          Path to visual model (default: "
               "output/xh2/qwen3-vl-8b_visual_448x448x2.hmm)"
            << std::endl;
  std::cout << "  --prefill <path>         Path to prefill model (default: "
               "output/xh2/qwen3-vl-8b_prefill.hmm)"
            << std::endl;
  std::cout << "  --decode <path>          Path to decode model (default: "
               "output/xh2/qwen3-vl-8b_decode.hmm)"
            << std::endl;
  std::cout << "  --tokenizer <path>       Path to tokenizer.json (default: "
               "Qwen3-VL-8B-Instruct/tokenizer.json)"
            << std::endl;
  std::cout << "  --embedding <path>       Path to embedding weights (default: "
               "output/xh2/hmquant/quant_embedding.bin)"
            << std::endl;
  std::cout << "  --repetition-penalty <f> Repetition penalty (default: 1.0)\n"
            << std::endl;
  std::cout << "  --presence-penalty <f>   Presence penalty (default: 1.5)\n"
            << std::endl;
  std::cout
      << "  --temperature <f>        Sampling temperature (default: 1.0)\n"
      << std::endl;
  std::cout << "  --top-k <n>              Top-k sampling (default: 1) \n"
            << std::endl;
  std::cout << "  --top-p <f>              Top-p sampling (default: 1.0)"
            << std::endl;
  std::cout << "  --it                     Enable interactive chat mode"
            << std::endl;
  std::cout << "  --history                Keep chat history across messages"
            << std::endl;
  std::cout << "  --ngram                  Enable N-gram repetition blocking "
               "(default params: size=8, window=128, threshold=3)"
            << std::endl;
  std::cout << "  -h, --help               Show this help message" << std::endl;
  std::cout << std::endl;
  std::cout << "Interactive Mode Usage:" << std::endl;
  std::cout << "  - Pure text: just type your question" << std::endl;
  std::cout << "  - With image(s): image_path1 [image_path2 ...] -- prompt text"
            << std::endl;
  std::cout << "  - Exit: type 'stop', 'exit', 'quit', or press Ctrl+C"
            << std::endl;
}

int main(int argc, char *argv[]) {
  // Default paths
  std::string visual_model_path = "output/xh2/qwen3-vl-8b_visual_448x448x2.hmm";
  std::string prefill_model_path = "output/xh2/qwen3-vl-8b_prefill.hmm";
  std::string decode_model_path = "output/xh2/qwen3-vl-8b_decode.hmm";
  std::string tokenizer_path = "Qwen3-VL-8B-Instruct/tokenizer.json";
  std::string embedding_path = "output/xh2/hmquant/quant_embedding.bin";
  std::string prompt = DEFAULT_IMAGE_PROMPT;
  std::vector<std::string> image_paths = {"../../../data/pic/beach.jpeg"};
  float repetition_penalty = 1.0f;
  float temperature = 1.0f;
  int top_k = 1;
  float top_p = 1.0f;
  float presence_penalty = 1.5f;
  bool interactive_mode = false;
  bool keep_history = false;  // Default: do NOT keep history (same as Python)
  bool enable_ngram = false;  // N-gram repetition blocking

  // Parse command line arguments
  // Support both: --image path1 path2 ... (multiple paths after --image)
  //            or: --image path1 --image path2 (repeated --image)
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "--image") {
      // Clear default values when --image is explicitly provided
      if (i == 1 || (i > 1 && std::string(argv[i - 1]) != "--image")) {
        image_paths.clear();
      }
      // Collect all paths after --image until next option or end
      while (i + 1 < argc && argv[i + 1][0] != '-') {
        image_paths.push_back(argv[++i]);
      }
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
    } else if (arg == "--presence-penalty" && i + 1 < argc) {
      presence_penalty = std::stof(argv[++i]);
    } else if (arg == "--temperature" && i + 1 < argc) {
      temperature = std::stof(argv[++i]);
    } else if (arg == "--top-k" && i + 1 < argc) {
      top_k = std::stoi(argv[++i]);
    } else if (arg == "--top-p" && i + 1 < argc) {
      top_p = std::stof(argv[++i]);
    } else if (arg == "--it") {
      interactive_mode = true;
    } else if (arg == "--history") {
      keep_history = true;
    } else if (arg == "--ngram") {
      enable_ngram = true;
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
                                   repetition_penalty, presence_penalty);

  // Configure N-gram repetition blocking if enabled
  if (enable_ngram) {
    sampling_manager.setNoRepeatNgramSize(8);     // Block N-grams of size 8
    sampling_manager.setRepeatNgramSize(8);       // Detect repeats with N=8
    sampling_manager.setRepeatCountThreshold(3);  // Trigger after 3 repeats
    std::cout << "N-gram repetition blocking enabled: ngram_size=8, threshold=3"
              << std::endl;
  }

  try {
    // Initialize inference engine
    std::cout << "Initializing Qwen3-VL inference engine..." << std::endl;
    std::unique_ptr<HmQwenVLInfer> infer(new HmQwenVLInfer(
        visual_model_path, prefill_model_path, decode_model_path,
        tokenizer_path, embedding_path, sampling_manager));
    infer->SetKeepHistory(keep_history);
    infer->SetEnablePerfReport(!interactive_mode);

    // Main loop: interactive or single-shot mode
    while (true) {
      std::vector<std::string> current_image_paths;
      std::string current_prompt;

      if (interactive_mode) {
        // Interactive mode: read user input
        std::cout << "Input your instruction here (or image paths): ";
        std::string user_input;
        std::getline(std::cin, user_input);
        user_input = trim(user_input);

        // Check for exit commands
        std::string lower_input = user_input;
        std::transform(lower_input.begin(), lower_input.end(),
                       lower_input.begin(), ::tolower);
        if (lower_input == "stop" || lower_input == "exit" ||
            lower_input == "quit" || user_input.empty()) {
          std::cout << std::endl;
          std::cout << "程序结束" << std::endl;
          break;
        }

        // Parse input for images and prompt
        ParseInteractiveInput(user_input, current_image_paths, current_prompt);
      } else {
        // Single-shot mode: use command line arguments
        current_image_paths = image_paths;
        current_prompt = prompt;
      }

      // Print question (same format as Python)
      std::cout << "[SUCCESS] question:" << std::endl;
      std::cout << current_prompt << std::endl;

      try {
        // Run chat
        std::string response = infer->Chat(current_image_paths, current_prompt);

        // Reset color after response
        std::cout << std::endl;

        // In single-shot mode, exit after one iteration
        if (!interactive_mode) {
          break;
        }
      } catch (const std::exception &e) {
        std::cerr << "聊天过程中出错: " << e.what() << std::endl;
        std::cout << std::endl;
        if (!interactive_mode) {
          return -3;
        }
        // In interactive mode, continue to next iteration
        continue;
      }
    }

    // Cleanup
    infer.reset();

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -3;
  }

  std::cout << "Program finished successfully" << std::endl;
  return 0;
}
