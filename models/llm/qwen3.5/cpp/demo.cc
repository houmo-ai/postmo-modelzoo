/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: demo.cc
 * Description:
 *   Qwen3.5 MLLM inference demo using Houmo framework.
 *
 *   Feature showcase:
 *     1. Basic inference - StreamingDecoder streaming output
 *     2. Image understanding - Qwen35MLLMContext image processing
 *     3. Advanced parameters - max tokens and model info
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

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "modules/streaming_decoder.h"
#include "qwen35_mllm_model.h"

namespace fs = std::filesystem;

// ============================================================================
// Helper functions
// ============================================================================

static void printUsage(const char* program_name) {
  std::cout << "Qwen3.5 MLLM C++ Inference Demo\n\n";
  std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
  std::cout << "Options:\n";
  std::cout
      << "  --prefill <path>      Path to prefill model (.hmm) [required]\n";
  std::cout
      << "  --decode <path>       Path to decode model (.hmm) [required]\n";
  std::cout << "  --visual <path>       Path to visual model (.hmm), required "
               "for image input\n";
  std::cout
      << "  --embedding <path>    Path to embedding file (.bin) [required]\n";
  std::cout << "  --tokenizer <path>    Path to tokenizer json or tokenizer "
               "directory [required]\n";
  std::cout << "  --prompt <text>       Prompt text\n";
  std::cout << "  --image <path>        Image path, can be specified multiple "
               "times\n";
  std::cout
      << "  --max-tokens <num>    Maximum generated tokens, default 256\n";
  std::cout << "  --device <id>         Device id, default 0\n";
  std::cout << "  -h, --help            Show this help message\n\n";
  std::cout << "Examples:\n";
  std::cout << "  # Text inference\n";
  std::cout << "  " << program_name
            << " --prefill model_prefill.hmm --decode model_decode.hmm "
               "--embedding quant_embedding.bin --tokenizer tokenizer.json "
               "--prompt \"Introduce yourself\"\n\n";
  std::cout << "  # Image understanding\n";
  std::cout << "  " << program_name
            << " --prefill model_prefill.hmm --decode model_decode.hmm "
               "--visual model_visual.hmm --embedding quant_embedding.bin "
               "--tokenizer tokenizer.json --image test.jpg "
               "--prompt \"Describe this image\"\n";
}

static void printSeparator(const char* title) {
  std::cout << "\n========== " << title << " ==========\n";
}

static bool requireFile(const std::string& flag, const std::string& path) {
  if (path.empty() || !fs::exists(path)) {
    std::cerr << "Error: " << flag << " path does not exist: " << path << "\n";
    return false;
  }
  return true;
}

struct Message {
  std::string role;
  std::string content;
};

static std::string ApplyChatTemplate(const std::vector<Message>& msgs,
                                     bool add_generation_prompt,
                                     bool enable_thinking) {
  std::string out;
  out.reserve(1024);

  for (const auto& m : msgs) {
    out.append("<|im_start|>");
    out.append(m.role);
    out.push_back('\n');
    out.append(m.content);
    out.append("<|im_end|>\n");
  }

  if (add_generation_prompt) {
    out.append("<|im_start|>assistant\n");
  }

  if (!enable_thinking) {
    out.append("<think>\n\n</think>\n\n");
  }

  return out;
}

static void printModelInfo(houmo::Qwen35MLLMModel& model) {
  printSeparator("Model Information");

  auto info = model.model_info();
  std::cout << "Model Name:       " << info.model_name << "\n";
  std::cout << "Model Type:       "
            << (info.type == houmo::ModelType::VLM ? "VLM" : "LLM") << "\n";
  std::cout << "n_batch:          " << info.n_batch << "\n";
  std::cout << "n_vocab:          " << info.n_vocab << "\n";
  std::cout << "n_embd:           " << info.n_embd << "\n";
  std::cout << "n_layer:          " << info.n_layer << "\n";
  std::cout << "n_ctx:            " << info.n_ctx << "\n";
  std::cout << "prefill_length:   " << info.prefill_length << "\n";
  std::cout << "kv_cache_layers:  " << info.kv_cache_layers << "\n";
  std::cout << "max_ctx_available:" << model.max_ctx_available() << "\n";
  std::cout << "vocab_size:       " << model.vocab_size() << "\n";
  std::cout << "embedding_dim:    " << model.embedding_dim() << "\n";
}

static std::string BuildImagePromptPrefix(size_t image_count) {
  std::string content;
  for (size_t i = 0; i < image_count; ++i) {
    content += "<|vision_start|><|image_pad|><|vision_end|>";
  }
  return content;
}

static void printGenerationMetrics(houmo::Context& ctx) {
  std::cout << "\n";
  ctx.profiler().print_summary();

  const auto stats = ctx.perf_stats();
  std::cout << "\n=== Generation Metrics ===\n"
            << "Input Tokens:         " << stats.n_input_tokens << "\n"
            << "Output Tokens:        " << stats.n_output_tokens << "\n"
            << "Prefill Time:         " << stats.prefill_time_ms << " ms\n"
            << "Decode Time:          " << stats.decode_time_ms << " ms\n"
            << "Total Time:           " << stats.total_time_ms << " ms\n"
            << "TTFT:                 " << stats.ttft_ms << " ms\n"
            << "TPS:                  " << stats.tps << " tok/s\n";
}

// ============================================================================
// Example features
// ============================================================================

static void exampleBasicInference(houmo::Qwen35MLLMModel& model,
                                  const std::string& prompt,
                                  const houmo::SamplingParams& params,
                                  bool enable_thinking) {
  printSeparator("Basic Inference");

  auto ctx = model.create_context();
  std::vector<Message> msgs = {{"user", prompt}};
  std::string rendered = ApplyChatTemplate(msgs, true, enable_thinking);
  auto tokens = model.tokenize(rendered, false, false);

  std::cout << "Prompt: " << prompt << "\n";
  std::cout << "Output: ";

  houmo::StreamingDecoder decoder(model.tokenizer());
  ctx->generate(tokens, params, [&decoder](houmo::Token token) {
    std::string chunk = decoder.decode(token);
    if (!chunk.empty()) {
      std::cout << chunk << std::flush;
    }
    return true;
  });
  std::cout << "\n";

  printGenerationMetrics(*ctx);
}

static void exampleImageUnderstanding(houmo::Qwen35MLLMModel& model,
                                      const std::vector<std::string>& images,
                                      const std::string& prompt,
                                      const houmo::SamplingParams& params,
                                      bool enable_thinking) {
  printSeparator("Image Understanding");

  auto ctx = model.create_context();
  auto* qwen_ctx = dynamic_cast<houmo::Qwen35MLLMContext*>(ctx.get());
  if (!qwen_ctx) {
    std::cerr << "Error: failed to create Qwen35MLLMContext\n";
    return;
  }

  qwen_ctx->set_images(images);
  std::cout << "Loaded " << images.size() << " images (Qwen35MLLM)\n";

  std::string content = BuildImagePromptPrefix(images.size()) + prompt;
  std::vector<Message> msgs = {{"user", content}};
  std::string rendered = ApplyChatTemplate(msgs, true, enable_thinking);
  auto tokens = model.tokenize(rendered, false, false);

  std::cout << "Prompt: " << prompt << "\n";
  std::cout << "Output: ";

  houmo::StreamingDecoder decoder(model.tokenizer());
  qwen_ctx->generate(tokens, params, [&decoder](houmo::Token token) {
    std::string chunk = decoder.decode(token);
    if (!chunk.empty()) {
      std::cout << chunk << std::flush;
    }
    return true;
  });
  std::cout << "\n";

  printGenerationMetrics(*qwen_ctx);
}

// ============================================================================
// Main function
// ============================================================================

int main(int argc, char* argv[]) {
  std::string prefill_path;
  std::string decode_path;
  std::string visual_path;
  std::string embedding_path;
  std::string tokenizer_path;
  std::string prompt = "你好，请介绍一下自己。";
  std::vector<std::string> image_paths;
  int max_tokens = 0;
  int device_id = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "--prefill" && i + 1 < argc) {
      prefill_path = argv[++i];
    } else if (arg == "--decode" && i + 1 < argc) {
      decode_path = argv[++i];
    } else if (arg == "--visual" && i + 1 < argc) {
      visual_path = argv[++i];
    } else if (arg == "--embedding" && i + 1 < argc) {
      embedding_path = argv[++i];
    } else if (arg == "--tokenizer" && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--image" && i + 1 < argc) {
      image_paths.emplace_back(argv[++i]);
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--device" && i + 1 < argc) {
      device_id = std::stoi(argv[++i]);
    } else {
      std::cerr << "Unknown or incomplete option: " << arg << "\n\n";
      printUsage(argv[0]);
      return 1;
    }
  }

  if (!requireFile("--prefill", prefill_path) ||
      !requireFile("--decode", decode_path) ||
      !requireFile("--embedding", embedding_path) ||
      !requireFile("--tokenizer", tokenizer_path)) {
    printUsage(argv[0]);
    return 1;
  }

  for (const auto& image_path : image_paths) {
    if (!requireFile("--image", image_path)) {
      return 1;
    }
  }

  if (!image_paths.empty() && !requireFile("--visual", visual_path)) {
    printUsage(argv[0]);
    return 1;
  }

  houmo::ModelConfig config;
  config.devices = {device_id};
  config.prefill_path = prefill_path;
  config.decode_path = decode_path;
  config.vision_path = visual_path;
  config.embedding_path = embedding_path;
  config.tokenizer_path = tokenizer_path;

  try {
    std::cout << "========================================\n";
    std::cout << "  Qwen3.5 MLLM C++ Inference Demo\n";
    std::cout << "  Houmo Inference Framework v" << houmo::version() << "\n";
    std::cout << "========================================\n\n";

    std::cout << "Loading Qwen3.5 MLLM model...\n";
    houmo::Qwen35MLLMModel model(config);
    std::cout << "Model loaded successfully!\n";

    printModelInfo(model);

    houmo::SamplingParams params;
    params.max_tokens = max_tokens;

    bool enable_thinking = false;
    if (!image_paths.empty()) {
      exampleImageUnderstanding(model, image_paths, prompt, params,
                                enable_thinking);
    } else {
      exampleBasicInference(model, prompt, params, enable_thinking);
    }

    std::cout << "\nDone!\n";
  } catch (const houmo::Exception& e) {
    std::cerr << "Houmo Error: " << e.what() << "\n";
    return 3;
  } catch (const std::exception& e) {
    std::cerr << "Standard Error: " << e.what() << "\n";
    return 3;
  }

  return 0;
}
