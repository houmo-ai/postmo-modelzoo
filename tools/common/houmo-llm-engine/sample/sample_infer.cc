/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: sample_infer.cc
 * Description:
 *   Unified inference example - supports all LLM/VLM models
 *
 *   Uses ModelFactory to automatically detect model type and create instances.
 *
 *   Feature showcase:
 *     1. Basic inference - StreamingDecoder streaming output
 *     2. Multi-turn dialogue - Context history retention
 *     3. Image understanding - VLM image processing
 *     4. Advanced parameters - sampling parameters, model info
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

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "core/model_factory.h"
#include "models/qwen35_mllm_model.h"
#include "models/qwen3_vlm_model.h"
#include "modules/streaming_decoder.h"

namespace fs = std::filesystem;

// ============================================================================
// Helper functions
// ============================================================================

static void printUsage(const char* program_name) {
  std::cout << "Houmo Inference Framework - 统一推理示例\n\n";
  std::cout << "用法: " << program_name << " [选项]\n\n";
  std::cout << "选项:\n";
  std::cout << "  --model <series>      模型系列: qwen3_llm, qwen35_mllm, "
               "qwen3_vlm\n";
  std::cout << "  --prompt <text>       用户提示词 (默认: \"你好\")\n";
  std::cout << "  --image <path>        图片路径 (VLM 模型，可多次指定)\n";
  std::cout << "  --multi-turn          启用多轮对话模式\n";
  std::cout << "  --max-tokens <n>      最大生成 token 数 (默认: 256)\n";
  std::cout << "  --temperature <f>     采样温度 (默认: 1.0)\n";
  std::cout << "  --top-k <n>           Top-k 采样 (默认: 1, greedy)\n";
  std::cout << "  --prefill <path>      Prefill 模型路径\n";
  std::cout << "  --decode <path>       Decode 模型路径\n";
  std::cout << "  --vision <path>       Vision 模型路径 (VLM)\n";
  std::cout << "  --tokenizer <path>    Tokenizer 路径\n";
  std::cout << "  --embedding <path>    Embedding 路径\n";
  std::cout << "  --info                显示模型信息后退出\n";
  std::cout << "  -h, --help            显示帮助信息\n";
  std::cout << "\n";
  std::cout << "示例:\n";
  std::cout << "  # LLM 推理 (自动检测)\n";
  std::cout << "  " << program_name << " --prompt \"介绍一下你自己\"\n";
  std::cout << "\n";
  std::cout << "  # VLM 图像理解\n";
  std::cout
      << "  " << program_name
      << " --model qwen35_mllm --image test.jpg --prompt \"描述这张图片\"\n";
  std::cout << "\n";
  std::cout << "  # 多轮对话\n";
  std::cout << "  " << program_name << " --multi-turn --prompt \"你好\"\n";
}

static void printSeparator(const char* title) {
  std::cout << "\n========== " << title << " ==========\n";
}

// Get base path (from environment variable or current directory)
static std::string getBasePath() {
  const char* env_path = std::getenv("HM_ENGINE_PATH");
  if (env_path && env_path[0] != '\0') {
    return std::string(env_path);
  }
  return "..";
}

// Message struct
struct Message {
  std::string role;
  std::string content;
};

// Build Chat Template (aligned with sample_qwen35_llm.cc)
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

  // Add special token when thinking mode is disabled
  if (!enable_thinking) {
    out.append("<think>\n\n</think>\n\n");
  }

  return out;
}

// ============================================================================
// Example features
// ============================================================================

// Basic inference
static void exampleBasicInference(houmo::LLMModel& model,
                                  const std::string& prompt,
                                  const houmo::SamplingParams& params,
                                  bool enable_thinking) {
  printSeparator("基本推理");

  auto ctx = model.create_context();

  std::vector<Message> msgs = {{"user", prompt}};
  std::string rendered = ApplyChatTemplate(msgs, true, enable_thinking);
  auto tokens = model.tokenize(rendered, false, false);

  std::cout << "提示: " << prompt << "\n";
  std::cout << "输出: ";

  houmo::StreamingDecoder decoder(model.tokenizer());
  ctx->generate(tokens, params, [&decoder](houmo::Token token) {
    std::string chunk = decoder.decode(token);
    if (!chunk.empty()) {
      std::cout << chunk << std::flush;
    }
    return true;
  });
  std::cout << "\n";

  // Output performance statistics
  ctx->profiler().print_summary();
}

// Multi-turn dialogue
static void exampleMultiTurn(houmo::LLMModel& model,
                             const std::string& initial_prompt,
                             const houmo::SamplingParams& params,
                             bool enable_thinking) {
  printSeparator("多轮对话");

  auto ctx = model.create_context();
  houmo::StreamingDecoder decoder(model.tokenizer());
  ctx->set_keep_history(true);  // Keep history context
  ctx->reset();                 // Ensure context state is cleared

  // Round 1
  std::vector<std::string> images = {"tests/data/a.png"};
  std::string content;
  for (size_t i = 0; i < images.size(); i++) {
    ctx->set_image(images[i]);
    content += "<|vision_start|><|image_pad|><|vision_end|>";
  }
  content += initial_prompt;
  std::vector<Message> msgs1 = {{"user", content}};
  std::string prompt1 = ApplyChatTemplate(msgs1, true, enable_thinking);
  auto tokens1 = model.tokenize(prompt1, false, false);

  std::cout << "用户: " << initial_prompt << "\n";
  std::cout << "助手: ";

  std::string response1;
  ctx->generate(tokens1, params, [&](houmo::Token token) {
    std::string chunk = decoder.decode(token);
    if (!chunk.empty()) {
      std::cout << chunk << std::flush;
      response1 += chunk;
    }
    return true;
  });
  std::cout << "\n";
  std::cout << "  [Context length: " << ctx->context_length() << "]\n";

  // Round 1 performance statistics
  std::cout << "\n--- Round 1 Performance ---\n";
  ctx->profiler().print_summary();

  // Round 2 - continue dialogue
  std::string follow_up = "那 2 + 2 等于多少？";
  std::vector<Message> msgs2 = {{"user", follow_up}};
  std::string prompt2 = ApplyChatTemplate(msgs2, true, enable_thinking);
  auto tokens2 = model.tokenize(prompt2, false, false);

  std::cout << "\n用户: " << follow_up << "\n";
  std::cout << "助手: ";

  decoder.reset();
  ctx->generate(tokens2, params, [&](houmo::Token token) {
    std::string chunk = decoder.decode(token);
    if (!chunk.empty()) {
      std::cout << chunk << std::flush;
    }
    return true;
  });
  std::cout << "\n";
  std::cout << "  [Context length: " << ctx->context_length() << "]\n";

  // Round 2 performance statistics
  std::cout << "\n--- Round 2 Performance ---\n";
  ctx->profiler().print_summary();

  // Round 3 - dialogue again
  std::string follow_up3 = "先介绍下图片内容，然后详细总结下历史对话";

  std::vector<std::string> images3 = {"tests/data/b.jpg"};
  std::string content3;
  for (size_t i = 0; i < images3.size(); i++) {
    ctx->set_image(images3[i]);
    content3 += "<|vision_start|><|image_pad|><|vision_end|>";
  }
  content3 += follow_up3;
  std::vector<Message> msgs3 = {{"user", content3}};
  std::string prompt3 = ApplyChatTemplate(msgs3, true, enable_thinking);
  auto tokens3 = model.tokenize(prompt3, false, false);

  decoder.reset();
  ctx->generate(tokens3, params, [&](houmo::Token token) {
    std::string chunk = decoder.decode(token);
    if (!chunk.empty()) {
      std::cout << chunk << std::flush;
    }
    return true;
  });
  std::cout << "\n";
  std::cout << "  [Context length: " << ctx->context_length() << "]\n";

  // Round 3 performance statistics
  std::cout << "\n--- Round 3 Performance ---\n";
  ctx->profiler().print_summary();
}

// Image understanding (VLM)
static void exampleImageUnderstanding(houmo::LLMModel& model,
                                      const std::vector<std::string>& images,
                                      const std::string& prompt,
                                      const houmo::SamplingParams& params,
                                      bool enable_thinking) {
  printSeparator("图像理解");

  // Check if VLM is supported
  if (model.type() != houmo::ModelType::VLM) {
    std::cerr << "错误: 当前模型不支持图像输入\n";
    return;
  }

  auto ctx = model.create_context();

  // Try Qwen35MLLM
  auto* qwen35_ctx = dynamic_cast<houmo::Qwen35MLLMContext*>(ctx.get());
  if (qwen35_ctx) {
    for (const auto& img : images) {
      ctx->set_image(img);
    }
    std::cout << "已加载 " << images.size() << " 张图片 (Qwen35MLLM)\n";
  }

  // Try Qwen3VLM
  auto* qwen3vl_ctx = dynamic_cast<houmo::Qwen3VLMContext*>(ctx.get());
  if (qwen3vl_ctx) {
    qwen3vl_ctx->set_images(images);
    std::cout << "已加载 " << images.size() << " 张图片 (Qwen3VLM)\n";
  }

  if (!qwen35_ctx && !qwen3vl_ctx) {
    std::cerr << "警告: 无法识别的 VLM Context 类型\n";
    return;
  }

  // Build VLM prompt (with image placeholders)
  std::string content;
  for (size_t i = 0; i < images.size(); i++) {
    content += "<|vision_start|><|image_pad|><|vision_end|>";
  }
  content += prompt;

  std::vector<Message> msgs = {{"user", content}};
  std::string rendered = ApplyChatTemplate(msgs, true, enable_thinking);
  auto tokens = model.tokenize(rendered, false, false);

  std::cout << "提示: " << prompt << "\n";
  std::cout << "输出: ";

  houmo::StreamingDecoder decoder(model.tokenizer());
  ctx->generate(tokens, params, [&decoder](houmo::Token token) {
    std::string chunk = decoder.decode(token);
    if (!chunk.empty()) {
      std::cout << chunk << std::flush;
    }
    return true;
  });
  std::cout << "\n";

  // Output performance statistics
  ctx->profiler().print_summary();
}

// Model information
static void printModelInfo(houmo::LLMModel& model) {
  printSeparator("模型信息");

  auto info = model.model_info();
  std::cout << "模型名称:     " << info.model_name << "\n";
  std::cout << "模型类型:     "
            << (info.type == houmo::ModelType::LLM ? "LLM" : "VLM") << "\n";
  std::cout << "----------------------------------------\n";
  std::cout << "n_batch:       " << info.n_batch << "\n";
  std::cout << "n_vocab:       " << info.n_vocab << "\n";
  std::cout << "n_embd:        " << info.n_embd << "\n";
  std::cout << "n_layer:       " << info.n_layer << "\n";
  std::cout << "n_ctx:         " << info.n_ctx << "\n";
  std::cout << "prefill_length: " << info.prefill_length << "\n";
  std::cout << "kv_cache_layers: " << info.kv_cache_layers << "\n";
  std::cout << "----------------------------------------\n";
  std::cout << "运行时信息:\n";
  std::cout << "max_ctx_available: " << model.max_ctx_available() << "\n";
  std::cout << "vocab_size:        " << model.vocab_size() << "\n";
  std::cout << "embedding_dim:     " << model.embedding_dim() << "\n";

  // Show registered model types
  auto types = houmo::ModelFactory<houmo::LLMModel>::ListRegisteredTypes();
  std::cout << "----------------------------------------\n";
  std::cout << "已注册模型类型: ";
  for (const auto& t : types) {
    std::cout << t << " ";
  }
  std::cout << "\n";
}

// ============================================================================
// Main function
// ============================================================================

int main(int argc, char* argv[]) {
  std::string base_path = getBasePath();

  // Default parameters
  std::string model_series = "auto";  // auto, qwen3_llm, qwen35_mllm, qwen3_vlm
  std::string prompt = "你好，介绍下图片中的内容。";
  std::vector<std::string> images;
  int max_tokens = 0;
  float temperature = 1.0f;
  int top_k = 1;
  bool multi_turn = false;
  bool show_info = false;

  // Model paths (optional, defaults to environment variable)
  std::string prefill_path;
  std::string decode_path;
  std::string vision_path;
  std::string tokenizer_path;
  std::string embedding_path;

  // Parse command-line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "--model" && i + 1 < argc) {
      model_series = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--image" && i + 1 < argc) {
      images.push_back(argv[++i]);
    } else if (arg == "--multi-turn") {
      multi_turn = true;
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--temperature" && i + 1 < argc) {
      temperature = std::stof(argv[++i]);
    } else if (arg == "--top-k" && i + 1 < argc) {
      top_k = std::stoi(argv[++i]);
    } else if (arg == "--prefill" && i + 1 < argc) {
      prefill_path = argv[++i];
    } else if (arg == "--decode" && i + 1 < argc) {
      decode_path = argv[++i];
    } else if (arg == "--vision" && i + 1 < argc) {
      vision_path = argv[++i];
    } else if (arg == "--tokenizer" && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (arg == "--embedding" && i + 1 < argc) {
      embedding_path = argv[++i];
    } else if (arg == "--info") {
      show_info = true;
    } else {
      std::cerr << "未知选项: " << arg << "\n";
      printUsage(argv[0]);
      return -1;
    }
  }

  // Set default paths based on model series
  houmo::ModelSeries series = houmo::StringToModelSeries(model_series);

  if (vision_path.empty() && (series == houmo::ModelSeries::kQwen35MLLM ||
                              series == houmo::ModelSeries::kQwen3VLM)) {
    if (series == houmo::ModelSeries::kQwen35MLLM) {
      vision_path = base_path +
                    "/output/xh2/qwen3.5-2b/"
                    "qwen3.5-2b_visual_448x448x2.hmm";
    } else if (series == houmo::ModelSeries::kQwen3VLM) {
      vision_path = base_path +
                    "/output/xh2/qwen3-vl-4b/"
                    "qwen3-vl-4b_visual_448x448x2.hmm";
    }
  }

  // Check if model files exist
  if (!fs::exists(prefill_path)) {
    std::cerr << "Prefill 模型未找到: " << prefill_path << "\n";
    std::cerr << "请设置 HM_ENGINE_PATH 环境变量或使用 --prefill 参数\n";
    return -2;
  }
  if (!fs::exists(decode_path)) {
    std::cerr << "Decode 模型未找到: " << decode_path << "\n";
    return -2;
  }
  if (!fs::exists(embedding_path)) {
    std::cerr << "Embedding 未找到: " << embedding_path << "\n";
    return -2;
  }

  // Configure model
  houmo::ModelConfig config;
  config.devices = {0};
  config.prefill_path = prefill_path;
  config.decode_path = decode_path;
  config.embedding_path = embedding_path;
  config.tokenizer_path = tokenizer_path;
  config.vision_path = vision_path;

  try {
    std::cout << "========================================\n";
    std::cout << "  Houmo Inference Framework v" << houmo::version() << "\n";
    std::cout << "========================================\n\n";

    // Create model using factory (user must specify model type)
    std::unique_ptr<houmo::LLMModel> model;

    if (model_series == "auto") {
      std::cerr << "错误: 必须使用 --model 指定模型类型\n";
      std::cerr << "可用类型: ";
      auto types = houmo::ModelFactory<houmo::LLMModel>::ListRegisteredTypes();
      for (const auto& t : types) {
        std::cerr << t << " ";
      }
      std::cerr << "\n";
      return -3;
    }

    model = houmo::ModelFactory<houmo::LLMModel>::Create(model_series, config);

    if (!model) {
      std::cerr << "错误: 无法创建模型\n";
      return -3;
    }

    std::cout << "模型加载成功!\n";

    // Sampling parameters
    houmo::SamplingParams params;
    params.max_tokens = max_tokens;
    params.temperature = temperature;
    params.top_k = top_k;

    // Enable thinking mode for qwen3_vlm
    bool enable_thinking = (model_series == "qwen3_vlm");

    // Show model information
    if (show_info) {
      printModelInfo(*model);
      return 0;
    }

    // Select feature based on parameters
    if (!images.empty() && model->type() == houmo::ModelType::VLM) {
      // Image understanding
      for (const auto& img : images) {
        if (!fs::exists(img)) {
          std::cerr << "图片未找到: " << img << "\n";
          return -2;
        }
      }
      exampleImageUnderstanding(*model, images, prompt, params, enable_thinking);
    } else if (multi_turn) {
      // Multi-turn dialogue
      exampleMultiTurn(*model, prompt, params, enable_thinking);
    } else {
      // Basic inference
      exampleBasicInference(*model, prompt, params, enable_thinking);
    }

    std::cout << "\n完成!\n";

  } catch (const houmo::Exception& e) {
    std::cerr << "Houmo 错误: " << e.what() << "\n";
    return -3;
  } catch (const std::exception& e) {
    std::cerr << "标准错误: " << e.what() << "\n";
    return -3;
  }

  return 0;
}
